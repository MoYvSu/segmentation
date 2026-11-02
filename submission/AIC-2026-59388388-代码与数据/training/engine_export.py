# -*- coding: utf-8 -*-
"""三模型权重到五个固定输入TensorRT引擎的独立构建入口。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import importlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
from torch.nn import functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = CODE_ROOT / 'training' / 'source'
MODEL_ROOT = CODE_ROOT.parent / 'AIC-2026-59388388-模型文件'
ROLES = ('trunk', 'semantic', 'affinity', 'naf', 'd5a')
FEATURES = ('f0', 'f1', 'f2', 'f3')
FEATURE_SHAPES = ((1, 112, 256, 256), (1, 224, 128, 128),
                  (1, 448, 64, 64), (1, 896, 32, 32))
NAF_BOUNDS = ((1, 1, 256, 256), (16, 1, 256, 256), (16, 1, 256, 256))
WORKSPACE_BYTES = 2 * 2 ** 30


def _activate_source():
    """将随包训练源码明确加入一次导入路径。"""
    source = SOURCE_ROOT.resolve(strict=True)
    if not (source / 'models' / 'backend_adaptation.py').is_file():
        raise FileNotFoundError('缺少training/source/models/backend_adaptation.py')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


@contextmanager
def fp32_runtime():
    """构建样本和等价检查统一采用FP32，退出后恢复调用方设置。"""
    precision = torch.get_float32_matmul_precision()
    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = old


def load_sources(model_dir, *, device='cuda'):
    """完整backend直接严格载入；NAF只恢复网络，不恢复训练状态。"""
    folder = Path(model_dir).resolve(strict=True)
    paths = {name: (folder / (name + '.pt')).resolve(strict=True)
             for name in ('backend', 'restorer', 'naf')}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError('模型目录须包含backend.pt、restorer.pt、naf.pt')
    source = _activate_source()
    sam2 = source / 'segment-anything-2'
    if not (sam2 / 'sam2').is_dir():
        raise FileNotFoundError('缺少随包SAM2源码: training/source/segment-anything-2/sam2')
    backend_api = importlib.import_module('models.backend_adaptation')
    rgb_api = importlib.import_module('models.rgb_restoration')
    boundary_api = importlib.import_module('models.boundary_rgb')
    config = dict(paths=dict(project_root=str(source)),
                  sam2=dict(input_normalization='legacy_none', sam2_repo_path=str(sam2)))
    backend, _ = backend_api.load_backend(str(paths['backend']), config, device)
    if any(getattr(backend, name, None) is not None for name in
           ('semantic_lora', 'geometry_feature_adapter', 'geometry_highres_refiner')):
        raise ValueError('五引擎接口需要共享主干、语义头和affinity头结构')
    restorer = rgb_api.load_rgb_restorer(str(paths['restorer']), device)
    if restorer.num_steps != 16 or restorer.kappa != .03:
        raise ValueError('D5a接口需要16步、kappa=.03的首步恢复器')
    saved = torch.load(paths['naf'], map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get('architecture'), dict):
        raise ValueError('NAF权重缺少architecture')
    naf = boundary_api.BoundaryRGBNAFNet(**saved['architecture'])
    if naf.input_mode != 'boundary' or naf.input_channels != 1:
        raise ValueError('NAF引擎输入固定为单通道二值边界')
    naf.load_state_dict(saved['model_state_dict'], strict=True)
    naf = naf.to(device)
    roots = (backend, restorer, naf)
    for model in roots:
        model.eval().requires_grad_(False)
        for value in model.state_dict().values():
            if value.is_floating_point() and value.dtype != torch.float32:
                raise TypeError('构建源网络须使用FP32张量')
    if sum(p.numel() for model in roots for p in model.parameters()) >= 500_000_000:
        raise ValueError('三模型参数总量须小于500M')
    return dict(trunk=backend.encoder.trunk, semantic=backend.semantic_decoder,
                affinity=backend.affinity_decoder, d5a=restorer, naf=naf)


class Trunk(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, image):
        return tuple(self.module(image))


class Semantic(Trunk):
    def forward(self, f0, f1, f2, f3, image):
        return self.module([f0, f1, f2, f3], image, return_features=True)


class Affinity(Trunk):
    def forward(self, f0, f1, f2, f3):
        result = self.module([f0, f1, f2, f3])
        return result['affinity_logits'], result['affinity_feature']


class NAF(Trunk):
    def forward(self, inputs):
        return self.module.forward_logits(inputs)


class D5a(Trunk):
    def forward(self, xt, condition, t_index):
        return self.module.denoise(xt, condition, t_index)


@contextmanager
def fixed_trunk_position(wrapper):
    """固定1024输入的位置图使用原bicubic结果，保持原算术。"""
    model = wrapper.module
    original = model._get_pos_embed
    with torch.no_grad():
        position = original((256, 256)).detach().clone()
        if not torch.equal(original((256, 256)), position):
            raise RuntimeError('原位置图计算未逐元素复现')
    wrapper.register_buffer('export_position_constant', position)

    def fixed(hw):
        if tuple(int(value) for value in hw) != (256, 256):
            raise ValueError('主干引擎固定1024输入、256位置网格')
        return wrapper.export_position_constant

    with patch.object(model, '_get_pos_embed', new=fixed):
        yield dict(position_precomputed=True, position_torch_equal=True)


@contextmanager
def fixed_naf_channels(model, sample):
    """只将LayerNorm通道维改写为常量，核对B1/B16原logits后导出。"""
    norms = [module for module in model.modules()
             if type(module).__name__ == 'BoundaryChannelLayerNorm']
    if not norms or sample.shape[0] != 16:
        raise ValueError('NAF导出需要原ChannelLayerNorm及B16样本')

    def fixed(owner):
        channels = int(owner.weight.numel())

        def forward(value):
            value = value.permute(0, 2, 3, 1)
            return F.layer_norm(value, (channels,), owner.weight, owner.bias, 1e-6).permute(0, 3, 1, 2)
        return forward

    with torch.no_grad(), torch.autocast(device_type=sample.device.type, enabled=False):
        expected = {batch: model.forward_logits(sample[:batch]) for batch in (1, 16)}
        with ExitStack() as stack:
            for norm in norms:
                stack.enter_context(patch.object(norm, 'forward', new=fixed(norm)))
            if any(not torch.equal(logits, model.forward_logits(sample[:batch]))
                   for batch, logits in expected.items()):
                raise RuntimeError('固定LayerNorm通道后logits发生变化')
            yield dict(channel_shape_constant=True, logits_torch_equal=True, checked_batches=[1, 16])


def synthetic_inputs(modules):
    image = torch.linspace(0, 1, 3 * 1024 * 1024, device='cuda',
                           dtype=torch.float32).reshape(1, 3, 1024, 1024)
    features = tuple(modules['trunk'](image))
    if len(features) != 4 or any(tuple(t.shape) != shape or t.dtype != torch.float32
                                for t, shape in zip(features, FEATURE_SHAPES)):
        raise ValueError('主干四级特征形状与固定推理接口不符')
    axis = torch.arange(256, device='cuda') // 8
    binary = ((axis[:, None] + axis[None, :]) % 2).float()[None, None]
    return dict(trunk=(image,), semantic=(*features, image), affinity=features,
                d5a=(image.clone(), image, torch.full((1,), 16, dtype=torch.int64, device='cuda')),
                naf=(binary.expand(16, 1, 256, 256).contiguous(),))


def export_onnx(role, wrapper, sample, path, onnx):
    names = dict(trunk=['image'], semantic=[*FEATURES, 'image'], affinity=list(FEATURES),
                 naf=['inputs'], d5a=['xt', 'condition', 't_index'])
    outputs = dict(trunk=list(FEATURES), semantic=['semantic_logits', 'semantic_feature'],
                   affinity=['affinity_logits', 'affinity_feature'], naf=['logits'], d5a=['x0'])
    context = (fixed_trunk_position(wrapper) if role == 'trunk' else
               fixed_naf_channels(wrapper.module, sample[0]) if role == 'naf' else nullcontext({}))
    with context as adaptation, torch.no_grad(), torch.autocast('cuda', enabled=False):
        torch.onnx.export(wrapper.eval(), sample, str(path), input_names=names[role],
            output_names=outputs[role], opset_version=17, dynamo=False, do_constant_folding=True,
            external_data=False,
            dynamic_axes={'inputs': {0: 'batch'}, 'logits': {0: 'batch'}} if role == 'naf' else None)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    if role == 'trunk' and any(node.op_type == 'Resize' and any(
            attr.name == 'mode' and attr.s == b'cubic' for attr in node.attribute)
            for node in graph.graph.node):
        raise RuntimeError('主干ONNX仍含TensorRT不支持的cubic Resize')
    return adaptation


def _check_io(role, io):
    feature_shapes = dict(zip(FEATURES, FEATURE_SHAPES))
    inputs = dict(trunk={'image': (1, 3, 1024, 1024)}, semantic=feature_shapes,
                  affinity=feature_shapes, naf={'inputs': (-1, 1, 256, 256)},
                  d5a={'xt': (1, 3, 1024, 1024), 'condition': (1, 3, 1024, 1024), 't_index': (1,)})[role]
    if role == 'semantic' and 'image' in io:
        inputs = dict(inputs, image=(1, 3, 1024, 1024))
    outputs = dict(trunk=feature_shapes,
        semantic={'semantic_logits': (1, 1, 1024, 1024), 'semantic_feature': (1, 256, 256, 256)},
        affinity={'affinity_logits': (1, 8, 512, 512), 'affinity_feature': (1, 128, 512, 512)},
        naf={'logits': (-1, 1, 256, 256)}, d5a={'x0': (1, 3, 1024, 1024)})[role]
    if set(io) != set(inputs) | set(outputs):
        raise ValueError('引擎输入输出名称不符: ' + role)
    for direction, rows in (('INPUT', inputs), ('OUTPUT', outputs)):
        for name, shape in rows.items():
            actual = tuple(io[name]['shape'])
            if io[name]['mode'] != direction or actual != shape:
                raise ValueError('引擎输入输出shape不符: ' + role + '.' + name
                                 + f'; actual={actual}, expected={shape}; '
                                 + f'mode={io[name]["mode"]}, expected_mode={direction}')


def build_engine(role, onnx_path, engine_path, trt):
    if engine_path.exists():
        raise FileExistsError(engine_path)
    logger = builder = network = parser = config = profile = serialized = tensor = getter = None
    runtime = engine = None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError('ONNX解析失败: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        for name in ('TF32', 'FP16', 'INT8'):
            flag = getattr(trt.BuilderFlag, name)
            config.clear_flag(flag)
            if config.get_flag(flag):
                raise RuntimeError('TensorRT精度开关未关闭: ' + name)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
        config.max_aux_streams = 0
        profile_metadata = None
        if role == 'naf':
            profile = builder.create_optimization_profile()
            profile.set_shape('inputs', *NAF_BOUNDS)
            if tuple(map(tuple, profile.get_shape('inputs'))) != NAF_BOUNDS or not bool(profile):
                raise RuntimeError('NAF动态batch profile不符')
            if config.add_optimization_profile(profile) != 0:
                raise RuntimeError('NAF需要单个profile，索引0')
            profile_metadata = dict(min=list(NAF_BOUNDS[0]), opt=list(NAF_BOUNDS[1]), max=list(NAF_BOUNDS[2]))
        io = {}
        for direction, count, getter in (('INPUT', network.num_inputs, network.get_input),
                                          ('OUTPUT', network.num_outputs, network.get_output)):
            for index in range(count):
                tensor = getter(index)
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                expected = trt.int64 if role == 'd5a' and tensor.name == 't_index' else trt.float32
                if tensor.dtype != expected:
                    raise TypeError('引擎I/O类型不符: ' + role + '.' + tensor.name)
                io[tensor.name] = dict(mode=direction, shape=list(tensor.shape))
        _check_io(role, io)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT构建失败: ' + role)
        payload = bytes(serialized)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(payload)
        if engine is None:
            raise RuntimeError('TensorRT引擎反序列化失败: ' + role)
        with engine_path.open('xb') as stream:
            stream.write(payload)
        return dict(file=engine_path.name, profile=profile_metadata, io=io)
    finally:
        engine = runtime = serialized = parser = config = profile = tensor = getter = network = None
        builder = None
        logger = None


def destination(path, model_dir):
    target, source = Path(path).resolve(), Path(model_dir).resolve()
    if target.exists():
        raise FileExistsError('引擎输出目录已存在，请选择新目录: ' + str(target))
    for protected in (source, SOURCE_ROOT.resolve()):
        if target.is_relative_to(protected) or protected.is_relative_to(target):
            raise ValueError('引擎输出目录须与权重及源码目录分离')
    return target


def run(args):
    model_dir = Path(args.model_dir).resolve(strict=True)
    target = destination(args.output_dir, model_dir)
    if not torch.cuda.is_available():
        raise RuntimeError('构建TensorRT引擎需要CUDA GPU')
    trt, onnx = importlib.import_module('tensorrt'), importlib.import_module('onnx')
    target.parent.mkdir(parents=True, exist_ok=True)
    # 临时目录仅由本次构建创建；成功发布时只移出五个engine和清单。
    with tempfile.TemporaryDirectory(prefix='.engine-build-', dir=target.parent) as work:
        work = Path(work)
        stage = work / 'engines'
        stage.mkdir()
        with fp32_runtime():
            modules = load_sources(model_dir)
            samples = synthetic_inputs(modules)
            wrappers = dict(trunk=Trunk, semantic=Semantic, affinity=Affinity, naf=NAF, d5a=D5a)
            records = {}
            for role in ROLES:
                print('BUILD_ENGINE', role, flush=True)
                onnx_path = work / (role + '.onnx')
                export_onnx(role, wrappers[role](modules[role]), samples[role], onnx_path, onnx)
                records[role] = build_engine(role, onnx_path, stage / (role + '.engine'), trt)
        manifest = dict(format='segmentation_engines_v1', precision='float32',
                        trt_flags=dict(TF32=False, FP16=False, INT8=False), engines=records)
        (stage / 'engines.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + '\n', encoding='utf-8')
        if target.exists():
            raise FileExistsError('输出目录已出现，请选择新目录: ' + str(target))
        stage.rename(target)
    return dict(engine_dir=str(target), manifest=str(target / 'engines.json'), roles=list(ROLES))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-dir', default=str(MODEL_ROOT / 'weights'),
                        help='含backend.pt、restorer.pt、naf.pt的完整权重目录')
    parser.add_argument('--output-dir', default=str(MODEL_ROOT / 'engines'),
                        help='新引擎目录；已有目录保留')
    return parser.parse_args(argv)


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result
