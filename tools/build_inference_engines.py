# -*- coding: utf-8 -*-
"""Build the fixed old-C FP32 TensorRT engines; this public builder has CPU-only authoring tests.

显式build才使用GPU/TensorRT；不读图片/cache，不训练，不宣称最终输出或官方得分
已验证。四模块固定B1/1024，NAF profile固定1/16/16；成功发布engines.json。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.boundary_naf import BoundaryChannelLayerNorm
from tools import infer_compact as app
from tools import infer_accelerated as accelerated

ROLES = accelerated.ROLES
NAF_BOUNDS = ((1, 1, 256, 256), (16, 1, 256, 256), (16, 1, 256, 256))
WORKSPACE_BYTES = 2 * 2 ** 30


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def destination(path):
    target = Path(path).resolve()
    if not any(target != base and target.is_relative_to(base)
               for base in (ROOT / 'output', ROOT / 'outputs')):
        raise ValueError('Output must be a new subdirectory inside this project output/ or outputs/')
    if target.exists():
        raise FileExistsError('Preserve existing output; choose a fresh directory: ' + str(target))
    return target


class Trunk(torch.nn.Module):
    def __init__(self, module):
        super().__init__(); self.module = module
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
    """保留原bicubic的真实结果，只把固定1024所需位置图作为常量导出。"""
    module = wrapper.module
    original = module._get_pos_embed
    with torch.no_grad():
        position = original((256, 256)).detach().clone()
        if not torch.equal(original((256, 256)), position):
            raise RuntimeError('Original fixed-grid position embedding did not reproduce exactly')
    wrapper.register_buffer('export_position_constant', position)
    metadata = dict(adaptation='original_exact_bicubic_position_precomputed_for_static1024',
        position_torch_equal=True, position_shape=list(position.shape),
        position_sha256=hashlib.sha256(position.cpu().contiguous().numpy().tobytes()).hexdigest())

    def fixed(hw):
        if tuple(int(value) for value in hw) != (256, 256):
            raise ValueError('Only the fixed 1024 input / 256 position grid is supported')
        return wrapper.export_position_constant

    try:
        with patch.object(module, '_get_pos_embed', new=fixed):
            yield metadata
    finally:
        metadata['original_method_restored'] = module._get_pos_embed == original


@contextmanager
def fixed_naf_channels(model, sample):
    """只固定LayerNorm的原通道数，保留算子/eps/权重并严格核logits。"""
    norms = [module for module in model.modules() if isinstance(module, BoundaryChannelLayerNorm)]
    if not norms or sample.shape[0] != 16:
        raise ValueError('Expected current NAF LayerNorm modules and a batch16 export sample')
    before = app.entry.tensor_digest(model.state_dict().items())
    originals = [(module, module.forward) for module in norms]
    metadata = dict(adaptation='fixed_original_LayerNorm_channel_shape',
                    adaptation_logits_torch_equal=False, checked_batches=[1, 16])

    def fixed(owner):
        channels = int(owner.weight.numel())
        def forward(value):
            value = value.permute(0, 2, 3, 1)
            return F.layer_norm(value, (channels,), owner.weight, owner.bias, 1e-6).permute(0, 3, 1, 2)
        return forward

    try:
        with torch.no_grad(), torch.autocast(device_type=sample.device.type, enabled=False):
            expected = {batch: model.forward_logits(sample[:batch]) for batch in (1, 16)}
            with ExitStack() as stack:
                for norm in norms:
                    stack.enter_context(patch.object(norm, 'forward', new=fixed(norm)))
                for batch, logits in expected.items():
                    if not torch.equal(logits, model.forward_logits(sample[:batch])):
                        raise RuntimeError('NAF fixed-channel adaptation changed original logits')
                metadata['adaptation_logits_torch_equal'] = True
                yield metadata
    finally:
        metadata['original_methods_restored'] = all(module.forward == method for module, method in originals)
        if app.entry.tensor_digest(model.state_dict().items()) != before:
            raise RuntimeError('NAF state changed during export adaptation')


def _load_sources(args):
    config_path = Path(args.config).resolve(strict=True)
    config = app.entry.load_config(str(config_path))
    config['paths']['project_root'] = str(app.entry.ROOT)
    app.zero.validate_baseline(config)
    app.flat.execution_config(config)
    model_dir = Path(args.model_dir).resolve() if args.model_dir else app.entry.ROOT / 'model'
    paths = app.entry.model_paths(config, model_dir)
    app.entry.configure_runtime()
    backend, restorer, terminal = app.entry.load_models(config, paths, 'cuda')
    if (terminal is not None or backend.semantic_lora is not None
            or backend.geometry_feature_adapter is not None or backend.geometry_highres_refiner is not None):
        raise ValueError('Builder requires the unchanged shared old-C backend')
    naf, provenance = app.naf.load_naf_checkpoint(args.naf_checkpoint, 'cuda',
                                                expected_model_sha=app.naf.EXPECTED_MODEL_SHA)
    modules = dict(trunk=backend.encoder.trunk, semantic=backend.semantic_decoder,
                   affinity=backend.affinity_decoder, d5a=restorer, naf=naf)
    if any(module.training or any(parameter.requires_grad for parameter in module.parameters())
           for root in modules.values() for module in root.modules()):
        raise RuntimeError('All source modules must be frozen and eval')
    files = dict(paths, naf=Path(args.naf_checkpoint).resolve(strict=True))
    source = dict(config_sha256=app.entry.sha(config_path),
                  model_file_sha256={name: app.entry.sha(path) for name, path in files.items()},
                  NAF_state_sha256=provenance['model_sha256'])
    return modules, source, files, config_path


def synthetic_inputs(modules):
    """独立于图像/缓存的固定ABI样本；不生成随机噪声。"""
    image = torch.linspace(0, 1, 3 * 1024 * 1024, dtype=torch.float32, device='cuda').reshape(1, 3, 1024, 1024)
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        features = tuple(modules['trunk'](image))
    if len(features) != 4:
        raise ValueError('Expected four original Hiera features')
    axis = torch.arange(256, device='cuda') // 8
    binary = ((axis[:, None] + axis[None, :]) % 2).float()[None, None]
    naf_input = binary.expand(16, 1, 256, 256).contiguous()
    return dict(trunk=(image,), semantic=(*features, image), affinity=features,
                d5a=(image.clone(), image, torch.full((1,), 16, dtype=torch.int64, device='cuda')),
                naf=(naf_input,))


def export_onnx(role, wrapper, sample, path, onnx):
    names = dict(trunk=['image'], semantic=[*accelerated.FEATURES, 'image'],
                 affinity=list(accelerated.FEATURES), naf=['inputs'], d5a=['xt', 'condition', 't_index'])
    outputs = dict(trunk=list(accelerated.FEATURES), semantic=['semantic_logits', 'semantic_feature'],
                   affinity=['affinity_logits', 'affinity_feature'], naf=['logits'], d5a=['x0'])
    context = (fixed_trunk_position(wrapper) if role == 'trunk' else
               fixed_naf_channels(wrapper.module, sample[0]) if role == 'naf' else nullcontext(dict(adaptation='none')))
    runtime = app.naf.naf_runtime_context() if role == 'naf' else nullcontext()
    started = time.monotonic()
    with runtime, context as adaptation, torch.no_grad(), torch.autocast('cuda', enabled=False):
        torch.onnx.export(wrapper.eval(), sample, str(path), input_names=names[role], output_names=outputs[role],
            opset_version=17, dynamo=False, do_constant_folding=True, external_data=False,
            dynamic_axes={'inputs': {0: 'batch'}, 'logits': {0: 'batch'}} if role == 'naf' else None)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    if role == 'trunk' and any(node.op_type == 'Resize' and any(
            attr.name == 'mode' and attr.s == b'cubic' for attr in node.attribute) for node in graph.graph.node):
        raise RuntimeError('Trunk ONNX still contains unsupported cubic Resize')
    return dict(adaptation, onnx_sha256=app.entry.sha(path), onnx_file=path.name,
                onnx_nodes=len(graph.graph.node), export_seconds=time.monotonic() - started,
                opset=17, dynamo=False, external_data=False,
                sample=[dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype)) for t in sample])


def build_engine(role, onnx_path, engine_path, trt):
    if engine_path.exists():
        raise FileExistsError(engine_path)
    logger = builder = network = parser = config = profile = serialized = tensor = getter = None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError('TensorRT ONNX parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        for name in ('TF32', 'FP16', 'INT8'):
            flag = getattr(trt.BuilderFlag, name)
            config.clear_flag(flag)
            if config.get_flag(flag):
                raise RuntimeError('TensorRT precision flag remained enabled: ' + name)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
        config.max_aux_streams = 0
        profile_metadata = None
        if role == 'naf':
            if (network.num_inputs != 1 or network.get_input(0).name != 'inputs'
                    or tuple(network.get_input(0).shape) != (-1, 1, 256, 256)):
                raise ValueError('Expected NAF input dynamic batch x 1 x 256 x 256')
            profile = builder.create_optimization_profile()
            returned = profile.set_shape('inputs', *NAF_BOUNDS)
            if tuple(map(tuple, profile.get_shape('inputs'))) != NAF_BOUNDS or not bool(profile):
                raise RuntimeError('TensorRT profile min/opt/max differs')
            if config.add_optimization_profile(profile) != 0:
                raise RuntimeError('Expected a single NAF optimization profile at index 0')
            profile_metadata = dict(min=list(NAF_BOUNDS[0]), opt=list(NAF_BOUNDS[1]), max=list(NAF_BOUNDS[2]),
                                    get_shape_verified=True, set_shape_return_repr=repr(returned))
        io = {}
        for direction, count, getter in (('INPUT', network.num_inputs, network.get_input),
                                          ('OUTPUT', network.num_outputs, network.get_output)):
            for index in range(count):
                tensor = getter(index)
                tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
                expected = trt.int64 if role == 'd5a' and tensor.name == 't_index' else trt.float32
                if tensor.dtype != expected:
                    raise TypeError('Unexpected engine runtime I/O dtype: ' + tensor.name)
                if role != 'naf' and any(int(size) <= 0 for size in tensor.shape):
                    raise ValueError('Non-NAF engines require fully static tensor shapes')
                io[tensor.name] = dict(mode=direction, shape=list(tensor.shape), dtype=str(tensor.dtype), format='LINEAR')
        started = time.monotonic()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT engine construction returned None: ' + role)
        with engine_path.open('xb') as stream:
            stream.write(bytes(serialized))
        return dict(file=engine_path.name, sha256=app.entry.sha(engine_path), bytes=engine_path.stat().st_size,
                    build_seconds=time.monotonic() - started, io=io, profile=profile_metadata,
                    TensorRT=str(trt.__version__), precision='float32', TF32=False, FP16=False, INT8=False,
                    workspace_bytes=WORKSPACE_BYTES, max_aux_streams=0)
    finally:
        # 先销毁子对象和绑定方法，再释放builder/logger；失败路径同样执行。
        serialized = parser = config = profile = tensor = getter = network = None
        builder = None
        logger = None


def run(args):
    target = destination(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError('Explicit engine build requires CUDA; --help and CPU tests do not')
    trt, onnx = importlib.import_module('tensorrt'), importlib.import_module('onnx')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '.building-', dir=target.parent))
    started = time.monotonic()
    try:
        modules, source, source_files, config_path = _load_sources(args)
        before = {role: app.entry.tensor_digest(module.state_dict().items()) for role, module in modules.items()}
        samples = synthetic_inputs(modules)
        classes = dict(trunk=Trunk, semantic=Semantic, affinity=Affinity, naf=NAF, d5a=D5a)
        engines = {}
        for role in ROLES:
            _write(stage / 'build_state.json', dict(complete=False, phase='building', role=role))
            metadata = export_onnx(role, classes[role](modules[role]), samples[role], stage / (role + '.onnx'), onnx)
            built = build_engine(role, stage / (role + '.onnx'), stage / (role + '.engine'), trt)
            engines[role] = dict(built, source_state_sha256=before[role], onnx=metadata)
            _write(stage / 'builds.json', engines)
        if any(app.entry.tensor_digest(module.state_dict().items()) != before[role] for role, module in modules.items()):
            raise RuntimeError('Original model states changed during engine construction')
        if (app.entry.sha(config_path) != source['config_sha256'] or any(
                app.entry.sha(path) != source['model_file_sha256'][name] for name, path in source_files.items())):
            raise RuntimeError('Source model/config files changed during construction')
        manifest = dict(format=accelerated.MANIFEST_FORMAT, complete=True, cpu_integer_readout=True,
            precision='float32', trt_flags=dict(TF32=False, FP16=False, INT8=False), engines=engines,
            source=source, versions=dict(torch=torch.__version__, TensorRT=str(trt.__version__),
                onnx=str(onnx.__version__), CUDA=torch.version.cuda,
                GPU_name=torch.cuda.get_device_name(), GPU_compute_capability=list(torch.cuda.get_device_capability())),
            fixed_profiles=dict(network_input=[1, 3, 1024, 1024], NAF_batch=dict(min=1, opt=16, max=16)),
            build=dict(complete=True, weights_unchanged=True, no_training=True, no_input_images=True,
                no_random_restoration_noise=True, full_inference_verified=False,
                wall_seconds=time.monotonic() - started, builder_sha256=app.entry.sha(__file__),
                at_utc=datetime.now(timezone.utc).isoformat()))
        _write(stage / 'engines.json', manifest)
        accelerated.load_manifest(stage / 'engines.json')
        _write(stage / 'build_state.json', dict(complete=True, phase='complete', roles=list(ROLES)))
        manifest_sha = app.entry.sha(stage / 'engines.json')
        if target.exists():
            raise FileExistsError('Output appeared during build; preserved staging directory')
        stage.rename(target)
        return dict(complete=True, manifest=str(target / 'engines.json'),
                    manifest_sha256=manifest_sha, build=manifest['build'])
    except Exception as exc:
        _write(stage / 'build_failed.json', dict(complete=False, error=repr(exc), traceback=traceback.format_exc()))
        _write(stage / 'build_state.json', dict(complete=False, phase='failed', error=repr(exc)))
        # 出错时不保留任何看似已发布的成功manifest；中间engine/ONNX保留诊断。
        manifest_path = stage / 'engines.json'
        if manifest_path.exists():
            manifest_path.rename(stage / 'unpublished_engines.json')
        print('Build failed; diagnostics preserved at ' + str(stage), file=sys.stderr, flush=True)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build',), help='explicit GPU engine build; no automatic inference')
    parser.add_argument('--config', default=str(ROOT / 'config/inference/area_floor_200.yaml'))
    parser.add_argument('--model-dir')
    parser.add_argument('--naf-checkpoint', required=True)
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/inference_engines/current_C'),
                        help='fresh project output/ or outputs/ subdirectory; never overwritten')
    return parser.parse_args(argv)


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
