# -*- coding: utf-8 -*-
"""原生1024部署窗的零噪D5a语义头接续训练；冻结基座，仅保存语义增量。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import backend_adaptation
from models.rgb_restoration import load_rgb_restorer
from train_direct_semantic_affinity import build_semantic_criterion, compute_semantic_loss
from utils import restoration_noise, restoration_probe
from utils.config import load_config, project_path

FORMAT = 'semantic_native_zero_delta_v1'
BACKEND_SHA = '13b3ab1bc509fa45848e2eb8a40ee458b178fe747fc83eb65333c527a8161f04'
RESTORER_SHA = '9315da872ead66735f799e27f85eb97f8c2c265d44d64c6f8a271be50417007b'
HEAD_PARAMETERS = 6_302_010
FIXED = dict(epochs=20, expected_sources=32, draws_per_source=4, batch_size=1,
             learning_rate=3e-5, minimum_lr_ratio=.1, warmup_epochs=2,
             weight_decay=1e-4, adam_eps=1e-4, grad_clip=1., num_workers=0,
             expected_trainable_parameters=HEAD_PARAMETERS, input_size=1024, overlap=.25,
             restoration_noise='explicit_zero_v1', augmentation='d4_only',
             train_precision='bfloat16', frozen_precision='float32', inference_precision='float32',
             minimum_free_gib=10., monitor_epochs=[0, 1, 5, 10, 15, 20])
LOSS = dict(dice_weight=.30, instance_weight=.75, core_radius=3, core_min_pixels=12,
            core_boundary_threshold=.20, class_balance=True, ferrite_weight=1.,
            hard_gamma=.5, hard_floor=.50, pool_weight=.50, thin_weight=1.50)
RESERVE_BYTES = 10 * 1024**3


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_sha(named_tensors):
    digest = hashlib.sha256()
    for name, value in sorted(named_tensors):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_sha(model):
    return tensor_sha((name, value) for name, value in model.state_dict().items()
                      if not name.startswith('semantic_decoder.'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8')


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def forbid_noise(*args, **kwargs):
    raise RuntimeError('训练恢复仅允许显式零噪声；禁止随机噪声与旧restore_first路径')


@contextmanager
def no_restoration_rng():
    # 普通函数替换，不生成MagicMock调用历史；仅覆盖恢复/encoder区间，不禁语义dropout。
    with ExitStack() as stack:
        for target, name in ((torch, 'randn'), (torch, 'randn_like'), (torch, 'normal'),
                             (restoration_noise, 'restoration_noise'),
                             (restoration_probe, 'restoration_noise'),
                             (backend_adaptation, 'restoration_noise'),
                             (backend_adaptation, 'restore_first')):
            stack.enter_context(patch.object(target, name, new=forbid_noise))
        yield


def validate_config(config):
    cfg = config['semantic_native_zero']
    for key, expected in FIXED.items():
        if cfg.get(key, expected) != expected:
            raise ValueError(f'固定训练合同 {key}={expected!r}，收到 {cfg[key]!r}')
    supplied_loss = config['direct_semantic_affinity']['semantic_loss']
    if any(supplied_loss.get(key) != expected for key, expected in LOSS.items()):
        raise ValueError('semantic_loss必须显式保留旧完整pixel/core/pool系数')
    if cfg.get('precision', 'BF16') not in ('BF16', 'bf16'):
        raise ValueError('语义头训练固定BF16；D5a/encoder与loss固定FP32')
    result = {**FIXED, **cfg, 'seed': int(cfg.get('seed', 20261031)), 'precision': 'BF16'}
    for key in ('output_dir', 'checkpoint_dir'):
        result.pop(key, None)
    return result


def learning_rate(epoch):
    if not 1 <= epoch <= 20:
        raise ValueError('epoch must be 1..20')
    factor = epoch / 2 if epoch <= 2 else .1 + .9 * (1 + math.cos(math.pi * (epoch-3) / 17)) / 2
    return 3e-5 * factor


def freeze_contract(model, restorer, *, expected_parameters=HEAD_PARAMETERS):
    for name in ('semantic_lora', 'geometry_feature_adapter', 'geometry_highres_refiner'):
        if getattr(model, name, None) is not None:
            raise ValueError(f'当前基座要求 {name}=None')
    model.eval().float().requires_grad_(False)
    restorer.eval().float().requires_grad_(False)
    model.encoder.trainable_lora = False
    model.semantic_decoder.requires_grad_(True)
    count = sum(p.numel() for p in model.semantic_decoder.parameters())
    if count != expected_parameters:
        raise ValueError(f'完整语义头参数数目不符: {count} != {expected_parameters}')
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(not name.startswith('semantic_decoder.') for name in trainable):
        raise RuntimeError('语义头外出现可训练参数')
    return dict(trainable_parameters=count, trainable_names=trainable,
                semantic_submodules={name: sum(p.numel() for p in module.parameters())
                                     for name, module in model.semantic_decoder.named_children()},
                frozen_backend_sha256=frozen_sha(model),
                frozen_restorer_sha256=tensor_sha(restorer.state_dict().items()),
                head_buffer_names=list(dict(model.semantic_decoder.named_buffers())),
                naf_loaded=False, rgb_residual_input='same_restored_RGB_as_encoder',
                restoration='first_x0_zero', encoder_dtype='FP32', decoder_autocast='BF16', loss_dtype='FP32')


def assert_frozen(model, restorer, initial):
    if frozen_sha(model) != initial['frozen_backend_sha256']:
        raise RuntimeError('冻结backend参数或buffer发生变化')
    if tensor_sha(restorer.state_dict().items()) != initial['frozen_restorer_sha256']:
        raise RuntimeError('冻结D5a参数或buffer发生变化')
    if any(module.training for name, module in model.named_modules()
           if name and not name.startswith('semantic_decoder')):
        raise RuntimeError('冻结模块进入train模式')
    if restorer.training:
        raise RuntimeError('D5a进入train模式')
    if any(p.grad is not None for name, p in model.named_parameters()
           if not name.startswith('semantic_decoder.')):
        raise RuntimeError('冻结参数产生梯度')


def train_mode(model):
    model.eval()
    model.semantic_decoder.train()


@torch.no_grad()
def restored_features(model, restorer, image):
    if image.dtype != torch.float32 or tuple(image.shape) != (1, 3, 1024, 1024):
        raise ValueError('训练必须是batch1原生1024窗FP32 RGB')
    audit = {}
    with no_restoration_rng(), torch.autocast(device_type=image.device.type, enabled=False):
        restored, _ = restoration_probe.restore_probe(restorer, image, 0, mode='zero', audit=audit)
        features = model.encoder(restored)
    if (audit.get('noise_nonzero') != 0 or audit.get('initial_equals_condition') is not True
            or audit.get('q_sample_calls') != 1 or audit.get('initial_max_abs') != 0):
        raise RuntimeError('D5a零噪声实际执行证据失败')
    if restored.dtype != torch.float32 or any(value.dtype != torch.float32 for value in features):
        raise RuntimeError('D5a/encoder必须保持FP32')
    return restored, features, audit


def single_batch(sample, device):
    # 不构造DataLoader iterator：构造iterator本身会消耗全局torch RNG，破坏中途恢复的dropout。
    batch = default_collate([{key: sample[key] for key in (
        'image', 'semantic_target', 'semantic_boundary', 'semantic_instance_map')}])
    return {key: value.to(device) for key, value in batch.items()}


def semantic_update(model, restorer, criterion, optimizer, sample, device):
    batch = single_batch(sample, device)
    optimizer.zero_grad(set_to_none=True)
    restored, features, noise = restored_features(model, restorer, batch['image'])
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model.semantic_decoder(features, restored)
    with torch.autocast(device_type=device.type, enabled=False):
        logits = logits.float()
        if logits.shape[-2:] != batch['semantic_target'].shape[-2:]:
            logits = F.interpolate(logits, size=batch['semantic_target'].shape[-2:],
                                   mode='bilinear', align_corners=True)
        loss = compute_semantic_loss(criterion, logits, batch)
    if not torch.isfinite(loss):
        raise RuntimeError('非有限semantic loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.semantic_decoder.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), grad_norm_before_clip=float(norm), noise=noise,
                input_sha256=tensor_sha([('image', batch['image'])]),
                restored_sha256=tensor_sha([('image', restored)]),
                logits_sha256=tensor_sha([('logits', logits)]))


def source_seal():
    # 覆盖实际项目算术与目标生成依赖；相对路径使同一源码快照可跨目录核验。
    files = {Path(__file__).resolve(), ROOT/'train_direct_semantic_affinity.py'}
    for directory in ('models', 'utils', 'data'):
        files.update((ROOT/directory).rglob('*.py'))
    return {str(path.relative_to(ROOT)).replace('\\', '/'): file_sha(path) for path in sorted(files)}


def config_seal(path):
    import yaml
    result = {}
    path = Path(path).resolve()
    while True:
        if str(path) in result:
            raise ValueError('循环config _base')
        result[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name] = file_sha(path)
        base = yaml.safe_load(path.read_text(encoding='utf-8')).get('_base')
        if not base:
            break
        path = (path.parent / base).resolve()
    return result


def configure_runtime(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                numpy=np.__version__, opencv=cv2.__version__, gpu=torch.cuda.get_device_name(0),
                threads=4, cudnn_benchmark=False, cudnn_deterministic=True, tf32=False,
                deterministic_algorithms=True, cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                decoder_train_precision='BF16', frozen_precision='FP32', monitor_precision='FP32')


def ensure_space(paths, *, checkpoint_bytes=0):
    # 原子替换保留旧last，同时需要临时新文件；逐个挂载点检查，避免假定两目录同盘。
    for path in paths:
        free = shutil.disk_usage(path).free
        if free - checkpoint_bytes < RESERVE_BYTES:
            raise RuntimeError(f'磁盘保留10GiB不足: {path}, free={free}, planned={checkpoint_bytes}')


def save_delta(path, model, optimizer, identity, progress, *, output_dir):
    state = {key: value.detach().cpu().clone() for key, value in model.semantic_decoder.state_dict().items()}
    bundle = dict(format=FORMAT, identity=deepcopy(identity), progress=deepcopy(progress),
                  semantic_decoder=state, semantic_sha256=tensor_sha(state.items()),
                  optimizer=optimizer.state_dict(), rng=rng_state())
    estimate = max(100_000_000, HEAD_PARAMETERS * 4 * 4)
    ensure_space([Path(path).parent], checkpoint_bytes=estimate)
    ensure_space([output_dir])
    temporary = Path(path).with_suffix('.tmp')
    if temporary.exists():
        raise FileExistsError(f'保留未完成checkpoint，拒绝覆盖: {temporary}')
    try:
        torch.save(bundle, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path),
                semantic_sha256=bundle['semantic_sha256'], completed_updates=progress['completed_updates'])


def load_head_delta(model, path, *, expected_backend_sha256=BACKEND_SHA):
    """部署用增量加载接口；只替换语义头并返回不可混淆的严格加载回执。"""
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle.get('format') != FORMAT:
        raise ValueError('不支持的语义增量格式')
    identity = bundle['identity']
    if identity['base_backend_file_sha256'] != expected_backend_sha256:
        raise ValueError('语义增量的不可变backend身份不符')
    if frozen_sha(model) != identity['frozen']['frozen_backend_sha256']:
        raise ValueError('实际载入模型的非语义基座与增量不同')
    initial = tensor_sha(model.semantic_decoder.state_dict().items())
    before = frozen_sha(model)
    if tensor_sha(bundle['semantic_decoder'].items()) != bundle['semantic_sha256']:
        raise ValueError('语义增量tensor摘要损坏')
    model.semantic_decoder.load_state_dict(bundle['semantic_decoder'], strict=True)
    loaded = tensor_sha(model.semantic_decoder.state_dict().items())
    after = frozen_sha(model)
    if loaded != bundle['semantic_sha256'] or before != after:
        raise RuntimeError('增量加载未保持完整tensor/frozen合同')
    return dict(format=FORMAT, checkpoint_file_sha256=file_sha(path), strict_load=True,
                completed_updates=bundle['progress']['completed_updates'],
                base_backend_file_sha256=identity['base_backend_file_sha256'],
                base_semantic_sha256=identity['base_semantic_sha256'],
                semantic_before_sha256=initial, loaded_semantic_sha256=loaded,
                nonsemantic_before_sha256=before, nonsemantic_after_sha256=after,
                architecture=identity['semantic_architecture'])


@torch.no_grad()
def monitor(model, restorer, dataset, output, epoch):
    destination = Path(output)/'monitor'/f'e{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    device = next(model.parameters()).device
    was_training = model.semantic_decoder.training
    with preserve_rng():
        model.eval()
        try:
            for sample in dataset.monitor_samples():
                batch = single_batch(sample, device)
                restored, features, noise = restored_features(model, restorer, batch['image'])
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model.semantic_decoder(features, restored).float()
                    if logits.shape[-2:] != batch['semantic_target'].shape[-2:]:
                        logits = F.interpolate(logits, size=batch['semantic_target'].shape[-2:],
                                               mode='bilinear', align_corners=True)
                    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
                rgb = np.rint(restored[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                target = sample['semantic_target'][0].numpy()
                colour = np.zeros((*target.shape, 3), dtype=np.uint8)
                colour[:] = (96, 96, 96)
                colour[target == 0] = (226, 125, 70)
                colour[target == 1] = (64, 190, 244)
                heat = cv2.cvtColor(cv2.applyColorMap(np.rint(probability*255).astype(np.uint8),
                                                     cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                panels = []
                for image, label in ((rgb, 'zero D5a RGB'), (colour, 'GT P/F; grey ignore'), (heat, 'P(F) 0..1')):
                    panel = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                    panel = cv2.copyMakeBorder(panel, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
                    cv2.putText(panel, label, (4,16), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1, cv2.LINE_AA)
                    panels.append(panel)
                name = Path(sample.get('source', sample['name'])).stem
                image_path = destination/f'{name}.png'
                if not cv2.imwrite(str(image_path), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)):
                    raise IOError(f'训练monitor写入失败: {image_path}')
                rows.append(dict(source=name, file=image_path.name, file_sha256=file_sha(image_path), noise=noise))
        finally:
            model.semantic_decoder.train(was_training)
    write_json(destination/'report.json', dict(epoch=epoch, training_sources_only=True,
               deployment_validation=False, precision='FP32_eval', samples=rows))
    return dict(epoch=epoch, sources=[row['source'] for row in rows])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'backend', 'restorer', 'raw-dir', 'gt-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--smoke-updates', type=int, default=None,
                        help='全局终止更新数；resume --smoke-updates 3在step3停止，不是再加3步')
    parser.add_argument('--resume', default=None)
    return parser.parse_args(argv)


def run(args):
    from data.semantic_native_zero import NativeSemanticDataset
    config = load_config(args.config)
    cfg = validate_config(config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('正式训练需要sam2_env中的BF16 CUDA；不以CPU结果替代')
    backend_path, restorer_path = (Path(project_path(config, value)) for value in (args.backend, args.restorer))
    if file_sha(backend_path) != BACKEND_SHA or file_sha(restorer_path) != RESTORER_SHA:
        raise ValueError('必须使用严格绑定的当前13b3 backend及9315 D5a导出')
    output = Path(args.output).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else output/'checkpoints'
    if args.smoke_updates is not None and args.smoke_updates <= 0:
        raise ValueError('--smoke-updates必须大于0，且不会改变完整20轮学习率计划')
    if not args.resume:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('新实验output必须为空，拒绝覆盖已有产物')
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError('新实验checkpoint-dir必须为空')
    elif not (output/'identity.json').is_file():
        raise FileNotFoundError('resume必须使用原output及其identity.json')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ensure_space([output, checkpoint_dir])
    runtime = configure_runtime(cfg['seed'])
    device = torch.device('cuda:0')
    dataset = NativeSemanticDataset(project_path(config, args.raw_dir), project_path(config, args.gt_dir),
        seed=cfg['seed'], expected_sources=32, epochs=20, draws_per_source=4,
        monitor_sources=cfg.get('monitor_sources'))
    if len(dataset) != 128 or dataset.identity()['total_updates'] != 2560:
        raise ValueError('数据合同必须32源×4窗×20轮=2560次更新')
    model, bundle = backend_adaptation.load_backend(str(backend_path), config, device)
    restorer = load_rgb_restorer(str(restorer_path), device)
    if restorer.num_steps != 16 or restorer.kappa != .03:
        raise ValueError('当前D5a首步恢复schedule不符')
    frozen = freeze_contract(model, restorer)
    criterion = build_semantic_criterion(config, device)
    optimizer = torch.optim.AdamW(model.semantic_decoder.parameters(), lr=learning_rate(1),
                                 weight_decay=1e-4, eps=1e-4, foreach=False)
    effective = deepcopy(config)
    effective['paths']['project_root'] = '<runtime_project_root>'
    for key in ('output_dir', 'checkpoint_dir'):
        effective.get('semantic_native_zero', {}).pop(key, None)
    identity = dict(format=FORMAT, recipe=cfg, effective_config=effective,
        config_source_seal=config_seal(args.config), source_seal=source_seal(), runtime=runtime,
        data=dataset.identity(), data_source_seal=dataset.source_seal(),
        base_backend_file_sha256=BACKEND_SHA, base_restorer_file_sha256=RESTORER_SHA,
        base_semantic_sha256=tensor_sha(model.semantic_decoder.state_dict().items()),
        semantic_architecture=bundle['architecture']['semantic_decoder'], frozen=frozen,
        loss=deepcopy(LOSS), monitoring=dataset.monitor_manifest(),
        training_input='native1024_deployment_grid_zeroD5a_first_x0_online_D4',
        optimizer='AdamW_foreach_false', full150_inference=False, total_updates=2560)
    identity = json.loads(json.dumps(identity, ensure_ascii=False, allow_nan=False))
    progress = dict(completed_updates=0, next_epoch=1, next_index=0, monitors=[],
                    training_zero_calls=0, training_encoder_calls=0, training_semantic_calls=0,
                    affinity_calls=0, monitor_zero_calls=0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (checkpoint.get('format') != FORMAT or checkpoint['identity'] != identity
                or json.loads((output/'identity.json').read_text(encoding='utf-8')) != identity):
            raise ValueError('resume的源码/配置/数据/模型/运行环境身份不符')
        receipt = load_head_delta(model, args.resume)
        optimizer.load_state_dict(checkpoint['optimizer'])
        progress = deepcopy(checkpoint['progress'])
        completed = progress['completed_updates']
        if not 0 <= completed <= 2560 or (progress['next_epoch'], progress['next_index']) != (completed//128+1, completed%128):
            raise ValueError('checkpoint更新游标不一致')
        # resume只接受最后一次安全写盘的游标，拒绝从旧epoch覆盖已有较新日志。
        log_path = output/'updates.jsonl'
        if log_path.exists():
            rows = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line]
            if len(rows) != completed or (rows and rows[-1]['step'] != completed):
                raise ValueError('日志与checkpoint更新数不同；保留证据，拒绝静默覆盖较新更新')
        write_json(output/'resume_receipt.json', receipt)
        restore_rng(checkpoint['rng'])
        del checkpoint
    else:
        write_json(output/'identity.json', identity)
        write_json(output/'data_manifest.json', dataset.manifest())
        write_json(output/'runtime_paths.json', dict(backend=str(backend_path), restorer=str(restorer_path),
                   raw_dir=str(Path(args.raw_dir).resolve()), gt_dir=str(Path(args.gt_dir).resolve()),
                   output=str(output), checkpoint_dir=str(checkpoint_dir)))
        result = monitor(model, restorer, dataset, output, 0)
        progress['monitors'].append(result)
        progress['monitor_zero_calls'] += len(result['sources'])
        save_delta(checkpoint_dir/'initial.pt', model, optimizer, identity, progress, output_dir=output)
    # 实际模块hook计数，普通闭包不保留输入/feature/logit引用。
    calls = dict(encoder=0, semantic=0, affinity=0)
    def count(name):
        def hook(module, inputs, result):
            calls[name] += 1
        return hook
    handles = [module.register_forward_hook(count(name)) for name, module in (
        ('encoder', model.encoder), ('semantic', model.semantic_decoder), ('affinity', model.affinity_decoder))]
    start = progress['completed_updates']
    stop = min(2560, args.smoke_updates) if args.smoke_updates else 2560
    if stop < start:
        raise ValueError('--smoke-updates全局终止数小于checkpoint已完成更新数')
    started = time.time()
    try:
        with (output/'updates.jsonl').open('a', encoding='utf-8', buffering=1) as log:
            while progress['completed_updates'] < stop:
                step = progress['completed_updates']
                epoch, index = step//128+1, step%128
                dataset.set_epoch(epoch)
                train_mode(model)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(epoch)
                before = dict(calls)
                sample = dataset[index]
                metrics = semantic_update(model, restorer, criterion, optimizer, sample, device)
                if {name: calls[name]-before[name] for name in calls} != dict(encoder=1, semantic=1, affinity=0):
                    raise RuntimeError('每更新必须一次D5a/encoder/semantic；affinity不得前向')
                step += 1
                progress.update(completed_updates=step, next_epoch=step//128+1, next_index=step%128)
                for key in ('training_zero_calls', 'training_encoder_calls', 'training_semantic_calls'):
                    progress[key] += 1
                metadata = {key: sample[key] for key in ('source', 'draw_index', 'global_draw', 'crop_box',
                            'rotation_k', 'horizontal_flip', 'vertical_flip', 'ledger')}
                log.write(json.dumps(dict(step=step, epoch=epoch, index=index, lr=learning_rate(epoch),
                    sample=metadata, **metrics), ensure_ascii=False, allow_nan=False)+'\n')
                print(f'epoch={epoch:02d} index={index:03d} step={step}/2560 loss={metrics["loss"]:.6f}', flush=True)
                if step % 128 == 0:
                    assert_frozen(model, restorer, frozen)
                    if epoch in (1,5,10,15,20):
                        result = monitor(model, restorer, dataset, output, epoch)
                        progress['monitors'].append(result)
                        progress['monitor_zero_calls'] += len(result['sources'])
                    if source_seal() != identity['source_seal']:
                        raise RuntimeError('训练期间源码发生变化')
                    save_delta(checkpoint_dir/'last.pt', model, optimizer, identity, progress, output_dir=output)
            assert_frozen(model, restorer, frozen)
            final = save_delta(checkpoint_dir/'last.pt', model, optimizer, identity, progress, output_dir=output)
            if progress['completed_updates'] == 2560:
                final = save_delta(checkpoint_dir/'final.pt', model, optimizer, identity, progress, output_dir=output)
    finally:
        for handle in handles:
            handle.remove()
    report = dict(format=FORMAT, status='complete' if stop == 2560 else 'smoke_checkpoint',
                  progress=progress, last_checkpoint=final, frozen=frozen,
                  elapsed_seconds=time.time()-started, full150_inference=False,
                  remaining_updates=2560-progress['completed_updates'], identity_sha256=json_sha(identity))
    write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    run(parse_args())
