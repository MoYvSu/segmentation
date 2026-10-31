# -*- coding: utf-8 -*-
"""离线关系分类短训：随机共用初态，仅RGB输入开/关，不加载主线网络。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from html import escape
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent


def local_module(relative, name):
    """不经旧models/utils的eager __init__，只加载本轮明确需要的纯文件。"""
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_model = local_module('models/boundary_relation.py', '_relation_model_runtime')
_config = local_module('utils/config.py', '_relation_config_runtime')
FORMAT, INITIAL_FORMAT = _model.FORMAT, _model.INITIAL_FORMAT
BoundaryRelationNet, model_from_architecture = _model.BoundaryRelationNet, _model.model_from_architecture
state_sha256 = _model.state_sha256
load_config, project_path = _config.load_config, _config.project_path


def data_module():
    if '_relation_data_runtime' not in sys.modules:
        local_module('data/boundary_relation.py', '_relation_data_runtime')
    return sys.modules['_relation_data_runtime']


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('续训CUDA随机状态拓扑不同')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def seed_runtime(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def validated_recipe(config):
    cfg = deepcopy(config['boundary_relation'])
    for key in ('seed', 'epochs', 'batch_size', 'num_workers'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key in ('seed', 'num_workers') else 1):
            raise ValueError('训练整数参数无效：' + key)
    if cfg.get('precision') != 'float32' or cfg.get('apply_predicted_merges', False):
        raise ValueError('首轮只允许FP32关系评分，不自动合并实例')
    if not cfg.get('no_validation_holdout', False):
        raise ValueError('本轮使用全部可靠监督，不另划人工留出')
    if (not 0 < float(cfg['minimum_learning_rate']) <= float(cfg['learning_rate'])
            or not 0 <= float(cfg['weight_decay']) < float('inf')
            or not 0 < float(cfg['max_ram_gib']) <= 16):
        raise ValueError('学习率、weight decay或16GiB RAM预算无效')
    cfg.setdefault('schedule_epochs', cfg['epochs'])
    cfg.setdefault('checkpoint_every_updates', 0)
    if type(cfg['schedule_epochs']) is not int or cfg['schedule_epochs'] < 1:
        raise ValueError('cosine schedule_epochs须为正整数')
    if type(cfg['checkpoint_every_updates']) is not int or cfg['checkpoint_every_updates'] < 0:
        raise ValueError('checkpoint间隔须为非负整数')
    explicit = 'max_updates' in cfg or 'schedule_updates' in cfg
    if explicit:
        if not all(type(cfg.get(k)) is int and cfg[k] > 0 for k in ('max_updates', 'schedule_updates')):
            raise ValueError('更新预算须显式同时提供正整数max_updates/schedule_updates')
        if cfg['schedule_updates'] < cfg['max_updates']:
            raise ValueError('schedule_updates不得短于实际更新预算')
        if cfg.get('monitor_epochs') != []:
            raise ValueError('更新预算模式必须显式monitor_epochs=[]，只按更新步监视')
        cfg.setdefault('checkpoint_updates', deepcopy(cfg.get('monitor_updates')))
        for key in ('monitor_updates', 'checkpoint_updates'):
            values = cfg.get(key)
            if (not isinstance(values, list) or not values
                    or not all(type(v) is int and 0 <= v <= cfg['max_updates'] for v in values)
                    or values != sorted(set(values)) or not {0, cfg['max_updates']}.issubset(values)):
                raise ValueError(key+'须为包含0和终态的递增不重复更新步')
    else:
        if 'monitor_updates' in cfg or 'checkpoint_updates' in cfg:
            raise ValueError('步监视必须配合显式更新预算，禁止静默混用epoch模式')
        epochs = cfg.get('monitor_epochs')
        if not isinstance(epochs, list) or len(set(epochs)) != len(epochs) or not all(type(e) is int and 0 <= e <= cfg['epochs'] for e in epochs):
            raise ValueError('monitor epochs须为阶段内不重复整数')
        if not {0, 1, cfg['epochs']}.issubset(epochs):
            raise ValueError('过程monitor必须覆盖e0/e1/阶段末轮')
    return cfg


def update_budget(cfg, steps_per_epoch):
    """显式更新预算不按epoch凑整；旧配方继续使用原有epoch日程。"""
    if 'max_updates' in cfg:
        return int(cfg['max_updates']), int(cfg['schedule_updates'])
    return cfg['epochs']*steps_per_epoch, cfg['schedule_epochs']*steps_per_epoch


def sampler_cursor(identity, cfg, step, steps_per_epoch):
    count = sum(identity['class_counts'].values())
    epoch, batch = step//steps_per_epoch+1, step % steps_per_epoch
    order = epoch_order(count, cfg['seed'], epoch)
    return dict(format='relation_epoch_cursor_v1', next_epoch=epoch,
                batches_consumed_in_next_epoch=batch, samples_consumed_in_next_epoch=batch*cfg['batch_size'],
                samples_per_epoch=count, full_epoch_order=order, order_sha256=canonical_sha(order),
                drop_last=False, stateless_D4=True)


def create_initial(path, model):
    path = Path(path)
    if path.exists() or state_sha256(model) != model.initial_state_sha256:
        raise ValueError('初态只能从新的随机模型生成，不能覆盖或继承训练结果')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format=INITIAL_FORMAT, architecture=model.architecture,
                   model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   model_sha256=model.initial_state_sha256, initial_state_sha256=model.initial_state_sha256,
                   global_step=0, random_initialization=True, trained_checkpoint_inherited=False)
    temporary = path.with_suffix(path.suffix + '.pending')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_initial(path, architecture):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = model_from_architecture(architecture)
    if (saved.get('format') != INITIAL_FORMAT or saved.get('global_step') != 0
            or saved.get('architecture') != model.architecture
            or saved.get('model_sha256') != model.initial_state_sha256
            or saved.get('initial_state_sha256') != model.initial_state_sha256
            or saved.get('trained_checkpoint_inherited') is not False):
        raise ValueError('共同初态必须是本架构/seed的完整随机e0，禁止继承训练权重')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != model.initial_state_sha256:
        raise ValueError('共同随机初态张量SHA不一致')
    return model


def epoch_order(count, seed, epoch):
    # 单独随机流，不受模型、monitor或DataLoader预取影响。
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, 991])).permutation(count).tolist()


def learning_rate(step, schedule_updates, cfg):
    position = min(max(step, 0), max(schedule_updates - 1, 0))
    ratio = position / max(schedule_updates - 1, 1)
    return float(cfg['minimum_learning_rate']) + .5 * (float(cfg['learning_rate']) - float(cfg['minimum_learning_rate'])) * (1 + math.cos(math.pi * ratio))


def draw_sha(batch):
    digest = hashlib.sha256(json.dumps(batch['sample_id'], ensure_ascii=False).encode('utf8'))
    for value in (batch['input'][:, 3:], batch['masks'], batch['label']):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def optimization_step(model, optimizer, batch, *, pos_weight, lr, mode):
    if mode == 'maps' and bool((batch['input'][:, :3] != 0).any()):
        raise ValueError('maps控制必须在归一后将三个RGB通道严格置0')
    labels = batch['label'].reshape(-1)
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('无GT monitor或未知标签不得进入训练loss')
    signature = draw_sha(batch)
    device = next(model.parameters()).device
    x = batch['input'].to(device=device, dtype=torch.float32, non_blocking=True)
    masks = batch['masks'].to(device=device, dtype=torch.float32, non_blocking=True)
    target = labels.to(device=device, dtype=torch.float32, non_blocking=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = lr
    logits = model(x, masks)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(pos_weight))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('关系分类loss不是有限值')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('关系分类gradient不是有限值')
    optimizer.step()
    predicted = logits.detach() >= 0
    target_bool = target.bool()
    return dict(loss=float(loss.detach()), samples=len(labels), keep=int(target_bool.sum()),
                merge=int((~target_bool).sum()), correct=int((predicted == target_bool).sum()),
                keep_as_merge=int((~predicted & target_bool).sum()), merge_as_keep=int((predicted & ~target_bool).sum()),
                learning_rate=lr, grad_norm=float(norm), shared_draw_sha256=signature)


def build_identity(model, dataset, cfg, mode, initial_path):
    keep = int(np.count_nonzero(dataset.labels == 1))
    merge = int(np.count_nonzero(dataset.labels == 0))
    if not keep or not merge or keep + merge != len(dataset):
        raise ValueError('可靠监督需要两个类别，不能把未知标签当merge补足')
    keys = ['seed', 'batch_size', 'learning_rate', 'minimum_learning_rate', 'weight_decay', 'precision', 'augment_d4']
    keys += ['max_updates', 'schedule_updates'] if 'max_updates' in cfg else ['schedule_epochs']
    recipe = {key: cfg[key] for key in keys}
    source_paths = [Path(__file__).resolve(), Path(inspect.getsourcefile(BoundaryRelationNet)).resolve(),
                    Path(inspect.getsourcefile(type(dataset))).resolve(), Path(_config.__file__).resolve()]
    result = dict(format=FORMAT, architecture=model.architecture, mode=mode, recipe=recipe,
                initial_state_sha256=model.initial_state_sha256, initial_checkpoint_sha256=file_sha(initial_path),
                data=dataset.identity, data_identity_sha256=dataset.identity_sha256,
                data_manifest_path=str(dataset.manifest_path), data_manifest_sha256=file_sha(dataset.manifest_path),
                source_sha256={path.relative_to(ROOT).as_posix(): file_sha(path) for path in source_paths},
                class_counts=dict(keep=keep, merge=merge), pos_weight=merge/keep,
                sample_ids_sha256=canonical_sha(dataset.sample_ids), no_C_weights_loaded=True,
                no_GT_features=True, no_automatic_merges=True, no_validation_holdout=True)
    if 'max_updates' in cfg:
        result['update_protocol'] = dict(format='relation_updates_v1', monitor_updates=cfg['monitor_updates'],
                                         checkpoint_updates=cfg['checkpoint_updates'])
    return result


def save_checkpoint(path, model, optimizer, identity, cfg, step, steps_per_epoch, monitor_epochs, *, smoke_only=False):
    target, schedule = update_budget(cfg, steps_per_epoch)
    explicit = 'max_updates' in cfg
    payload = dict(format=FORMAT, architecture=model.architecture,
                   model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   model_sha256=state_sha256(model), initial_state_sha256=model.initial_state_sha256,
                   identity=identity, identity_sha256=canonical_sha(identity), optimizer=optimizer.state_dict(),
                   rng=capture_rng(), global_step=step, epoch=step//steps_per_epoch,
                   batch_in_epoch=step % steps_per_epoch, steps_per_epoch=steps_per_epoch,
                   scheduler=dict(kind='cosine_update', next_step=step,
                                  total_updates=schedule),
                   stage_epochs=cfg['epochs'], monitor_epochs=[] if explicit else list(monitor_epochs), smoke_only=bool(smoke_only))
    if explicit:
        payload.update(update_budget=dict(format='relation_updates_v1', max_updates=target, schedule_updates=schedule),
                       monitor_updates=list(monitor_epochs), sampler=sampler_cursor(identity, cfg, step, steps_per_epoch))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume(path, model, optimizer, identity, cfg, steps_per_epoch, *, allow_smoke=False):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    step = saved.get('global_step')
    target, schedule = update_budget(cfg, steps_per_epoch)
    if (saved.get('format') != FORMAT or (saved.get('smoke_only', False) and not allow_smoke)
            or saved.get('architecture') != model.architecture
            or saved.get('identity') != identity or saved.get('identity_sha256') != canonical_sha(identity)
            or saved.get('initial_state_sha256') != model.initial_state_sha256
            or type(step) is not int or not 0 <= step <= target
            or saved.get('steps_per_epoch') != steps_per_epoch
            or saved.get('epoch') != step//steps_per_epoch or saved.get('batch_in_epoch') != step % steps_per_epoch
            or saved.get('scheduler') != dict(kind='cosine_update', next_step=step,
                                            total_updates=schedule)):
        raise ValueError('续训数据、架构、随机初态、配方或采样游标不匹配')
    if 'max_updates' in cfg:
        if (saved.get('update_budget') != dict(format='relation_updates_v1', max_updates=target, schedule_updates=schedule)
                or saved.get('sampler') != sampler_cursor(identity, cfg, step, steps_per_epoch)
                or not isinstance(saved.get('monitor_updates'), list)
                or saved['monitor_updates'] != sorted(set(saved['monitor_updates']))
                or any(v not in cfg['monitor_updates'] or v > step for v in saved['monitor_updates'])):
            raise ValueError('显式更新预算、monitor或完整采样顺序不匹配')
    elif 'update_budget' in saved:
        raise ValueError('不能将更新预算checkpoint伪装成旧epoch配方')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != saved['model_sha256']:
        raise ValueError('续训模型SHA不一致')
    states = saved['optimizer'].get('state', {})
    parameter_ids = [p for group in saved['optimizer']['param_groups'] for p in group['params']]
    if (len(parameter_ids) != len(list(model.parameters()))
            or (step > 0 and set(states) != set(parameter_ids))
            or any(int(value['step']) != step for value in states.values())):
        raise ValueError('AdamW参数或内部step与续训步数不一致')
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    return step, list(saved['monitor_updates'] if 'max_updates' in cfg else saved['monitor_epochs'])


@contextmanager
def monitor_state(model):
    modes = [(module, module.training) for module in model.modules()]
    rng, before = capture_rng(), state_sha256(model)
    try:
        model.eval()
        with torch.no_grad():
            yield
        if state_sha256(model) != before:
            raise RuntimeError('monitor改变了模型状态')
    finally:
        restore_rng(rng)
        for module, mode in modes:
            module.training = mode


def _heat(value):
    return cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(value, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def render_monitor(model, train_data, monitor_data, cfg, output, epoch, mode, *, global_step=None):
    settings = train_data.manifest.get('monitor', {})
    selected = settings.get('train_sample_ids', [])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('训练monitor必须预先固定不重复sample IDs')
    lookup = {name: i for i, name in enumerate(train_data.sample_ids)}
    if not set(selected) <= set(lookup):
        raise ValueError('固定monitor样本不在可靠训练池中')
    parent = Path(output)/'monitor'
    parent.mkdir(exist_ok=True)
    tag = f'e{epoch:03d}' if global_step is None else f's{global_step:06d}'
    target = parent/tag
    model_sha = state_sha256(model)
    if target.exists():
        existing = json.loads((target/'receipt.json').read_text(encoding='utf8'))
        if existing.get('complete') and existing.get('model_sha256') == model_sha:
            return existing
        raise ValueError('已有monitor不属于当前模型，拒绝覆盖')
    staging = parent/f'.{tag}_{uuid.uuid4().hex[:8]}'
    staging.mkdir()
    rows = []
    groups = [('train_seen_not_validation', train_data, [lookup[name] for name in selected])]
    if monitor_data is not None and len(monitor_data):
        groups.append(('unlabeled_no_GT', monitor_data, list(range(len(monitor_data)))))
    with monitor_state(model):
        for kind, dataset, indices in groups:
            for index in indices:
                sample = dataset.tensors(index, augment=False, mode=mode)
                device = next(model.parameters()).device
                logits = model(sample['input'][None].to(device), sample['masks'][None].to(device))
                probability = float(logits.sigmoid()[0])
                raw = dataset.raw(index)
                rgb = raw['rgb'].transpose(1, 2, 0)
                maps, masks = raw['maps'].astype(np.float32), raw['masks']
                overlay = rgb.astype(np.float32)
                for mask, color in ((masks[0], (20, 255, 255)), (masks[1], (255, 170, 30))):
                    overlay[mask > 0] = .6*overlay[mask > 0]+.4*np.array(color)
                overlay[masks[2] > 0] = (255, 70, 210)
                cells = [rgb, _heat(maps[0]), _heat(maps[1]), overlay.astype(np.uint8)]
                width = int(cfg.get('thumbnail_width', 144))
                canvas = np.full((width+128, width*4, 3), 248, np.uint8)
                label = 'unknown' if kind == 'unlabeled_no_GT' else ('keep' if raw['label'] == 1 else 'merge')
                group_title = 'TRAIN SEEN (not validation)' if kind != 'unlabeled_no_GT' else 'UNLABELED (NO GT)'
                title = f'{group_title} | {raw["source"]} | {mode} {tag}'
                cv2.putText(canvas, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .35, (25, 25, 25), 1, cv2.LINE_AA)
                cv2.putText(canvas, f'KEEP SCORE={probability:.4f} | label={label}', (6, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (8, 55), (canvas.shape[1]-9, 70), (220, 220, 220), -1)
                right = 8+round((canvas.shape[1]-17)*probability)
                cv2.rectangle(canvas, (8, 55), (right, 70), (35, 160, 75) if probability >= .5 else (220, 100, 35), -1)
                cv2.putText(canvas, '0 MERGE <-------- keep score --------> 1 KEEP', (8, 87),
                            cv2.FONT_HERSHEY_SIMPLEX, .34, (30, 30, 30), 1, cv2.LINE_AA)
                for i, cell in enumerate(cells):
                    canvas[96:96+width, i*width:(i+1)*width] = cv2.resize(cell, (width, width), interpolation=cv2.INTER_AREA)
                cv2.putText(canvas, 'RGB reference | frozen N1 | frozen NAF | A cyan / B orange / seam pink; scores only',
                            (5, width+117), cv2.FONT_HERSHEY_SIMPLEX, .30, (25, 25, 25), 1, cv2.LINE_AA)
                filename = f'{len(rows):03d}_{"train" if dataset is train_data else "unlabeled"}.jpg'
                ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise OSError('monitor缩略图编码失败')
                (staging/filename).write_bytes(encoded.tobytes())
                rows.append(dict(kind=kind, sample_id=raw['sample_id'], source=raw['source'],
                                 label=None if kind == 'unlabeled_no_GT' else raw['label'],
                                 keep_probability=probability, file=filename, sha256=file_sha(staging/filename)))
    receipt = dict(complete=True, epoch=epoch, mode=mode, model_sha256=model_sha, rows=rows,
                   original_instances_unmodified=True, not_accuracy=True, no_test_GT=True)
    if global_step is not None:
        receipt.update(global_step=global_step, format='relation_update_monitor_v1',
                       keep_probability_visualized=True, sample_selection='original manifest monitor, unchanged')
    body = ['<!doctype html><meta charset="utf-8"><title>Relation keep scores</title>',
            '<style>body{font-family:Arial;max-width:1000px;margin:20px auto}img{width:100%;margin-bottom:24px}</style>',
            f'<h1>{escape(mode)} {escape(tag)}: keep scores</h1>',
            '<p>Seen training samples are not validation. Unlabeled samples have no GT. Scores are not calibrated probabilities; no masks are changed.</p>']
    for row in rows:
        label = 'NO GT' if row['label'] is None else ('KEEP' if row['label'] == 1 else 'MERGE')
        body.append(f'<p>{escape(row["kind"])} | {escape(row["source"])} | KEEP SCORE {row["keep_probability"]:.4f} | label {label}</p>'
                    f'<img loading="lazy" src="{escape(row["file"])}">')
    (staging/'index.html').write_text('\n'.join(body), encoding='utf8')
    write_json(staging/'receipt.json', receipt)
    staging.replace(target)
    return receipt


def reconcile_log(path, step):
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    kept = [row for row in rows if row['global_step'] <= step]
    start = kept[0]['global_step'] if kept else 1
    if (any(row['global_step'] != i+start for i, row in enumerate(kept))
            or (kept and kept[-1]['global_step'] != step)):
        raise ValueError('续训step日志缺失或重复，不静默补造记录')
    if len(kept) != len(rows):
        path.rename(path.with_name(path.name+f'.after_checkpoint_{uuid.uuid4().hex[:8]}'))
        path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in kept), encoding='utf8')


def train(model, dataset, monitor_data, config, cfg, output, mode, initial_path, *, resume=None):
    collate_boundary_relation = data_module().collate_boundary_relation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise ValueError('新训练输出目录必须为空')
    if resume is None and state_sha256(model) != model.initial_state_sha256:
        raise ValueError('新模块必须从共同完整随机初态启动')
    identity = build_identity(model, dataset, cfg, mode, initial_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
    updates = math.ceil(len(dataset)/cfg['batch_size'])
    target_updates, schedule_updates = update_budget(cfg, updates)
    explicit = 'max_updates' in cfg
    step, monitored = (0, []) if resume is None else load_resume(resume, model, optimizer, identity, cfg, updates)
    if resume:
        reconcile_log(output/'steps.jsonl', step)
        write_json(output/'resume_source.json', dict(path=str(Path(resume).resolve()), sha256=file_sha(resume),
            global_step=step, prior_monitor_epochs=[] if explicit else monitored,
            prior_monitor_updates=monitored if explicit else [], same_training_identity=True))
    write_json(output/'identity.json', identity)
    write_json(output/'resolved_config.json', config)
    write_json(output/'initialization.json', dict(initial_checkpoint_sha256=file_sha(initial_path),
        initial_state_sha256=model.initial_state_sha256, random_initialization=True, external_trained_weights_inherited=False,
        C_networks_loaded=False, parameters=sum(p.numel() for p in model.parameters()), mode=mode, resume=resume is not None))
    def save(name):
        save_checkpoint(output/name, model, optimizer, identity, cfg, step, updates, monitored)
    def monitor(epoch):
        if epoch in cfg['monitor_epochs'] and epoch not in monitored:
            render_monitor(model, dataset, monitor_data, cfg, output, epoch, mode)
            monitored.append(epoch)
            save('last.pt')
    def monitor_update():
        if step in cfg['monitor_updates'] and step not in monitored:
            epoch = (step-1)//updates+1 if step else 0
            render_monitor(model, dataset, monitor_data, cfg, output, epoch, mode, global_step=step)
            monitored.append(step)
        if step in cfg['checkpoint_updates']:
            save(f'u{step:06d}.pt')
        if step in cfg['monitor_updates'] or step in cfg['checkpoint_updates']:
            save('last.pt')
    if step == 0:
        save('last.pt')
        monitor_update() if explicit else monitor(0)
    elif explicit:
        monitor_update()
    elif step % updates == 0:
        monitor(step//updates)
    started = time.monotonic()
    while step < target_updates:
        epoch, cursor = step//updates+1, step % updates
        dataset.set_epoch(epoch)
        order = epoch_order(len(dataset), cfg['seed'], epoch)
        ids = [dataset.sample_ids[index] for index in order]
        write_json(output/f'sampling_e{epoch:03d}.json', dict(epoch=epoch, sample_ids=ids, order_sha256=canonical_sha(ids),
            stateless_D4=True, no_replacement=True, samples=len(ids), optimizer_updates=updates))
        generator = torch.Generator().manual_seed(cfg['seed']+epoch+771)
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], sampler=order[cursor*cfg['batch_size']:],
                            collate_fn=collate_boundary_relation, num_workers=cfg['num_workers'],
                            pin_memory=next(model.parameters()).is_cuda, generator=generator, drop_last=False)
        for batch in loader:
            before = time.monotonic()
            row = optimization_step(model, optimizer, batch, pos_weight=identity['pos_weight'],
                                    lr=learning_rate(step, schedule_updates, cfg), mode=mode)
            step += 1
            row.update(global_step=step, epoch=epoch, batch_in_epoch=(step-1) % updates+1,
                       sample_ids=batch['sample_id'], seconds=time.monotonic()-before)
            with (output/'steps.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            interval = cfg['checkpoint_every_updates']
            if interval and step % interval == 0:
                save('last.pt')
            if explicit:
                monitor_update()
            if step == target_updates:
                break
        if step % updates and step != target_updates:
            raise RuntimeError('训练epoch没有完整遍历所有可靠样本')
        save('last.pt')
        if not explicit:
            monitor(epoch)
            if epoch in (1, cfg['epochs']):
                save(f'e{epoch:03d}.pt')
        print(json.dumps(dict(epoch=epoch, global_step=step, mode=mode, model_sha256=state_sha256(model))), flush=True)
    if file_sha(initial_path) != identity['initial_checkpoint_sha256'] or dataset.identity_sha256 != identity['data_identity_sha256']:
        raise RuntimeError('共同初态或训练数据身份改变')
    for relative, digest in identity['source_sha256'].items():
        if file_sha(ROOT/relative) != digest:
            raise RuntimeError('训练源码改变：' + relative)
    for data in (dataset, monitor_data):
        if data is None:
            continue
        if file_sha(data.manifest_path) != data.identity['manifest_sha256']:
            raise RuntimeError('数据清单改变')
        for shard in data.identity['shards']:
            if file_sha(data.manifest_path.parent/shard['path']) != shard['sha256']:
                raise RuntimeError('冻结数据分片改变')
    completion = dict(complete=True, format=FORMAT, mode=mode, epochs=cfg['epochs'],
        optimizer_updates=step, steps_per_epoch=updates, samples_per_epoch=len(dataset),
        monitor_epochs=[] if explicit else monitored, model_sha256=state_sha256(model), initial_state_sha256=model.initial_state_sha256,
        identity_sha256=canonical_sha(identity), seconds=time.monotonic()-started,
        no_C_weights_loaded=True, original_instances_unmodified=True, official_score=None, convergence_claimed=False)
    if explicit:
        completion.update(global_step=step, target_updates=target_updates, schedule_updates=schedule_updates,
            monitor_updates=monitored, checkpoint_updates=cfg['checkpoint_updates'],
            configured_epochs=cfg['epochs'], completed_epochs=step//updates, last_epoch=(step-1)//updates+1 if step else 0,
            last_batch_in_epoch=(step-1) % updates+1 if step else 0,
            samples_consumed=(step//updates)*len(dataset)+(step % updates)*cfg['batch_size'],
            update_budget=dict(format='relation_updates_v1', max_updates=target_updates, schedule_updates=schedule_updates))
    write_json(output/'complete.json', completion)
    return model


def smoke(model, dataset, monitor_data, config, cfg, output, mode, initial_path):
    """两个实际更新验证梯度/载入/下一步续训；不会成为正式训练初态。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('smoke输出须为空，不能覆盖正式训练')
    collate = data_module().collate_boundary_relation
    identity = build_identity(model, dataset, cfg, mode, initial_path)
    updates = math.ceil(len(dataset)/cfg['batch_size'])
    order = epoch_order(len(dataset), cfg['seed'], 1)[:cfg['batch_size']]
    dataset.set_epoch(1)
    batch = collate([dataset[index] for index in order])
    render_monitor(model, dataset, monitor_data, cfg, output, 0, mode,
                   global_step=0 if 'max_updates' in cfg else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    device = next(model.parameters()).device
    with monitor_state(model):
        x, masks = batch['input'].to(device), batch['masks'].to(device)
        a = model(x, masks)
        swapped = masks[:, [1, 0, 2, 3]]
        if not torch.equal(a, model(x, swapped)):
            raise RuntimeError('交换A/B后logit不完全一致')
    before = state_sha256(model)
    first = optimization_step(model, optimizer, batch, pos_weight=identity['pos_weight'], lr=cfg['learning_rate'], mode=mode)
    if state_sha256(model) == before or first['grad_norm'] <= 0:
        raise RuntimeError('smoke没有实际非零梯度/参数更新')
    checkpoint = output/'smoke_resume.pt'
    save_checkpoint(checkpoint, model, optimizer, identity, cfg, 1, updates, [0], smoke_only=True)
    construction_rng = capture_rng()
    restored = load_initial(initial_path, model.architecture).to(device)
    _model.load_boundary_relation(checkpoint, device=device)
    after_construction = capture_rng()
    if (not torch.equal(construction_rng['torch'], after_construction['torch'])
            or len(construction_rng['cuda']) != len(after_construction['cuda'])
            or any(not torch.equal(a, b) for a, b in zip(construction_rng['cuda'], after_construction['cuda']))):
        raise RuntimeError('CPU构造/strict load改变了CPU或已初始化CUDA随机流')
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    saved_step, _ = load_resume(checkpoint, restored, restored_optimizer, identity, cfg, updates, allow_smoke=True)
    if saved_step != 1 or state_sha256(restored) != state_sha256(model):
        raise RuntimeError('smoke不能严格重载模型/优化器')
    with monitor_state(model), monitor_state(restored):
        if not torch.equal(model(x, masks), restored(x, masks)):
            raise RuntimeError('重载预测不完全一致')
    rng = capture_rng()
    lr = learning_rate(1, update_budget(cfg, updates)[1], cfg)
    second = optimization_step(model, optimizer, batch, pos_weight=identity['pos_weight'], lr=lr, mode=mode)
    expected = state_sha256(model)
    restore_rng(rng)
    optimization_step(restored, restored_optimizer, batch, pos_weight=identity['pos_weight'], lr=lr, mode=mode)
    if state_sha256(restored) != expected:
        raise RuntimeError('续训下一次AdamW更新不完全一致')
    write_json(output/'smoke.json', dict(complete=True, smoke_only=True, optimizer_updates=2,
        resume_probe_updates=1, AB_swap_exact=True, strict_reload_prediction_exact=True,
        resumed_next_update_exact=True, constructors_and_strict_load_preserve_RNG=True,
        initialized_CUDA_RNG_checked=bool(construction_rng['cuda']), initial_state_sha256=model.initial_state_sha256,
        shared_initial_checkpoint_sha256=file_sha(initial_path), gradients=[first['grad_norm'], second['grad_norm']],
        mode=mode, actual_batch_size=len(batch['label']), no_C_weights_loaded=True,
        formal_training_started=False, original_instances_unmodified=True))
    print(json.dumps(dict(smoke=True, complete=True, mode=mode, resumed_next_update_exact=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-manifest', help='只覆盖本次运行的数据清单，不改YAML源文件')
    parser.add_argument('--mode', choices=('maps', 'rgb'))
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda')
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument('--initial-create')
    initial.add_argument('--initial-load')
    parser.add_argument('--initial-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_manifest:
        config['boundary_relation']['data_manifest'] = str(Path(args.data_manifest).resolve())
    cfg = validated_recipe(config)
    mode = args.mode or cfg.get('mode', 'rgb')
    if mode not in ('maps', 'rgb'):
        raise ValueError('mode须为maps或rgb')
    cfg['mode'] = mode
    config['boundary_relation'] = deepcopy(cfg)
    config['boundary_relation_runtime'] = dict(source_YAML_path=str(Path(args.config).resolve()),
                                              source_YAML_sha256=file_sha(args.config))
    if not args.initial_create and not args.initial_load:
        raise ValueError('必须显式创建或加载共同随机初态，不能加载训练checkpoint冒充初态')
    seed_runtime(cfg['seed'])
    torch.set_num_threads(int(cfg.get('torch_threads', 4)))
    cv2.setNumThreads(2)
    model = BoundaryRelationNet(**cfg.get('model', {}), seed=cfg['seed'])
    initial_path = Path(args.initial_create or args.initial_load).resolve()
    if args.initial_create:
        create_initial(initial_path, model)
    else:
        model = load_initial(initial_path, model.architecture)
    if args.initial_only:
        print(json.dumps(dict(initial_only=True, initial_state_sha256=model.initial_state_sha256,
                              checkpoint_sha256=file_sha(initial_path), parameters=sum(p.numel() for p in model.parameters()))))
        return
    BoundaryRelationDataset = data_module().BoundaryRelationDataset
    manifest = project_path(config, cfg['data_manifest'])
    dataset = BoundaryRelationDataset(manifest, mode=mode, augment=cfg['augment_d4'], seed=cfg['seed'],
                                     max_ram_gib=cfg['max_ram_gib'], split='train')
    remaining = cfg['max_ram_gib'] - dataset.ram_bytes/1024**3
    if remaining <= 0:
        raise MemoryError('训练RAM已耗尽，不能再加载固定monitor')
    monitor_data = BoundaryRelationDataset(manifest, mode=mode, augment=False, seed=cfg['seed'],
                                          max_ram_gib=remaining, split='monitor')
    model.to(args.device)
    output = Path(args.output) if args.output else Path(project_path(config, cfg['output_dir']))/mode
    if args.smoke:
        if args.resume:
            raise ValueError('smoke只从共同随机初态启动，不继续正式训练')
        smoke(model, dataset, monitor_data, config, cfg, output, mode, initial_path)
    else:
        train(model, dataset, monitor_data, config, cfg, output, mode, initial_path, resume=args.resume)


if __name__ == '__main__':
    main()
