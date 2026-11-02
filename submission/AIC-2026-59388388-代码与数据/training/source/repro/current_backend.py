# -*- coding: utf-8 -*-
"""本轮 D5a/R1/导出适配器；不借用历史消融日志，也不修改旧训练器。

仅由显式训练编排调用。R1 的视野表由本轮 DataLoader 的实际抽样顺序生成，
因此不需要把旧 control/native/mixed 的回执冒充为本轮证据。
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from utils.config import _deep_merge, load_config

CONTEXT_FORMAT = 'current_training_context_v1'
STAGES = ('d5a', 'r1', 'export_backend')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _absolute(value, label):
    path = Path(value)
    _require(path.is_absolute(), label + ' must be an absolute path')
    return path.resolve()


def _context(context):
    _require(context.get('format') == CONTEXT_FORMAT, 'Unsupported training context format')
    for key in ('project_root', 'run_dir'):
        _absolute(context[key], key)
    _require(isinstance(context.get('settings'), dict), 'Expanded settings are required')
    _require(isinstance(context.get('assets'), dict) and isinstance(context.get('artifacts'), dict),
             'Explicit assets/artifacts are required')
    return context


def _path(context, group, key, *, existing=False):
    path = _absolute(context[group][key], group + '.' + key)
    if existing:
        _require(path.exists(), 'Missing ' + group + '.' + key + ': ' + str(path))
    return path


def _options(context, stage):
    options = context['settings'].get('stages', {}).get(stage, {})
    _require(isinstance(options, dict), 'Stage settings must be a mapping')
    return options


def _config(context, stage, default):
    options = _options(context, stage)
    source = Path(options.get('config', default))
    if not source.is_absolute():
        source = _path_root(context) / source
    config = load_config(str(source))
    overrides = options.get('overrides', {})
    _require(isinstance(overrides, dict), 'Stage overrides must be a mapping')
    config = _deep_merge(config, overrides)
    config.setdefault('paths', {})['project_root'] = str(_path_root(context))
    return config


def _path_root(context):
    return _absolute(context['project_root'], 'project_root')


def _output(context, key, stage):
    path = _path(context, 'artifacts', key)
    run = _absolute(context['run_dir'], 'run_dir')
    _require(path.is_relative_to(run), 'Stage output must stay inside run_dir')
    output = path.parent if stage != 'export_backend' else path.parent.parent
    _require(output != run, 'Stage output needs its own directory')
    return output


def _save_config(context, stage, config):
    path = _absolute(context['run_dir'], 'run_dir') / 'configs' / (stage + '.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), 'Generated config already exists: ' + str(path))
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return path


def _monitor_sources(context, count=2):
    from data.mim_dataset import list_images
    directory = _path(context, 'assets', 'monitor_dir', existing=True)
    raw_dir = _path(context, 'assets', 'raw_dir', existing=True)
    # 默认且只允许人工训练源目录；不让测试集成为重新训练的隐含依赖。
    _require(directory == raw_dir, 'Process monitor must use the manual training-source directory')
    files = [Path(p) for p in list_images(str(directory)) if Path(p).with_suffix('.json').is_file()]
    _require(bool(files), 'No manual training images for fixed thumbnails')
    return directory, [p.stem for p in files[:count]]


def prepare_d5a_config(context):
    """仅生成配置；不会构造数据集、模型或启动训练。"""
    _context(context)
    config = _config(context, 'd5a', 'config/train/rgb_diffusion_d5a.yaml')
    cfg = config['rgb_restoration']
    cfg['data_dir'] = str(_path(context, 'assets', 'unlabeled_dir', existing=True))
    cfg['output_dir'] = str(_output(context, 'd5a', 'd5a'))
    cfg['seed'] = int(_options(context, 'd5a').get('seed', 42))
    directory, names = _monitor_sources(context, 4)
    cfg.setdefault('monitor', {})['real'] = dict(enabled=True, data_dir=str(directory), images=names)
    train, model = cfg['train'], cfg['model']
    return config


def prepare_r1_config(context):
    """将本轮输入逐项绑定到严格的 S-align/V 初始化合同。"""
    _context(context)
    config = _config(context, 'r1', 'config/train/affinity_balance_mixed.yaml')
    options = _options(context, 'r1')
    joint, semantic, affinity, d5a = [_path(context, 'artifacts', k, existing=True)
                                    for k in ('joint', 'semantic', 'affinity', 'd5a')]
    backbone = _path(context, 'assets', 'backbone', existing=True)
    sam2_repo = _path(context, 'assets', 'sam2_repo', existing=True)
    config['paths'].update(raw_data_dir=str(_path(context, 'assets', 'raw_dir', existing=True)),
                           weights_dir=str(backbone.parent), sam2_ckpt=backbone.name)
    config['sam2']['sam2_repo_path'] = str(sam2_repo)
    config['affinity_geometry_g1'].update(reference_checkpoint=str(joint),
                                          geometry_init_checkpoint=str(affinity))
    # 本轮 joint 是几何和 S-align 的共同特征基座；不携带旧 decoder override 逃生口。
    config['affinity_geometry_g1'].pop('semantic_decoder_override_base', None)
    config['affinity_deployment']['checkpoint'] = str(affinity)
    config['semantic_challenger'].update(checkpoint=str(semantic), enabled=True, replace_reference_semantic=True)
    cfg = config['backend_adaptation']
    teacher = context['assets'].get('teacher_gt_dir')
    teacher_manifest = context['assets'].get('teacher_gt_manifest')
    _require(bool(teacher) == bool(teacher_manifest), 'teacher_gt_dir/manifest must be supplied together')
    gt_key = 'teacher_gt_dir' if teacher else 'completed_gt_dir'
    manifest_key = 'teacher_gt_manifest' if teacher else 'completed_gt_manifest'
    cfg.update(output_dir=str(_output(context, 'r1', 'r1')), seed=int(options.get('seed', 20260928)),
               completed_gt_dir=str(_path(context, 'assets', gt_key, existing=True)))
    supervision_path = _path(context, 'assets', manifest_key, existing=True)
    supervision = json.loads(supervision_path.read_text(encoding='utf-8'))
    _require(not teacher or supervision.get('format') == 'teacher_supervision_v1', 'Teacher manifest format differs')
    if supervision.get('format') == 'teacher_supervision_v1':
        cfg['supervision_source'] = dict(format='teacher_supervision_v1', source_kind='model_teacher',
            manifest_sha256=_sha(supervision_path),
            manual_source_images_only=True, manual_GT_targets_used=False, ID0='ignore')
    cfg.pop('baseline_prediction_dir', None)
    cfg.pop('d4_prediction_dir', None)
    geometry_dir = _path(context, 'assets', 'sam2_geometry_dir', existing=True)
    cfg['sam2_geometry'].update(dataset_dir=str(geometry_dir),
        source_dir=str(_path(context, 'assets', 'sam2_source_dir', existing=True)),
        manifest_sha256=_sha(geometry_dir / 'manifest.jsonl'),
        expected_sources=64)
    d5a_config = _absolute(context['run_dir'], 'run_dir') / 'configs' / 'd5a.yaml'
    _require(d5a_config.is_file(), 'R1 requires this run\'s generated D5a recipe')
    cfg['restoration'].update(checkpoint=str(d5a), checkpoint_sha256=_sha(d5a),
                              degradation_config=str(d5a_config), mode='first_x0')
    directory, names = _monitor_sources(context)
    cfg['monitor'].update(images=names)
    config['inference']['test_dir'] = str(directory)
    # 旧键只属于历史消融审计。本轮只保留实际裁窗与视野计划选项。
    previous = config['affinity_native']
    config['affinity_native'] = {k: deepcopy(previous[k]) for k in
        ('overlap', 'crop_attempts', 'minimum_known_pairs', 'monitor_width')}
    config['affinity_native']['sizes'] = [1024]
    config['affinity_native']['mixed_view_seed'] = int(options.get('mixed_view_seed', 20260930))
    config.pop('affinity_connectivity', None)
    config.pop('affinity_interface', None)
    return config


def sampler_order(count, seed):
    """包含 DataLoader 的 iterator seed 消耗，避免用裸 randperm 猜测抽样顺序。"""
    generator = torch.Generator().manual_seed(int(seed))
    return [int(batch.item()) for batch in DataLoader(range(count), batch_size=1, shuffle=True,
                                                    num_workers=0, generator=generator)]


class FreshMixedViewDataset(Dataset):
    """同一更新的人工/几何流使用同一视野，本轮独立生成 draw→view 映射。"""
    def __init__(self, original, kind, options, loader_seed):
        from data.affinity_mixed import MixedViewDataset
        wrapped = MixedViewDataset(original, kind, options, receipts=[])
        self.whole, self.native = wrapped.whole, wrapped.native
        self.kind, self.options, self.loader_seed = kind, options, int(loader_seed)
        self.mapping, self.draw_order, self.plan = None, None, None

    def __len__(self):
        return len(self.whole)

    def set_epoch(self, epoch):
        from data.affinity_mixed import view_plan
        self.plan = view_plan(self.options['mixed_view_seed'], epoch, len(self))
        self.draw_order = sampler_order(len(self), self.loader_seed + epoch * 2)
        self.mapping = dict(zip(self.draw_order, self.plan))
        _require(len(self.mapping) == len(self), 'Repeated or missing DataLoader draw')
        self.whole.set_epoch(epoch)
        self.native.set_epoch(epoch)

    def __getitem__(self, index):
        _require(self.mapping is not None, 'Set the epoch before reading mixed views')
        size = self.mapping[int(index)]
        sample = (self.whole if size == 0 else self.native)[index]
        sample['training_view'] = 'whole' if size == 0 else 'native1024'
        return sample


def build_fresh_datasets(config):
    from train_backend_adaptation import build_datasets
    manual, pseudo, parent = build_datasets(config)
    _require(parent['updates_per_epoch'] == len(manual) == len(pseudo),
             'R1 requires exactly 64 paired updates per epoch')
    seed = config['backend_adaptation']['seed']
    options = config['affinity_native']
    if getattr(manual.base_dataset, 'teacher_supervision', False):
        parent = dict(parent, manual_cohort_target_source='model_teacher',
                      manual_GT_targets_used=False, teacher_ID0='ignore')
    return (FreshMixedViewDataset(manual, 'manual', options, seed),
            FreshMixedViewDataset(pseudo, 'pseudo', options, seed + 1),
            dict(parent_data=parent, updates_per_epoch=64, training_views=['whole', 'native1024'],
                 per_epoch_view_updates={'whole': 32, 'native1024': 32},
                 view_seed=options['mixed_view_seed'], native_crop_size=1024,
                 planning='fresh_DataLoader_order_seed_epoch_v1', historical_control_reused=False))


def _tensor_digest(named_tensors):
    from train_backend_adaptation import tensor_digest
    return tensor_digest(named_tensors)


def _r1_update(model, restorer, raw, pseudo, config, seed, device, optimizer, *, draw_saver=None):
    """真实单更新；两次反向后只 step 一次。CPU 小模型测试可替换恢复/loss。"""
    from models.backend_adaptation import restore_first
    from train_direct_semantic_affinity import compute_affinity_loss, move_batch
    cfg = config['backend_adaptation']
    optimizer.zero_grad(set_to_none=True)
    losses, supervision = [], []
    for is_pseudo, source in ((False, raw), (True, pseudo)):
        batch = move_batch(source, device)
        with torch.no_grad():
            restored = restore_first(restorer, batch['image'], seed + (2000000 if is_pseudo else 1000000))
            if not is_pseudo and draw_saver is not None:
                draw_saver(batch, restored)
            features = model.encoder(restored)
        logits = model.affinity_decoder(features)['affinity_logits']
        _require(logits.dtype == torch.float32, 'R1 must use FP32 logits without AMP')
        loss, details = compute_affinity_loss(logits, batch,
            config['direct_semantic_affinity']['affinity_loss'], pseudo=is_pseudo)
        if not torch.isfinite(loss) or details['positive_edges'] + details['negative_edges'] <= 0:
            raise FloatingPointError('Nonfinite or unsupervised R1 loss')
        (loss * (cfg['pseudo_weight'] if is_pseudo else 1.)).backward()
        losses.append(float(loss.detach()))
        supervision.append({k: details[k] for k in
                            ('positive_edges', 'negative_edges', 'precision', 'recall', 'specificity')})
    grad = float(torch.nn.utils.clip_grad_norm_(model.affinity_decoder.parameters(), cfg['grad_clip'],
                                               error_if_nonfinite=True))
    _require(not any(p.grad is not None for name, p in model.named_parameters()
                                  if not name.startswith('affinity_decoder.')),
             'R1 requires affinity gradients and no frozen gradients')
    _require(not any(p.grad is not None for p in restorer.parameters()), 'Frozen D5a acquired gradients')
    optimizer.step()
    return dict(manual_bce=losses[0], pseudo_bce=losses[1], grad_norm=grad,
                supervision=supervision, auxiliary_weight=0.)




def train_fresh_r1(config, context, output):
    """独立 fresh R1；从未把历史 control 回执作为新训练证明。"""
    import cv2
    from models.backend_adaptation import build_backend, load_backend, save_backend
    from tools.affinity_native_views import save_training_draw
    from train_affinity_connectivity import append_json, configure_training, draw_receipt, frozen_digest, get_restorer
    from train_backend_adaptation import monitor
    from train_direct_semantic_affinity import set_seed
    _require(torch.cuda.is_available(), 'R1 training requires the GPU server sam2_env')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = config['backend_adaptation']
    set_seed(cfg['seed'])
    device = torch.device('cuda')
    model, initialization = build_backend(config, device)
    restorer = get_restorer(config, device)
    _require(restorer.num_steps == 16 and restorer.kappa == .03, 'D5a first-x0 contract differs')
    backend_parameters = sum(p.numel() for p in model.parameters())
    restorer_parameters = sum(p.numel() for p in restorer.parameters())
    total = backend_parameters + restorer_parameters
    _require(total < 500_000_000, 'Model parameter limit exceeded')
    manual, pseudo, data = build_fresh_datasets(config)
    initial_head = _tensor_digest(model.affinity_decoder.state_dict().items())
    status = dict(format='current_r1_fresh_training_v1', status='preflight', config=config,
        initialization=initialization, initial_state_sha256=_tensor_digest(model.state_dict().items()),
        initial_affinity_sha256=initial_head, frozen_state_sha256=frozen_digest(model),
        restorer_state_sha256=_tensor_digest(restorer.state_dict().items()), data=data,
        total_parameters=total, backend_parameters=backend_parameters,
        restorer_parameters=restorer_parameters, precision='FP32',
        historical_control_reused=False, historical_parity_claimed=False,
        selection='fixed_e20_all_sources_no_holdout', official_score=None, updates=0)
    _write(output / 'status.json', status)
    configure_training(model)
    torch.save(dict(format='current_r1_initial_head_v1', geometry_state_dict=model.affinity_decoder.state_dict(),
                    initial_affinity_sha256=initial_head, initialization=initialization), output / 'initial_head.pt')
    optimizer = torch.optim.AdamW(model.affinity_decoder.parameters(), lr=cfg['learning_rates']['affinity'],
                                  weight_decay=cfg['weight_decay'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: cfg['minimum_lr_ratio'] +
        (1 - cfg['minimum_lr_ratio']) * (1 + math.cos(math.pi * e / cfg['epochs'])) / 2)
    monitor(model, restorer, config, output, 0)
    _require(frozen_digest(model) == status['frozen_state_sha256']
             and _tensor_digest(model.affinity_decoder.state_dict().items()) == initial_head,
             'Initial process monitor changed model state')
    updates, started = 0, time.time()
    for epoch in range(1, int(cfg['epochs']) + 1):
        for dataset in (manual, pseudo):
            dataset.set_epoch(epoch)
        _require(manual.plan == pseudo.plan and manual.plan.count(0) == manual.plan.count(1024),
                 'R1 requires 32 whole and 32 native1024 paired updates')
        append_json(output / 'view_plans.jsonl', dict(epoch=epoch, sizes=manual.plan,
            manual_draw_order=manual.draw_order, pseudo_draw_order=pseudo.draw_order,
            manual_loader_seed=cfg['seed'] + epoch * 2, pseudo_loader_seed=cfg['seed'] + epoch * 2 + 1))
        loaders = [DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
                              generator=torch.Generator().manual_seed(cfg['seed'] + epoch * 2 + i))
                   for i, ds in enumerate((manual, pseudo))]
        configure_training(model)
        sums, count, views = [0., 0.], 0, Counter()
        for index, (raw, raw_pseudo) in enumerate(zip(*loaders)):
            seed = cfg['seed'] + epoch * 10000 + index
            set_seed(seed)
            _require(int(raw['draw_index'].item()) == manual.draw_order[index]
                     and int(raw_pseudo['draw_index'].item()) == pseudo.draw_order[index],
                     'Actual DataLoader order differs from fresh view plan')
            view = 'whole' if manual.plan[index] == 0 else 'native1024'
            _require(raw['training_view'] == raw_pseudo['training_view'] == [view],
                     'Actual manual/pseudo view differs from the paired plan')
            saver = None
            if index < 2 and (epoch == 1 or epoch % 5 == 0):
                saver = lambda batch, restored: save_training_draw(output, batch, restored, epoch, index + 1)
            row = dict(epoch=epoch, step=index + 1, seed=seed, manual=raw['name'], pseudo=raw_pseudo['name'],
                manual_draw=draw_receipt(raw), pseudo_draw=draw_receipt(raw_pseudo), training_view=view,
                manual_crop=raw['crop_box'].tolist(), pseudo_crop=raw_pseudo['crop_box'].tolist(),
                crop_attempts=[raw['crop_attempt'].tolist(), raw_pseudo['crop_attempt'].tolist()],
                learning_rate=optimizer.param_groups[0]['lr'])
            row.update(_r1_update(model, restorer, raw, raw_pseudo, config, seed, device, optimizer,
                                  draw_saver=saver))
            updates += 1
            count += 1
            views[view] += 1
            sums[0] += row['manual_bce']
            sums[1] += row['pseudo_bce']
            row['updates'] = updates
            append_json(output / 'steps.jsonl', row)
        _require(count == len(manual) and views == Counter('whole' if v == 0 else 'native1024' for v in manual.plan), 'Incomplete R1 epoch')
        scheduler.step()
        epoch_row = dict(epoch=epoch, updates=updates, manual_bce=sums[0] / count, pseudo_bce=sums[1] / count,
                         training_views=dict(views), next_lr=optimizer.param_groups[0]['lr'],
                         elapsed_seconds=time.time() - started)
        append_json(output / 'epochs.jsonl', epoch_row)
        torch.save(dict(format='current_r1_head_v1', epoch=epoch, updates=updates,
                        geometry_state_dict=model.affinity_decoder.state_dict(), config=config,
                        initialization=initialization, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict()), output / 'last_head.pt')
        if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == int(cfg['epochs']):
            monitor(model, restorer, config, output, epoch)
        status.update(status='running', **epoch_row)
        _write(output / 'status.json', status)
        print(json.dumps(epoch_row), flush=True)
    status.update(frozen_unchanged=frozen_digest(model) == status['frozen_state_sha256'],
                  restorer_unchanged=_tensor_digest(restorer.state_dict().items()) == status['restorer_state_sha256'],
                  affinity_changed=_tensor_digest(model.affinity_decoder.state_dict().items()) != initial_head)
    _require(all(status[k] for k in ('frozen_unchanged', 'restorer_unchanged')), 'Frozen model state changed')
    final_digest = _tensor_digest(model.state_dict().items())
    save_backend(model, config, initialization, output / 'final.pt', int(cfg['epochs']),
                 extra=dict(updates=updates, data=data, fresh_training=True, historical_control_reused=False))
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(output / 'final.pt', config, device)
    _require(_tensor_digest(reloaded.state_dict().items()) == final_digest, 'Strict R1 reload differs')
    status.update(status='complete', strict_reload_equal=True, final_state_sha256=final_digest,
                  final_checkpoint_sha256=_sha(output / 'final.pt'))
    _write(output / 'status.json', status)
    return status


def run_stage(stage, context):
    """返回小回执；配置与产物只写 context 指定的本轮私有目录。"""
    _context(context)
    _require(stage in STAGES, 'Unsupported backend stage: ' + str(stage))
    if stage == 'd5a':
        config = prepare_d5a_config(context)
        output = _output(context, 'd5a', stage)
        _require(not output.exists(), 'Fresh D5a output already exists')
        config_path = _save_config(context, stage, config)
        from train_rgb_diffusion_restoration import train
        summary = train(config, device=torch.device('cuda'), output_dir=str(output))
        target = _path(context, 'artifacts', 'd5a', existing=True)
        _require(summary.get('training_complete') is True and summary.get('total_updates', 0) > 0,
                 'D5a did not complete 60 epochs / 15000 updates')
        return dict(format='current_training_stage_receipt_v1', stage=stage, complete=True,
                    config=str(config_path), config_sha256=_sha(config_path), checkpoint=str(target),
                    checkpoint_sha256=_sha(target), updates=summary['total_updates'])
    if stage == 'r1':
        config = prepare_r1_config(context)
        config_path = _save_config(context, stage, config)
        report = train_fresh_r1(config, context, _output(context, 'r1', stage))
        return dict(format='current_training_stage_receipt_v1', stage=stage, complete=True,
                    config=str(config_path), config_sha256=_sha(config_path),
                    checkpoint=str(_path(context, 'artifacts', 'r1')),
                    checkpoint_sha256=report['final_checkpoint_sha256'], updates=report['updates'],
                    strict_reload_equal=report['strict_reload_equal'], historical_control_reused=False)
    from tools.build_semifinal_bundle import export_checkpoint
    backend = _path(context, 'artifacts', 'backend')
    restorer = _path(context, 'artifacts', 'restorer')
    output = _absolute(context['run_dir'], 'run_dir') / 'export_backend'
    _require(restorer.parent == backend.parent,
             'Backend/restorer exports must share the stage model directory')
    _require(not output.exists(), 'Fresh export output already exists')
    r1 = _path(context, 'artifacts', 'r1', existing=True)
    d5a = _path(context, 'artifacts', 'd5a', existing=True)
    status = json.loads((r1.parent / 'status.json').read_text(encoding='utf-8'))
    _require(status.get('format') == 'current_r1_fresh_training_v1' and status.get('status') == 'complete'
             and status.get('strict_reload_equal') is True
             and status.get('final_checkpoint_sha256') == _sha(r1),
             'Export requires the complete strictly reloaded fresh R1 checkpoint')
    rows = [export_checkpoint(r1, backend, 'backend'),
            export_checkpoint(d5a, restorer, 'restorer')]
    for row in rows:
        row['path'] = str(backend if row['kind'] == 'backend' else restorer)
        row['parameters'] = int(status[row['kind'] + '_parameters'])
        row['parameter_upper_bound'] = row['tensor_elements']
        _require(0 < row['parameters'] <= row['parameter_upper_bound'], 'Invalid exported parameter count')
        expected_state = status['final_state_sha256'] if row['kind'] == 'backend' else status['restorer_state_sha256']
        _require(row['state_sha256'] == expected_state, 'Exported model differs from the R1 audited state')
    _require(sum(row['parameters'] for row in rows) < 500_000_000,
             'Backend/restorer exceed the parameter budget before adding NAF')
    report = dict(format='current_training_stage_receipt_v1', stage=stage, complete=True,
                  precision='FP32', strict_tensor_export=True, models={row['kind']: row for row in rows})
    _write(output / 'export_receipt.json', report)
    _write(output / 'export_manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--context', type=Path, required=True)
    args = parser.parse_args(argv)
    context = json.loads(args.context.read_text(encoding='utf-8'))
    print(json.dumps(run_stage(args.stage, context), ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
