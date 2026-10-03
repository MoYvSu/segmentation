# -*- coding: utf-8 -*-
"""只读定位距离监督与融合；固定模型，不训练、不扫描部署阈值。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


class TrustedManualView(Dataset):
    """保留训练读图/裁块，只把原标未知域从诊断关系中移除。"""
    def __init__(self, base, target_dir):
        self.base, self.target_dir = base, Path(target_dir)

    def __len__(self):
        return len(self.base)

    @property
    def epoch(self):
        return getattr(self.base, 'epoch', 0)

    @epoch.setter
    def epoch(self, value):
        self.base.epoch = value

    @property
    def draw(self):
        return getattr(self.base, 'draw', 0)

    @draw.setter
    def draw(self, value):
        self.base.draw = value

    def __getitem__(self, index):
        from utils.offset_letterbox import letterbox_instance_geometry
        sample = dict(self.base[index])
        stem = Path(sample['image_path']).stem
        with np.load(self.target_dir / (stem + '_gt.npz'), allow_pickle=False) as blob:
            ids = blob['instance_map']
            trusted = blob['original_covered'].astype(bool) & (ids > 0)
            ids = np.where(trusted, ids, 0)
        y, x, y1, x1 = map(int, sample['crop_box'])
        labels, valid, _ = letterbox_instance_geometry(ids[y:y1, x:x1], 1024, 512)
        label_tensor = torch.from_numpy(labels.astype(np.int64))
        valid_tensor = torch.from_numpy(valid)[None]
        sample.update(diagnostic_instance_map=label_tensor, diagnostic_valid_content=valid_tensor)
        return sample


def distance_reader_class():
    from data.affinity_native import NativeDegradationDataset
    class DistanceDegradationDataset(NativeDegradationDataset):
        _SPATIAL_KEYS = (*NativeDegradationDataset._SPATIAL_KEYS,
                         'diagnostic_instance_map', 'diagnostic_valid_content')
    return DistanceDegradationDataset


def canonical_sample(sample, kind, view, base):
    """SAM2 whole没有native附加别名；显式使用各自的原数据接口。"""
    sample = dict(sample)
    if 'image_path' not in sample:
        if kind != 'pseudo' or view != 'whole':
            raise RuntimeError('Unexpected missing source path')
        source = base.base
        sample['image_path'] = str(source._source_path(source.rows[int(sample['source_index'])]))
    ids = sample['instance_map' if kind == 'pseudo' else 'affinity_instance_map']
    content = sample['valid_content' if kind == 'pseudo' else 'affinity_valid_content']
    return sample, ids, content


@contextmanager
def gradient_state(model, seed):
    """模拟affinity训练模式；只求梯度，退出恢复BN、模式和requires_grad。"""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.affinity_decoder.buffers()]
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32, matmul_tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            # TF32卷积反传的舍入会破坏三次梯度相加的数值线性；仅诊断使用严格FP32。
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)
            model.eval().requires_grad_(False)
            model.affinity_decoder.train()
            shared = model.affinity_decoder.affinity_head[0].weight
            shared.requires_grad_(True)
            yield {'affinity_decoder.affinity_head.0.weight': shared}
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = benchmark


def build_views(config, seed):
    from data.affinity_mixed import build_datasets
    from utils.config import project_path
    manual, pseudo, metadata = build_datasets(config, 1024)
    if len(manual.whole.base_dataset) != 32 or len(pseudo.whole.base_dataset) != 64:
        raise RuntimeError('Expected exactly 32 manual and 64 confirmed SAM2 sources')
    selected = dict(manual=list(range(32)), pseudo=sorted(np.random.default_rng(seed).choice(64, 8, replace=False).tolist()))
    gradient_manual = sorted(np.random.default_rng(seed + 1).choice(32, 8, replace=False).tolist())
    readers = {}
    for kind, original in [('manual', manual), ('pseudo', pseudo)]:
        for view, original_reader in [('whole', original.whole), ('native1024', original.native)]:
            for profile in ['clean', 'blur']:
                base = original_reader.base_dataset
                if kind == 'manual':
                    base = TrustedManualView(base, project_path(config, config['backend_adaptation']['completed_gt_dir']))
                recipe = deepcopy(original_reader.degradation)
                recipe['profile_probabilities'] = [1., 0., 0., 0.] if profile == 'clean' else [0., 1., 0., 0.]
                reader = distance_reader_class()(base, recipe, original_reader.noise,
                    original_reader.seed, original_reader.repeats, True)
                reader.set_epoch(1)
                readers[kind, view, profile] = reader
    return readers, selected, gradient_manual, metadata


def probe(models, restorer, config, out, seed):
    from models.backend_adaptation import restore_first
    from tools.probe_affinity_distance_fusion import fusion_diagnostic
    from tools.probe_affinity_distance_gradients import probe_distance_gradients
    from tools.probe_affinity_equal_recall import curve_metrics, equal_recall
    from train_backend_adaptation import fusion_options, tensor_digest
    from utils.affinity_loss import build_affinity_targets_torch
    readers, selected, gradient_manual, data = build_views(config, seed)
    mode, options = fusion_options(config)
    if mode != 'gated' or options['short_reduction'] != 'top2':
        raise RuntimeError('Expected fixed gated/top2 deployment')
    losses = config['direct_semantic_affinity']['affinity_loss']
    reference = {k: losses[k] for k in ['negative_weight', 'hard_negative_weight', 'hard_negative_gamma', 'normalize_edge_weights']}
    if any(reference[k] != v for k, v in [('negative_weight', 1.), ('hard_negative_weight', 1.), ('hard_negative_gamma', 2.)]):
        raise RuntimeError('Loss diagnostic differs from scored r1/h1/gamma2 control')
    rows, gradients, pooled, receipts, pair_receipts = [], [], {}, [], {}
    out.mkdir()
    for kind in ['manual', 'pseudo']:
        for index in selected[kind]:
            for view in ['whole', 'native1024']:
                for profile in ['clean', 'blur']:
                    reader = readers[kind, view, profile]
                    sample, training_ids, training_content = canonical_sample(reader[index], kind, view, reader.base_dataset)
                    target, valid = build_affinity_targets_torch(
                        sample.get('diagnostic_instance_map', training_ids)[None],
                        sample.get('diagnostic_valid_content', training_content)[None])
                    training_target, training_valid = build_affinity_targets_torch(
                        training_ids[None], training_content[None])
                    if sample['profile'] != ('identity' if profile == 'clean' else 'blur'):
                        raise RuntimeError('Forced profile failed')
                    paired = dict(target_sha256=digest(target.numpy()), valid_sha256=digest(valid.numpy()),
                        training_target_sha256=digest(training_target.numpy()), training_valid_sha256=digest(training_valid.numpy()),
                        crop=sample['crop_box'].tolist(), flips=[sample['horizontal_flip'], sample['vertical_flip'], sample['rotation_k']])
                    key = kind, index, view
                    if profile == 'clean':
                        pair_receipts[key] = paired
                    elif pair_receipts[key] != paired:
                        raise RuntimeError('Clean/blur geometry or supervision differs')
                    identity = dict(kind=kind, source=Path(sample['image_path']).stem, index=index, view=view, profile=profile)
                    tensor = sample['image'][None].to('cuda')
                    restoration_seed = seed + (0 if kind == 'manual' else 1000000) + index * 100 + (view == 'native1024') * 10 + (profile == 'blur')
                    with torch.no_grad():
                        restored = restore_first(restorer, tensor, restoration_seed)
                        features = models['control'].encoder(restored)
                    # 编码器完全相同且冻结，两个头复用同一份实际特征。
                    receipt = dict(**identity, **paired, input_sha256=tensor_digest([('input', tensor)]),
                        restored_sha256=tensor_digest([('restored', restored)]), restoration_seed=restoration_seed,
                        base_sigma=sample['base_sigma'], spatial=sample['is_spatial'], endpoint=sample['endpoint_applied'],
                        noise_sigma=sample['noise_sigma'])
                    receipts.append(receipt)
                    with torch.no_grad():
                        logits = models['control'].affinity_decoder(features)['affinity_logits'].float()
                    if logits.shape != (1, 8, 512, 512) or not torch.isfinite(logits).all():
                        raise RuntimeError('Affinity output grid or values invalid')
                    result = fusion_diagnostic(logits, target.to('cuda'), valid.to('cuda'), options)
                    rows.append(dict(**identity, **result['report']))
                    for variant, domains in result['histograms'].items():
                        for domain, hist in domains.items():
                            hkey = kind, view, profile, variant, domain
                            pooled[hkey] = pooled.get(hkey, np.zeros_like(hist)) + hist
                    if kind == 'pseudo' or index in gradient_manual:
                        for state, model in models.items():
                            with gradient_state(model, restoration_seed + 5000000) as parameters:
                                model.affinity_decoder.eval()
                                with torch.no_grad():
                                    strict_reference = model.affinity_decoder(features)['affinity_logits'].float()
                                model.affinity_decoder.train()
                                training_logits = model.affinity_decoder(features)['affinity_logits'].float()
                                if not torch.equal(training_logits.detach(), strict_reference):
                                    raise RuntimeError('Training/eval head logits differ; diagnostic contract requires review')
                                gradient_target, gradient_valid = training_target.to('cuda').float(), training_valid.to('cuda')
                                edge_weight = torch.ones_like(training_logits)
                                if kind == 'pseudo':
                                    edge_weight[gradient_valid & (gradient_target <= .5)] = float(losses['pseudo_negative_weight'])
                                gradient = probe_distance_gradients(training_logits, gradient_target,
                                    gradient_valid, parameters, edge_weight=edge_weight, loss_options=reference, retain_graph=False)
                            gradients.append(dict(**identity, state=state,
                                diagnostic_mode='strict FP32 training head; TF32/BN/buffers/RNG restored; no optimizer',
                                training_eval_logits_exact=True,
                                max_abs_difference_from_deployment_logits=(float((training_logits.detach() - logits).abs().max()) if state == 'control' else None), **gradient))
                    write(out / 'status.json', dict(status='running', fusion_views=len(rows), gradient_views=len(gradients)))
                    print('PROBE', kind, index, view, profile, len(rows), len(gradients), flush=True)
    summary = []
    for key, hist in pooled.items():
        kind, view, profile, variant, domain = key
        summary.append(dict(kind=kind, view=view, profile=profile, variant=variant, domain=domain,
                            **curve_metrics(hist), equal_recall=[equal_recall(hist, q) for q in [.9, .95, .97, .98]]))
    result = dict(complete=True, training=False, selected=selected, gradient_manual=gradient_manual, data=data,
        fusion_views=len(rows), gradient_views=len(gradients), receipts=receipts, rows=rows,
        gradients=gradients, pooled=summary, fusion_options=options, reference_loss=reference,
        clean_blur_geometry_equal=True, manual_unknown_original_covered_ignored=True,
        pseudo_semantic_labels_used=False, source_coefficients=dict(manual=1., pseudo=.5))
    result['gradient_supervision'] = 'existing completed canonical training targets; zero/covered-uncovered pairs ignored'
    result['fusion_supervision'] = 'original-covered manual targets / confirmed SAM2 geometry; independent common short-domain'
    write(out / 'report.json', result)
    np.savez_compressed(out / 'histograms.npz', **{'__'.join(k): v for k, v in pooled.items()})
    write(out / 'status.json', dict(status='complete', fusion_views=len(rows), gradient_views=len(gradients)))
    return result


def final_gt(model, restorer, config, out):
    from tools.affinity_native_views import predict_views, verify_prediction
    from tools.analyze_affinity_connectivity import load_directory, render
    from tools.analyze_affinity_sweep_gt import geometry, paired_changes
    from tools.probe_affinity_patch import load_gt
    from tools.probe_semantic_fixed import class_independent_pairs, fixed_confusion
    from utils.config import project_path
    from utils.patch_diagnostic import trusted_phase_metrics
    cases = [r for r in read(project_path(config, config['affinity_native']['prior_probe'], 'cases.json')) if r['kind'] == 'train']
    if len(cases) != 4:
        raise RuntimeError('Four unchanged seen-GT cases required')
    reference = read(ROOT / 'outputs/affinity_sweep/analysis/gt/report.json')
    refrows = {r['source']: r for r in reference['rows']}
    short_config = deepcopy(config)
    short_config['affinity_deployment']['fusion_mode'] = 'short'
    rows = []
    out.mkdir()
    for case in cases:
        target, trusted, classes = load_gt(case, config)
        fixed, short_fixed = {}, {}
        rgb, baseline = predict_views(model, restorer, case['path'], config, 'cuda', case['seed'], [1024], fixed_audit=fixed)
        _, candidate = predict_views(model, restorer, case['path'], short_config, 'cuda', case['seed'], [1024], fixed_audit=short_fixed)
        cache = ROOT / 'outputs/affinity_balance/analysis/train' / case['name'] / 'mixed_local'
        cached = load_directory(cache, case['name'])
        current = baseline['patch1024']
        if not np.array_equal(cached[0], current['instances']) or cached[1] != current['classes']:
            raise RuntimeError('Scored control final output mismatch: ' + case['name'])
        if fixed != short_fixed or fixed != refrows[case['name']]['fixed_semantic_inputs']:
            raise RuntimeError('Semantic inputs changed during fusion diagnostic')
        predictions = {'gated': (current['instances'], current['classes']),
                       'short': (candidate['patch1024']['instances'], candidate['patch1024']['classes'])}
        row = dict(source=case['name'], fixed_semantic_inputs=fixed, control_final_output_exact=True, views={})
        for name, (ids, labels) in predictions.items():
            verify_prediction(ids, labels, rgb.shape[:2])
            pairs, summary = class_independent_pairs(target, trusted, ids)
            row['views'][name] = dict(geometry=geometry(target, trusted, ids, pairs, summary), pairs=pairs,
                class_aware=trusted_phase_metrics(target, classes, trusted, ids, labels),
                fixed_confusion=fixed_confusion(pairs, classes, labels))
        if row['views']['gated'] != {k: refrows[case['name']]['views']['r1'][k] for k in row['views']['gated']}:
            raise RuntimeError('Reference GT metrics no longer reproduce cache')
        row['paired_changes'] = paired_changes(row['views']['gated'], row['views']['short'], classes, predictions['gated'][1], predictions['short'][1])
        shown = np.where(trusted, target, 0).astype(np.uint16)
        shown_classes = {str(int(g)): classes[int(g)] for g in np.unique(shown) if g > 0}
        for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
            render(case['path'], [(shown, shown_classes), *predictions.values()], ['GT known only', 'control gated', 'control short only'],
                   out / (case['name'] + '_' + suffix + '.png'), case['name'] + ' SEEN / unknown ignored', crop=crop, width=380)
        rows.append(row)
        print('GT', case['name'], flush=True)
    result = dict(complete=True, seen_training_not_validation=True, unknown_ignored=True, rows=rows)
    write(out / 'report.json', result)
    return result


def main(args):
    from models.backend_adaptation import build_backend, load_backend
    from tools.analyze_affinity_p100 import preflight, verify_sources
    from tools.run_affinity_sweep import runtime_contract
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_backend_adaptation import sha, tensor_digest
    from train_direct_semantic_affinity import set_seed
    from utils.config import load_config
    if not torch.cuda.is_available():
        raise RuntimeError('Use GPU server sam2_env')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    out = ROOT / args.output
    if out.exists() or not out.resolve().is_relative_to((ROOT / 'outputs').resolve()):
        raise RuntimeError('Use a fresh ignored outputs directory')
    out.mkdir()
    started = time.time()
    write(out / 'status.json', dict(status='running', stage='preflight', training=False))
    try:
        pconfig = load_config('config/train/affinity_source_p100.yaml')
        checks = preflight(pconfig, ROOT / 'outputs/affinity_p100')
        config = checks['configs']['r1']
        sources = {name: sha(ROOT / name) for name in ['tools/run_affinity_distance_diagnostic.py', 'tools/probe_affinity_distance_fusion.py', 'tools/probe_affinity_distance_gradients.py']}
        for name in sources:
            path = out / 'source' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        set_seed(config['backend_adaptation']['seed'])
        initial, metadata = build_backend(config, 'cuda')
        control, _ = load_backend(checks['folders']['r1'] / 'final.pt', config, 'cuda')
        models = {'initial': initial, 'control': control}
        state = checks['states']['r1']
        if tensor_digest(initial.state_dict().items()) != state['initial_state_sha256']:
            raise RuntimeError('Common original initialization mismatch')
        if metadata != state['initialization'] or tensor_digest(control.state_dict().items()) != state['final_state_sha256']:
            raise RuntimeError('Initialization provenance or control final state differs')
        if any(frozen_digest(m) != state['frozen_state_sha256'] for m in models.values()):
            raise RuntimeError('Encoder/LoRA/semantic frozen dependency differs')
        restorer = get_restorer(config, 'cuda')
        total = sum(p.numel() for p in control.parameters()) + sum(p.numel() for p in restorer.parameters())
        if total >= 500_000_000:
            raise RuntimeError('Competition parameter budget exceeded')
        if tensor_digest(restorer.state_dict().items()) != state['restorer_state_sha256']:
            raise RuntimeError('D5a state differs')
        before = {name: tensor_digest(model.state_dict().items()) for name, model in models.items()}
        restorer_before = tensor_digest(restorer.state_dict().items())
        report = dict(training=False, official_accuracy=False, seed=args.seed, config=config, sources=sources,
            parent_pipeline_sha256=checks['pipeline_sha256'], parent_sources=checks['sources'],
            runtime=checks['runtime'], control=checks['identities']['r1'], initial_metadata=metadata)
        write(out / 'preflight.json', report)
        report['probe'] = probe(models, restorer, config, out / 'probe', args.seed)
        if report['probe']['data'] != state['data']:
            raise RuntimeError('Actual source registration differs from scored control training')
        report['gt'] = final_gt(control, restorer, config, out / 'gt')
        if any(tensor_digest(m.state_dict().items()) != before[name] for name, m in models.items()) or tensor_digest(restorer.state_dict().items()) != restorer_before:
            raise RuntimeError('Diagnostic mutated model or restoration tensors')
        verify_sources(ROOT / 'outputs/affinity_p100', checks['sources'])
        if runtime_contract(config) != checks['runtime']:
            raise RuntimeError('Runtime changed during diagnostic')
        if (sha(ROOT / 'outputs/affinity_p100/pipeline_status.json') != checks['pipeline_sha256']
                or any(sha(ROOT / name) != expected or sha(out / 'source' / name) != expected for name, expected in sources.items())):
            raise RuntimeError('Source or formal receipt changed during diagnostic')
        report.update(complete=True, elapsed_seconds=time.time() - started, model_states_unchanged=True,
            source_and_parent_receipts_unchanged=True, no_optimizer_steps=True,
            parameter_count=sum(p.numel() for p in control.parameters()) + sum(p.numel() for p in restorer.parameters()),
            caveats=['已见可信人工域与确认SAM2几何候选，不是独立验证或官方成绩。',
                     '梯度取训练模式，同次forward比较三损失；BN/随机数已恢复，无参数更新。',
                     '距离以512输出格点计算；只读short剥离不改变当前最佳部署。'])
        write(out / 'report.json', report)
        gallery = ['<!doctype html><meta charset="utf-8"><title>Affinity distance diagnostic</title>',
                   '<h1>固定control：短距离与门控融合</h1><p>已见GT机制诊断，未知区忽略；无训练／正式分数。</p><a href="report.json">报告</a>']
        for path in sorted((out / 'gt').glob('*.png')):
            rel = path.relative_to(out).as_posix()
            gallery.append('<p>' + html.escape(path.stem) + '<br><img width="1520" src="' + rel + '"></p>')
        (out / 'index.html').write_text('\n'.join(gallery), encoding='utf8')
        write(out / 'status.json', dict(status='complete', stage='done', training=False, elapsed_seconds=report['elapsed_seconds']))
        with zipfile.ZipFile(out / 'report.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.png', '.html', '.npz'} and 'source' not in path.relative_to(out).parts:
                    archive.write(path, path.relative_to(out).as_posix())
        print('COMPLETE distance diagnostic', report['elapsed_seconds'], flush=True)
    except Exception as exc:
        write(out / 'status.json', dict(status='failed', stage='error', training=False, error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/affinity_distance')
    parser.add_argument('--seed', type=int, default=20261002)
    main(parser.parse_args())
