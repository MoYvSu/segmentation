# -*- coding: utf-8 -*-
"""新 native/mixed balance 的同视野连续边界固定融合；不训练、不自动提交。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REGISTERED = {
    'native': '2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b',
    'mixed': '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434',
}
D5A_SHA256 = '048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a'
EXPECTED_PARAMETERS = 90_621_069
VARIANTS = ('native', 'mixed', 'half')
HYPOTHESIS = ('相同视野和输入下，两种训练视野历史产生的边界误差可能互补；'
              '固定各半连续概率融合有望降低单头偶发响应，同时保留共同支持的真实边界。'
              '也可能削弱仅单头识别的弱边界，因此需检查新增粘连和误切。')
SOURCE_NAMES = (
    'tools/probe_affinity_balance_fusion.py', 'tools/affinity_native_views.py',
    'tools/probe_affinity_patch.py', 'tools/affinity_connectivity_views.py',
    'tools/analyze_affinity_connectivity.py', 'tools/analyze_affinity_pair_gt.py',
    'tools/probe_semantic_fixed.py', 'tools/run_affinity_preserve.py',
    'tools/semantic_crossover.py', 'tools/render_backend_ablation.py',
    'models/backend_adaptation.py', 'models/fused_deployment.py',
    'models/affinity_geometry.py', 'models/sam2_encoder.py', 'models/lora.py',
    'models/semantic_lora.py', 'models/fpn_decoder.py',
    'models/rgb_restoration.py', 'models/rgb_diffusion_restoration.py',
    'train_affinity_connectivity.py', 'train_backend_adaptation.py',
    'utils/affinity_fusion.py', 'utils/affinity_deployment.py',
    'utils/patch_diagnostic.py', 'utils/post_process.py',
    'utils/semantic_vote.py', 'utils/marker_restoration.py', 'utils/offset_letterbox.py',
    'utils/semantic_challenger.py', 'utils/config.py',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_receipt(value, *, hash_values=True):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
        raise ValueError('Expected a finite tensor')
    result = dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                  device=str(value.device), contiguous=value.is_contiguous())
    if hash_values:
        result['sha256'] = tensor_sha(value)
    return result


def named_feature_tensors(value, prefix='features'):
    """只读遍历特征，不改变容器、张量布局或数值。"""
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, dict):
        result = []
        for name, item in sorted(value.items()):
            result.extend(named_feature_tensors(item, prefix + '.' + str(name)))
        return result
    if isinstance(value, (list, tuple)):
        return [entry for index, item in enumerate(value)
                for entry in named_feature_tensors(item, prefix + '.' + str(index))]
    raise ValueError('Unexpected encoder feature container')


def feature_receipt(features, *, hash_values=True):
    return {name: tensor_receipt(value, hash_values=hash_values)
            for name, value in named_feature_tensors(features)}


def fuse_probability(native, mixed, mixed_weight=.5):
    """两个同坐标连续概率图；端点原样复制，禁止先阈值化。"""
    native, mixed = np.asarray(native), np.asarray(mixed)
    weight = float(mixed_weight)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Invalid fusion weight')
    if native.ndim != 2 or native.shape != mixed.shape:
        raise ValueError('Fusion needs equal two-dimensional maps')
    if any(not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
           for p in (native, mixed)):
        raise ValueError('Invalid boundary probability')
    if weight == 0:
        return native.copy()
    if weight == 1:
        return mixed.copy()
    return ((1 - weight) * native + weight * mixed).astype(np.float32)


@torch.no_grad()
def shared_boundary_grids(model, mixed_head, image, *, mode, fusion_kwargs, output_grid=None, audit_values=True):
    """同一次 encoder 的同一特征对象交给两头，分别在原格点融合成标量。"""
    from utils.affinity_fusion import affinity_boundary_probability

    if (image.dtype != torch.float32 or image.ndim != 4 or image.shape[:2] != (1, 3)
            or not torch.isfinite(image).all().item()):
        raise ValueError('Expected finite FP32 [1,3,H,W] image')
    if model.affinity_decoder is mixed_head:
        raise ValueError('Two independent affinity heads are required')
    for name in ('geometry_feature_adapter', 'geometry_highres_refiner', 'semantic_lora'):
        if getattr(model, name, None) is not None:
            raise ValueError('Unexpected deployed extra feature route: ' + name)
    features = model.encoder(image)
    before = feature_receipt(features, hash_values=audit_values)
    maps, receipts = {}, {}
    for name, head in (('native', model.affinity_decoder), ('mixed', mixed_head)):
        output = head(features)
        if not isinstance(output, dict) or 'affinity_logits' not in output:
            raise ValueError('Affinity decoder must return affinity_logits')
        logits = output['affinity_logits'].float()
        if (logits.ndim != 4 or logits.shape[:2] != (1, 8)
                or not torch.isfinite(logits).all().item()
                or (output_grid is not None and logits.shape[-2:] != (output_grid, output_grid))):
            raise ValueError('Wrong affinity logits shape or nonfinite value')
        grid = affinity_boundary_probability(logits, mode=mode, **fusion_kwargs)
        if (grid.shape != (1, 1, *logits.shape[-2:]) or not torch.isfinite(grid).all().item()
                or (grid < 0).any().item() or (grid > 1).any().item()):
            raise ValueError('Invalid fused boundary probability')
        maps[name] = grid
        receipts[name] = dict(logits=tensor_receipt(logits, hash_values=audit_values),
                              boundary=tensor_receipt(grid, hash_values=audit_values))
    if maps['native'].shape != maps['mixed'].shape:
        raise ValueError('Two affinity heads use different output grids')
    if before != feature_receipt(features, hash_values=audit_values):
        raise RuntimeError('Affinity head changed shared features')
    return maps, dict(features=before, heads=receipts, encoder_calls=1,
                      native_head_calls=1, mixed_head_calls=1, shared_feature_layout_unchanged=True,
                      shared_features_value_check=audit_values)


def deployment_parameters(model, mixed_head, restorer):
    """计实际部署：共享部分一次、两个头各一次、恢复器一次。"""
    if model.affinity_decoder is mixed_head:
        raise ValueError('Two independent affinity heads are required')
    groups = dict(shared_without_affinity=[p for name, p in model.named_parameters()
                          if not name.startswith('affinity_decoder.')],
                  native_affinity=list(model.affinity_decoder.parameters()),
                  mixed_affinity=list(mixed_head.parameters()), d5a=list(restorer.parameters()))
    used, result = set(), {}
    for name, values in groups.items():
        identities = {id(p) for p in values}
        if len(identities) != len(values) or used & identities:
            raise ValueError('Deployment component parameters overlap: ' + name)
        used |= identities
        result[name] = sum(p.numel() for p in values)
    if result['shared_without_affinity'] + result['native_affinity'] != sum(p.numel() for p in model.parameters()):
        raise ValueError('Shared model parameter accounting is incomplete')
    result['total'] = sum(result.values())
    actual = {id(p): p for module in (model, mixed_head, restorer) for p in module.parameters()}
    unique_count = sum(p.numel() for p in actual.values())
    if set(actual) != used or unique_count != result['total']:
        raise ValueError('Deployment sum differs from actual unique parameters')
    result['sum'] = result['total']
    result['actual_unique_parameters'] = unique_count
    result['constraint_passed'] = result['total'] < 500_000_000
    if not result['constraint_passed']:
        raise ValueError('Combined deployment exceeds competition parameter limit')
    return result


def module_receipt(module):
    from train_backend_adaptation import tensor_digest
    return dict(state_sha256=tensor_digest(module.state_dict().items()),
                requires_grad={name: p.requires_grad for name, p in module.named_parameters()},
                training={name: child.training for name, child in module.named_modules()},
                trainable_lora=getattr(module, 'trainable_lora', None))


@contextmanager
def frozen_guard(modules):
    """包括异常路径的只读检查；不切换头、不更改原模块的模式。"""
    before = {name: module_receipt(module) for name, module in modules.items()}
    if any(any(row['requires_grad'].values()) or any(row['training'].values())
           for row in before.values()):
        raise ValueError('All deployment modules must already be frozen and eval')
    audit = dict(before=before)
    try:
        yield audit
    finally:
        after = {name: module_receipt(module) for name, module in modules.items()}
        audit.update(after=after, unchanged=before == after)
        if before != after:
            raise RuntimeError('Frozen deployment state or mode changed')


def test_cases(config):
    from data.mim_dataset import list_images
    from utils.config import project_path
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    if len(files) != 100 or len({path.stem for path in files}) != 100:
        raise RuntimeError('Expected exactly 100 unique semifinal test sources')
    seed = int(config['backend_adaptation']['restoration']['inference_seed'])
    return [dict(kind='test', name=path.stem, path=str(path), index=i, seed=seed+i)
            for i, path in enumerate(files)]


def probe_cases(config, tests):
    from utils.config import project_path
    prior = Path(project_path(config, config['affinity_native']['prior_probe']))
    train = [case for case in read(prior/'cases.json') if case['kind'] == 'train']
    if len(train) != 4 or {case['index'] for case in train} != {0, 10, 20, 31}:
        raise RuntimeError('Expected the original four seen GT sources')
    fixed = ['test_101', 'test_116']
    excluded = set(config['backend_adaptation']['monitor']['images'])
    random = sorted(np.random.default_rng(20261001).choice(
        [case['name'] for case in tests if case['name'] not in excluded], 6, replace=False).tolist())
    selected = set(fixed + random)
    if len(selected) != 8 or not selected.issubset({case['name'] for case in tests}):
        raise RuntimeError('Incomplete fixed test selection')
    return train + [case for case in tests if case['name'] in selected], dict(
        fixed=fixed, random=random, selection_seed=20261001, monitor_excluded=sorted(excluded))


def case_identity(case, config):
    from utils.config import project_path
    result = dict(**case, source_sha256=sha(case['path']))
    if case['kind'] == 'train':
        folder = Path(project_path(config, config['backend_adaptation']['completed_gt_dir']))
        result['gt_sources'] = {name: dict(path=str(folder/name), sha256=sha(folder/name))
                                for name in (case['name']+'_gt.npz', case['name']+'_class.json')}
    return result


def reference_paths(config, case):
    from utils.config import project_path
    if case['kind'] == 'test':
        return {name: Path(project_path(config, 'outputs/affinity_balance', name,
                                       'deployment/patch1024')) for name in REGISTERED}
    return {name: Path(project_path(config, 'outputs/affinity_balance/analysis/train',
                                   case['name'], name+'_local')) for name in REGISTERED}


def cache_identity(config, tests, probe_sources=()):
    """绑定已有100图端点缓存的清单及每个文件，不以新推理替换参考。"""
    from utils.config import project_path
    result = {}
    names = {case['name'] for case in tests}
    for name, digest in REGISTERED.items():
        deployment = Path(project_path(config, 'outputs/affinity_balance', name, 'deployment'))
        manifest = read(deployment/'manifest.json')
        selected = [row['source'] for row in manifest['images'] if row['view'] == 'patch1024']
        if (manifest.get('complete') is not True or manifest.get('epoch') != 20
                or manifest.get('checkpoint_sha256') != digest or manifest.get('precision') != 'FP32'
                or 1024 not in manifest.get('sizes', []) or len(selected) != 100 or set(selected) != names
                or manifest.get('semantic') != 'fixed full-image D5a geometry plus raw final vote'):
            raise RuntimeError('Wrong registered endpoint deployment manifest: ' + name)
        folder = deployment/'patch1024'
        expected = {stem+suffix for stem in names for suffix in ('_inst.png', '_class.json')}
        actual = {p.name for p in folder.iterdir() if p.is_file() and p.suffix in ('.png', '.json')}
        if actual != expected:
            raise RuntimeError('Endpoint cache does not contain exactly 100 PNG/class pairs: ' + name)
        result[name] = dict(path=str(folder), manifest_sha256=sha(deployment/'manifest.json'),
                            files={file: sha(folder/file) for file in sorted(expected)})
    result['seen_training'] = {}
    for case in probe_sources:
        if case['kind'] != 'train':
            continue
        result['seen_training'][case['name']] = {}
        for name, folder in reference_paths(config, case).items():
            result['seen_training'][case['name']][name] = dict(path=str(folder), files={
                case['name']+suffix: sha(folder/(case['name']+suffix))
                for suffix in ('_inst.png', '_class.json')})
    return result


def source_identity(config_path):
    """推理和GT诊断实际依赖、全部继承配置均保留哈希和私有源码副本。"""
    import yaml
    result = {name: sha(ROOT/name) for name in SOURCE_NAMES}
    path = Path(config_path).resolve()
    seen = set()
    while path not in seen:
        seen.add(path)
        name = path.relative_to(ROOT).as_posix()
        result[name] = sha(path)
        base = yaml.safe_load(path.read_text(encoding='utf-8')).get('_base')
        if not base:
            break
        path = (path.parent/base).resolve()
    else:
        raise ValueError('Configuration inheritance cycle')
    return result


def validate_full_gate(probe, config, identities, sources, caches, probe_sources, full_ok):
    """只核技术兼容和显式人工放行；不自动根据loss、GT或目检选择。"""
    if not full_ok:
        raise ValueError('Full inference requires explicit --full-ok after human review')
    state, report = read(probe/'status.json'), read(probe/'report.json')
    if (state.get('status') != 'complete' or report.get('complete') is not True
            or report.get('mode') != 'probe' or report.get('samples') != 12
            or report.get('frozen_exact') is not True or report.get('full_output_endpoint_parity') is not True
            or report.get('sources_unchanged') is not True):
        raise ValueError('Completed 12-source technical probe is required')
    for key, expected in (('checkpoints', identities), ('sources', sources),
                          ('endpoint_caches', caches), ('case_identities', probe_sources)):
        if report.get(key) != expected:
            raise ValueError('Probe identity changed: ' + key)
    if read(probe/'config.json') != json.loads(json.dumps(config)):
        raise ValueError('Resolved configuration differs from the completed probe')
    rows = read(probe/'rows.json')
    if (len(rows) != 12 or {r['source'] for r in rows} != {c['name'] for c in probe_sources}
            or any(r.get('endpoint_parity') != {'native': True, 'mixed': True} for r in rows)):
        raise ValueError('Missing complete final-output endpoint probe receipts')
    if report.get('parameters', {}).get('total') != EXPECTED_PARAMETERS:
        raise ValueError('Probe deployment parameter count differs')
    return dict(human_stage_release=True, release_flag='--full-ok',
                technical_only=True, probe_path=str(probe), probe_report_sha256=sha(probe/'report.json'),
                probe_rows_sha256=sha(probe/'rows.json'), probe_status_sha256=sha(probe/'status.json'))


def geometry_metrics(gt, trust, ids, gt_classes, classes):
    from tools.analyze_affinity_pair_gt import geometry
    from tools.probe_semantic_fixed import class_independent_pairs, fixed_confusion
    from utils.patch_diagnostic import trusted_phase_metrics
    pairs, summary = class_independent_pairs(gt, trust, ids)
    return dict(geometry=geometry(gt, trust, ids, pairs, summary), pairs=pairs,
                fixed_confusion=fixed_confusion(pairs, gt_classes, classes),
                class_aware=trusted_phase_metrics(gt, gt_classes, trust, ids, classes))


def aggregate(records):
    totals = {}
    for name in VARIANTS:
        phase_totals = {}
        for phase in ('ferrite', 'pearlite'):
            values = [r['variants'][name]['phases'][phase] for r in records]
            phase_totals[phase] = {key: sum(v[key] for v in values) for key in ('instances', 'pixels')}
            if phase == 'ferrite':
                phase_totals[phase]['tiny_under200'] = sum(v['tiny_under200'] for v in values)
                phase_totals[phase]['pooled_mean_area'] = (
                    phase_totals[phase]['pixels']/phase_totals[phase]['instances']
                    if phase_totals[phase]['instances'] else None)
        labelled = [r['variants'][name]['known_domain'] for r in records if r['kind'] == 'train']
        known = {}
        if labelled:
            geom = {key: sum(v['geometry'][key] for v in labelled) for key in (
                'gt_instances', 'predicted_instances', 'matched', 'unmatched_gt', 'unmatched_pred',
                'iou_sum', 'split_gt', 'merged_pred')}
            geom.update(matched_iou=geom['iou_sum']/max(1, geom['matched']),
                        gt_penalized_iou=geom['iou_sum']/max(1, geom['gt_instances']))
            known['geometry'] = geom
            known['class_aware'] = {}
            for phase in ('ferrite', 'pearlite'):
                sums = {key: sum(v['class_aware'][phase][key] for v in labelled) for key in (
                    'gt_instances', 'pred_instances_in_known_domain', 'matches', 'iou_sum',
                    'split_gt', 'merged_pred')}
                sums.update(matched_iou=sums['iou_sum']/max(1, sums['matches']),
                            gt_penalized_iou=sums['iou_sum']/max(1, sums['gt_instances']))
                known['class_aware'][phase] = sums
            known['fixed_confusion'] = {key: sum(v['fixed_confusion'][key] for v in labelled)
                                       for key in ('true_F_pred_F', 'true_F_pred_P', 'true_P_pred_F',
                                                   'true_P_pred_P', 'correct', 'wrong')}
        totals[name] = dict(predicted_distribution=phase_totals, known_training_domain=known)
    return totals


def draw_case(out, case, rgb, maps, predictions, gt):
    import cv2
    from tools.analyze_affinity_connectivity import render
    from tools.render_backend_ablation import text_line, write_png
    gallery = out/'gallery'
    shown = [predictions[name] for name in VARIANTS]
    for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
        render(case['path'], shown, ['native balance', 'mixed balance', 'fixed 50/50'],
               gallery/(case['name']+'_'+suffix+'.png'),
               case['name']+' prediction only; no inferred test GT', crop=crop, width=420)
        if gt is not None:
            target, trust, classes = gt
            known = np.where(trust, target, 0).astype(np.uint16)
            known_classes = {str(int(g)): classes[int(g)] for g in np.unique(known) if g > 0}
            render(case['path'], [(known, known_classes), *shown],
                   ['GT original covered only', 'native balance', 'mixed balance', 'fixed 50/50'],
                   gallery/(case['name']+'_known_'+suffix+'.png'),
                   case['name']+' SEEN training / unknown GT ignored', crop=crop, width=350)
    panels = []
    for name in VARIANTS:
        value = cv2.resize(maps[name], (420, round(420*rgb.shape[0]/rgb.shape[1])),
                           interpolation=cv2.INTER_AREA)
        heat = cv2.applyColorMap(np.rint(np.clip(value, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.copyMakeBorder(heat, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        text_line(heat, name+' boundary: fixed 0..1', (5, 21), 420, size=.45)
        panels.append(heat)
    write_png(gallery/(case['name']+'_boundary.png'), np.concatenate(panels, axis=1))


def finish_gallery(out):
    pages = ['<!doctype html><meta charset="utf-8"><title>同视野 balance 固定融合</title>',
             '<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:100%;background:white}section{margin:28px 0}</style>',
             '<h1>native balance / mixed balance / 固定50%融合</h1>',
             '<p>'+html.escape(HYPOTHESIS)+'</p>',
             '<p>仅12源诊断，不训练、不提交。金=铁素体，蓝=珠光体；训练GT原标域以外忽略。测试无GT。</p>',
             '<p><a href="report.json">汇总与技术门槛</a> · <a href="rows.json">逐图诊断</a></p>']
    for path in sorted((out/'gallery').glob('*.png')):
        name = html.escape(path.relative_to(out).as_posix())
        pages.append(f'<section><h2>{html.escape(path.stem)}</h2><img loading="lazy" src="{name}"></section>')
    (out/'index.html').write_text('\n'.join(pages), encoding='utf-8')
    # 仅可传输图册和回执；不加入source/raw/model/预测PNG/NPZ或提交包。
    with zipfile.ZipFile(out/'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in [*out.glob('*.json'), out/'index.html', *(out/'gallery').glob('*.png')]:
            archive.write(path, path.relative_to(out).as_posix())
    with zipfile.ZipFile(out/'report.zip') as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Gallery archive CRC failed')


def run(args):
    import cv2
    from models.backend_adaptation import load_backend, restore_first
    from tools.affinity_connectivity_views import _inference_state, decode_native
    from tools.affinity_native_views import verify_prediction
    from tools.analyze_affinity_connectivity import load_directory, relation
    from tools.probe_affinity_patch import load_gt, tensor_from_rgb
    from tools.run_affinity_preserve import trusted_changes
    from tools.semantic_crossover import revote_instances
    from train_affinity_connectivity import get_restorer
    from train_backend_adaptation import fusion_options, prediction_maps, tensor_digest
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    from utils.config import load_config, project_path
    from utils.patch_diagnostic import BlendMap, phase_description, tile_boxes

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = load_config(args.config)
    opt = config['affinity_balance_fusion']
    if float(opt['mixed_weight']) != .5:
        raise ValueError('Only the authorized fixed 50/50 candidate is supported')
    if (config['sam2'].get('input_normalization', 'legacy_none') != 'legacy_none'
            or float(config['affinity_native']['overlap']) != .25):
        raise ValueError('Deployed normalization or tile overlap changed')
    infer = config['inference']
    if (infer.get('semantic_vote_mode') != 'probability_mean'
            or int(infer.get('semantic_vote_erode_width', 0)) != 0
            or float(infer.get('semantic_vote_threshold', .5)) != .5):
        raise ValueError('The existing raw terminal-vote contract is required')
    restoration = config['backend_adaptation']['restoration']
    if restoration['checkpoint_sha256'] != D5A_SHA256 or restoration['mode'] != 'first_x0':
        raise ValueError('Registered D5a e60 first output is required')
    base = Path(project_path(config, args.output or opt['output_dir']))
    out = base/args.mode
    tests = test_cases(config)
    selected, selection = probe_cases(config, tests)
    cases = selected if args.mode == 'probe' else tests
    identities = {}
    for name, expected in REGISTERED.items():
        path = Path(project_path(config, opt[name+'_checkpoint']))
        if opt[name+'_sha256'] != expected or sha(path) != expected:
            raise RuntimeError('Wrong registered e20 checkpoint: ' + name)
        identities[name] = dict(path=str(path), epoch=20, sha256=expected)
    d5a_path = Path(project_path(config, restoration['checkpoint']))
    if sha(d5a_path) != D5A_SHA256:
        raise RuntimeError('D5a file changed')
    identities['d5a'] = dict(path=str(d5a_path), epoch=60, sha256=D5A_SHA256, mode='first_x0')
    sources = source_identity(args.config)
    caches = cache_identity(config, tests, selected)
    selected_identities = [case_identity(case, config) for case in selected]
    release = dict(human_stage_release=False, technical_only=True)
    if args.mode == 'full':
        probe = Path(project_path(config, args.probe_dir)) if args.probe_dir else base/'probe'
        release = validate_full_gate(probe, config, identities, sources, caches,
                                     selected_identities, args.full_ok)
    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env GPU for the actual deployment diagnostic')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', config)
    write(out/'cases.json', cases)
    write(out/'status.json', dict(status='running', mode=args.mode, training=False, completed=0))
    try:
        model, native_bundle = load_backend(identities['native']['path'], config, 'cuda')
        duplicate, mixed_bundle = load_backend(identities['mixed']['path'], config, 'cpu')
        if (native_bundle['epoch'] != 20 or mixed_bundle['epoch'] != 20
                or native_bundle['architecture'] != mixed_bundle['architecture']
                or native_bundle['deployment'] != mixed_bundle['deployment']):
            raise RuntimeError('Endpoint epoch/architecture/deployed contract differs')
        if (native_bundle['deployment']['fusion'] != config['affinity_deployment']
                or native_bundle['deployment']['inference'] != config['inference']):
            raise RuntimeError('Runtime deployment differs from the saved endpoints')
        shared_digest = lambda m: tensor_digest((n, v) for n, v in m.state_dict().items()
                                                if not n.startswith('affinity_decoder.'))
        shared = shared_digest(model)
        if shared != shared_digest(duplicate):
            raise RuntimeError('Encoder/LoRA/semantic/shared state differs between endpoints')
        if model.encoder.trainable_lora or duplicate.encoder.trainable_lora:
            raise RuntimeError('Frozen encoder must not retain a trainable LoRA route')
        architecture = native_bundle['architecture']
        mixed_head = duplicate.affinity_decoder.to('cuda').eval().requires_grad_(False)
        del duplicate, native_bundle, mixed_bundle
        model.eval().requires_grad_(False)
        restorer = get_restorer(config, 'cuda')
        parameters = deployment_parameters(model, mixed_head, restorer)
        if parameters['total'] != EXPECTED_PARAMETERS:
            raise RuntimeError('Registered deployment parameter count changed: '+str(parameters))
        modules = dict(encoder=model.encoder, semantic=model.semantic_decoder,
                       native_affinity=model.affinity_decoder, mixed_affinity=mixed_head, d5a=restorer)
        source_dir = out/'source'
        for name in sources:
            target = source_dir/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        if args.mode == 'probe':
            (out/'gallery').mkdir()
        mode, kwargs = fusion_options(config)
        write(out/'state.json', dict(checkpoints=identities, architecture=architecture,
              shared_frozen_sha256=shared, parameters=parameters, sources=sources,
              endpoint_caches=caches, stage_release=release))
        records, started = [], time.time()
        # 全量保留布局/调用数/最终端点；12源诊断另记逐张量数值hash，避免全量反复CPU复制。
        audit_values = args.mode == 'probe'
        receipt = lambda tensor: tensor_receipt(tensor, hash_values=audit_values)
        with frozen_guard(modules) as frozen, _inference_state(model, restorer, 'cuda'):
            complete_before = tensor_digest(model.state_dict().items())
            for case in cases:
                current_identity = case_identity(case, config)
                global_inputs = []
                hook = model.encoder.register_forward_pre_hook(
                    lambda module, inp: global_inputs.append(receipt(inp[0])))
                try:
                    rgb, semantic, _ = prediction_maps(model, case['path'], config, 'cuda', restorer, case['seed'])
                finally:
                    hook.remove()
                if len(global_inputs) != 1:
                    raise RuntimeError('Whole semantic route unexpectedly re-encoded the image')
                _, raw, ph, pw = prepare_image(case['path'], 1024, 'cuda')
                raw_logits = model.semantic_decoder(model.encoder(raw.float()), raw.float())
                if isinstance(raw_logits, dict):
                    raw_logits = raw_logits['semantic_logits']
                raw_native = crop_letterbox_output(raw_logits.float(), 1024, ph, pw, rgb.shape[:2]).cpu()
                probability = raw_native.sigmoid()[0, 0].numpy()
                record = dict(source=case['name'], kind=case['kind'], seed=case['seed'],
                              source_identity=current_identity, shape=list(rgb.shape[:2]), tiles=[], variants={},
                              endpoint_parity={}, endpoint_cache_receipts={}, fixed_inputs=dict(
                                  full_image_d5a_encoder_input=global_inputs[0], raw_input=receipt(raw),
                                  geometry_semantic=tensor_receipt(semantic), raw_native_logits=tensor_receipt(raw_native),
                                  raw_probability=tensor_receipt(torch.from_numpy(probability))))
                boxes = tile_boxes(rgb.shape[:2], 1024, config['affinity_native']['overlap'])
                blends = {name: BlendMap(rgb.shape[:2]) for name in REGISTERED}
                for index, box in enumerate(boxes):
                    y, x, y1, x1 = box
                    image, pad_h, pad_w = tensor_from_rgb(rgb[y:y1, x:x1], 'cuda')
                    seed = case['seed'] + 100000 + 1024*1000 + index
                    restored = restore_first(restorer, image, seed)
                    grids, audit = shared_boundary_grids(model, mixed_head, restored, mode=mode,
                                                        fusion_kwargs=kwargs, output_grid=512,
                                                        audit_values=audit_values)
                    for name, grid in grids.items():
                        value = crop_letterbox_output(grid, 1024, pad_h, pad_w, (y1-y, x1-x)).cpu()[0, 0].numpy()
                        blends[name].add(box, value)
                    record['tiles'].append(dict(index=index, box=list(box), seed=seed,
                                                padding=[pad_h, pad_w], input=receipt(image),
                                                restored=receipt(restored), shared_forward=audit))
                completed = {name: blend.finish() for name, blend in blends.items()}
                native, mixed = completed['native'][0], completed['mixed'][0]
                maps = dict(native=fuse_probability(native, mixed, 0), mixed=fuse_probability(native, mixed, 1),
                            half=fuse_probability(native, mixed, .5))
                record['blend'] = {name: value[1] for name, value in completed.items()}
                record['boundaries'] = {name: tensor_receipt(torch.from_numpy(value)) for name, value in maps.items()}
                predictions = {}
                labelled = load_gt(case, config) if args.mode == 'probe' else None
                for name in VARIANTS:
                    ids, _ = decode_native(semantic, torch.from_numpy(maps[name])[None, None], config)
                    classes, _ = revote_instances(ids, probability)
                    verify_prediction(ids, classes, rgb.shape[:2])
                    predictions[name] = ids, classes
                    row = dict(phases=phase_description(ids, classes),
                               ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
                    if labelled is not None:
                        row['known_domain'] = geometry_metrics(*labelled[:2], ids, labelled[2], classes)
                    record['variants'][name] = row
                references = reference_paths(config, case)
                for name, directory in references.items():
                    ids, classes = load_directory(directory, case['name'])
                    verify_prediction(ids, classes, rgb.shape[:2])
                    actual, mapping = predictions[name]
                    exact = bool(np.array_equal(ids, actual) and classes == mapping)
                    record['endpoint_parity'][name] = exact
                    record['endpoint_cache_receipts'][name] = dict(
                        path=str(directory), png_sha256=sha(directory/(case['name']+'_inst.png')),
                        classes_sha256=sha(directory/(case['name']+'_class.json')),
                        changed_pixels=int((ids != actual).sum()), classes_equal=classes == mapping)
                    if not exact:
                        write(out/'failed_endpoint.json', record)
                        raise RuntimeError('Complete final endpoint differs: '+case['name']+'/'+name)
                record['relations'] = {name: relation(predictions[name][0], predictions['half'][0],
                                                       predictions[name][1], predictions['half'][1])
                                       for name in REGISTERED}
                if args.mode == 'probe':
                    folder = out/'predictions'/case['name']
                    folder.mkdir(parents=True)
                    np.savez_compressed(folder/'fullmaps.npz', **maps)
                    for name, (ids, classes) in predictions.items():
                        target = folder/name
                        target.mkdir()
                        if not cv2.imwrite(str(target/(case['name']+'_inst.png')), ids):
                            raise RuntimeError('Could not save final uint16 PNG')
                        write(target/(case['name']+'_class.json'), classes)
                    if labelled is not None:
                        gt, trust, _ = labelled
                        record['trust'] = tensor_receipt(torch.from_numpy(trust))
                        record['trusted_changes'] = {}
                        for name in REGISTERED:
                            forward = trusted_changes(gt, trust, predictions[name][0], predictions['half'][0])
                            reverse = trusted_changes(gt, trust, predictions['half'][0], predictions[name][0])
                            changes = dict(reference_to_half=forward, half_to_reference=reverse)
                            write(folder/('trusted_changes_'+name+'.json'), changes)
                            record['trusted_changes'][name] = dict(reference_to_half=forward['summary'],
                                                                   half_to_reference=reverse['summary'])
                    draw_case(out, case, rgb, maps, predictions, labelled)
                else:
                    folder = out/'patch1024'
                    folder.mkdir(exist_ok=True)
                    ids, classes = predictions['half']
                    if not cv2.imwrite(str(folder/(case['name']+'_inst.png')), ids):
                        raise RuntimeError('Could not save final uint16 PNG')
                    write(folder/(case['name']+'_class.json'), classes)
                if case_identity(case, config) != current_identity:
                    raise RuntimeError('Source image or trusted GT changed while running')
                records.append(record)
                write(out/'rows.json', records)
                write(out/'status.json', dict(status='running', mode=args.mode, training=False,
                      completed=len(records), total=len(cases), source=case['name']))
                print(json.dumps(dict(source=case['name'], endpoints=record['endpoint_parity'],
                      phases={name: row['phases'] for name, row in record['variants'].items()})), flush=True)
            if complete_before != tensor_digest(model.state_dict().items()) or shared != shared_digest(model):
                raise RuntimeError('Complete shared model changed during diagnostic')
        unchanged = (sources == source_identity(args.config) and caches == cache_identity(config, tests, selected)
                     and selected_identities == [case_identity(case, config) for case in selected]
                     and all(sha(row['path']) == row['sha256'] for row in identities.values()))
        if not unchanged:
            raise RuntimeError('Source/checkpoint/reference identity changed during run')
        count = 12 if args.mode == 'probe' else 100
        if len(records) != count:
            raise RuntimeError('Incomplete fixed source cohort')
        report = dict(complete=True, mode=args.mode, training=False, official_submission=False,
                      full_test_inference=args.mode == 'full', samples=len(records), hypothesis=HYPOTHESIS,
                      scope='Seen training original-covered GT only; test distributions/visuals have no GT or accuracy.',
                      mixed_weight=.5, checkpoints=identities, sources=sources, sources_unchanged=True,
                      case_identities=selected_identities, selection=selection, endpoint_caches=caches,
                      parameters=parameters, frozen_exact=frozen['unchanged'], frozen=frozen,
                      full_output_endpoint_parity=True, shared_frozen_sha256=shared, stage_release=release,
                      architecture=architecture, fusion=dict(mode=mode, kwargs=kwargs), totals=aggregate(records),
                      elapsed_seconds=time.time()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
                      matching='Class-independent one-to-one IoU>=0.5, then semantic confusion; unknown ignored.',
                      trusted_changes='Both directions report existing anchor-merge repairs and new split/merge relations; not held-out accuracy.')
        write(out/'report.json', report)
        write(out/'summary.json', dict(scope=report['scope'], totals=report['totals'], images=len(records)))
        if args.mode == 'full':
            pngs = list((out/'patch1024').glob('*_inst.png'))
            classes = list((out/'patch1024').glob('*_class.json'))
            if len(pngs) != 100 or len(classes) != 100:
                raise RuntimeError('Full output must contain exactly 100 PNG/class pairs')
            write(out/'manifest.json', dict(complete=True, epoch=20, precision='FP32', sizes=[1024],
                  checkpoint_sha256={name: row['sha256'] for name, row in identities.items()},
                  mixed_weight=.5, parameters=parameters, source_config=str(args.config),
                  semantic='fixed full-image D5a geometry plus raw final vote', stage_release=release,
                  output_contract='Single uint16 <=65535, pearl0/ferrite1, class IDs exact',
                  full_output_endpoint_parity=True, official_score=None,
                  images=[dict(source=r['source'], view='patch1024', phases=r['variants']['half']['phases'],
                               png_sha256=sha(out/'patch1024'/(r['source']+'_inst.png')),
                               classes_sha256=sha(out/'patch1024'/(r['source']+'_class.json')))
                          for r in records]))
        write(out/'state.json', dict(checkpoints=identities, architecture=architecture,
              shared_frozen_sha256=shared, parameters=parameters, sources=sources,
              endpoint_caches=caches, stage_release=release, frozen=frozen))
        write(out/'status.json', dict(status='complete', mode=args.mode, training=False,
                                     completed=len(records), endpoint_parity=True, frozen_exact=True))
        if args.mode == 'probe':
            finish_gallery(out)
        print('BALANCE FUSION COMPLETE '+json.dumps(dict(mode=args.mode, samples=len(records),
              output=str(out), parameters=parameters, totals=report['totals'])), flush=True)
        return report
    except BaseException as error:
        write(out/'status.json', dict(status='failed', mode=args.mode, training=False, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/inference/affinity_balance_fusion.yaml')
    parser.add_argument('--mode', choices=['probe', 'full'], default='probe')
    parser.add_argument('--output', help='Root containing independent probe/ and full/ directories')
    parser.add_argument('--probe-dir', help='Completed probe gate; default output_root/probe')
    parser.add_argument('--full-ok', action='store_true', help='Explicit human release after inspecting probe')
    run(parser.parse_args())
