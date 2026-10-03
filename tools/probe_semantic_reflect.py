# -*- coding: utf-8 -*-
"""固定已评分reflect实例PNG，仅比较四条已有语义路线；不训练或重跑几何。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_semantic_fixed import (
    ROUTES, changes, class_independent_pairs, copy_fixed_mask, file_sha,
    fixed_confusion, probability_details, write,
)

FORMAT = 'semantic_reflect_probe_v1'
INPUT_FORMAT = 'semantic_reflect_inputs_v1'
SCORED_ZIP_SHA = 'c812fbf32a87ae3a22973b2cd5548a4ab4bf9643dfc4b3136a584942d54da1e2'
NATIVE_SHA = '2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b'
CHECKPOINTS = {
    'baseline': ('outputs/affinity_balance/mixed/final.pt', '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434', 20),
    'lora': ('outputs/semantic_lora/candidate/epoch_020.pt', '23b23d1f0cecb268091943610ceddffd1dffc4371ca548ce725639e2dfc4068b', 20),
    'full': ('outputs/semantic_d5a/full/epoch_060.pt', 'cce91a6643b919e69f6cd1f6f4b331265a33c80842e1587a241386d964f7d2a7', 60),
}
GROUPS = ('unchanged', 'reshaped', 'split', 'merged', 'complex', 'new')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_prediction(ids, classes, shape=None):
    if (ids is None or ids.ndim != 2 or ids.dtype != np.uint16
            or (shape is not None and ids.shape != tuple(shape))
            or int(ids.max()) > 65535
            or set(classes) != {str(int(i)) for i in np.unique(ids) if i > 0}
            or any(type(v) is not int or v not in (0, 1) for v in classes.values())):
        raise RuntimeError('Final single-channel uint16/classes contract differs')


def load_fixed(directory, name):
    directory = Path(directory)
    png, labels = directory/(name+'_inst.png'), directory/(name+'_class.json')
    ids, classes = cv2.imread(str(png), cv2.IMREAD_UNCHANGED), read(labels)
    validate_prediction(ids, classes)
    return ids, classes, png, labels


def partition_mapping(old, current):
    """用像素重叠建立旧新二部对应，不把背景当作父实例或只比较ID。"""
    if old.shape != current.shape or old.ndim != 2:
        raise ValueError('Partition mapping requires identical native grids')
    for value in (old, current):
        if value.dtype != np.uint16:
            raise ValueError('Partition mapping requires uint16 instance maps')
    encoded = (old.astype(np.uint64) << 16) | current.astype(np.uint64)
    values, counts = np.unique(encoded, return_counts=True)
    old_area = np.bincount(old.ravel().astype(np.int64), minlength=65536)
    new_area = np.bincount(current.ravel().astype(np.int64), minlength=65536)
    parents, children, edges = {}, {}, []
    old_background, new_background = {}, {}
    for code, count in zip(values, counts):
        a, b, n = int(code >> np.uint64(16)), int(code & np.uint64(65535)), int(count)
        if a and b:
            parents.setdefault(b, []).append(a)
            children.setdefault(a, []).append(b)
            edges.append(dict(old=a, current=b, pixels=n))
        elif b:
            new_background[b] = n
        elif a:
            old_background[a] = n
    rows = {}
    for b in np.flatnonzero(new_area):
        b = int(b)
        if not b:
            continue
        incoming = parents.get(b, [])
        if not incoming:
            kind = 'new'
        elif len(incoming) == 1:
            a = incoming[0]
            if len(children[a]) == 1:
                kind = ('unchanged' if new_area[b] == old_area[a]
                        and not new_background.get(b, 0) and not old_background.get(a, 0)
                        else 'reshaped')
            else:
                kind = 'split'
        elif all(len(children[a]) == 1 for a in incoming):
            kind = 'merged'
        else:
            kind = 'complex'
        rows[str(b)] = dict(kind=kind, old_ids=incoming, area=int(new_area[b]),
                           old_background_pixels=new_background.get(b, 0))
    # 额外保留显著像素重叠，排除一两个边缘像素造成的多对多假象；
    # 两种图均是分区对应诊断，不代表物理晶界正确性。
    significant = [edge for edge in edges if edge['pixels'] >=
                   max(50, .1*min(old_area[edge['old']], new_area[edge['current']]))]
    significant_parents, significant_children = {}, {}
    for edge in significant:
        significant_parents.setdefault(edge['current'], []).append(edge['old'])
        significant_children.setdefault(edge['old'], []).append(edge['current'])
    for key, row in rows.items():
        incoming = significant_parents.get(int(key), [])
        if row['kind'] == 'unchanged':
            significant_kind = 'unchanged'
        elif not incoming:
            significant_kind = 'no_significant_correspondence'
        elif len(incoming) == 1:
            significant_kind = 'split' if len(significant_children[incoming[0]]) > 1 else 'reshaped'
        else:
            significant_kind = 'merged' if all(len(significant_children[a]) == 1 for a in incoming) else 'complex'
        row.update(significant_old_ids=incoming, significant_kind=significant_kind)
    return dict(current=rows, overlaps=edges, significant_overlaps=significant,
                significant_overlap_rule='pixels>=max(50, .1*min(old_area,current_area)); diagnostic only',
                old_to_current={str(k):v for k, v in children.items()},
                lost_old_ids=[int(a) for a in np.flatnonzero(old_area) if a and a not in children],
                old_pixels_now_background={str(k):v for k, v in old_background.items()},
                group_counts=dict(Counter(row['kind'] for row in rows.values())),
                significant_group_counts=dict(Counter(row['significant_kind'] for row in rows.values())))


def grouped_changes(original, current, details, base_details, mapping):
    result = {}
    for group in GROUPS:
        keys = [key for key, value in mapping['current'].items() if value['kind'] == group]
        if not keys:
            result[group] = dict(instances=0, pixels=0, changed_instances=0, changed_pixels=0,
                                 P_to_F_instances=0, P_to_F_pixels=0, F_to_P_instances=0,
                                 F_to_P_pixels=0, low_confidence_changed=0)
            continue
        subset = lambda value: {key:value[key] for key in keys}
        row = changes(subset(original), subset(current), subset(details), subset(base_details))
        row['pixels'] = sum(details[key]['area'] for key in keys)
        result[group] = row
    return result


def ferrite_composition(classes, details):
    keys = [key for key, value in classes.items() if value == 1]
    areas = np.array([details[key]['area'] for key in keys], dtype=np.int64)
    result = dict(instances=len(keys), pixels=int(areas.sum()),
                  mean_area=float(areas.mean()) if len(areas) else None,
                  quantiles={str(q):float(np.quantile(areas, q)) for q in (.05, .25, .5, .75, .95)} if len(areas) else {},
                  ids=keys, tails={})
    for low, high, label in ((0, 200, 'lt200'), (200, 1000, '200to999'),
                             (1000, 10000, '1000to9999'), (10000, None, 'ge10000')):
        chosen = [key for key in keys if details[key]['area'] >= low
                  and (high is None or details[key]['area'] < high)]
        result['tails'][label] = dict(instances=len(chosen), pixels=sum(details[key]['area'] for key in chosen), ids=chosen)
    return result


def probability_mixture(ids, probability, details):
    """核心/外圈与整粒分歧仅诊断；所有正式类别仍严格整粒均值>0.5。"""
    rows = {}
    for key, value in details.items():
        pixels = probability[ids == int(key)]
        rows[key] = dict(core_vs_all_disagree=value['core_class'] != int(value['probability_all'] > .5),
                         outer_vs_all_disagree=value['outer_class'] is not None
                            and value['outer_class'] != int(value['probability_all'] > .5),
                         ferrite_pixel_fraction=float((pixels > .5).mean()),
                         uncertain_pixel_fraction=float((np.abs(pixels-.5) < .1).mean()),
                         probability_quantiles={str(q):float(np.quantile(pixels, q)) for q in (.1, .5, .9)})
    return dict(instances=rows,
                core_vs_all_disagreements=sum(row['core_vs_all_disagree'] for row in rows.values()),
                outer_vs_all_disagreements=sum(row['outer_vs_all_disagree'] for row in rows.values()))


def touches_frame(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def gt_contract(gt, trust, classes, ids):
    """处理后的GT保持原标信任域；未知、补齐、图框与混合相分别记录。"""
    if gt.shape != ids.shape or trust.shape != ids.shape or gt.dtype != np.uint16:
        raise ValueError('Processed GT and fixed prediction native coordinates differ')
    trust = np.asarray(trust, bool) & (gt > 0)
    if set(classes) != {int(x) for x in np.unique(gt) if x} or any(v not in (0, 1) for v in classes.values()):
        raise ValueError('Processed GT class mapping differs')
    pairs, stats = class_independent_pairs(gt, trust, ids)
    gt_rows, pred_rows = {}, {}
    for value in np.unique(gt):
        if not value:
            continue
        mask = gt == value
        original = mask & trust
        gt_rows[str(int(value))] = dict(area=int(mask.sum()), original_pixels=int(original.sum()),
            filled_pixels=int((mask & ~trust).sum()), frame_censored=touches_frame(mask), cls=int(classes[int(value)]),
            entirely_original=bool(original.sum() == mask.sum()))
    for value in np.unique(ids):
        if not value:
            continue
        mask = ids == value
        covered = mask & trust
        gt_ids = [int(x) for x in np.unique(gt[covered]) if x]
        phases = sorted({classes[x] for x in gt_ids})
        flags = dict(area=int(mask.sum()), original_pixels=int(covered.sum()),
                     unknown_pixels=int((mask & (gt == 0)).sum()),
                     filled_pixels=int((mask & (gt > 0) & ~trust).sum()),
                     frame_censored=touches_frame(mask), known_gt_ids=gt_ids,
                     known_phases=phases, mixed_phase=len(phases) > 1)
        # 不完整、填充或图框区域不自动得到类别真值。
        eligible = covered.sum() == mask.sum() and not flags['frame_censored'] and len(phases) == 1
        flags.update(oracle_eligible=bool(eligible), oracle_class=int(phases[0]) if eligible else None)
        pred_rows[str(int(value))] = flags
    audited = []
    for pair in pairs:
        grow, prow = gt_rows[str(pair['gt'])], pred_rows[str(pair['pred'])]
        flags = []
        if not grow['entirely_original']:
            flags.append('GT_contains_filled_pixels')
        if grow['frame_censored'] or prow['frame_censored']:
            flags.append('frame_censored')
        if prow['unknown_pixels']:
            flags.append('prediction_contains_unknown')
        if prow['filled_pixels']:
            flags.append('prediction_contains_filled_pixels')
        if prow['mixed_phase']:
            flags.append('mixed_phase_prediction')
        audited.append(dict(pair, censorship=flags, uncensored=not flags))
    return dict(pairs=audited, matching=stats, gt_instances=gt_rows, predictions=pred_rows,
                original_pixels=int(trust.sum()), filled_pixels=int(((gt > 0) & ~trust).sum()),
                unknown_pixels=int((gt == 0).sum()),
                oracle_eligible_instances=sum(row['oracle_eligible'] for row in pred_rows.values()),
                pair_scope='class-independent IoU>=.5 in original_covered & GT>0, not official evaluation')


def training_probability_diagnostic(contract, gt, trust, gt_classes, classes, probability):
    pairs = contract['pairs']
    eligible = [pair for pair in pairs if pair['uncensored']]
    confusion = fixed_confusion(pairs, gt_classes, classes)
    clean_confusion = fixed_confusion(eligible, gt_classes, classes)
    truth_ids = [int(key) for key, value in contract['gt_instances'].items() if value['original_pixels'] > 0]
    gt_probability, gt_confusion = {}, Counter()
    for value in [int(key) for key in contract['gt_instances']]:
        mask = gt == value
        known, filled = mask & trust, mask & ~trust
        row = contract['gt_instances'][str(value)]
        score = float(probability[known].mean()) if known.any() else None
        guess = int(score > .5) if score is not None else None
        truth = int(gt_classes[value])
        gt_probability[str(value)] = dict(row, probability_original=score, predicted_original=guess,
            probability_filled=float(probability[filled].mean()) if filled.any() else None,
            probability_completed_region=float(probability[mask].mean()),
            completed_region_class=int(float(probability[mask].mean()) > .5))
        if guess is not None:
            gt_confusion['true_'+('F' if truth else 'P')+'_pred_'+('F' if guess else 'P')] += 1
    correct = [pair for pair in pairs if classes[str(pair['pred'])] == gt_classes[pair['gt']]]
    # 固定GT列表包含未匹配/错相计0；条件均值并列，不能以退出匹配后的均值称改善。
    conditional = float(np.mean([pair['iou'] for pair in correct])) if correct else None
    fixed_sum = sum(pair['iou'] for pair in correct)
    oracle = {key:int(value['oracle_class']) for key, value in contract['predictions'].items()
              if value['oracle_eligible']}
    oracle_flips = [key for key in oracle if oracle[key] != classes[key]]
    oracle_classes = {**classes, **oracle}
    def composition(mapping):
        positive = [key for key, value in mapping.items() if value == 1]
        pixels = sum(contract['predictions'][key]['area'] for key in positive)
        return dict(instances=len(positive), pixels=pixels, mean_area=pixels/len(positive) if positive else None)
    return dict(fixed_pair_confusion=confusion, uncensored_fixed_pair_confusion=clean_confusion,
                fixed_list_class_sensitive_score=fixed_sum/len(truth_ids) if truth_ids else None,
                class_correct_valid_matches=len(correct), conditional_iou_mean=conditional,
                fixed_gt_list_size=len(truth_ids), fixed_geometry_matches=len(pairs),
                GT_region_probability=gt_probability, original_region_class_confusion=dict(gt_confusion),
                pure_known_nonframe_oracle=dict(eligible=len(oracle), changed=len(oracle_flips),
                    changed_pixels=sum(contract['predictions'][key]['area'] for key in oracle_flips),
                    changed_ids=oracle_flips, classes=oracle,
                    original_ferrite_composition=composition(classes), oracle_ferrite_composition=composition(oracle_classes)),
                caveat='Seen training sources; original trusted domain mechanism only. Completed GT probabilities are diagnostic, never added as trusted truth.')


def verify_scored_bytes(archive, case, png, labels):
    """已评分测试PNG和JSON必须逐字节等同，拒绝同像素不同文件的冒充。"""
    by_name = {}
    for name in archive.namelist():
        if name.endswith('/'):
            continue
        stem = Path(name).name
        if stem in by_name:
            raise RuntimeError('Duplicate scored ZIP entry basename')
        by_name[stem] = name
    if len(by_name) != 200 or any(not (name.endswith('_inst.png') or name.endswith('_class.json')) for name in by_name):
        raise RuntimeError('Scored ZIP requires exactly100 PNG/JSON pairs')
    for path in (png, labels):
        if path.name not in by_name or archive.read(by_name[path.name]) != path.read_bytes():
            raise RuntimeError('Fixed scored reflect bytes differ: '+case['name'])


def validate_cases(config, inputs):
    from tools.probe_affinity_scale_fusion import sources
    expected, selection = sources(config)
    cases = inputs.get('cases', [])
    names = [(row['kind'], row['name'], row['seed']) for row in cases]
    wanted = [(row['kind'], row['name'], row['seed']) for row in expected]
    if (inputs.get('format') != INPUT_FORMAT or inputs.get('complete') is not True
            or names != wanted or len(cases) != 12 or len(set(names)) != 12):
        raise RuntimeError('Expected unchanged predetermined four train/eight test source manifest')
    if inputs.get('selection') != selection:
        raise RuntimeError('Source selection metadata differs from predetermined original selection')
    for actual, prior in zip(cases, expected):
        if Path(actual['path']).resolve() != Path(prior['path']).resolve():
            raise RuntimeError('Source path differs from predetermined original source')
        if actual['kind'] == 'test' and actual.get('index') != prior['index']:
            raise RuntimeError('Test full-cohort restoration seed/index differs')
    return cases, selection


def protected_sources(output):
    """封存实际加载的项目源码；动态虚拟路径单列，不保护实验产物。"""
    paths = {Path(__file__).resolve()}
    virtual = []
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and path.relative_to(ROOT).parts[0] not in {'output', 'outputs'}:
                if path.exists():
                    paths.add(path)
                else:
                    virtual.append(path.relative_to(ROOT).as_posix())
    hashes = {}
    for path in sorted(paths):
        key = path.relative_to(ROOT).as_posix()
        target = output/'source'/key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[key] = file_sha(path)
        if file_sha(target) != hashes[key]:
            raise RuntimeError('Copied source snapshot differs')
    return hashes, sorted(set(virtual))


def summarize(records):
    result = {}
    for route in ROUTES:
        train = [row['routes'][route]['training'] for row in records if row['kind'] == 'train']
        test = [row['routes'][route] for row in records if row['kind'] == 'test']
        change_keys = ('instances', 'ferrite', 'ferrite_pixels', 'changed_instances', 'changed_pixels',
                       'P_to_F_instances', 'P_to_F_pixels', 'F_to_P_instances', 'F_to_P_pixels',
                       'low_confidence_changed', 'high_confidence_changed')
        total = {key:sum(row['changes'][key] for row in test) for key in change_keys}
        total['pooled_ferrite_mean_area'] = total['ferrite_pixels']/total['ferrite'] if total['ferrite'] else None
        groups = {}
        for group in GROUPS:
            fields = ('instances', 'pixels', 'changed_instances', 'changed_pixels', 'P_to_F_instances',
                      'P_to_F_pixels', 'F_to_P_instances', 'F_to_P_pixels', 'low_confidence_changed')
            groups[group] = {key:sum(row['partition_group_changes'][group][key] for row in test) for key in fields}
        tails = {key:dict(instances=sum(row['ferrite_composition']['tails'][key]['instances'] for row in test),
                         pixels=sum(row['ferrite_composition']['tails'][key]['pixels'] for row in test))
                 for key in ('lt200', '200to999', '1000to9999', 'ge10000')}
        result[route] = dict(test_unlabeled_changes=total, test_partition_groups=groups,
            test_ferrite_tails=tails,
            training_original_domain_confusion={key:sum(row['fixed_pair_confusion'][key] for row in train)
                for key in train[0]['fixed_pair_confusion']},
            training_uncensored_confusion={key:sum(row['uncensored_fixed_pair_confusion'][key] for row in train)
                for key in train[0]['uncensored_fixed_pair_confusion']},
            test_core_vs_all_disagreements=sum(row['probability_mixture']['core_vs_all_disagreements'] for row in test),
            training_scope='Seen processed training GT, original_covered only; incomplete/filled/frame/mixed separately reported')
    return result


def run(args):
    import torch
    from models.backend_adaptation import load_backend, restore_first
    from models.semantic_lora import semantic_features
    from tools.analyze_affinity_connectivity import render
    from tools.probe_affinity_patch import load_gt
    from tools.run_marker_reflect import config_gate, precision_record, validate_manifest
    from tools.semantic_crossover import revote_instances
    from train_affinity_connectivity import get_restorer
    from train_backend_adaptation import tensor_digest
    from tools.run_affinity_sweep import runtime_contract
    from data.mim_dataset import list_images
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    from utils.config import load_config, project_path

    canonical = load_config(args.config)
    resolve = lambda value: Path(project_path(canonical, value)).resolve()
    inputs_path = resolve(args.inputs)
    inputs = read(inputs_path)
    cases, selection = validate_cases(canonical, inputs)
    out = resolve(args.output)
    if out.exists() or not out.is_relative_to((ROOT/'outputs').resolve()):
        raise RuntimeError('A fresh ignored outputs directory is required')
    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env CUDA for actual semantic model forwards')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    out.mkdir(parents=True)
    gallery = out/'gallery'
    gallery.mkdir()
    write(out/'status.json', dict(status='running', training=False, no_optimizer_steps=True))
    protected = {}

    def protect(path, expected=None):
        path = resolve(path)
        digest = file_sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError('Protected input SHA mismatch: '+str(path))
        protected[str(path)] = digest
        return path

    try:
        protect(inputs_path)
        config_path = protect(args.config)
        prior_rows_path = protect(inputs['prior_semantic_rows']['path'], inputs['prior_semantic_rows']['sha256'])
        prior_rows = read(prior_rows_path)
        prior_records = {row['source']:row for row in prior_rows}
        if len(prior_rows) != 12 or len(prior_records) != 12 or set(prior_records) != {row['name'] for row in cases}:
            raise RuntimeError('Previous predetermined semantic forward records differ')
        zipped = protect(inputs['scored_zip']['path'], SCORED_ZIP_SHA)
        if inputs['scored_zip'].get('sha256') != SCORED_ZIP_SHA:
            raise RuntimeError('Wrong scored reflect ZIP registration')
        reflect_path = protect(inputs['reflect_report']['path'], inputs['reflect_report']['sha256'])
        reflected = read(reflect_path)
        config = reflected['config']
        config_gate(config, canonical)
        if (reflected.get('format') != 'marker_reflect_run_v1' or not reflected.get('complete')
                or reflected.get('full100_complete') is not True or reflected.get('no_seams') is not True
                or reflected.get('reflection_padding') != 32
                or reflected.get('checkpoint_sha256') != CHECKPOINTS['baseline'][1]
                or config['affinity_native']['overlap'] != .25
                or config['backend_adaptation']['restoration']['inference_seed'] != 314159):
            raise RuntimeError('Scored complete mixed R1 reflect32 report contract differs')
        precision = precision_record()
        runtime = runtime_contract(config)
        if precision != reflected['precision'] or runtime != reflected['runtime']:
            raise RuntimeError('Actual precision/runtime differs from scored reflect capture')
        test_files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
        original_manifest_path = protect(inputs.get('old_manifest', 'outputs/affinity_balance/mixed/deployment/manifest.json'))
        original_manifest = read(original_manifest_path)
        original_records = validate_manifest(original_manifest, test_files, CHECKPOINTS['baseline'][1])
        reflect_records = {row['source']:row for row in reflected['images']}
        if len(reflect_records) != 100 or len(reflected['images']) != 100:
            raise RuntimeError('Reflect full100 source set differs')

        models, identities, state_digests = {}, {}, {}
        native_path = protect(inputs.get('native_checkpoint', {}).get('path', 'outputs/affinity_balance/native/final.pt'), NATIVE_SHA)
        native_bundle = torch.load(native_path, map_location='cpu', weights_only=False)
        native_state = {name:value for name, value in native_bundle['model_state_dict'].items()
                        if not name.startswith('affinity_decoder.')}
        native_shared = tensor_digest(native_state.items())
        if native_bundle.get('epoch') != 20:
            raise RuntimeError('Old native semantic control epoch differs')
        for key, (relative, expected, epoch) in CHECKPOINTS.items():
            actual_path = inputs.get('checkpoints', {}).get(key, {}).get('path', relative)
            path = protect(actual_path, expected)
            model, bundle = load_backend(path, config, 'cuda')
            if bundle.get('epoch') != epoch:
                raise RuntimeError('Unexpected checkpoint epoch: '+key)
            model.eval().requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError('Semantic route is not frozen')
            if key == 'baseline':
                actual_shared = {name:value for name, value in bundle['model_state_dict'].items()
                                 if not name.startswith('affinity_decoder.')}
                if (actual_shared.keys() != native_state.keys()
                        or any(not torch.equal(actual_shared[name], native_state[name]) for name in native_state)
                        or tensor_digest(actual_shared.items()) != native_shared):
                    raise RuntimeError('Mixed baseline changed shared encoder/semantic versus old native control')
            models[key] = model
            state_digests[key] = tensor_digest(model.state_dict().items())
            identities[key] = dict(path=str(path), sha256=expected, epoch=epoch,
                architecture=bundle['architecture'], parameters=sum(p.numel() for p in model.parameters()),
                model_state_sha256=state_digests[key])
            del bundle
        del native_bundle, native_state
        encoder_digests = {key:tensor_digest(model.encoder.state_dict().items()) for key, model in models.items()}
        if len(set(encoder_digests.values())) != 1:
            raise RuntimeError('Geometry encoder/LoRA differs between semantic sources')
        if models['baseline'].semantic_lora is not None or models['full'].semantic_lora is not None:
            raise RuntimeError('Unexpected independent LoRA in baseline/full route')
        adapter = models['lora'].semantic_lora
        if adapter is None or adapter.VERSION != 'independent_lora_v1':
            raise RuntimeError('Actual independent semantic LoRA missing')
        geometry_lora = dict(models['lora'].encoder.trunk.named_parameters())
        if all(torch.equal(value, geometry_lora[name]) for name, value in zip(adapter.parameter_names, adapter.weights)):
            raise RuntimeError('Private semantic LoRA equals geometry LoRA')
        private_calls, geometry_calls = [], []
        hooks = [adapter.register_forward_hook(lambda module, inp, output: private_calls.append(True))]
        def forbidden_geometry(module, inp, output):
            geometry_calls.append(True)
            raise RuntimeError('Fixed semantic experiment must never call affinity head')
        hooks += [model.affinity_decoder.register_forward_hook(forbidden_geometry) for model in models.values()]
        restoration = config['backend_adaptation']['restoration']
        protect(restoration['checkpoint'], restoration['checkpoint_sha256'])
        restorer = get_restorer(config, 'cuda')
        restorer.eval().requires_grad_(False)
        restore_digest = tensor_digest(restorer.state_dict().items())
        if restore_digest != reflected['restorer_state_sha256']:
            raise RuntimeError('Frozen D5a state differs from scored reflect report')
        restorer_parameters = sum(p.numel() for p in restorer.parameters())
        resident = sum(row['parameters'] for row in identities.values()) + restorer_parameters
        deployment_upper = {key:identities['baseline']['parameters']+row['parameters']+restorer_parameters
                            for key, row in identities.items() if key != 'baseline'}
        deployment_upper['baseline'] = identities['baseline']['parameters']+restorer_parameters
        if max(resident, *deployment_upper.values()) >= 500000000:
            raise RuntimeError('Combined model parameter budget exceeded')
        # 封存所有实际导入的模型/工具源码，而不是只列出新入口。
        code_hashes, virtual = protected_sources(out)
        write(out/'cases.json', cases)
        write(out/'inputs.json', inputs)
        write(out/'config.json', config)
        records, restore_calls = [], 0
        started = time.time()
        with zipfile.ZipFile(zipped) as archive, torch.no_grad(), torch.autocast(device_type='cuda', enabled=False), \
                patch('cv2.watershed', side_effect=RuntimeError('No watershed allowed in fixed semantic experiment')):
            for case in cases:
                source_path = protect(case['path'], case.get('source_sha256'))
                ids, original_classes, png, labels = load_fixed(resolve(case['fixed_dir']), case['name'])
                old, old_classes, old_png, old_labels = load_fixed(resolve(case['old_dir']), case['name'])
                if old.shape != ids.shape:
                    raise RuntimeError('Old/new partitions use different native coordinates')
                for path, key in ((png, 'fixed_png_sha256'), (labels, 'fixed_classes_sha256'),
                                  (old_png, 'old_png_sha256'), (old_labels, 'old_classes_sha256')):
                    protect(path, case[key])
                if case['kind'] == 'test':
                    verify_scored_bytes(archive, case, png, labels)
                    ref = reflect_records[case['name']]
                    if (ref['seed'] != case['seed'] or ref['full_source_index'] != case['index']
                            or ref['source_sha256'] != file_sha(source_path)
                            or ref['output']['png_sha256'] != file_sha(png)
                            or ref['output']['classes_sha256'] != file_sha(labels)
                            or ref['reference']['png_sha256'] != file_sha(old_png)
                            or ref['reference']['classes_sha256'] != file_sha(old_labels)
                            or case['name'] not in original_records):
                        raise RuntimeError('Source does not match scored reflect/original report')
                else:
                    origin = case['reflect_origin']
                    origin_path = protect(origin['path'], origin['sha256'])
                    origin_report = read(origin_path)
                    origin_rows = {row['source']:row for row in origin_report.get('images', [])}
                    origin_row = origin_rows.get(case['name'], {})
                    if (origin_report.get('format') != 'semantic_reflect_train_cache_v1'
                            or origin_report.get('complete') is not True
                            or origin_report.get('checkpoint_sha256') != CHECKPOINTS['baseline'][1]
                            or origin_report.get('restorer_checkpoint_sha256') !=
                                config['backend_adaptation']['restoration']['checkpoint_sha256']
                            or origin_report.get('reflection_padding') != 32
                            or origin_report.get('no_seams') is not True
                            or origin_report.get('no_training') is not True
                            or origin_row.get('seed') != case['seed']
                            or origin_row.get('baseline_exact') is not True
                            or origin_row.get('real_barrier_elevation_exact') is not True
                            or origin_row.get('output', {}).get('png_sha256') != file_sha(png)
                            or origin_row.get('output', {}).get('classes_sha256') != file_sha(labels)
                            or origin_row.get('reference', {}).get('png_sha256') != file_sha(old_png)
                            or origin_row.get('reference', {}).get('classes_sha256') != file_sha(old_labels)):
                        raise RuntimeError('Current training reflect cache identity differs')
                    config_gate(origin_report['config'], canonical)
                gt = load_gt(case, config)
                if gt is not None:
                    gtroot = resolve(config['backend_adaptation']['completed_gt_dir'])
                    gt_path = protect(gtroot/(case['name']+'_gt.npz'), case.get('gt_npz_sha256'))
                    protect(gtroot/(case['name']+'_class.json'), case.get('gt_classes_sha256'))
                    with np.load(gt_path, allow_pickle=False) as blob:
                        original = blob['original_covered'].astype(bool)
                        filled = blob['filled'].astype(bool)
                        unknown = blob['residual_unknown'].astype(bool)
                        if (original.shape != gt[0].shape or filled.shape != gt[0].shape
                                or unknown.shape != gt[0].shape or np.any(original & (gt[0] == 0))
                                or not np.array_equal(filled, (gt[0] > 0) & ~original)
                                or not np.array_equal(unknown, gt[0] == 0)):
                            raise RuntimeError('Processed GT original/filled/unknown partition differs')
                rgb, image, ph, pw = prepare_image(source_path, 1024, 'cuda')
                image = image.float()  # 原布局，不新增contiguous调用。
                if ids.shape != rgb.shape[:2]:
                    raise RuntimeError('Fixed masks and image native coordinates differ')
                restored = restore_first(restorer, image, case['seed']).float()
                restore_calls += 1
                restored_sha = tensor_digest([('restored', restored)])
                prior = prior_records[case['name']]
                if (prior['kind'] != case['kind'] or prior['seed'] != case['seed']
                        or prior['shape'] != list(ids.shape)
                        or prior['restored_tensor_sha256'] != restored_sha
                        or prior['image_stride'] != list(image.stride())
                        or prior['restored_stride'] != list(restored.stride())):
                    raise RuntimeError('Shared restoration/layout differs from previous fixed semantic deployment: '+case['name'])
                mapping = partition_mapping(old, ids)
                contract = gt_contract(*gt, ids) if gt is not None else None
                record = dict(source=case['name'], kind=case['kind'], seed=case['seed'], shape=list(ids.shape),
                    fixed_png_sha256=file_sha(png), fixed_classes_sha256=file_sha(labels),
                    old_png_sha256=file_sha(old_png), old_classes_sha256=file_sha(old_labels),
                    partition_mapping=mapping, restored_tensor_sha256=restored_sha,
                    image_stride=list(image.stride()), restored_stride=list(restored.stride()),
                    routes={}, training_GT_contract=contract)
                predictions, base_details = {}, None
                for route in ROUTES:
                    key = 'baseline' if route.startswith('salign') else route.split('_')[0]
                    value = image if route == 'salign_raw' else restored
                    before_calls = len(private_calls)
                    model = models[key]
                    logits = model.semantic_decoder(semantic_features(model, value), value)
                    if isinstance(logits, dict):
                        logits = logits['semantic_logits']
                    probability = crop_letterbox_output(logits.float(), 1024, ph, pw, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()
                    probability = np.ascontiguousarray(probability, dtype=np.float32)
                    if route == 'lora_d5a' and len(private_calls) != before_calls+1:
                        raise RuntimeError('Semantic forward bypassed private LoRA')
                    classes, scores = revote_instances(ids, probability)
                    validate_prediction(ids, classes, rgb.shape[:2])
                    details = probability_details(ids, probability, scores)
                    if route == 'salign_raw':
                        if classes != original_classes:
                            raise RuntimeError('Scored reflect original classes fail exact reproduction: '+case['name'])
                        # 用同一概率在旧分区上复投票，验证只改变几何而未改变语义计算。
                        if revote_instances(old, probability)[0] != old_classes:
                            raise RuntimeError('Same-chain old mixed classes fail exact reproduction: '+case['name'])
                        base_details = details
                    folder = out/'predictions'/case['name']/route
                    folder.mkdir(parents=True)
                    copy_fixed_mask(png, folder/png.name)
                    write(folder/(case['name']+'_class.json'), classes)
                    write(folder/'instances.json', details)
                    mixture = probability_mixture(ids, probability, details)
                    row = dict(changes=changes(original_classes, classes, details, base_details),
                        partition_group_changes=grouped_changes(original_classes, classes, details, base_details, mapping),
                        ferrite_composition=ferrite_composition(classes, details), probability_mixture=mixture,
                        probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                        copied_png_sha256=file_sha(folder/png.name),
                        input='raw' if route == 'salign_raw' else 'shared_D5a_first_x0_tensor',
                        semantic_features_route='private_LoRA' if route == 'lora_d5a' else 'original_encoder')
                    if gt is not None:
                        row['training'] = training_probability_diagnostic(contract, *gt, classes, probability)
                    record['routes'][route] = row
                    predictions[route] = ids, classes
                if tensor_digest([('restored', restored)]) != restored_sha:
                    raise RuntimeError('Shared restored tensor changed during semantic routing')
                for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
                    render(source_path, [predictions[route] for route in ROUTES],
                        ['reflect S-align/raw', 'S-align/D5a', 'private LoRA/D5a', 'full e60/D5a'],
                        gallery/(case['name']+'_'+suffix+'.png'),
                        case['name']+' FIXED scored reflect PNG; only classes change', crop=crop, width=380)
                records.append(record)
                write(out/'rows.json', records)
                write(out/'status.json', dict(status='running', completed=len(records), total=len(cases), training=False))
                print('SEMANTIC_REFLECT', len(records), case['name'], 'baseline_exact', flush=True)
        for hook in hooks:
            hook.remove()
        if len(private_calls) != len(cases) or restore_calls != len(cases) or geometry_calls:
            raise RuntimeError('Forward count/geometry isolation audit failed')
        if any(tensor_digest(model.state_dict().items()) != state_digests[key] for key, model in models.items()):
            raise RuntimeError('Frozen semantic source changed')
        if tensor_digest(restorer.state_dict().items()) != restore_digest:
            raise RuntimeError('Frozen D5a changed')
        for path, expected in protected.items():
            if file_sha(path) != expected:
                raise RuntimeError('Protected input changed: '+path)
        for name, expected in code_hashes.items():
            if file_sha(ROOT/name) != expected or file_sha(out/'source'/name) != expected:
                raise RuntimeError('Protected code changed: '+name)
        if precision_record() != precision or runtime_contract(config) != runtime:
            raise RuntimeError('Precision/runtime changed during fixed semantic experiment')
        report = dict(format=FORMAT, complete=True, training=False, no_optimizer_steps=True,
            geometry_forward_calls=0, watershed_calls=0, full_test_inference=False, official_submission=False,
            best_deployment_changed=False, sample_count=len(cases), selection=selection,
            scored_reflect=dict(zip_sha256=SCORED_ZIP_SHA, user_reported_score=[.8645, .9006], report_sha256=file_sha(reflect_path)),
            checkpoints=identities, native_control_shared_state_sha256=native_shared,
            shared_native_mixed_exact=True, common_geometry_encoder_sha256=encoder_digests,
            parameter_counts=dict(diagnostic_resident=resident, restorer=restorer_parameters,
                                  conservative_deployment_upper_bound=deployment_upper),
            private_lora_forward_calls=len(private_calls), private_lora_distinct=True,
            restoration_calls=restore_calls, restored_once_per_source=True,
            restoration_and_layout_exact_to_prior_semantic=True,
            baseline_reflect_classes_exact=True, baseline_old_mixed_classes_exact=True,
            original_png_bytes_unchanged=True, cpu_sigmoid_after_native_logit_restore=True,
            votes='float32 arithmetic mean >0.5; equal 0.5 is pearlite', precision=precision, runtime=runtime,
            protected_inputs=protected, protected_inputs_exact=True, protected_sources=code_hashes,
            protected_code_exact=True, virtual_module_paths=virtual, totals=summarize(records),
            elapsed_seconds=time.time()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
            caveat='Unlabeled test prediction changes are not correctness; seen processed training GT is mechanism diagnosis, not generalization ranking. Unknown/filled/frame/mixed predictions receive no forced oracle class.')
        write(out/'report.json', report)
        write(out/'status.json', dict(status='complete', completed=len(cases), training=False, no_optimizer_steps=True))
        page = ['<!doctype html><meta charset="utf-8"><title>固定reflect语义诊断</title>',
            '<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:100%}section{margin:28px 0}</style>',
            '<h1>固定已评分reflect实例图，仅比较语义</h1>',
            '<p>左至右：原图/S-align、D5a/S-align、D5a/私有LoRA、D5a/full e60。金=铁素体，蓝=珠光体。</p>',
            '<p>四张已见GT仅定位机制；八张测试图无GT，不据翻转方向指定正确类别。</p>',
            '<p><a href="report.json">汇总</a> · <a href="rows.json">逐图与新旧分区对应</a></p>']
        for path in sorted(gallery.glob('*_detail.png')):
            page.append(f'<section><h2>{html.escape(path.stem)}</h2><img src="{html.escape(path.relative_to(out).as_posix())}"></section>')
        (out/'index.html').write_text('\n'.join(page), encoding='utf-8')
        with zipfile.ZipFile(out/'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in [*out.glob('*.json'), out/'index.html', *gallery.glob('*.png'), *out.glob('predictions/*/*/instances.json')]:
                archive.write(path, path.relative_to(out).as_posix())
        print('SEMANTIC_REFLECT_COMPLETE', len(cases), out, flush=True)
    except Exception as error:
        write(out/'status.json', dict(status='failed', training=False, error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_balance_mixed.yaml')
    parser.add_argument('--inputs', required=True, help='私有semantic_reflect_inputs_v1完整输入回执')
    parser.add_argument('--output', default='outputs/semantic_reflect')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
