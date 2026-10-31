# -*- coding: utf-8 -*-
"""冻结正式后端，只模糊局部 D5a/affinity 输入的32训练源诊断。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_ferrite_native import array_sha, protect_manifest, read, verify_protected
from train_backend_adaptation import sha, tensor_digest, write_json

PARENT_REPORT_SHA = 'c47b46c1d08a2b54fd5f6d9c230b82fac41b75051906cbee5be57d85deb3957d'
BLUR_RECIPE = dict(format='native_global_endpoint_sigma_v1', weak_sigma=1.0, strong_sigma=4.0,
                   transition_fraction=.30, min_region_fraction=.05, levels=8,
                   seed_namespace=20261003, quantization='float32 0..255 -> rint/clip -> uint8 once',
                   sigma_regions=dict(weak='sigma<=1.5', transition='1.5<sigma<3.5', strong='sigma>=3.5'),
                   scope='one whole-original blur before cropping; sigma in original pixels')
GALLERY_FIXED = ('train_005.jpg', 'train_008.jpg', 'train_171.jpg', 'train_176.jpg',
                 'train_659.jpg', 'train_686.jpg', 'train_707.jpg')
GALLERY_SEED = 20261003
GALLERY_LOCAL_GTS = {'train_005.jpg': [91], 'train_008.jpg': [91], 'train_659.jpg': [69, 70]}


def make_global_blur(rgb, seed):
    """沿用现有平滑端点场，整图只退化一次，不使用全局随机流。"""
    from data.rgb_spatial_blur import apply_sigma_map_blur, make_endpoint_sigma_map
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError('Original RGB uint8 HxWx3 with both dimensions >=2 required')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Nonnegative integer source seed required')
    cfg = dict(enabled=True, weak_sigma=[1., 1.], strong_sigma=[4., 4.],
               transition_fraction=[.30, .30], min_region_fraction=.05)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), BLUR_RECIPE['seed_namespace']]))
    h, w = rgb.shape[:2]
    sigma = make_endpoint_sigma_map((h, w), (0, 0, h, w), cfg, rng, [1., 4.])
    blurred = apply_sigma_map_blur(rgb.astype(np.float32), sigma, [1., 4.], levels=8)
    if not np.isfinite(blurred).all():
        raise RuntimeError('Nonfinite spatial blur')
    # 保持正式 tensor_from_rgb 的 uint8 /255 契约，不把0..255浮点直接输入模型。
    result = np.ascontiguousarray(np.rint(np.clip(blurred, 0., 255.)).astype(np.uint8))
    receipt = dict(recipe=dict(BLUR_RECIPE), source_seed=int(seed), sigma_sha256=array_sha(sigma),
                   input_sha256=array_sha(rgb), output_sha256=array_sha(result),
                   sigma_min=float(sigma.min()), sigma_max=float(sigma.max()),
                   sigma_mean_square=float(np.mean(sigma.astype(np.float64)**2)),
                   weak_fraction=float(np.mean(sigma <= 1.5)),
                   strong_fraction=float(np.mean(sigma >= 3.5)),
                   exact_weak_endpoint_fraction=float(np.mean(sigma == 1.)),
                   exact_strong_endpoint_fraction=float(np.mean(sigma == 4.)),
                   additional_illumination_or_noise=False, full_image_applications=1)
    return result, sigma, receipt


def gallery_sources(names):
    names = sorted(names)
    if len(names) != 32 or len(set(names)) != 32 or not set(GALLERY_FIXED).issubset(names):
        raise ValueError('Expected fixed 32-source cohort')
    pool = [n for n in names if n not in GALLERY_FIXED]
    extra = str(np.random.default_rng(GALLERY_SEED).choice(pool))
    return list(GALLERY_FIXED) + [extra], dict(fixed=list(GALLERY_FIXED), random_seed=GALLERY_SEED,
                                            random_pool=pool, random_source=extra,
                                            selection_before_inference=True)


@torch.no_grad()
def clean_whole_semantics(model, restorer, path, config, device, seed):
    """精确照 predict_views 重建原图的 geometry语义与raw末端投票概率。"""
    from train_backend_adaptation import prediction_maps
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    rgb, semantic, unused_global_boundary = prediction_maps(model, path, config, device, restorer, seed)
    del unused_global_boundary
    _, tensor, ph, pw = prepare_image(path, 1024, device)
    raw = model.semantic_decoder(model.encoder(tensor.float()), tensor.float())
    if isinstance(raw, dict):
        raw = raw['semantic_logits']
    probability = crop_letterbox_output(raw.float(), 1024, ph, pw, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()
    fixed = dict(geometry_semantic_sha256=array_sha(semantic.contiguous().numpy()),
                 raw_probability_sha256=array_sha(probability))
    return rgb, semantic, probability, fixed


@torch.no_grad()
def native_affinity_boundary(model, restorer, rgb, config, device, seed):
    """原native1024窗口/种子/首步D5a/融合/拼接；只接受已退化的uint8整图。"""
    from models.backend_adaptation import restore_first
    from tools.probe_affinity_patch import boundary_grid, tensor_from_rgb
    from utils.affinity_deployment import crop_letterbox_output
    from utils.patch_diagnostic import BlendMap, tile_boxes
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Native local path requires uint8 RGB; floating 0..255 is forbidden')
    size = 1024
    blend = BlendMap(rgb.shape[:2])
    boxes = tile_boxes(rgb.shape[:2], size, config['affinity_native']['overlap'])
    for index, (y, x, y1, x1) in enumerate(boxes):
        image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], device)
        restored = restore_first(restorer, image, int(seed) + 100000 + size*1000 + index)
        grid = boundary_grid(model, restored, config)
        native = crop_letterbox_output(grid, size, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
        blend.add((y, x, y1, x1), native)
    boundary, audit = blend.finish()
    audit['tiles'] = len(boxes)
    return boundary, audit


def decode_with_signed_capture(semantic, boundary, probability, config):
    """捕获真实有符号分水岭；不将-1边界与原始0/过滤区提前混在一起。"""
    from tools.affinity_connectivity_views import decode_native
    from tools.affinity_native_views import verify_prediction
    from tools.semantic_crossover import revote_instances
    boundary = np.asarray(boundary)
    if (boundary.dtype != np.float32 or boundary.ndim != 2 or not np.isfinite(boundary).all()
            or boundary.min() < 0 or boundary.max() > 1):
        raise ValueError('Finite native float32 boundary in [0,1] required')
    if (np.asarray(probability).shape != boundary.shape or not np.isfinite(probability).all()
            or np.min(probability) < 0 or np.max(probability) > 1):
        raise ValueError('Finite clean native voting probability required')
    if config['inference'].get('marker_partition_restore', {}).get('enabled', False):
        raise ValueError('One-pass frozen marker restoration disabled contract required')
    original = cv2.watershed
    captures = []

    def capture(image, markers):
        before = markers.copy()
        result = original(image, markers)
        captures.append(dict(markers=before, signed_watershed=result.copy()))
        return result

    with patch('utils.post_process.cv2.watershed', side_effect=capture):
        instances, _ = decode_native(semantic, torch.from_numpy(boundary)[None, None], config)
    if len(captures) != 1:
        raise RuntimeError('Exactly one actual watershed call required')
    classes, _ = revote_instances(instances, probability)
    verify_prediction(instances, classes, boundary.shape)
    value = captures[0]
    value.update(instances=instances, classes=classes, boundary=boundary)
    signed = value['signed_watershed']
    value['zero_sources'] = dict(
        watershed_boundary=int(np.sum((instances == 0) & (signed == -1))),
        original_watershed_zero=int(np.sum((instances == 0) & (signed == 0))),
        removed_positive_region=int(np.sum((instances == 0) & (signed > 0))),
        final_zero=int(np.sum(instances == 0)),
        signed_watershed_sha256=array_sha(signed),
        zero_sources_exact=True)
    if (np.any(signed < -1) or np.any((instances > 0) & (signed <= 0))
            or sum(value['zero_sources'][k] for k in ('watershed_boundary', 'original_watershed_zero',
                                                       'removed_positive_region')) != value['zero_sources']['final_zero']):
        raise RuntimeError('Actual pre/post watershed zero provenance does not close')
    return value


def require_clean_parity(value, boundary, expected, old_ids, old_classes, fixed, old_markers):
    """清晰缓存及实际每层输出都必须完全一致，否则禁止解释blur差异。"""
    checks = dict(boundary=array_sha(boundary) == expected['boundary_sha256'],
                  markers=array_sha(value['markers']) == expected['actual_marker_sha256']
                      and np.array_equal(value['markers'], old_markers),
                  watershed=array_sha(np.maximum(value['signed_watershed'], 0)) == expected['actual_watershed_sha256'],
                  final=np.array_equal(value['instances'], old_ids),
                  classes=value['classes'] == old_classes,
                  whole_semantics=fixed == expected['fixed_whole_semantics'])
    if not all(checks.values()):
        raise RuntimeError('Clean formal replay parity failed: ' + ', '.join(k for k, v in checks.items() if not v))
    return dict(exact=True, checks=checks, clean_affinity_forward_repeated=False,
                cached_boundary_used=True, actual_signed_watershed_collected=True)


def phase_pixels(gt, gt_classes, pred, pred_classes, valid_content=None):
    """全部已知域相位矩阵；预测ID0单列，包含只落在珠光体上的预测。"""
    from tools.analyze_ferrite_native import _classes, _labels
    gt, pred = _labels(gt, 'GT'), _labels(pred, 'prediction')
    if gt.shape != pred.shape:
        raise ValueError('Same native GT/pred grid required')
    gc, pc = _classes(gt_classes, gt, 'GT classes'), _classes(pred_classes, pred, 'prediction classes')
    content = np.ones(gt.shape, bool) if valid_content is None else np.asarray(valid_content)
    if content.dtype != np.bool_ or content.shape != gt.shape:
        raise ValueError('Valid content must be a same-grid boolean array')
    known = (gt > 0) & content
    gl = np.full(int(gt.max())+1, -1, np.int8)
    pl = np.full(int(pred.max())+1, 2, np.int8)
    for k, v in gc.items(): gl[k] = v
    for k, v in pc.items(): pl[k] = v
    matrix = np.bincount(gl[gt[known]].astype(np.int64)*3+pl[pred[known]], minlength=6).reshape(2, 3)
    if matrix.sum() != known.sum():
        raise RuntimeError('Known phase matrix does not close')
    return dict(matrix=matrix.tolist(), rows=['GT_P_0', 'GT_F_1'],
                columns=['pred_P_0', 'pred_F_1', 'unassigned_ID0'], closure_exact=True)


def matched_phase_changes(clean, blurred, clean_classes, blurred_classes, gt_classes):
    """同GT两次几何匹配的末端类别变化；不是固定区域的语义头因果实验。"""
    before = {x['gt']: x for x in clean['pairs']}
    after = {x['gt']: x for x in blurred['pairs']}
    shared = sorted(set(before) & set(after))
    changes = []
    for g in shared:
        a, b = before[g], after[g]
        old = int(clean_classes[str(a['pred'])]); new = int(blurred_classes[str(b['pred'])])
        if old != new:
            changes.append(dict(gt=g, GT_phase=int(gt_classes[g]), clean_pred=a['pred'], blur_pred=b['pred'],
                                clean_phase=old, blur_phase=new, clean_iou=a['iou'], blur_iou=b['iou']))
    return dict(shared_matched_GT=len(shared), phase_flips=len(changes), changed=changes,
                clean_only_matched_GT=sorted(set(before)-set(after)),
                blur_only_matched_GT=sorted(set(after)-set(before)),
                interpretation='clean probability fixed; changed partitions can change instance vote')


def paired_diagnostic_delta(clean, blurred):
    """清晰safe名单固定；不重选blur易样本冒充面积改善。"""
    before = {r['gt']: r for r in clean['gt_rows']}
    after = {r['gt']: r for r in blurred['gt_rows']}
    if set(before) != set(after) or any(before[g]['area'] != after[g]['area'] for g in before):
        raise ValueError('Same complete processed ferrite GT cohort required')
    eligible = [g for g, r in before.items() if r['safe_one_to_one_area_eligible']]
    pred_rows = {r['pred']: r for r in blurred['pred_rows']}
    survived, lost = [], []
    for g in eligible:
        a, b = before[g], after[g]
        if b['safe_one_to_one_area_eligible']:
            survived.append(dict(gt=g, clean_error=a['area_relative_error'], blurred_error=b['area_relative_error'],
                                 change=b['area_relative_error']-a['area_relative_error']))
        else:
            lost.append(dict(gt=g, unmatched=b['matched_pred'] is None, GT_censored=b['censored'],
                             topology_censored=b['topology_censored'],
                             matched_pred_censored=pred_rows[b['matched_pred']]['censored']
                                 if b['matched_pred'] is not None else None,
                             predicted_phase=b['predicted_phase'], significant_split=b['significant_geometry_split'],
                             matched_significant_merge=b['matched_pred_significant_geometry_merge'],
                             matched_pred=b['matched_pred']))
    keys = clean['summary']['mechanism_counts']
    return dict(mechanism_delta={k: blurred['summary']['mechanism_counts'][k]-v for k, v in keys.items()},
                geometry_matched_F_delta=blurred['summary']['counts']['geometry_matched_F']
                    - clean['summary']['counts']['geometry_matched_F'],
                clean_safe_fixed_n=len(eligible), clean_safe_survived_n=len(survived),
                clean_safe_lost_n=len(lost), clean_safe_survived=survived, clean_safe_lost=lost,
                blur_new_safe_gt=sorted(g for g in before if g not in eligible and after[g]['safe_one_to_one_area_eligible']))


def regional_diagnostics(gt, gt_classes, clean, candidate, clean_diag, blur_diag, sigma):
    """预设sigma三区的像素闭合与GT整体中位sigma分层；区域切片不是完整面积。"""
    if sigma.shape != gt.shape or not np.isfinite(sigma).all() or sigma.min() < 1 or sigma.max() > 4:
        raise ValueError('Same native sigma field in [1,4] required')
    masks = dict(weak=sigma <= 1.5, transition=(sigma > 1.5) & (sigma < 3.5), strong=sigma >= 3.5)
    known = gt > 0
    gids, inverse, counts = np.unique(gt[known], return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    ordered_sigma = sigma[known][order]
    offset = 0; assignment = {}
    for g, count in zip(gids, counts):
        median = float(np.median(ordered_sigma[offset:offset+count])); offset += count
        assignment[int(g)] = dict(median_sigma=median, region='weak' if median <= 1.5 else 'strong' if median >= 3.5 else 'transition')
    output = {}
    for name, mask in masks.items():
        ids = [g for g, a in assignment.items() if a['region'] == name]
        fids = {g for g in ids if int(gt_classes[g]) == 1}
        arms = {}
        for arm, value, diagnostic in (('clean', clean, clean_diag), ('blur', candidate, blur_diag)):
            f = [r for r in diagnostic['gt_rows'] if r['gt'] in fids]
            ff = {g for r in diagnostic['ff_merge_rows'] for g in r['gt_ids'] if g in fids}
            safe_ff = {g for r in diagnostic['ff_merge_rows'] if not r['censored'] and r['whole_grain_cooccupancy']
                       for g in r['gt_ids'] if g in fids}
            arms[arm] = dict(phase_pixels=phase_pixels(gt, gt_classes, value['instances'], value['classes'], mask),
                F_GT_group_counts=dict(F_GT=len(f), GT_censored=sum(x['censored'] for x in f),
                    GT_uncensored=sum(not x['censored'] for x in f),
                    topology_censored=sum(x['topology_censored'] for x in f),
                    matched=sum(x['matched_pred'] is not None for x in f),
                    unmatched=sum(x['matched_pred'] is None for x in f),
                    significant_split=sum(x['significant_geometry_split'] for x in f),
                    same_parent_partition_candidate=sum(x['same_parent_internal_partition_candidate'] for x in f),
                    safe_same_parent_partition_candidate=sum(x['same_parent_internal_partition_candidate']
                        and not x['topology_censored'] for x in f),
                    strict_contained_split_candidate=sum(x['internal_split_candidate'] for x in f),
                    significant_foreign_invasion=sum(x['significant_foreign_neighbor_invasion_candidate'] for x in f),
                    FF_merge_participating_F_GT=len(ff),
                    safe_whole_grain_FF_merge_participating_F_GT=len(safe_ff),
                    matched_vote_wrong=sum(x['vote_wrong'] for x in f),
                    safe_area_eligible=sum(x['safe_one_to_one_area_eligible'] for x in f)),
                all_phase_GT_group_counts=dict(GT=len(ids), matched=sum(r['gt'] in ids for r in diagnostic['pairs'])))
        output[name] = dict(known_region_pixels=int((known & mask).sum()),
            clean=arms['clean'], blur=arms['blur'],
            mechanism_count_delta={k: arms['blur']['F_GT_group_counts'][k]-v
                for k, v in arms['clean']['F_GT_group_counts'].items()})
    if sum(x['known_region_pixels'] for x in output.values()) != int(known.sum()):
        raise RuntimeError('Fixed sigma-region known pixels do not close')
    for arm, value in (('clean', clean), ('blur', candidate)):
        total = sum((np.asarray(x[arm]['phase_pixels']['matrix'], np.int64) for x in output.values()), np.zeros((2, 3), np.int64))
        if not np.array_equal(total, phase_pixels(gt, gt_classes, value['instances'], value['classes'])['matrix']):
            raise RuntimeError('Sigma-region phase matrices do not close to the whole known matrix')
    return dict(regions=output, GT_median_sigma_assignment={str(g): a for g, a in assignment.items()},
                rules=dict(weak='sigma<=1.5', transition='1.5<sigma<3.5', strong='sigma>=3.5',
                           GT_group='whole known GT support median sigma; each GT belongs to one group'),
                caveat='Region pixels are slices, not complete grain areas; GT groups use full known GT support.')


def save_thumbnail(path, rgb, blurred, sigma, gt, clean, candidate, *, box_yxyx=None):
    from tools.analyze_ferrite_native import _color_panel
    from tools.render_backend_ablation import text_line, write_png
    h, w = gt.shape
    if box_yxyx is not None:
        y0, x0, y1, x1 = box_yxyx
        h, w = y1-y0, x1-x0
    width = 320; height = max(1, round(h*width/w))
    heat = lambda a: cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(a, 0, 1)*255).astype(np.uint8),
                                                    cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    panels = [('Clean RGB', rgb), ('Spatial blur RGB', blurred), ('sigma 1..4 native px', heat((sigma-1)/3)),
              ('Processed newGT', _color_panel(rgb, gt, gt, gt > 0)),
              ('Clean final', _color_panel(rgb, clean['instances'], gt, gt > 0)),
              ('Blur final / clean semantics', _color_panel(rgb, candidate['instances'], gt, gt > 0)),
              ('Clean continuous boundary', heat(clean['boundary'])),
              ('Blur continuous boundary', heat(candidate['boundary']))]
    canvas = np.full((height+28, width*len(panels), 3), 255, np.uint8)
    for index, (label, image) in enumerate(panels):
        if box_yxyx is not None:
            image = image[y0:y1, x0:x1]
        canvas[28:, index*width:(index+1)*width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        text_line(canvas, label, (index*width+3, 18), width-6)
    write_png(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def run(output, *, smoke=False):
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend
    from tools.affinity_connectivity_views import _inference_state
    from tools.analyze_affinity_tail_gt import CHECKPOINTS, existing_project_sources, runtime_files_contract
    from tools.analyze_ferrite_native import analyze_ferrite_native
    from tools.analyze_ferrite_seams import analyze_ferrite_seams
    from tools.render_backend_ablation import read_prediction
    from tools.run_affinity_short import precision_record
    from tools.run_affinity_tail import verify_snapshot
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import get_restorer
    from train_affinity_tail import CONFIG_PATH
    from utils.config import load_config

    output = Path(output).resolve()
    expected = ROOT / ('outputs/ferrite_blur_smoke' if smoke else 'outputs/ferrite_blur')
    if output != expected or output.exists() or not torch.cuda.is_available():
        raise RuntimeError('Fresh prescribed output and actual server GPU required')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    folder = ROOT / 'outputs/ferrite_native/analysis'
    if sha(folder / 'report.json') != PARENT_REPORT_SHA:
        raise RuntimeError('Fixed32 clean parent report identity differs')
    parent_report = read(folder / 'report.json'); capture = read(folder / 'capture.json')
    if (not parent_report['complete'] or not capture['complete'] or len(capture['rows']) != 32
            or sha(folder / 'capture.json') != parent_report['capture_sha256']
            or parent_report['checkpoint_sha256'] != CHECKPOINTS['r1']):
        raise RuntimeError('Complete accepted clean native32 parent required')
    config = load_config(CONFIG_PATH)
    if json.loads(json.dumps(config)) != capture['config']:
        raise RuntimeError('Formal fixed configuration changed')
    verify_snapshot(ROOT / 'outputs/affinity_tail', capture['sealed_sources'])
    runtime = runtime_files_contract(capture['sealed_runtime'], config)
    precision = precision_record(torch)
    if runtime != capture['runtime_audit'] or precision != capture['precision'] or audit_new_gt(config) != capture['newgt']:
        raise RuntimeError('Clean parent runtime/precision/newGT identity differs')
    checkpoint = ROOT / capture['checkpoint']
    if sha(checkpoint) != CHECKPOINTS['r1']:
        raise RuntimeError('Best R1 checkpoint changed')
    protected = dict(capture['protected_sha256'])
    protect_manifest(folder, protected)
    new_sources = {name: sha(ROOT / name) for name in
                   ('tools/probe_ferrite_blur.py', 'tools/analyze_ferrite_seams.py', 'tools/analyze_ferrite_native.py')}
    if new_sources['tools/analyze_ferrite_native.py'] != parent_report['diagnostic_core_sha256']:
        raise RuntimeError('Clean and blur must use the same sealed ferrite diagnostic core')
    protected.update({str(ROOT / name): digest for name, digest in new_sources.items()})
    verify_protected(protected)
    names = [r['source'] for r in capture['rows']]
    if len(set(names)) != 32 or set(names) != set(capture['seed_map']):
        raise RuntimeError('Exact original source ordering and seed map required')
    selected, selection_receipt = gallery_sources(names)
    smoke_record = None
    if not smoke:
        smoke_folder = ROOT / 'outputs/ferrite_blur_smoke'
        smoke_record = read(smoke_folder / 'report.json')
        protect_manifest(smoke_folder, protected)
        if (not smoke_record['complete'] or not smoke_record['smoke'] or smoke_record['sources'] != ['train_686.jpg']
                or smoke_record['new_sources'] != new_sources or smoke_record['recipe'] != BLUR_RECIPE
                or smoke_record['parent_report_sha256'] != PARENT_REPORT_SHA
                or smoke_record['config'] != json.loads(json.dumps(config))
                or smoke_record['runtime_audit'] != runtime or smoke_record['precision'] != precision
                or smoke_record['frozen_states_sha256'] != capture['frozen_states_sha256']):
            raise RuntimeError('Fresh exact clean686 smoke gate required')
    model = load_backend(checkpoint, config, 'cuda')[0]
    restorer = get_restorer(config, 'cuda')
    states = dict(model=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if (states != capture['frozen_states_sha256'] or any(p.requires_grad for p in model.parameters())
            or any(p.requires_grad for p in restorer.parameters())):
        raise RuntimeError('Frozen backend or D5a state differs')
    output.mkdir(parents=True); (output/'sources').mkdir(); (output/'gallery').mkdir()
    for name in new_sources: shutil.copyfile(ROOT/name, output/'sources'/Path(name).name)
    parent_rows = {r['source']: r['diagnostic'] for r in parent_report['rows']}
    rows = []; phase_total = {k: np.zeros((2, 3), np.int64) for k in ('clean', 'blur')}
    planned = ['train_686.jpg'] if smoke else names
    for name in planned:
        source = next(r for r in capture['rows'] if r['source'] == name)
        if source['seed'] != capture['seed_map'][name]:
            raise RuntimeError('Per-source seed map changed')
        if smoke_record is not None and name == 'train_686.jpg':
            row = dict(smoke_record['rows'][0]); row['reused_smoke_exact'] = True
            rows.append(row)
            for arm in phase_total: phase_total[arm] += np.asarray(row[arm]['phase_pixels']['matrix'], np.int64)
            for p in (ROOT/'outputs/ferrite_blur_smoke/gallery').glob('*.png'): shutil.copyfile(p, output/'gallery'/p.name)
            continue
        path = ROOT/source['source_path']; gtpath = ROOT/source['gt_path']
        if sha(path) != source['source_file_sha256'] or sha(gtpath) != source['gt_file_sha256']:
            raise RuntimeError('Actual original source or complete processed GT changed')
        gt, lookup = load_completed_semantic_source(gtpath, source['shape'])
        gc = {int(v): int(lookup[v]) for v in np.unique(gt) if v > 0}
        if gc != {int(k): int(v) for k, v in source['gt_classes'].items()}:
            raise RuntimeError('Processed GT class identity changed')
        old_ids, class_int, _ = read_prediction(folder/'masks', path.stem, gt.shape)
        old_classes = {str(k): v for k, v in class_int.items()}
        with np.load(folder/source['map_file'], allow_pickle=False) as array:
            clean_boundary = array['boundary'].copy(); old_markers = array['markers'].copy()
        with _inference_state(model, restorer, 'cuda'):
            rgb, semantic, probability, fixed = clean_whole_semantics(model, restorer, path, config, 'cuda', source['seed'])
            if array_sha(rgb) != array_sha(read_rgb(str(path))):
                raise RuntimeError('Clean source decoding changed')
            clean = decode_with_signed_capture(semantic, clean_boundary, probability, config)
            parity = require_clean_parity(clean, clean_boundary, source, old_ids, old_classes, fixed, old_markers)
            blurred_rgb, sigma, blur_receipt = make_global_blur(rgb, source['seed'])
            boundary, blend = native_affinity_boundary(model, restorer, blurred_rgb, config, 'cuda', source['seed'])
            if any(blend[k] != source['blend'][k] for k in ('tiles', 'tiles_max_overlap', 'covered_fraction', 'overlap_fraction')):
                raise RuntimeError('Fixed native window/blending geometry changed')
            candidate = decode_with_signed_capture(semantic, boundary, probability, config)
            if fixed != dict(geometry_semantic_sha256=array_sha(semantic.contiguous().numpy()),
                             raw_probability_sha256=array_sha(probability)):
                raise RuntimeError('Shared clean semantic tensor/probability mutated')
        clean_diag = parent_rows[name]
        blur_diag = analyze_ferrite_native(gt, candidate['instances'], gc, candidate['classes'])
        row = dict(source=name, seed=source['seed'], source_file_sha256=sha(path), gt_file_sha256=sha(gtpath),
                   fixed_whole_semantics=fixed, clean_parity=parity, blur=blur_receipt,
                   blend=blend, delta=paired_diagnostic_delta(clean_diag, blur_diag), clean={}, reused_smoke_exact=False)
        row['matched_phase_changes'] = matched_phase_changes(clean_diag, blur_diag, clean['classes'], candidate['classes'], gc)
        row['regional_diagnostics'] = regional_diagnostics(gt, gc, clean, candidate, clean_diag, blur_diag, sigma)
        # blur配方与blur诊断分开，避免重复/覆盖同名JSON键。
        row['blur_recipe_receipt'] = row.pop('blur'); row['blur'] = {}
        for arm, value, diagnostic in (('clean', clean, clean_diag), ('blur', candidate, blur_diag)):
            allocation = phase_pixels(gt, gc, value['instances'], value['classes'])
            matrix = np.asarray(allocation['matrix'], np.int64)
            if (int(matrix[1].sum()) != sum(x['area'] for x in diagnostic['gt_rows'])
                    or int(matrix[1, 0]) != sum(x['F_pixels_predicted_P'] for x in diagnostic['gt_rows'])
                    or int(matrix[1, 2]) != sum(x['unassigned_known_pixels'] for x in diagnostic['gt_rows'])):
                raise RuntimeError('Sealed core and all-phase allocation do not close')
            phase_total[arm] += matrix
            seam = analyze_ferrite_seams(gt, value['instances'], gc, value['classes'], value['markers'], diagnostic,
                                         watershed=value['signed_watershed'], watershed_format='signed')
            row[arm] = dict(diagnostic=diagnostic, phase_pixels=allocation, seams=seam,
                            zero_sources=value['zero_sources'], final_sha256=array_sha(value['instances']),
                            classes_sha256=hashlib.sha256(json.dumps(value['classes'], sort_keys=True).encode()).hexdigest(),
                            boundary_sha256=array_sha(value['boundary']), marker_sha256=array_sha(value['markers']))
        if name in selected:
            save_thumbnail(output/'gallery'/f'{path.stem}.png', rgb, blurred_rgb, sigma, gt, clean, candidate)
        if name in GALLERY_LOCAL_GTS:
            mask = np.isin(gt, GALLERY_LOCAL_GTS[name])
            if not mask.any() or not set(GALLERY_LOCAL_GTS[name]).issubset(gc):
                raise RuntimeError('Predeclared local labelled gallery GT missing')
            y, x = np.where(mask)
            box = [max(0, int(y.min())-32), max(0, int(x.min())-32),
                   min(gt.shape[0], int(y.max())+33), min(gt.shape[1], int(x.max())+33)]
            save_thumbnail(output/'gallery'/f'{path.stem}_detail.png', rgb, blurred_rgb, sigma, gt, clean, candidate, box_yxyx=box)
            row['fixed_local_gallery'] = dict(GT_ids=GALLERY_LOCAL_GTS[name], box_yxyx=box,
                                              selected_before_inference=True, margin_native_pixels=32)
        rows.append(row)
        write_json(output/'progress.json', dict(complete=False, rows=rows))
        print(json.dumps(dict(index=len(rows), total=len(planned), source=name, clean_exact=True,
                              delta=row['delta']['mechanism_delta'])), flush=True)
        del clean, candidate, old_markers, clean_boundary, boundary, rgb, blurred_rgb, sigma, semantic, probability
    if (states != dict(model=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
            or runtime_files_contract(capture['sealed_runtime'], config) != runtime or precision_record(torch) != precision
            or audit_new_gt(config) != capture['newgt']):
        raise RuntimeError('Frozen states/runtime/precision/processed GT changed during diagnostic')
    verify_protected(protected); verify_snapshot(ROOT/'outputs/affinity_tail', capture['sealed_sources'])
    if not smoke and any(int(v.sum()) != capture['newgt']['totals']['completed_known'] for v in phase_total.values()):
        raise RuntimeError('All32 full processed known phase pixels do not close')
    imported = existing_project_sources(list(sys.modules.values()), ROOT)
    report = dict(complete=True, smoke=bool(smoke), no_training=True, no_blackbox=True, deployment_changed=False,
                  parent_report_sha256=PARENT_REPORT_SHA, parent_capture_sha256=sha(folder/'capture.json'),
                  sources=planned, rows=rows, checkpoint_sha256=CHECKPOINTS['r1'], epoch=20,
                  recipe=BLUR_RECIPE, gallery_selection=selection_receipt,
                  local_gallery_fixed_GT=GALLERY_LOCAL_GTS,
                  config=config, runtime_audit=runtime, precision=precision,
                  frozen_states_sha256=states, new_sources=new_sources, imported_sources=imported,
                  source148_unchanged=True, formal_pipeline_unchanged=True,
                  formal_pipeline_algorithm_unchanged=True, local_affinity_input_changed=True,
                  whole_semantic_inputs_unchanged=True,
                  all_phase_pixels={k: v.tolist() for k, v in phase_total.items()},
                  clean686_smoke_reused=smoke_record is not None,
                  scope='local restoration+affinity blur diagnostic; both whole-image semantic paths stay clean',
                  caveats=['Seen labelled training sources, not independent validation or official scores.',
                           'Connected processed GT may omit real boundaries; split candidates are not physical false-edge proof.',
                           'Blur is a sigma-squared Gaussian mixture with one uint8 rounding, not an exact optical PSF.',
                           'Training already includes spatial/endpoint blur after view transformation; native blur failure is not evidence that blur augmentation was absent.',
                           'No full-size new maps, predictions, training, threshold change or test GT produced.'])
    write_json(output/'report.json', report)
    write_json(output/'download_manifest.json', dict(complete=True, files={p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'download_manifest.json'}))
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    run(ROOT/(args.output or ('outputs/ferrite_blur_smoke' if args.smoke else 'outputs/ferrite_blur')), smoke=args.smoke)
