# -*- coding: utf-8 -*-
"""多源已见新GT的关键通路反事实；固定真实模型，不生成比赛提交。

仅纠正受实际原生通路支持的512网格短关系，再重算原CUDA融合、插值、
窗口拼接、marker、分水岭和原图语义投票。GT0/填充/第三GT不提供监督。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.affinity_bottleneck_capture import capture_native_tiles
from tools.analyze_affinity_coverage import fixed_area_cohort
from tools.analyze_affinity_fused_gt import diagnose_partition, gallery_panel
from tools.analyze_ferrite_native import analyze_ferrite_native
from tools.probe_affinity_partition import decode_capture, representative
from tools.run_marker_reflect import (array_sha, config_gate, precision_record,
                                     read, save_prediction, sha, source_snapshot, write)
from tools.semantic_crossover import revote_instances
from utils.affinity_bottleneck import constrained_bottleneck_path
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.patch_diagnostic import BlendMap, phase_description

FORMAT = 'affinity_bottleneck_probe_v1'


def project_native_support(tile, native_gt, support):
    """沿真实align_corners插值中心投影，并检查所有短边的原生像素段。

    训练GT采用nearest，部署边界采用bilinear，两者坐标不能直接当相同。
    因此只保留两种坐标的GT一致且真实native支持连续的关系。
    """
    y0, x0, y1, x1 = tile['box']
    crop = np.asarray(native_gt)[y0:y1, x0:x1]
    domain = np.asarray(support)[y0:y1, x0:x1]
    if domain.dtype != np.bool_ or domain.shape != crop.shape:
        raise ValueError('native support must be bool and match GT')
    grid_gt, valid = tile['grid_gt'], tile['valid']
    meta = tile['geometry']; ch, cw = meta['content_height'], meta['content_width']
    gh, gw = grid_gt.shape; h, w = crop.shape
    if min(ch, cw, h, w) < 1 or ch > gh or cw > gw:
        raise ValueError('Invalid actual geometry')
    ys = np.rint(np.linspace(0, h - 1, ch)).astype(np.int32)
    xs = np.rint(np.linspace(0, w - 1, cw)).astype(np.int32)
    yy = np.zeros((gh, gw), np.int32); xx = yy.copy()
    yy[:ch, :cw] = ys[:, None]; xx[:ch, :cw] = xs[None, :]
    projected = valid & domain[yy, xx] & (crop[yy, xx] == grid_gt) & (grid_gt > 0)
    edges = np.zeros((4, gh, gw), bool)
    # 相邻grid中心在native中距离很短；逐段等距采样是8邻原生像素路径。
    for c, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        ay, ax = np.indices((gh, gw)); by, bx = ay + dy, ax + dx
        inside = (by >= 0) & (by < gh) & (bx >= 0) & (bx < gw)
        sy, sx = ay[inside], ax[inside]; ty, tx = by[inside], bx[inside]
        keep = projected[sy, sx] & projected[ty, tx]
        sy,sx,ty,tx=sy[keep],sx[keep],ty[keep],tx[keep]
        if not len(sy): continue
        keep=np.ones(len(sy),bool)
        ny0, nx0 = yy[sy, sx], xx[sy, sx]
        ny1, nx1 = yy[ty, tx], xx[ty, tx]
        ga, gb = grid_gt[sy, sx], grid_gt[ty, tx]
        steps = np.maximum(np.abs(ny1 - ny0), np.abs(nx1 - nx0))
        for k in range(int(steps.max(initial=0)) + 1):
            fraction = np.minimum(k, steps) / np.maximum(1, steps)
            py = np.rint(ny0 + fraction * (ny1 - ny0)).astype(np.int32)
            px = np.rint(nx0 + fraction * (nx1 - nx0)).astype(np.int32)
            keep &= domain[py, px]
            # 同GT网格边不得跳过两中心之间的异GT细条后被错误clamp为1。
            keep &= (crop[py, px] == ga) | (crop[py, px] == gb)
        edges[c, sy, sx] = keep
    return projected, edges, yy + y0, xx + x0


def near_point(mask, point, yy, xx):
    """选择GT/支持域合法且最靠近原生固定端点的网格点。"""
    gy, gx = np.nonzero(mask)
    if not len(gy):
        return None
    distance = (yy[gy, gx].astype(np.int64) - point[0]) ** 2 + (xx[gy, gx].astype(np.int64) - point[1]) ** 2
    index = int(np.argmin(distance))
    return [int(gy[index]), int(gx[index])]


def native_segment_path(path, yy, xx):
    result = []
    for a, b in zip(path, path[1:]):
        p = np.array([yy[tuple(a)], xx[tuple(a)]], np.int32)
        q = np.array([yy[tuple(b)], xx[tuple(b)]], np.int32)
        steps = int(np.abs(q - p).max())
        points = np.rint(p + np.linspace(0, 1, steps + 1)[:, None] * (q - p)).astype(int).tolist()
        result.extend(points if not result else points[1:])
    if not result and path:
        result = [[int(yy[tuple(path[0])]), int(xx[tuple(path[0])])]]
    return result


def case_relation(decoded, case):
    points = case['points']; labels = decoded['instances']; markers = decoded['actual_markers']
    values = [int(labels[tuple(p)]) for p in points]
    mvalues = [int(markers[tuple(p)]) for p in points]
    same = values[0] > 0 and values[0] == values[1]
    desired = (not same and min(values) > 0) if case['direction'] == 'negative' else same
    return dict(instance_ids=values, marker_ids=mvalues, same_instance=same,
                endpoint_goal=desired, caveat='Endpoint relation only; full GT contributions also audited')


def case_quality(decoded, gt, case):
    """核对整块已知归属，避免固定端点进入小腔被误报整粒修复。"""
    ids=decoded['instances']; rows=[]
    for g in case['gt_ids']:
        mask=gt==g; deep=cv2.erode(mask.astype(np.uint8),np.ones((9,9),np.uint8),
            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        values,counts=np.unique(ids[mask],return_counts=True)
        order=sorted(zip(values.tolist(),counts.tolist()),key=lambda item:(-item[1],item[0]))
        positive=[(p,n) for p,n in order if p>0]; dominant=positive[0][0] if positive else 0
        rows.append(dict(gt=int(g),known_pixels=int(mask.sum()),deep_pixels=int(deep.sum()),
            dominant_pred=int(dominant),dominant_fraction=float((mask&(ids==dominant)).sum()/max(1,mask.sum())) if dominant else 0.,
            dominant_deep_fraction=float((deep&(ids==dominant)).sum()/max(1,deep.sum())) if dominant else 0.,
            significant_pred_ids=[int(p) for p,n in positive if n>=max(50,.1*int(mask.sum()))],
            contributions=[dict(pred=int(p),pixels=int(n),deep_pixels=int((deep&(ids==p)).sum())) for p,n in order]))
    return dict(endpoint_relation=case_relation(decoded,case),gt_rows=rows,
        dominant_instances_separate=len(rows)==2 and min(r['dominant_pred'] for r in rows)>0 and rows[0]['dominant_pred']!=rows[1]['dominant_pred'],
        all_dominant_deep_fraction_at_least_95=all(r['dominant_deep_fraction']>=.95 for r in rows),
        whole_object_censored=case.get('censored',True),known_domain_only=True)


def select_bottlenecks(tile, native_gt, decoded, case, probabilities):
    direction = case['direction']; groups = list(map(int, case['gt_ids'])); points = case['points']
    relation = case_relation(decoded, case)
    if relation['endpoint_goal']:
        return dict(found=False, reason='endpoint_goal_already_met', relation=relation)
    if not all(np.any(tile['grid_gt'] == g) for g in groups):
        return dict(found=False, reason='tile_lacks_target_gt', relation=relation)
    if direction == 'negative':
        m = relation['marker_ids']
        if m[0] <= 0 or m[0] != m[1]:
            return dict(found=False, reason='no_shared_positive_actual_marker', relation=relation)
        support = (decoded['actual_markers'] == m[0]) & np.isin(native_gt, groups)
    else:
        support = native_gt == groups[0]
    allowed, edge_valid, yy, xx = project_native_support(tile, native_gt, support)
    grid_gt = tile['grid_gt']
    masks = [allowed & (grid_gt == groups[0]), allowed & (grid_gt == groups[-1])]
    # 正样本必须对应原生同GT的两个实际片；不能以窗口内任意两点替代。
    if direction == 'positive':
        ids = relation['instance_ids']
        masks = [masks[i] & (decoded['instances'][yy, xx] == ids[i]) for i in range(2)]
    start = near_point(masks[0], points[0], yy, xx)
    end = near_point(masks[1], points[1], yy, xx)
    if start is None or end is None:
        return dict(found=False, reason='tile_lacks_both_legal_endpoints', relation=relation)
    result = constrained_bottleneck_path(probabilities, grid_gt, tile['valid'], start, end,
        direction, allowed=allowed, edge_valid=edge_valid)
    result.update(relation=relation, tile_index=tile['index'],
                  allowed_grid_sha256=array_sha(allowed), edge_valid_sha256=array_sha(edge_valid),
                  tile_contains_fixed_points=[tile['box'][0]<=p[0]<tile['box'][2] and tile['box'][1]<=p[1]<tile['box'][3] for p in points],
                  selected_native_endpoints=[[int(yy[tuple(p)]),int(xx[tuple(p)])] for p in (start,end)],
                  endpoint_native_distances=[float(np.hypot(yy[tuple(q)]-p[0],xx[tuple(q)]-p[1])) for p,q in zip(points,(start,end))],
                  witness_scope='window-local support path; does not claim the original fixed endpoints lie in this window')
    if result['found']:
        native_path = native_segment_path(result['path'], yy, xx)
        legal = all(support[tuple(p)] for p in native_path)
        if not legal:
            raise RuntimeError('Selected grid path does not have a legal actual native witness')
        result.update(native_path=native_path, native_path_known_support_exact=True,
                      native_endpoint_gt=[int(native_gt[tuple(p)]) for p in (native_path[0], native_path[-1])],
                      path_cross_gt_steps=sum(e['cross_gt'] for e in result['edges']),
                      interpretation='Constrained maximin; a direct cross-component path can reduce to reachable top1, not a minimum cut')
    return result


def corrected_logits(logits, negative, positive):
    if negative.shape != tuple(logits.shape) or positive.shape != tuple(logits.shape):
        raise ValueError('Correction mask shape differs from logits')
    if negative.dtype != np.bool_ or positive.dtype != np.bool_ or np.any(negative & positive):
        raise ValueError('Masks must be bool and mutually exclusive')
    if negative[:, 4:].any() or positive[:, 4:].any():
        raise ValueError('Only four original short channels may change')
    n = torch.zeros_like(logits,dtype=torch.bool); p = torch.zeros_like(logits,dtype=torch.bool)
    n.copy_(torch.from_numpy(negative)); p.copy_(torch.from_numpy(positive))
    value = torch.where(p, logits.new_tensor(16.), torch.where(n, logits.new_tensor(-16.), logits))
    if not torch.equal(value[~(n | p)], logits[~(n | p)]):
        raise RuntimeError('Unselected affinity logits changed')
    return value


def recompute(views, tiles, masks, config, shape, *, force_empty_cuda=False):
    from train_backend_adaptation import fusion_options
    from utils.affinity_fusion import affinity_boundary_probability
    mode, kwargs = fusion_options(config); blend = BlendMap(shape); probabilities = []
    with torch.no_grad():
        for tile, (negative, positive) in zip(tiles, masks):
            logits = corrected_logits(tile['logits'], negative, positive)
            probabilities.append(logits[0, :4].sigmoid().cpu().numpy().copy())
            if not force_empty_cuda and not negative.any() and not positive.any():
                native = tile['native'][0, 0]
            else:
                grid = affinity_boundary_probability(logits.float(), mode=mode, **kwargs)
                y, x, y1, x1 = tile['box']
                native = views.crop_letterbox_output(grid, 1024, tile['pad_h'], tile['pad_w'],
                                                     (y1-y, x1-x)).cpu()[0, 0].numpy()
                if force_empty_cuda and (not np.array_equal(grid.cpu().numpy(),tile['grid'])
                        or not np.array_equal(native,tile['native'][0,0])):
                    raise RuntimeError('Actual two-where empty intervention grid/native is not exact')
            blend.add(tile['box'], native)
    boundary, audit = blend.finish(); audit['tiles'] = len(tiles)
    return boundary, audit, probabilities


def error_sets(diag, partition):
    return dict(safe_same_parent={r['gt'] for r in diag['gt_rows'] if r['same_parent_internal_partition_candidate'] and not r['topology_censored']},
        safe_ff={tuple(sorted(p)) for r in diag['ff_merge_rows'] if not r['censored'] and r['whole_grain_cooccupancy'] for p in r['adjacent_gt_pairs']},
        conservative_split=set(partition['conservative']['split_gt_ids']),
        conservative_merge={tuple(sorted(p)) for p in partition['conservative']['merged_gt_pairs']})


def measure(decoded, raw, gt, classes, baseline=None):
    ids = decoded['instances']; votes, _ = revote_instances(ids, raw)
    diag = analyze_ferrite_native(gt, ids, classes, votes)
    partition = diagnose_partition(gt, ids, classes, votes)
    rows = {r['gt']: r for r in diag['gt_rows']}
    old = diag if baseline is None else baseline['full_ferrite']
    matched = [r['gt'] for r in old['gt_rows'] if r['matched_pred'] is not None]
    errors = error_sets(diag, partition); prior_errors = errors if baseline is None else error_sets(old, baseline['full_partition'])
    phase = np.full(int(gt.max()) + 1, -1, np.int8); pred_phase = np.full(int(ids.max()) + 1, -1, np.int8)
    for g, c in classes.items(): phase[int(g)] = int(c)
    for p, c in votes.items(): pred_phase[int(p)] = int(c)
    truth, predicted = phase[gt], pred_phase[ids]
    report = dict(instances=len(votes), maximum_id=int(ids.max()), markers=decoded['stages']['marker_count'],
        original_matched_F_count=len(matched), original_matched_F_missing=[g for g in matched if rows[g]['matched_pred'] is None],
        original_matched_F_mean_IoU_missing_as_zero=sum((rows[g]['matched_iou'] or 0) for g in matched)/max(1, len(matched)),
        all_F_count=len(rows), all_F_mean_IoU_missing_as_zero=sum((r['matched_iou'] or 0) for r in rows.values())/max(1, len(rows)),
        fixed_area=fixed_area_cohort(old, diag),
        error_changes={k:dict(added=sorted(errors[k]-prior_errors[k]), removed=sorted(prior_errors[k]-errors[k])) for k in errors},
        F_predicted_P=int(((gt > 0) & (truth == 1) & (predicted == 0)).sum()),
        P_predicted_F=int(((gt > 0) & (truth == 0) & (predicted == 1)).sum()),
        F_unassigned=int(((gt > 0) & (truth == 1) & (predicted < 0)).sum()),
        phases=phase_description(ids, votes), full_ferrite=diag, full_partition=partition)
    return report, votes


def validate_case(case, gt):
    groups = list(map(int, case['gt_ids'])); points = case['points']; direction = case['direction']
    expected = 2 if direction == 'negative' else 1
    if direction not in ('negative', 'positive') or len(groups) != expected or min(groups) <= 0:
        raise ValueError('Case must provide legal positive GT IDs and direction')
    if len(set(groups)) != len(groups) or len(points) != 2:
        raise ValueError('Case IDs/points invalid')
    for p, g in zip(points, [groups[0], groups[-1]]):
        if len(p) != 2 or any(not isinstance(v, int) for v in p):
            raise ValueError('Native endpoint must be integer [y,x]')
        if not (0 <= p[0] < gt.shape[0] and 0 <= p[1] < gt.shape[1]) or int(gt[tuple(p)]) != g:
            raise ValueError('Endpoint is not its promised known GT')


def save_variant(folder, stem, decoded, votes, boundary):
    save_prediction(folder, stem, decoded['instances'], votes)
    np.savez_compressed(folder / (stem + '_maps.npz'), boundary=boundary,
                        markers=decoded['actual_markers'], watershed=decoded['watershed'])


def analyze_source(views, model, restorer, source, config, output, max_rounds):
    from data.semantic_targets import load_completed_semantic_source
    path, gt_path = (ROOT / source[k] for k in ('source_path', 'gt_path'))
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None: raise RuntimeError('Source image missing')
    gt, lookup = load_completed_semantic_source(gt_path, bgr.shape[:2])
    classes = {int(g): int(lookup[g]) for g in np.unique(gt) if g > 0}
    for case in source['cases']: validate_case(case, gt)
    folder = output / path.stem; folder.mkdir()
    class_path=gt_path.with_name(gt_path.name.removesuffix('_gt.npz')+'_class.json')
    before_inputs = {str(p):sha(p) for p in (path, gt_path, class_path)}
    rgb, captured, tiles, capture_audit = capture_native_tiles(views, model, restorer, path, config, gt, 'cuda', source['seed'])
    base = decode_capture(captured['semantic'], captured['boundary'], config)
    raw = captured['raw_probability']; original = captured['baseline']
    base_metrics, base_votes = measure(base, raw, gt, classes)
    empty_masks=[(np.zeros(tuple(t['logits'].shape),bool),np.zeros(tuple(t['logits'].shape),bool)) for t in tiles]
    empty_boundary,empty_blend,_=recompute(views,tiles,empty_masks,config,gt.shape,force_empty_cuda=True)
    if not np.array_equal(empty_boundary,original['boundary']) or empty_blend!=original['blend']:
        raise RuntimeError('Actual two-where empty intervention stitched output is not exact')
    capture_audit['actual_two_where_empty_cuda_grid_native_stitch_exact']=True
    if (not np.array_equal(base['instances'], original['instances']) or base_votes != original['classes']
            or not np.array_equal(base['actual_markers'], captured['markers'])
            or not np.array_equal(base['watershed'], captured['watershed'])):
        raise RuntimeError('Original pipeline replay differs from actual captured baseline')
    refdir = ROOT / source['reference_dir']; prefix = source.get('reference_prefix', 'r1')
    reference = cv2.imread(str(refdir / (prefix + '_inst.png')), cv2.IMREAD_UNCHANGED)
    if reference is None or not np.array_equal(reference, original['instances']) or read(refdir / (prefix + '_class.json')) != base_votes:
        raise RuntimeError('Recomputed baseline differs from original source cohort')
    with np.load(refdir / (prefix + '_maps.npz'), allow_pickle=False) as ref:
        for key, value in [('boundary',original['boundary']),('semantic',captured['semantic'].numpy()),('raw_probability',raw),
                           ('markers',base['actual_markers']),('watershed',base['watershed'])]:
            if not np.array_equal(np.squeeze(ref[key]), np.squeeze(value)):
                raise RuntimeError('Original reference continuous/stage map changed: ' + key)
    save_variant(folder, 'baseline', base, base_votes, original['boundary'])
    report = dict(source=source, training=False, input_hashes=before_inputs, capture_audit=capture_audit,
        baseline_exact=True, baseline=base_metrics, arms={})
    # 单方向与双方向都使用同一捕获，分清补连/断连的贡献，不重跑控制。
    final_decoded = {}
    for arm, directions in [('negative', {'negative'}), ('positive', {'positive'}), ('combined', {'negative','positive'})]:
        cases = [c for c in source['cases'] if c['direction'] in directions]
        if not cases: continue
        masks = [(np.zeros(tuple(t['logits'].shape), bool), np.zeros(tuple(t['logits'].shape), bool)) for t in tiles]
        decoded = base; rounds = []; boundary = original['boundary']; probabilities = [t['logits'][0,:4].sigmoid().cpu().numpy() for t in tiles]
        for index in range(max_rounds):
            selections = []; added = 0
            for case_index, case in enumerate(cases):
                for tile_index, tile in enumerate(tiles):
                    result = select_bottlenecks(tile, gt, decoded, case, probabilities[tile_index])
                    # 负样本零容量已经被切断；不要重复向已断关系加监督。
                    if result['found'] and not (case['direction']=='negative' and result['zero_capacity']):
                        mask = masks[tile_index][0 if case['direction']=='negative' else 1]
                        for c, y, x in result['bottleneck_indices']:
                            if not mask[0,c,y,x]: mask[0,c,y,x] = True; added += 1
                    selections.append(dict(case_index=case_index, **result))
            if not added:
                rounds.append(dict(round=index+1, added=0, selections=selections, reason='no_new_legal_bottleneck')); break
            boundary, blend, probabilities = recompute(views, tiles, masks, config, gt.shape)
            decoded = decode_capture(captured['semantic'], torch.from_numpy(boundary[None,None]), config)
            measured, votes = measure(decoded, raw, gt, classes, base_metrics)
            entry = dict(round=index+1, added=added, selected_negative=sum(int(n.sum()) for n,p in masks),
                selected_positive=sum(int(p.sum()) for n,p in masks), selections=selections,
                boundary_sha256=array_sha(boundary), blend=blend, metrics=measured,
                relations=[case_relation(decoded,c) for c in cases],case_quality=[case_quality(decoded,gt,c) for c in cases])
            rounds.append(entry)
            save_variant(folder, arm + f'_{index+1:02d}', decoded, votes, boundary)
            print(json.dumps(dict(stage='intervention', source=path.name, arm=arm, round=index+1, added=added,
                endpoint_goals=sum(r['endpoint_goal'] for r in entry['relations']), lost_F=measured['original_matched_F_missing'])), flush=True)
        final_metrics, final_votes = measure(decoded, raw, gt, classes, base_metrics)
        report['arms'][arm] = dict(rounds=rounds, final_metrics=final_metrics,
            final_relations=[case_relation(decoded,c) for c in cases], cases=cases,
            baseline_case_quality=[case_quality(base,gt,c) for c in cases],
            final_case_quality=[case_quality(decoded,gt,c) for c in cases],
            selected_negative=sum(int(n.sum()) for n,p in masks), selected_positive=sum(int(p.sum()) for n,p in masks))
        final_decoded[arm] = decoded
        save_variant(folder, arm + '_final', decoded, final_votes, boundary)
        for tile, (negative, positive) in zip(tiles,masks):
            if negative.any() or positive.any():
                np.savez_compressed(folder / (arm+f'_tile_{tile["index"]:02d}_masks.npz'),negative=negative,positive=positive,
                    grid_gt=tile['grid_gt'],valid=tile['valid'])
    panels = [gallery_panel(rgb,base['instances'],gt>0,gt)]
    for key, decoded in final_decoded.items(): panels.append(gallery_panel(rgb,decoded['instances'],gt>0,gt))
    small = [cv2.resize(p,(500,round(p.shape[0]*500/p.shape[1])),interpolation=cv2.INTER_AREA) for p in panels]
    cv2.imwrite(str(folder/'compare.png'), cv2.cvtColor(np.concatenate(small,axis=1),cv2.COLOR_RGB2BGR))
    if any(sha(p) != digest for p,digest in before_inputs.items()): raise RuntimeError('Protected source/GT changed')
    write(folder/'report.json',report)
    del tiles; gc.collect(); torch.cuda.empty_cache()
    return report


def run(args):
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from train_backend_adaptation import tensor_digest
    from tools.run_affinity_sweep import runtime_contract
    from utils.config import load_config, project_path
    output=(ROOT/args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'outputs') or not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA sam2_env and fresh ignored outputs directory')
    if not 1<=args.rounds<=8: raise ValueError('Fixed path intervention budget must be 1..8')
    output.mkdir(parents=True); started=time.time(); torch.set_num_threads(4); cv2.setNumThreads(2)
    write(output/'status.json',dict(status='running',training=False,stage='preflight'))
    try:
        receipt=read(ROOT/args.receipt); r1=receipt['arms']['r1']; config=deepcopy(r1['config'])
        config_gate(config,load_config(str(ROOT/'config/train/affinity_balance_mixed.yaml')))
        plan=read(ROOT/args.plan)
        if not plan['complete'] or not plan['sources']: raise RuntimeError('Complete fixed source plan required')
        core=load_original_core(config); views=core.coverage_original_views
        checkpoint=Path(r1['checkpoint']); restore=Path(project_path(config,config['backend_adaptation']['restoration']['checkpoint']))
        if sha(checkpoint)!=r1['checkpoint_sha256'] or sha(restore)!=receipt['restorer_checkpoint_sha256']:
            raise RuntimeError('Frozen checkpoint identity differs')
        model,bundle=load_backend(checkpoint,config,'cuda'); model.eval().requires_grad_(False)
        restorer=get_restorer(config,'cuda').eval().requires_grad_(False)
        states=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items()))
        if states!=(r1['model_state_sha256'],receipt['restorer_state_sha256']) or bundle['epoch']!=20:
            raise RuntimeError('Frozen model states or epoch differ')
        if precision_record()!=receipt['precision'] or runtime_contract(config)!=receipt.get('current_runtime',receipt.get('runtime')):
            raise RuntimeError('Pinned original runtime/precision differs')
        if any(m.training for m in list(model.modules())+list(restorer.modules())): raise RuntimeError('All modules must be eval')
        report=dict(format=FORMAT,complete=False,training=False,no_optimizer_steps=True,
            official_accuracy=False,best_deployment_changed=False,submission_package_created=False,
            only_variable='constrained widest-path legal short-edge GT intervention; original readout recomputed',
            max_rounds=args.rounds,config=config,plan_sha256=sha(ROOT/args.plan),checkpoint_sha256=sha(checkpoint),
            model_state_sha256=states[0],restorer_state_sha256=states[1],restorer_checkpoint_sha256=sha(restore),
            parameter_count=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters()),
            runtime=runtime_contract(config),precision=precision_record(),sources=[],
            caveat='All known processed GT includes filled; training sources seen, not independent validation or official accuracy')
        if report['parameter_count']>=500_000_000: raise RuntimeError('Parameter budget exceeded')
        protected,_=source_snapshot(output); write(output/'preflight.json',report)
        selected=plan['sources'][:args.limit] if args.limit else plan['sources']
        for source in selected:
            result=analyze_source(views,model,restorer,source,config,output,args.rounds)
            report['sources'].append(result); write(output/'report.json',report)
        if states!=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items())):
            raise RuntimeError('Frozen model/restorer weights changed')
        for name,digest in protected.items():
            if sha(ROOT/name)!=digest: raise RuntimeError('Loaded project source changed during experiment: '+name)
        report.update(complete=True,elapsed_seconds=time.time()-started,frozen_states_exact=True)
        write(output/'report.json',report); write(output/'status.json',dict(status='complete',training=False,
            sources=len(report['sources']),elapsed_seconds=report['elapsed_seconds']))
    except BaseException as exc:
        write(output/'status.json',dict(status='failed',training=False,error=str(exc))); raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True); p.add_argument('--output',default='outputs/affinity_bottleneck')
    p.add_argument('--receipt',default='outputs/affinity_weak/capture.json')
    p.add_argument('--rounds',type=int,default=8); p.add_argument('--limit',type=int,default=0)
    run(p.parse_args())


if __name__=='__main__': main()
