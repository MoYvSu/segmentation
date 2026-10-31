# -*- coding: utf-8 -*-
"""冻结原生输出的ID0空间与面积闭合；不填缝、不修改预测、不进行推理。

final ID0和种子markers不能唯一识别分水岭界线或面积过滤来源。
只有调用方提供真实signed watershed，或normalized watershed及明确边界mask，
才能把最终零像素精确分成watershed边界、watershed原零、正ID后续删除。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.analyze_ferrite_native import FORMAT as FERRITE_FORMAT
from tools.analyze_ferrite_native import _classes, _labels, _unsafe_ids


FORMAT = 'ferrite_zero_seams_v1'
RULES = dict(zero_connectivity=8, positive_distance='exact chessboard distance bins 1/2/>2',
             zero_component_scope='whole final0 including unknown; known phase counts reported separately',
             eligible_F_scope='reuse caller verified parent safe_one_to_one_area_eligible GT identities',
             case_order='ascending GT or zero/WS component IDs, no error magnitude ranking',
             intervention=False)
KINDS = ('eligible_F_distance1', 'eligible_F_distance2', 'eligible_F_distance_gt2',
         'positive_ws_removed_F')


def _integer_grid(value, shape, name, minimum=0):
    value = np.asarray(value)
    if (value.shape != shape or not np.issubdtype(value.dtype, np.integer)
            or value.dtype == np.bool_ or value.min() < minimum
            or value.max() > np.iinfo(np.int32).max):
        raise ValueError(name + ' must be a same-grid integer array with valid stage IDs')
    return value


def _phase_lookup(labels, mapping):
    lookup = np.full(int(labels.max())+1, -1, np.int8)
    for value, phase in mapping.items():
        lookup[value] = phase
    return lookup[labels]


def _phase_counts(mask, phase):
    return dict(P=int(np.count_nonzero(mask & (phase == 0))),
                F=int(np.count_nonzero(mask & (phase == 1))))


def _distance_bins(pred):
    """只需两次方形膨胀得到精确8邻距离分组，无正ID时全归>2并明确标记。"""
    positive, zero = pred > 0, pred == 0
    adjacent1 = cv2.dilate(positive.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    adjacent2 = cv2.dilate(positive.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return {'distance1': zero & adjacent1, 'distance2': zero & adjacent2 & ~adjacent1,
            'distance_gt2': zero & ~adjacent2}


def _source_masks(watershed, watershed_format, watershed_boundary_mask, shape):
    if watershed is None:
        if watershed_format is not None or watershed_boundary_mask is not None:
            raise ValueError('Watershed format/mask requires actual watershed array')
        return None
    if watershed_format not in ('signed', 'zero_normalized'):
        raise ValueError('Explicit watershed_format signed/zero_normalized required')
    ws = _integer_grid(watershed, shape, 'watershed', -1 if watershed_format == 'signed' else 0)
    if watershed_format == 'signed':
        boundary = ws == -1
        if watershed_boundary_mask is not None:
            supplied = np.asarray(watershed_boundary_mask)
            if supplied.dtype != np.bool_ or supplied.shape != shape or not np.array_equal(supplied, boundary):
                raise ValueError('Signed watershed explicit boundary mask must exactly equal ws==-1')
        original_zero = ws == 0
    else:
        boundary = np.asarray(watershed_boundary_mask)
        if (boundary.dtype != np.bool_ or boundary.shape != shape
                or np.any(boundary & (ws != 0))):
            raise ValueError('Zero-normalized watershed requires explicit same-grid boolean boundary mask')
        original_zero = (ws == 0) & ~boundary
    return dict(ws=ws, ws_boundary=boundary, ws_original_zero=original_zero, positive_ws=ws > 0)


def _component_rows(zero, known, phase, bins, markers, sources):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(zero.astype(np.uint8), connectivity=8)
    frame = {int(v) for v in np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))) if v > 0}
    def counts(mask):
        return np.bincount(labels[mask], minlength=count)
    f, p = counts(known & (phase == 1)), counts(known & (phase == 0))
    positive_seed = counts(zero & (markers > 0))
    widths = {name: counts(mask & known) for name, mask in bins.items()}
    source_counts = ({name: counts(zero & mask & known) for name, mask in sources.items()
                      if name != 'ws'} if sources is not None else {})
    rows = []
    for i in range(1, count):
        x, y, width, height, pixels = map(int, stats[i])
        row = dict(component=i, pixels=pixels, bbox_yxyx=[y,x,y+height,x+width],
            known_F_pixels=int(f[i]), known_P_pixels=int(p[i]),
            unknown_or_invalid_pixels=pixels-int(f[i])-int(p[i]),
            touches_frame=i in frame, contains_unknown_or_invalid=bool(pixels-int(f[i])-int(p[i])),
            final0_on_positive_marker_pixels=int(positive_seed[i]),
            distance_bins_known={name:int(values[i]) for name, values in widths.items()})
        if source_counts:
            row['actual_ws_sources_known'] = {name:int(values[i]) for name, values in source_counts.items()}
        rows.append(row)
    return rows


def _safe_area_rows(gt, pred, known, parent, bins, sources, gc, pc):
    if parent.get('format') != FERRITE_FORMAT or parent.get('shape') != list(gt.shape):
        raise ValueError('Same-grid sealed parent ferrite diagnostic required')
    ids, gi = np.unique(gt[known], return_inverse=True)
    labels, pi = np.unique(pred[known], return_inverse=True)
    matrix = np.bincount(gi*len(labels)+pi, minlength=len(ids)*len(labels)).reshape(len(ids),len(labels))
    gindex, pindex = {int(v):i for i,v in enumerate(ids)}, {int(v):i for i,v in enumerate(labels)}
    ga, pa = matrix.sum(1), matrix.sum(0)
    all_pred, all_counts = np.unique(pred, return_counts=True)
    total = {int(v):int(n) for v,n in zip(all_pred,all_counts)}
    bad_g, bad_p = _unsafe_ids(gt,known), _unsafe_ids(pred,known)
    def by_gt(mask):
        return np.bincount(gi[mask[known]], minlength=len(ids))
    widths = {name:by_gt(mask & known) for name,mask in bins.items()}
    attributed = ({name:by_gt((pred==0)&mask&known) for name,mask in sources.items() if name!='ws'}
                  if sources is not None else {})
    rows = []
    for old in parent['gt_rows']:
        if not old['safe_one_to_one_area_eligible']:
            continue
        value, match = old['gt'], old['matched_pred']
        if value not in gindex or match not in pindex or gc[value]!=1 or pc[match]!=1:
            raise ValueError('Parent eligible F identity or same-class match differs')
        i,j = gindex[value],pindex[match]
        intersection, area, prediction_area = int(matrix[i,j]),int(ga[i]),total[match]
        iou = intersection/(area+int(pa[j])-intersection)
        if (value in bad_g or match in bad_p or area!=old['area']
                or prediction_area!=old['matched_pred_total_pixels']
                or abs(iou-old['matched_iou'])>1e-12
                or abs((prediction_area-area)/area-old['area_relative_error'])>1e-12):
            raise ValueError('Parent safe area/censor/matching contract differs')
        zero_pixels = int(sum(v[i] for v in widths.values()))
        other_pred_pixels = area-intersection-zero_pixels
        extra_pred_pixels = prediction_area-intersection
        deficit = area-prediction_area
        if deficit != zero_pixels+other_pred_pixels-extra_pred_pixels:
            raise RuntimeError('Exact safe F area balance failed')
        row = dict(gt=value,matched_pred=match,GT_pixels=area,pred_pixels=prediction_area,
            intersection_pixels=intersection,signed_area_deficit_pixels=deficit,
            relative_area_error=old['area_relative_error'],
            final0_inside_GT=zero_pixels,
            distance_bins_inside_GT={name:int(v[i]) for name,v in widths.items()},
            other_positive_pred_inside_GT=other_pred_pixels,matched_pred_outside_GT=extra_pred_pixels,
            area_balance_exact=True)
        if attributed:
            row['actual_ws_sources_final0_inside_GT'] = {name:int(v[i]) for name,v in attributed.items()}
        rows.append(row)
    return rows


def _area_summary(rows):
    def aggregate(selected):
        return dict(n=len(selected),GT_pixels=sum(r['GT_pixels'] for r in selected),
            pred_pixels=sum(r['pred_pixels'] for r in selected),
            signed_area_deficit_pixels=sum(r['signed_area_deficit_pixels'] for r in selected),
            final0_inside_GT=sum(r['final0_inside_GT'] for r in selected),
            distance_bins_inside_GT={name:sum(r['distance_bins_inside_GT'][name] for r in selected)
                for name in ('distance1','distance2','distance_gt2')},
            other_positive_pred_inside_GT=sum(r['other_positive_pred_inside_GT'] for r in selected),
            matched_pred_outside_GT=sum(r['matched_pred_outside_GT'] for r in selected),
            area_balance_exact=all(r['area_balance_exact'] for r in selected))
    return dict(all_eligible=aggregate(rows),shrinking=aggregate([r for r in rows if r['signed_area_deficit_pixels']>0]),
                expanding=aggregate([r for r in rows if r['signed_area_deficit_pixels']<0]),
                unchanged=aggregate([r for r in rows if r['signed_area_deficit_pixels']==0]))


def analyze_ferrite_seams(gt,pred,gt_classes,pred_classes,markers,ferrite_report,
                         valid_content=None,watershed=None,watershed_format=None,
                         watershed_boundary_mask=None):
    """同原生网格CPU诊断；parent名单必须对应实际同一预测，不能套给变化后的blur输出。

    signed watershed真实-1和原0分别保留；normalized若无边界mask拒绝三来源归因。
    正ws ID变final0证明发生后续删除，具体面积过滤原因须由外部固定管线回执确认。
    """
    gt,pred = _labels(gt,'gt'),_labels(pred,'pred')
    if gt.shape!=pred.shape:
        raise ValueError('Exact same native grids required')
    gc,pc = _classes(gt_classes,gt,'gt_classes'),_classes(pred_classes,pred,'pred_classes')
    markers = _integer_grid(markers,gt.shape,'actual markers')
    content = np.ones(gt.shape,bool) if valid_content is None else np.asarray(valid_content)
    if content.shape!=gt.shape or content.dtype!=np.bool_:
        raise ValueError('valid_content must be a same-grid boolean array')
    known = (gt>0)&content
    phase,pred_phase = _phase_lookup(gt,gc),_phase_lookup(pred,pc)
    zero,bins = pred==0,_distance_bins(pred)
    sources = _source_masks(watershed,watershed_format,watershed_boundary_mask,gt.shape)
    phase_matrix = [[int(np.count_nonzero(known&(phase==truth)&(pred_phase==guess)))
                     for guess in (0,1,-1)] for truth in (0,1)]
    widths = {name:_phase_counts(mask&known,phase) for name,mask in bins.items()}
    components = _component_rows(zero,known,phase,bins,markers,sources)
    eligible = _safe_area_rows(gt,pred,known,ferrite_report,bins,sources,gc,pc)
    attribution = dict(available=sources is not None,watershed_format=watershed_format,
        unidentified_from_final_and_markers=sources is None)
    if sources is not None:
        attributed = {name:_phase_counts(zero&mask&known,phase) for name,mask in sources.items() if name!='ws'}
        if any(sum(r[phase_name] for r in attributed.values())!=_phase_counts(zero&known,phase)[phase_name]
               for phase_name in ('P','F')):
            raise RuntimeError('Actual watershed final0 source partition is not closed')
        attribution.update(final0_known_sources=attributed,source_partition_exact=True,
            ws_nonpositive_changed_to_final_positive=_phase_counts((~zero)&(sources['ws']<=0)&known,phase),
            positive_ws_removed_description='actual ws>0 and final0; later deletion, not independently proof of a specific area-filter threshold')
    if sum(sum(row) for row in phase_matrix)!=int(known.sum()):
        raise RuntimeError('Known phase allocation is not closed')
    expected = ferrite_report['summary']['counts']
    if (sum(phase_matrix[1])!=sum(r['area'] for r in ferrite_report['gt_rows'])
            or phase_matrix[1][0]!=expected['F_pixels_predicted_P']
            or phase_matrix[1][2]!=sum(r['unassigned_known_pixels'] for r in ferrite_report['gt_rows'])):
        raise ValueError('Parent F known allocation differs')
    result = dict(format=FORMAT,rules=dict(RULES),shape=list(gt.shape),
        known_phase_pixels=phase_matrix,matrix_rows=['GT_P','GT_F'],matrix_columns=['pred_P','pred_F','pred0'],
        known_phase_allocation_closed=True,known_phase_parent_F_exact=True,
        final0_known=_phase_counts(zero&known,phase),distance_bins_known=widths,
        no_positive_output=not bool((pred>0).any()),
        marker_observation=dict(final0_on_positive_actual_marker_known=_phase_counts(zero&(markers>0)&known,phase),
            scope='seed overlap describes stage support; never uniquely classifies watershed seam or area filtering'),
        zero_components=components,
        component_summary=dict(n=len(components),touches_frame=sum(r['touches_frame'] for r in components),
            contains_unknown_or_invalid=sum(r['contains_unknown_or_invalid'] for r in components),
            known_F_pixels=sum(r['known_F_pixels'] for r in components),
            known_P_pixels=sum(r['known_P_pixels'] for r in components)),
        safe_area_rows=eligible,safe_area_summary=_area_summary(eligible),actual_ws_attribution=attribution,
        caveats=['Full processed newGT including filled; GT0/invalid ignored in known phase/area counts.',
            'Distance1/2/>2 measures 8-neighbor distance to positive final IDs, not geometric seam width or correctness.',
            'Final0 and markers cannot uniquely distinguish watershed boundary from later region deletion.',
            'Zero components are connected over the full output; frame/unknown status does not erase their known-domain contribution.',
            'Parent safe F area identities are fixed; zero, other-pred loss and extra-pred gains form an exact algebraic balance, not a causal intervention.',
            'No GT filling, output change, force counts, area fitting, new thresholds, training, test GT or official score.'])
    json.dumps(result,allow_nan=False)
    return result


def select_ferrite_seam_cases(report,max_per_kind=1):
    """事先固定规则，只选parent完整可信GT；不按最大亏面积选例。"""
    if isinstance(max_per_kind,bool) or not isinstance(max_per_kind,int) or max_per_kind<0:
        raise ValueError('max_per_kind must be a nonnegative integer')
    chosen=[]
    for kind,width in zip(KINDS[:3],('distance1','distance2','distance_gt2')):
        rows=[r for r in report['safe_area_rows'] if r['distance_bins_inside_GT'][width]>0]
        chosen.extend(dict(kind=kind,gt_ids=[r['gt']],pred_ids=[r['matched_pred']],
            observed_pixels=r['distance_bins_inside_GT'][width],evidence_rule='first eligible GT ID with nonzero predeclared distance bin')
            for r in sorted(rows,key=lambda q:q['gt'])[:max_per_kind])
    if report['actual_ws_attribution']['available']:
        rows=[r for r in report['safe_area_rows'] if r['actual_ws_sources_final0_inside_GT']['positive_ws']>0]
        chosen.extend(dict(kind=KINDS[3],gt_ids=[r['gt']],pred_ids=[r['matched_pred']],
            observed_pixels=r['actual_ws_sources_final0_inside_GT']['positive_ws'],
            evidence_rule='first eligible GT ID with actual ws-positive pixels removed in final')
            for r in sorted(rows,key=lambda q:q['gt'])[:max_per_kind])
    return chosen


def selftest():
    from tools.analyze_ferrite_native import analyze_ferrite_native
    gt=np.full((40,40),2,np.uint16);gt[10:30,10:30]=1
    pred=gt.copy();pred[10:30,28:30]=0
    parent=analyze_ferrite_native(gt,pred,{1:1,2:0},{1:1,2:0})
    markers=pred.astype(np.int32)
    result=analyze_ferrite_seams(gt,pred,{1:1,2:0},{1:1,2:0},markers,parent)
    assert result['final0_known']['F']==40
    assert result['safe_area_summary']['all_eligible']['signed_area_deficit_pixels']==40
    assert not result['actual_ws_attribution']['available']
    ws=pred.astype(np.int32);ws[10:30,29]=-1;ws[10:30,28]=1
    sources=analyze_ferrite_seams(gt,pred,{1:1,2:0},{1:1,2:0},markers,parent,watershed=ws,watershed_format='signed')
    assert sources['actual_ws_attribution']['final0_known_sources']['ws_boundary']['F']==20
    assert sources['actual_ws_attribution']['final0_known_sources']['positive_ws']['F']==20
    return dict(passed=True,format=FORMAT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if not args.selftest:
        parser.error('CPU array core only; standalone --selftest')
    print(json.dumps(selftest(),ensure_ascii=False))
