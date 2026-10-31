# -*- coding: utf-8 -*-
"""用可靠 GT 短缺段生成只增不减的局部纠错候选；不训练、不替代推理读出。

目标仍保留真实 N0 的幅值、底噪和已有响应。默认仅信原始标注安全区；
显式 processed_thin_fill 模式另允许距原标不超过匹配半径、且最近原标
同实例的窄填补区。后者是处理后 GT 信号，不是人工原始真值。所有模式
仍须双端强响应，且 unknown 不参与正监督；候选不宣称 GT 完全正确。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import cv2
import numpy as np
from scipy.ndimage import label, find_objects, distance_transform_edt
from skimage.morphology import skeletonize


DEFAULT_CORRECTION_CONFIG = {
    'support_policy': 'original_strict',
    'valid_margin': 8,
    'match_radius': 4,
    'low_threshold': .45,
    'anchor_threshold': .65,
    'min_gap_length': 8.,
    'max_gap_length': 64.,
    'anchor_search_length': 16.,
    'sigma': 3.,
    'peak': .90,
    'truncate_sigma': 3.,
    'junction_margin': 2,
    'offset_guard': True,
    'offset_reject_radius': 12,
    'outside_ridge_threshold': .45,
    'profile_tangent_half_points': 4,
}


def _configuration(config):
    value = dict(DEFAULT_CORRECTION_CONFIG)
    if config is not None:
        unknown = set(config) - set(value)
        if unknown:
            raise ValueError(f'Unknown boundary correction keys: {sorted(unknown)}')
        value.update(config)
    if type(value['offset_guard']) is not bool:
        raise ValueError('offset_guard must be bool')
    if value['support_policy'] not in ('original_strict','processed_thin_fill'):
        raise ValueError('Unknown GT support policy')
    integer_keys = {'valid_margin', 'match_radius', 'junction_margin',
                    'offset_reject_radius', 'profile_tangent_half_points'}
    for key in integer_keys:
        if type(value[key]) is not int or value[key] < (0 if key == 'junction_margin' else 1):
            raise ValueError(f'{key} must be an integer native-pixel radius')
    for key in set(value) - integer_keys - {'offset_guard','support_policy'}:
        if isinstance(value[key], bool) or not math.isfinite(float(value[key])):
            raise ValueError(f'{key} must be finite')
        value[key] = float(value[key])
    if not 0 < value['low_threshold'] < value['anchor_threshold'] < value['peak'] <= 1:
        raise ValueError('Require 0 < low < anchor < peak <= 1')
    if not 0 < value['min_gap_length'] <= value['max_gap_length']:
        raise ValueError('Positive ordered native gap length range required')
    if min(value['anchor_search_length'], value['sigma'], value['truncate_sigma']) <= 0:
        raise ValueError('Positive anchor distance and native rendering scale required')
    if value['offset_reject_radius'] <= value['match_radius']:
        raise ValueError('Offset guard must inspect outside the original matching radius')
    if not 0 < value['outside_ridge_threshold'] <= 1:
        raise ValueError('Outside-ridge response threshold must be in (0,1]')
    return value


def _binary(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or not np.isin(result, (0, 1)).all():
        raise ValueError(f'{name} must be a binary mask matching native shape')
    return result.astype(bool)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _disk(radius):
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (xx*xx+yy*yy <= radius*radius).astype(np.uint8)


def _known_boundary(ids):
    known = ids > 0
    result = np.zeros(ids.shape, bool)
    a = known[1:] & known[:-1] & (ids[1:] != ids[:-1])
    b = known[:, 1:] & known[:, :-1] & (ids[:, 1:] != ids[:, :-1])
    result[1:] |= a; result[:-1] |= a
    result[:, 1:] |= b; result[:, :-1] |= b
    return result


def _support_domain(ids, original, config):
    """支持域独立于模型置信度；不回写 GT，也不把大片填补视为原始标注。"""
    known = ids>0
    filled = known & ~original
    margin = config['valid_margin']
    kernel = np.ones((2*margin+1,2*margin+1),np.uint8)
    full_known_safe = cv2.erode(known.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,
                              borderValue=0).astype(bool)
    own_support = original.copy()
    near_filled = np.zeros(ids.shape,bool)
    same_nearest_id = np.zeros(ids.shape,bool)
    if config['support_policy']=='processed_thin_fill' and original.any():
        distance, nearest = distance_transform_edt(~original,return_indices=True)
        near_filled = filled & (distance<=config['match_radius'])
        same_nearest_id = filled & (ids[tuple(nearest)]==ids)
        own_support |= near_filled & same_nearest_id
    if config['support_policy']=='original_strict':
        valid = cv2.erode(original.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(bool)
    else:
        valid = full_known_safe.copy()
    distance_evaluated = config['support_policy']=='processed_thin_fill'
    receipt = dict(policy=config['support_policy'],original_pixels=int(original.sum()),
                   filled_pixels=int(filled.sum()),supported_original_pixels=int((own_support&original).sum()),
                   supported_filled_pixels=int((own_support&filled).sum()),
                   thin_fill_distance_evaluated=distance_evaluated,
                   filled_near_original_pixels=int(near_filled.sum()) if distance_evaluated else None,
                   filled_near_but_nearest_other_instance=int((near_filled&~same_nearest_id).sum()) if distance_evaluated else None,
                   filled_without_near_original=int((filled&~near_filled).sum()) if distance_evaluated else None,
                   thin_fill_max_native_distance=config['match_radius'],
                   nearest_rule='Nearest point in original-covered union must have the same instance ID; ties use deterministic SciPy EDT selection',
                   positive_support_role=('original-covered only' if config['support_policy']=='original_strict'
                                          else 'processed GT narrow-fill support, not original human truth'))
    return own_support,valid,full_known_safe,receipt


def _neighbours(point, pixels):
    """八邻接，但已有轴向连接时不增加斜向捷径，避免拐角被误认三叉。"""
    y, x = point
    result = []
    for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        other = (y+dy, x+dx)
        if other not in pixels:
            continue
        if dy and dx and ((y+dy, x) in pixels or (y, x+dx) in pixels):
            continue
        result.append(other)
    return result


def _paths(skeleton):
    """确定性地排序去分叉后的简单路径/环；绝不跨连通分量拼接。"""
    component_map, _ = label(skeleton, structure=np.ones((3,3), np.uint8))
    paths, invalid = [], 0
    for component_id, box in enumerate(find_objects(component_map), 1):
        if box is None:
            continue
        points = np.argwhere(component_map[box] == component_id)
        points += np.array([box[0].start, box[1].start])
        pixels = set(map(tuple, points.tolist()))
        graph = {point:_neighbours(point, pixels) for point in pixels}
        if any(len(ns)>2 for ns in graph.values()):
            invalid += 1
            continue
        ends = sorted(point for point, ns in graph.items() if len(ns)<=1)
        closed = not ends
        current = ends[0] if ends else min(pixels)
        previous = None
        ordered = []
        seen = set()
        while current not in seen:
            ordered.append(current); seen.add(current)
            following = sorted(p for p in graph[current] if p != previous and p not in seen)
            if not following:
                break
            previous, current = current, following[0]
        if len(seen) != len(pixels):
            invalid += 1
            continue
        paths.append((np.asarray(ordered, np.int32), closed))
    paths.sort(key=lambda item:tuple(item[0][0]))
    return paths, invalid


def _runs(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded)==1)
    stops = np.flatnonzero(np.diff(padded)==-1)
    return zip(starts.tolist(), stops.tolist())


def _anchor(path, response, index, direction, closed, limit, strong):
    """沿同一 GT 路径找最近的强响应锚点，距离按欧氏步长累计。"""
    current, travelled = int(index), 0.
    for _ in range(len(path)-1):
        nxt = current+direction
        if not closed and not 0 <= nxt < len(path):
            break
        nxt %= len(path)
        travelled += float(np.linalg.norm(path[nxt].astype(float)-path[current]))
        if travelled > limit + 1e-9:
            break
        if response[nxt] >= strong:
            return dict(path_index=int(nxt), yx=path[nxt].tolist(),
                        distance_native=travelled, matched_response=float(response[nxt]))
        current = nxt
    return None


def _response_at_anchor(boundary, point, radius, threshold):
    y, x = point
    y0, y1 = max(0,y-radius), min(boundary.shape[0],y+radius+1)
    x0, x1 = max(0,x-radius), min(boundary.shape[1],x+radius+1)
    yy, xx = np.ogrid[y0:y1,x0:x1]
    eligible = (yy-y)**2+(xx-x)**2 <= radius*radius
    values = boundary[y0:y1,x0:x1]
    score = np.where(eligible, values, -1)
    iy, ix = np.unravel_index(np.argmax(score),score.shape)
    return [int(y0+iy),int(x0+ix)], float(values[iy,ix]), bool(values[iy,ix]>=threshold)


def _soft_segment(points, shape, config):
    radius = int(math.ceil(config['sigma']*config['truncate_sigma']))
    y0 = max(0,int(points[:,0].min())-radius)
    y1 = min(shape[0],int(points[:,0].max())+radius+1)
    x0 = max(0,int(points[:,1].min())-radius)
    x1 = min(shape[1],int(points[:,1].max())+radius+1)
    seed = np.zeros((y1-y0,x1-x0),np.uint8)
    seed[points[:,0]-y0,points[:,1]-x0] = 1
    # OpenCV 的 precise EDT 在不同线程调度下可有 float32 ULP 漂移。
    # 候选目标绑定精确 SHA，因此使用确定性的 float64 EDT 后统一转回 float32。
    distance = distance_transform_edt(seed==0)
    soft = config['peak']*np.exp(-.5*(distance/config['sigma'])**2)
    soft[distance>config['truncate_sigma']*config['sigma']] = 0
    return (slice(y0,y1),slice(x0,x1)),soft.astype(np.float32)


def _offset_profile(path, start, stop, heat, valid, config):
    """整条缺段逐点检查法向，仅拒绝偏移线，不把模型线搬到 GT 上。

    中点只是展示用切片；局部平行线可能仅出现在端部，因此拒绝规则必须
    覆盖所有候选点，不能以中点未见响应推断整段都没有偏移线。
    """
    segment = path[start:stop]
    distance = np.r_[0,np.cumsum(np.linalg.norm(np.diff(segment.astype(float),axis=0),axis=1))]
    middle_local = int(np.argmin(np.abs(distance-distance[-1]/2)))
    half = config['profile_tangent_half_points']
    tangents, normals = [],[]
    for index in range(start,stop):
        points = path[max(0,index-half):min(len(path),index+half+1)].astype(float)
        centered = points-points.mean(axis=0)
        _, vectors = np.linalg.eigh(centered.T@centered)
        tangent = vectors[:,-1]
        tangents.append(tangent)
        normals.append(np.array([-tangent[1],tangent[0]]))
    tangents,normals = np.asarray(tangents),np.asarray(normals)
    offsets = np.arange(-config['offset_reject_radius'],config['offset_reject_radius']+1,dtype=np.float32)
    locations = segment.astype(float)[:,None,:]+offsets[None,:,None]*normals[:,None,:]
    mapx,mapy = locations[:,:,1].astype(np.float32),locations[:,:,0].astype(np.float32)
    response = cv2.remap(heat,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    support = cv2.remap(valid.astype(np.float32),mapx,mapy,cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0)>=1
    outside = (np.abs(offsets)[None]>config['match_radius']) & support
    # 这里保护已有读出弱响应，不能复用更高的双端强锚阈值。
    shifted = outside & (response>=config['outside_ridge_threshold'])
    return dict(center_yx=segment[middle_local].tolist(),tangent_yx=tangents[middle_local].tolist(),
                normal_yx=normals[middle_local].tolist(),offsets_native=offsets.tolist(),
                response=response[middle_local].tolist(),valid_samples=support[middle_local].tolist(),
                centers_yx=segment.tolist(),tangents_yx=tangents.tolist(),normals_yx=normals.tolist(),
                all_responses=response.tolist(),all_valid_samples=support.tolist(),
                checked_point_count=len(segment),scope='every_candidate_skeleton_point',
                outside_ridge_threshold=config['outside_ridge_threshold'],
                midpoint_would_reject=bool(shifted[middle_local].any()),
                outer_peak=float(response[outside].max()) if outside.any() else None,
                inner_peak=float(response[(np.abs(offsets)[None]<=config['match_radius'])&support].max()),
                shifted_strong_offsets_native=offsets[shifted.any(axis=0)].tolist(),
                shifted_point_indices=np.flatnonzero(shifted.any(axis=1)).tolist(),
                rejected_as_potential_offset=bool(shifted.any()),
                interpretation='Conservative rejection only: existing offset/parallel response, not proof of GT error')


def build_boundary_correction(boundary, instances, original_known, *, unknown=None,
                              config=None, source=None):
    """生成一张原生热图的可靠 GT 短段纠错候选，所有返回数组均不共享输入内存。

    `boundary` 必须为原始 float32 N0 热图；`instances` 是处理后实例，
    `original_known` 必须来自其 original_covered，而不是 instances>0。
    support_policy 默认 original_strict；processed_thin_fill 是明确分离的
    支持域对照，不改变置信度、缺段长度、锚点或偏移拒绝阈值。
    source 仅透传来源标识；文件身份由调用方用 manifest 核验。函数额外绑定
    所有输入数组的 dtype/shape/内容 SHA，不能用同名编号推断源图对应关系。
    返回 accepted records 与逐原因 rejection 计数，不进行阈值搜索。
    """
    cfg = _configuration(config)
    heat, ids = np.asarray(boundary),np.asarray(instances)
    if (heat.ndim!=2 or heat.dtype!=np.float32 or not np.isfinite(heat).all()
            or np.any(heat<0) or np.any(heat>1)):
        raise ValueError('Original N0 boundary must be finite float32 HW in [0,1]')
    if ids.shape!=heat.shape or ids.dtype.kind not in 'iu' or np.any(ids<0):
        raise ValueError('Nonnegative integer native instance map must match N0')
    original = _binary(original_known,heat.shape,'original_known')
    if np.any(original & (ids==0)):
        raise ValueError('Original GT coverage cannot include unknown zero IDs')
    if unknown is not None and not np.array_equal(_binary(unknown,heat.shape,'unknown'),ids==0):
        raise ValueError('Unknown provenance must match processed zero IDs')
    # 拒绝非 JSON 来源，防止记录悄然丢失或嵌入大型对象。
    json.dumps(source,allow_nan=False)
    own_support,valid,full_known_safe,support_receipt = _support_domain(ids,original,cfg)
    target_support = valid & own_support
    # 交界两端都要独立得到原标/同实例窄填补支持，不能只筛骨架中心。
    truth = _known_boundary(np.where(own_support,ids,0))
    initial = skeletonize(truth & valid)
    pixels = set(map(tuple,np.argwhere(initial).tolist()))
    junction = np.zeros(heat.shape,np.uint8)
    for point in pixels:
        if len(_neighbours(point,pixels))>2:
            junction[point] = 1
    if cfg['junction_margin']:
        junction = cv2.dilate(junction,_disk(cfg['junction_margin']))
    skeleton = initial & ~junction.astype(bool)
    paths, invalid_paths = _paths(skeleton)
    # 薄填补模式在完整 processed-known 安全域观察 N0，包括没有正监督资格
    # 的 filled；否则可能漏掉偏移旧线。严格模式维持原来的观察域不变。
    supported_heat = np.where(valid,heat,0).astype(np.float32)
    matched = cv2.dilate(supported_heat,_disk(cfg['match_radius']))
    target = heat.copy()
    candidate_mask = np.zeros(heat.shape,bool)
    records, rejected = [],[]
    weak_run_count, before_offset_guard = 0,0
    anchor_widths = None
    for path_index,(path,closed) in enumerate(paths):
        response = matched[path[:,0],path[:,1]]
        low = response < cfg['low_threshold']
        # 环从非弱点开始，避免一个缺段被任意排序起点分成两段。
        if closed and (~low).any():
            shift = int(np.flatnonzero(~low)[0])
            path,response,low = (np.roll(a,-shift,axis=0) for a in (path,response,low))
        for start,stop in _runs(low):
            weak_run_count += 1
            points = path[start:stop]
            length = float(np.linalg.norm(np.diff(points.astype(float),axis=0),axis=1).sum())
            base = dict(path_index=path_index,closed_path=bool(closed),start_yx=points[0].tolist(),
                        end_yx=points[-1].tolist(),native_length=length,points=int(len(points)))
            if not cfg['min_gap_length'] <= length <= cfg['max_gap_length']:
                rejected.append({**base,'reason':'length_outside_range'})
                continue
            before = _anchor(path,response,start,-1,closed,cfg['anchor_search_length'],cfg['anchor_threshold'])
            after = _anchor(path,response,stop-1,1,closed,cfg['anchor_search_length'],cfg['anchor_threshold'])
            if before is None or after is None or before['path_index']==after['path_index']:
                rejected.append({**base,'reason':'missing_distinct_two_sided_strong_anchors'})
                continue
            before_offset_guard += 1
            profile = _offset_profile(path,start,stop,heat,valid,cfg)
            if cfg['offset_guard'] and profile['rejected_as_potential_offset']:
                rejected.append({**base,'reason':'potential_offset_or_parallel_response',
                                 'offset_guard_scope':'midpoint_also' if profile['midpoint_would_reject'] else 'off_midpoint_only',
                                 'before_anchor':before,'after_anchor':after,'normal_profile':profile})
                continue
            if anchor_widths is None:
                # 连续响应带在锚点处的宽度代理，既不是真实晶界宽也不用于补线尺度拟合。
                anchor_widths = {str(t):2*distance_transform_edt(heat>=t) for t in (.45,.65)}
            for anchor in (before,after):
                location,peak,passed = _response_at_anchor(supported_heat,anchor['yx'],
                                                         cfg['match_radius'],cfg['anchor_threshold'])
                assert passed
                anchor.update(response_yx=location,response_peak=peak)
                anchor['response_bandwidth_2edt_at_peak'] = {t:float(width[tuple(location)]) for t,width in anchor_widths.items()}
            candidate_mask[points[:,0],points[:,1]] = True
            box,soft = _soft_segment(points,heat.shape,cfg)
            allowed = target_support[box] & (heat[box] < cfg['anchor_threshold'])
            support = soft>0
            before_target = target[box].copy()
            target[box] = np.where(allowed,np.maximum(target[box],soft),target[box])
            changed = target[box] > before_target
            raw = heat[points[:,0],points[:,1]]
            records.append({**base,'before_anchor':before,'after_anchor':after,'normal_profile':profile,
                            'path_yx':points.tolist(),'raw_center_mean':float(raw.mean()),
                            'raw_center_max':float(raw.max()),'radius_match_mean':float(response[start:stop].mean()),
                            'radius_match_max':float(response[start:stop].max()),
                            'target_sigma_native':cfg['sigma'],'target_peak':cfg['peak'],
                            'new_correction_pixels':int(changed.sum()),
                            'new_positive_delta_sum':float((target[box]-before_target).sum(dtype=np.float64)),
                            'soft_support_pixels':int(support.sum()),
                            'soft_support_clipped_by_valid':int((support & ~valid[box]).sum()),
                            'soft_support_clip_fraction':float((support & ~valid[box]).sum()/support.sum()),
                            'soft_support_clipped_by_own_support':int((support & valid[box] & ~own_support[box]).sum()),
                            'soft_support_total_clip_fraction':float((support & ~target_support[box]).sum()/support.sum()),
                            'support_box_yxyx':[box[0].start,box[1].start,box[0].stop,box[1].stop]})
    correction = target > heat
    assert np.array_equal(target[~valid],heat[~valid])
    assert np.array_equal(target[~target_support],heat[~target_support])
    assert np.array_equal(target[heat>=cfg['anchor_threshold']],heat[heat>=cfg['anchor_threshold']])
    assert np.all(target>=heat)
    summary = dict(format='boundary_gt_positive_candidates_v1',source=source,config=cfg,
                   training=False,target_role='candidate_positive_correction_not_verified_truth',
                   original_only=cfg['support_policy']=='original_strict',
                   filled_as_positive_support=cfg['support_policy']=='processed_thin_fill',test_gt=False,
                   support=support_receipt,
                   source_identity='Caller must verify source manifest/file identity; array digests below bind this call',
                   native_shape=list(heat.shape),input_sha256=_array_sha(heat),instances_sha256=_array_sha(ids),
                   original_known_sha256=_array_sha(original),target_sha256=_array_sha(target),
                   original_known_pixels=int(original.sum()),valid_pixels=int(valid.sum()),
                   full_known_safe_pixels=int(full_known_safe.sum()),
                   eligible_support_pixels=int(own_support.sum()),target_support_pixels=int(target_support.sum()),
                   gt_boundary_pixels=int((truth&valid).sum()),initial_skeleton_pixels=int(initial.sum()),
                   supported_boundary_original_centers=int((truth&valid&original).sum()),
                   supported_boundary_filled_centers=int((truth&valid&~original).sum()),
                   excluded_junction_pixels=int((initial&junction.astype(bool)).sum()),
                   eligible_skeleton_pixels=int(skeleton.sum()),path_count=len(paths),invalid_path_count=invalid_paths,
                   weak_run_count=weak_run_count,accepted_count=len(records),
                   accepted_before_offset_guard=before_offset_guard,
                   offset_guard_rejected=sum(r['reason']=='potential_offset_or_parallel_response' for r in rejected),
                   offset_guard_additional_vs_midpoint=sum(r.get('offset_guard_scope')=='off_midpoint_only' for r in rejected),
                   rejection_counts=dict(Counter(r['reason'] for r in rejected)),
                   candidate_center_pixels=int(candidate_mask.sum()),correction_pixels=int(correction.sum()),
                   candidate_original_centers=int((candidate_mask&original).sum()),
                   candidate_filled_centers=int((candidate_mask&~original).sum()),
                   correction_original_pixels=int((correction&original).sum()),
                   correction_filled_pixels=int((correction&~original).sum()),
                   positive_delta_sum=float((target-heat).sum(dtype=np.float64)),
                   max_positive_delta=float((target-heat).max()),
                   unchanged_high_response=True,unchanged_outside_valid=True,only_nonnegative_correction=True,
                   graph_adjacency='8-neighbour without diagonal shortcuts when an axial neighbour exists',
                   distance_backend='scipy.ndimage.distance_transform_edt_float64',
                   length_definition='Euclidean arc distance between first and last missing skeleton centers',
                   target_width_note='sigma is native Gaussian scale, not a measured physical grain-boundary width')
    return dict(input=heat.copy(),target=target,valid=valid,correction_mask=correction,
                own_support=own_support,eligible_support=own_support,target_support=target_support,
                full_known_safe=full_known_safe,
                candidate_mask=candidate_mask,skeleton=skeleton,matched_response=matched,
                records=records,rejections=rejected,summary=summary)
