# -*- coding: utf-8 -*-
"""连续边界的空间诊断；纯数组函数，不读GT文件、不训练、不改部署。

调用方负责核验本轮新GT及坐标身份。这里沿用四个短程关系的锚点定义，
不将空白标注补成边界。gap大小是连通组件的像素数，不是晶界的几何弧长。
内部高响应只列候选；闭环或多处接触不等于实际分水岭一定会误切。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, build_affinity_targets

FORMAT = 'affinity_spatial_v1'
HIGH_THRESHOLD = .65
CATEGORIES = ('isolated', 'branch_or_line', 'closed_loop_candidate',
              'multi_contact_candidate')
SHORT_OFFSETS = DEFAULT_AFFINITY_OFFSETS[:4]
EIGHT = np.ones((3, 3), dtype=bool)


def _binary(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must be a binary array with shape {shape}')
    if not np.isfinite(result).all() or not np.isin(result, (0, 1)).all():
        raise ValueError(f'{name} must be binary, not confidence weights')
    return result.astype(bool, copy=False)


def _inputs(boundary, instance_map, trusted, valid_content):
    scores = np.asarray(boundary)
    ids = np.asarray(instance_map)
    if scores.ndim != 2 or min(scores.shape, default=0) < 1:
        raise ValueError('boundary must be a nonempty two-dimensional array')
    if scores.dtype.kind not in 'fiu' or not np.isfinite(scores).all():
        raise ValueError('boundary must contain finite probabilities')
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('boundary probabilities must be in [0, 1]')
    if ids.shape != scores.shape or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('instance_map must contain nonnegative integer IDs with the boundary shape')
    trust = _binary(trusted, scores.shape, 'trusted')
    content = (np.ones(scores.shape, bool) if valid_content is None else
               _binary(valid_content, scores.shape, 'valid_content'))
    known = trust & content & (ids > 0)
    return scores, ids, known


def _slices(shape, dy, dx):
    h, w = shape
    sy, sx = slice(max(0, -dy), min(h, h-dy)), slice(max(0, -dx), min(w, w-dx))
    ty, tx = slice(max(0, -dy)+dy, min(h, h-dy)+dy), slice(max(0, -dx)+dx, min(w, w-dx)+dx)
    return (sy, sx), (ty, tx)


def _contacts(ids, known):
    """共享旧短程目标；按GT对归属锚点，同时保留对称的可信接触端点。"""
    target, edge_valid = build_affinity_targets(ids, known, offsets=SHORT_OFFSETS)
    eligible = edge_valid.all(axis=0)
    interface = (~target.astype(bool)).any(axis=0) & eligible
    symmetric = np.zeros(ids.shape, bool)
    entries = defaultdict(list)
    h, w = ids.shape
    for channel, (dy, dx) in enumerate(SHORT_OFFSETS):
        source, destination = _slices(ids.shape, dy, dx)
        cross = edge_valid[channel][source] & ~target[channel][source].astype(bool)
        symmetric[source] |= cross
        symmetric[destination] |= cross
        chosen = cross & eligible[source]
        if not chosen.any():
            continue
        yy, xx = np.nonzero(chosen)
        yy += source[0].start
        xx += source[1].start
        a, b = ids[yy, xx], ids[yy+dy, xx+dx]
        pairs = np.stack((np.minimum(a, b), np.maximum(a, b)), axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        flat = yy*w + xx
        for index, pair in enumerate(unique):
            entries[tuple(map(int, pair))].append(flat[inverse == index])
    masks = {}
    for pair in sorted(entries):
        mask = np.zeros(h*w, bool)
        mask[np.concatenate(entries[pair])] = True
        masks[pair] = mask.reshape(h, w)
    return interface, eligible, symmetric, masks


def _box(mask, origin=(0, 0)):
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    return [int(yy.min()+origin[0]), int(xx.min()+origin[1]),
            int(yy.max()+1+origin[0]), int(xx.max()+1+origin[1])]


def _frame(mask, origin, shape):
    box = _box(mask, origin)
    return bool(box and (box[0] == 0 or box[1] == 0 or
                         box[2] == shape[0] or box[3] == shape[1]))


def _topology(component, background_structure):
    """前景4/8连通与背景8/4配对；空洞不以骨架图的邻接环代替。"""
    padded = np.pad(component, 1)
    holes = ndi.binary_fill_holes(padded, structure=background_structure)[1:-1, 1:-1] & ~component
    _, hole_count = ndi.label(holes, structure=background_structure)
    # 骨架形态只作描述，统一8邻域；不声称其节点数是几何长度。
    skeleton = skeletonize(padded)[1:-1, 1:-1]
    degree = ndi.convolve(skeleton.astype(np.int16), EIGHT.astype(np.int16), mode='constant') - skeleton
    return holes, dict(hole_count=int(hole_count), hole_pixels=int(holes.sum()),
        skeleton_pixels=int(skeleton.sum()), skeleton_endpoint_pixels=int((skeleton & (degree == 1)).sum()),
        skeleton_junction_pixels=int((skeleton & (degree >= 3)).sum()))


def spatial_boundary_diagnostic(boundary, instance_map, trusted, *, valid_content=None,
                                threshold=HIGH_THRESHOLD, connectivity=8, deep_radius=4,
                                return_maps=False):
    """在同一网格上诊断真实输出的接触缺口和实例内部高响应。

    接触界面使用四短距均可信的锚点，任一跨GT关系即为界面，按无序GT对
    分组。一个交汇锚点可属于多个GT对，因此逐对像素和不能当唯一像素数。
    gap是界面上低于threshold的连通组件；触unknown/图框者单列censored。

    内部组件先移除对称的已知GT接触端点，再在每个GT完整可信区域追踪，
    只分析含deep_radius次3x3腐蚀内部像素的高响应组件，不先截成深内部。
    全组件或其封闭空洞触unknown/图框，或空洞含另一GT者不判内部假边界。
    多处接触指组件的一格邻域内有至少两个不相连的GT接触patch，不是两像素。
    类别和空洞均为候选结构，未运行marker/分水岭，不能断言实际误切。
    """
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError('connectivity must be 4 or 8')
    if type(deep_radius) is not int or deep_radius < 1:
        raise ValueError('deep_radius must be a positive integer in supplied grid pixels')
    if not isinstance(threshold, (float, int)) or isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('threshold must be finite and in (0, 1]')
    if type(return_maps) is not bool:
        raise ValueError('return_maps must be boolean')
    scores, ids, known = _inputs(boundary, instance_map, trusted, valid_content)
    structure = ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    background_connectivity = 8 if connectivity == 4 else 4
    background_structure = ndi.generate_binary_structure(2, 2 if connectivity == 4 else 1)
    interface, eligible, symmetric, pair_masks = _contacts(ids, known)
    # all-four-valid本身会裁去unknown旁的一层锚点；再看其邻域，避免把被裁断
    # 的gap剩余段当成完整可信组件。图框也视为未观测，不能延伸出图继续计量。
    unsafe = ~ndi.binary_erosion(known & eligible, structure=EIGHT, border_value=0)
    high = scores >= float(threshold)
    maps = {name: np.zeros(ids.shape, bool) for name in
            ('gap', 'accepted_gap', 'censored_gap', 'deep_interior',
             'internal_high_response', 'accepted_internal_high_response', 'censored_internal_high_response')}
    interface_rows, gap_rows = [], []
    for pair, mask in pair_masks.items():
        contact_labels, contact_count = ndi.label(mask, structure=structure)
        gaps, count = ndi.label(mask & ~high, structure=structure)
        lengths = []
        for label, section in enumerate(ndi.find_objects(gaps), start=1):
            local = gaps[section] == label
            pixels = int(local.sum())
            censored = bool(unsafe[section][local].any())
            contact_ids = np.unique(contact_labels[section][local])
            if len(contact_ids) != 1 or int(contact_ids[0]) == 0:
                raise AssertionError('A gap escapes its declared contact component')
            row = dict(gt_pair=list(pair), gap_component=label, contact_component=int(contact_ids[0]),
                component_pixels=pixels, bbox_yxyx=_box(local, (section[0].start, section[1].start)),
                censored_unknown_or_frame=censored)
            gap_rows.append(row)
            maps['gap'][section] |= local
            maps['censored_gap' if censored else 'accepted_gap'][section] |= local
            if not censored:
                lengths.append(pixels)
        interface_rows.append(dict(gt_pair=list(pair), interface_anchor_pixels=int(mask.sum()),
            contact_components=int(contact_count), low_response_anchor_pixels=int((mask & ~high).sum()),
            gap_components=int(count), accepted_gap_components=len(lengths),
            longest_accepted_gap_component_pixels=max(lengths, default=0)))

    instance_rows, internal_rows = [], []
    gt_ids, inverse = np.unique(ids[known], return_inverse=True)
    compact = np.zeros(ids.shape, np.int32)
    compact[known] = inverse + 1
    for index, section in enumerate(ndi.find_objects(compact), start=1):
        gt_id = int(gt_ids[index-1])
        gt_region = compact[section] == index
        # bbox外必不属于当前GT，腐蚀在bbox局部计算不会改变定义。
        deep = ndi.binary_erosion(gt_region, structure=EIGHT, iterations=deep_radius, border_value=0)
        maps['deep_interior'][section] |= deep
        candidates = high[section] & gt_region & ~symmetric[section]
        components, count = ndi.label(candidates, structure=structure)
        considered, accepted, censored_count = 0, 0, 0
        origin = (section[0].start, section[1].start)
        for component_id, local_section in enumerate(ndi.find_objects(components), start=1):
            local = components[local_section] == component_id
            if not (deep[local_section] & local).any():
                continue
            considered += 1
            # 一格扩展供接触和unknown检查；从原全图取，不能让GT bbox制造假边。
            y0 = origin[0]+local_section[0].start
            x0 = origin[1]+local_section[1].start
            y1, x1 = y0+local.shape[0], x0+local.shape[1]
            padded_box = (slice(max(0, y0-1), min(ids.shape[0], y1+1)),
                          slice(max(0, x0-1), min(ids.shape[1], x1+1)))
            component = np.zeros(known[padded_box].shape, bool)
            cy, cx = y0-padded_box[0].start, x0-padded_box[1].start
            component[cy:cy+local.shape[0], cx:cx+local.shape[1]] = local
            neighborhood = ndi.binary_dilation(component, structure=EIGHT)
            unknown_touch = bool((neighborhood & ~known[padded_box]).any())
            frame_touch = _frame(component, (padded_box[0].start, padded_box[1].start), ids.shape)
            holes, morphology = _topology(component, background_structure)
            enclosed_unknown = bool((holes & ~known[padded_box]).any())
            enclosed_other_gt = bool((holes & known[padded_box] & (ids[padded_box] != gt_id)).any())
            is_censored = unknown_touch or frame_touch or enclosed_unknown or enclosed_other_gt
            contact = neighborhood & symmetric[padded_box] & known[padded_box] & (ids[padded_box] == gt_id)
            contact_labels, patches = ndi.label(contact, structure=structure)
            patch_sizes = np.bincount(contact_labels.ravel(), minlength=int(patches)+1)[1:]
            multi_contact = int(patches) >= 2
            if morphology['hole_count']:
                category = 'closed_loop_candidate'
            elif multi_contact:
                category = 'multi_contact_candidate'
            elif morphology['skeleton_pixels'] <= 1:
                category = 'isolated'
            else:
                category = 'branch_or_line'
            reasons = [name for name, active in
                [('touches_unknown', unknown_touch), ('touches_frame', frame_touch),
                 ('encloses_unknown', enclosed_unknown), ('encloses_other_gt', enclosed_other_gt)] if active]
            row = dict(gt_instance=gt_id, component=component_id, component_pixels=int(local.sum()),
                deep_pixels=int((deep[local_section] & local).sum()), bbox_yxyx=[y0, x0, y1, x1],
                category=category, contact_patches=int(patches), contact_patch_pixels=patch_sizes.tolist(),
                multi_contact_candidate=multi_contact,
                censored=is_censored, censor_reasons=reasons, **morphology)
            internal_rows.append(row)
            maps['internal_high_response'][padded_box] |= component
            maps['censored_internal_high_response' if is_censored else 'accepted_internal_high_response'][padded_box] |= component
            accepted += int(not is_censored)
            censored_count += int(is_censored)
        instance_rows.append(dict(gt_instance=gt_id, trustworthy_pixels=int(gt_region.sum()),
            deep_pixels=int(deep.sum()), deep_high_response_pixels=int((deep & high[section]).sum()),
            considered_components=considered, accepted_components=accepted, censored_components=censored_count,
            nondeep_high_components=int(count-considered)))

    accepted_gaps = [row for row in gap_rows if not row['censored_unknown_or_frame']]
    accepted_internal = [row for row in internal_rows if not row['censored']]
    report = dict(format=FORMAT, settings=dict(threshold=float(threshold),
        component_connectivity=connectivity, hole_background_connectivity=background_connectivity,
        unknown_adjacency_connectivity=8, skeleton_description_connectivity=8, deep_radius=deep_radius,
        units='supplied_boundary_grid_pixels', gap_size='connected_component_pixel_count_not_arc_length',
        interface='any_cross_GT_of_four_short_offsets_all_four_valid',
        internal_component='high_response_per_GT_without_symmetric_true_contact_endpoints_before_deep_filter'),
        domain=dict(shape=list(ids.shape), known_pixels=int(known.sum()), ignored_pixels=int((~known).sum()),
            gt_instances=len(gt_ids), gt_instances_with_deep_interior=sum(row['deep_pixels'] > 0 for row in instance_rows),
            short_target_eligible_pixels=int(eligible.sum()),
            unique_interface_anchor_pixels=int(interface.sum()), deep_interior_pixels=int(maps['deep_interior'].sum())),
        interfaces=dict(gt_pairs=len(pair_masks), contact_components=sum(row['contact_components'] for row in interface_rows),
            unique_low_response_anchor_pixels=int((interface & ~high).sum()),
            gap_components=len(gap_rows), accepted_gap_components=len(accepted_gaps),
            censored_gap_components=len(gap_rows)-len(accepted_gaps),
            accepted_gap_pixels_pair_attributed=sum(row['component_pixels'] for row in accepted_gaps),
            accepted_gap_pixels_unique=int(maps['accepted_gap'].sum()),
            longest_accepted_gap_component_pixels=max((row['component_pixels'] for row in accepted_gaps), default=0)),
        internal=dict(considered_components=len(internal_rows), accepted_components=len(accepted_internal),
            censored_components=len(internal_rows)-len(accepted_internal),
            considered_response_pixels=int(maps['internal_high_response'].sum()),
            accepted_response_pixels=int(maps['accepted_internal_high_response'].sum()),
            accepted_deep_response_pixels=sum(row['deep_pixels'] for row in accepted_internal),
            categories={name: sum(row['category'] == name for row in accepted_internal) for name in CATEGORIES},
            multi_contact_candidates=sum(row['multi_contact_candidate'] for row in accepted_internal)),
        interface_rows=interface_rows, gap_rows=gap_rows, instance_rows=instance_rows, internal_rows=internal_rows,
        caveats=['Only caller-supplied audited GT/trust is used; no data or source identities are loaded here.',
            'Threshold gaps are response components, not boundary arc lengths or proven failed separations.',
            'GT-pair attribution may count a junction anchor in more than one pair.',
            'Unknown/frame-contacting components are censored, not diagnosed as false boundaries.',
            'Deep erosion excludes small/thin GT regions; zero considered components is not zero false responses.',
            'Loop/multi-contact categories do not prove actual marker splitting or watershed errors.',
            'These diagnostic labels are not official accuracy or independent generalization evidence.'])
    result = dict(report=report)
    if return_maps:
        result['maps'] = dict(**maps, true_interface=interface, short_target_eligible=eligible,
                              symmetric_true_contact=symmetric, unsafe_gap_anchor=unsafe, known=known)
    return result


def compare_spatial_boundaries(boundaries: Mapping[str, np.ndarray], instance_map, trusted, *,
                               reference='gated', valid_content=None, threshold=HIGH_THRESHOLD,
                               connectivity=8, deep_radius=4, return_maps=False):
    """同一GT口径比较已计算的连续图；不再融合logits或调用模型。"""
    if not isinstance(boundaries, Mapping) or len(boundaries) < 2:
        raise ValueError('boundaries must contain at least two named continuous maps')
    if any(not isinstance(name, str) or not name.strip() for name in boundaries) or reference not in boundaries:
        raise ValueError('A named reference map must be present')
    arms = {name: spatial_boundary_diagnostic(value, instance_map, trusted, valid_content=valid_content,
            threshold=threshold, connectivity=connectivity, deep_radius=deep_radius, return_maps=return_maps)
            for name, value in boundaries.items()}
    paths = [('interfaces', 'accepted_gap_components'), ('interfaces', 'longest_accepted_gap_component_pixels'),
             ('interfaces', 'accepted_gap_pixels_unique'), ('internal', 'accepted_components'),
             ('internal', 'accepted_deep_response_pixels'), ('internal', 'multi_contact_candidates')]
    before = arms[reference]['report']
    deltas = {name: {f'{group}.{key}': arm['report'][group][key]-before[group][key]
                    for group, key in paths} for name, arm in arms.items() if name != reference}
    result = dict(report=dict(format=FORMAT, reference=reference, settings=before['settings'],
        domain=before['domain'], arms={name: value['report'] for name, value in arms.items()},
        changes_relative_to_reference=deltas, interpretation='paired_output_structure_not_accuracy'))
    if return_maps:
        result['maps'] = {name: value['maps'] for name, value in arms.items()}
    return result


def self_test():
    """完全已知的CPU合成环和缺口；不访问赛题数据。"""
    ids = np.ones((31, 31), np.int64)
    scores = np.zeros(ids.shape, np.float32)
    scores[9:22, 9] = scores[9:22, 21] = .9
    scores[9, 9:22] = scores[21, 9:22] = .9
    result = spatial_boundary_diagnostic(scores, ids, np.ones(ids.shape, bool))['report']
    assert result['internal']['categories']['closed_loop_candidate'] == 1
    assert result['internal']['accepted_components'] == 1
    trust = np.ones(ids.shape, bool)
    trust[15, 15] = False
    censored = spatial_boundary_diagnostic(scores, ids, trust)['report']
    assert censored['internal']['accepted_components'] == 0
    assert censored['internal']['censored_components'] == 1
    json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    self_test()
    print(json.dumps(dict(status='passed', inputs='CPU_synthetic_only', format=FORMAT)))
