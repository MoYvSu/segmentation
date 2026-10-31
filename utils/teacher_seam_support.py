# -*- coding: utf-8 -*-
"""保守恢复已确认教师实例的窄零缝监督，不修改教师或人工GT文件。

恢复只利用教师几何：原人工GT仅提供known域和事后关系统计，不提供修补ID。
这不是纠正教师已经合并的实例，也不宣称新增监督是物理真值。
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
import torch

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.affinity_loss import build_affinity_targets_torch
from utils.affinity_teacher_targets import (_binary, _ids, _relations, _sha,
                                            build_teacher_pair)
from utils.offset_letterbox import letterbox_instance_geometry


def _disk(radius):
    extent = int(math.ceil(radius))
    axis = np.arange(-extent, extent + 1)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    return yy * yy + xx * xx <= radius * radius + 1.e-12


def repair_teacher_zero_seams(manual_ids, teacher_ids, original_covered, *,
                              source_name, allowed_training_sources,
                              max_distance_native=1.0, context_radius_native=None):
    """返回 ``pair, audit``；原生像素欧氏距离内、两实例之间的窄零缝补ID。

    三项门禁：距known内已有教师正ID不超过max_distance；context半径内
    存在至少两个不同教师ID；不靠近宽零洞核心（nearest距离超过上限）。
    宽零洞核心再膨胀max_distance，防止只给宽洞填一圈边。context默认
    两倍max_distance。最近源等距时按(y,x)字典序，而不是ID大小，打破平局。
    这是局部欧氏几何门禁，不添加整条原生连线known门禁或人工类别规则。
    """
    maximum = float(max_distance_native)
    context = 2.0 * maximum if context_radius_native is None else float(context_radius_native)
    if (not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(context)
            or context < maximum):
        raise ValueError('Finite positive max_distance and context>=max_distance required')
    old, old_audit = build_teacher_pair(
        manual_ids, teacher_ids, original_covered, source_name=source_name,
        allowed_training_sources=allowed_training_sources)
    manual = old['manual_original']
    teacher = _ids(teacher_ids, 'teacher', teacher=True)
    known = manual > 0
    seeds = old['common_known']
    zeros = known & (teacher == 0)
    repaired = old['teacher_masked'].copy()
    assigned = np.zeros(manual.shape, bool)
    coordinates = np.empty((0, 2), np.int32)
    sources = np.empty((0, 2), np.int32)
    assigned_distance = np.empty(0, np.float32)
    nearby = np.zeros(manual.shape, bool)
    two_ids = np.zeros(manual.shape, bool)
    wide_neighborhood = np.zeros(manual.shape, bool)
    if zeros.any() and seeds.any():
        distance = ndimage.distance_transform_edt(~seeds)
        nearby = zeros & (distance <= maximum + 1.e-12)
        footprint = _disk(context)
        seed_ids = np.where(seeds, teacher, 0).astype(np.uint32)
        low = ndimage.minimum_filter(np.where(seeds, teacher.astype(np.uint32), np.uint32(65536)),
                                     footprint=footprint, mode='constant', cval=65536)
        high = ndimage.maximum_filter(seed_ids, footprint=footprint, mode='constant', cval=0)
        two_ids = zeros & (low < high)
        wide_core = zeros & (distance > maximum + 1.e-12)
        wide_neighborhood = ndimage.binary_dilation(wide_core, structure=_disk(maximum)) & zeros
        assigned = nearby & two_ids & ~wide_neighborhood
        ys, xs = np.nonzero(assigned)
        best_d2 = np.full(ys.size, np.inf)
        best_flat = np.full(ys.size, np.iinfo(np.int64).max, np.int64)
        extent = int(math.ceil(maximum))
        height, width = teacher.shape
        for dy in range(-extent, extent + 1):
            for dx in range(-extent, extent + 1):
                d2 = dy * dy + dx * dx
                if d2 > maximum * maximum + 1.e-12:
                    continue
                sy, sx = ys + dy, xs + dx
                in_image = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
                positions = np.flatnonzero(in_image)
                if not positions.size:
                    continue
                positions = positions[seeds[sy[positions], sx[positions]]]
                flat = sy[positions].astype(np.int64) * width + sx[positions]
                take = (d2 < best_d2[positions]) | ((d2 == best_d2[positions])
                                                        & (flat < best_flat[positions]))
                positions, flat = positions[take], flat[take]
                best_d2[positions], best_flat[positions] = d2, flat
        if np.any(~np.isfinite(best_d2)):
            raise AssertionError('An eligible seam has no known positive source within radius')
        sy, sx = best_flat // width, best_flat % width
        repaired[ys, xs] = teacher[sy, sx]
        coordinates = np.column_stack((ys, xs)).astype(np.int32)
        sources = np.column_stack((sy, sx)).astype(np.int32)
        assigned_distance = np.sqrt(best_d2).astype(np.float32)
    common_new = repaired > 0
    if (np.any(assigned & ~known) or np.any(repaired[~known])
            or not np.array_equal(repaired[seeds], teacher[seeds])
            or not np.array_equal(common_new, seeds | assigned)
            or not np.array_equal(np.unique(repaired[common_new]), np.unique(teacher[seeds]))):
        raise AssertionError('Seam repair changed known positive IDs, unknown, or instance set')
    pair = dict(old, teacher_original=teacher.copy(),
                old_manual_masked=old['manual_masked'].copy(),
                old_teacher_masked=old['teacher_masked'].copy(),
                old_common_known=seeds.copy(), common_known=common_new,
                teacher_repaired=repaired.copy(), teacher_masked=repaired,
                manual_masked=np.where(common_new, manual, 0).astype(np.uint16),
                assigned_mask=assigned, assigned_distance_native=assigned_distance,
                assignment_coordinates_native=coordinates, nearest_sources_native=sources)
    audit = dict(format='teacher_seam_support_v1', source_name=old['source_name'],
                 allowlist_sha256=old['allowlist_sha256'], native_shape=list(manual.shape),
                 parameters=dict(max_distance_native=maximum, context_radius_native=context,
                                 wide_core_exclusion_radius_native=maximum),
                 pixels=dict(processed_known=int(known.sum()), old_common=int(seeds.sum()),
                             teacher_zero_known=int(zeros.sum()), within_distance=int(nearby.sum()),
                             nearby_two_teacher_ids=int((nearby & two_ids).sum()),
                             rejected_wide_void_neighbor=int((nearby & two_ids & wide_neighborhood).sum()),
                             assigned=int(assigned.sum()), assigned_original=int((assigned & old['original_covered']).sum()),
                             assigned_filled=int((assigned & old['filled']).sum()),
                             remaining_teacher_zero_known=int((known & ~common_new).sum()),
                             new_common=int(common_new.sum())),
                 hashes={key: _sha(pair[key]) for key in
                         ('manual_original', 'teacher_original', 'teacher_repaired', 'assigned_mask',
                          'assigned_distance_native', 'assignment_coordinates_native', 'nearest_sources_native')},
                 original_pair_audit=old_audit, existing_positive_ids_preserved=True,
                 known_instance_id_set_preserved=True, unknown_as_ignore=True,
                 nearest_tie_rule='minimum squared Euclidean distance, then source (y,x) lexicographic',
                 manual_instance_answers_used_for_assignment=False, semantic_used_for_assignment=False,
                 source_files_modified=False,
                 caveat='Only accepted local narrow zeros gain supervision; teacher merges are not repaired; additions are not asserted physically correct')
    return pair, audit


def _endpoint_mask(nodes, legal):
    selected = torch.zeros_like(legal)
    height, width = nodes.shape
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        if height <= abs(dy) or width <= abs(dx):
            continue
        source, dest = _edge_slices(height, width, dy, dx)
        selected[0, channel][source] = legal[0, channel][source] & torch.from_numpy(nodes[source] & nodes[dest])
    return selected


def resize_seam_pair(pair, *, input_size=1024, output_grid=512, semantic=None):
    """旧/新增端点监督统计；nearest等比letterbox保持现有512网格合同。

    semantic是可选人工类别审计(-1未知/0珠光体/1铁素体)，绝不参加补ID。
    新边按原人工同/异关系、original-original/涉及filled和可选FF/FP/PP分层。
    """
    manual = _ids(pair['manual_original'], 'manual_original')
    names = ('old_manual_masked', 'old_teacher_masked', 'manual_masked', 'teacher_masked')
    maps = {name: _ids(pair[name], name, teacher='teacher' in name) for name in names}
    if any(value.shape != manual.shape for value in maps.values()):
        raise ValueError('All native maps must have matching shape')
    old_known = _binary(pair['old_common_known'], manual.shape, 'old_common_known')
    known = _binary(pair['common_known'], manual.shape, 'common_known')
    assigned = _binary(pair['assigned_mask'], manual.shape, 'assigned_mask')
    original = _binary(pair['original_covered'], manual.shape, 'original_covered')
    filled = _binary(pair['filled'], manual.shape, 'filled')
    if (not np.array_equal(known, old_known | assigned) or np.any(old_known & assigned)
            or np.any(known & (manual == 0)) or not np.array_equal(original | filled, manual > 0)
            or np.any(original & filled)):
        raise ValueError('Native seam support/provenance inconsistent')
    for name, values in maps.items():
        expected_known = old_known if name.startswith('old_') else known
        if not np.array_equal(values > 0, expected_known):
            raise ValueError('Paired maps must share old/new support respectively')
    if (not np.array_equal(maps['old_teacher_masked'][old_known], maps['teacher_masked'][old_known])
            or not np.array_equal(maps['old_manual_masked'][old_known], manual[old_known])
            or not np.array_equal(maps['manual_masked'][known], manual[known])):
        raise ValueError('Original labels changed on existing support')
    grids, targets, legal_masks = {}, {}, {}
    for name, values in dict(maps, manual_original=manual).items():
        grid, valid, meta = letterbox_instance_geometry(values, input_size, output_grid)
        grids[name] = grid
        targets[name], legal_masks[name] = build_affinity_targets_torch(
            torch.from_numpy(grid.astype(np.int64))[None], torch.from_numpy(valid)[None, None])
    oldlegal, newlegal = legal_masks['old_teacher_masked'], legal_masks['teacher_masked']
    if (not torch.equal(oldlegal, legal_masks['old_manual_masked'])
            or not torch.equal(newlegal, legal_masks['manual_masked'])
            or torch.any(oldlegal & ~newlegal)
            or not torch.equal(targets['old_teacher_masked'][oldlegal], targets['teacher_masked'][oldlegal])
            or torch.any(newlegal & ~legal_masks['manual_original'])):
        raise AssertionError('Old legal answers changed, old support lost, or unknown supervision added')
    added = newlegal & ~oldlegal
    original_grid, _, _ = letterbox_instance_geometry(original.astype(np.uint8), input_size, output_grid)
    original_edges = _endpoint_mask(original_grid > 0, added)
    strata = dict(all_added=_relations(targets['manual_masked'], targets['teacher_masked'], added),
                  original_original_added=_relations(targets['manual_masked'], targets['teacher_masked'], original_edges),
                  involving_filled_added=_relations(targets['manual_masked'], targets['teacher_masked'], added & ~original_edges))
    if semantic is not None:
        sem = np.asarray(semantic)
        if (sem.shape != manual.shape or not np.issubdtype(sem.dtype, np.integer)
                or np.any(~np.isin(sem, [-1, 0, 1]))):
            raise ValueError('semantic audit must be matching HW integers -1/0/1')
        semgrid, _, _ = letterbox_instance_geometry(sem, input_size, output_grid)
        ff = _endpoint_mask(semgrid == 1, added)
        pp = _endpoint_mask(semgrid == 0, added)
        classified = _endpoint_mask(semgrid >= 0, added)
        fp = classified & ~ff & ~pp
        for name, mask in (('FF_added', ff), ('FP_added', fp), ('PP_added', pp),
                           ('class_unknown_added', added & ~classified)):
            strata[name] = _relations(targets['manual_masked'], targets['teacher_masked'], mask)
    stats = dict(format='teacher_seam_relations_v1', source_name=pair['source_name'],
                 resize=meta.to_dict(), old_legal_edges=int(oldlegal.sum()),
                 new_legal_edges=int(newlegal.sum()), added_legal_edges=int(added.sum()),
                 full_manual_legal_edges=int(legal_masks['manual_original'].sum()),
                 old_legal_subset_verified=True, old_teacher_answers_unchanged=True,
                 masks_sha256=dict(old=_sha(oldlegal), new=_sha(newlegal), added=_sha(added)),
                 strata=strata, semantic_used_only_for_audit=semantic is not None,
                 unknown_padding_ignored=True, uncovered_as_boundary=False,
                 rule='existing eight offsets, nearest letterbox, both endpoints positive; added edges need not be physically correct')
    geometry = dict(manual_grid=grids['manual_masked'], teacher_grid=grids['teacher_masked'],
                    old_teacher_grid=grids['old_teacher_masked'], manual_targets=targets['manual_masked'],
                    teacher_targets=targets['teacher_masked'], edge_valid=newlegal,
                    old_edge_valid=oldlegal, added_edge_valid=added,
                    valid_content=valid, metadata=meta.to_dict())
    return geometry, stats
