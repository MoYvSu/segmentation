# -*- coding: utf-8 -*-
"""同32训练源的processed GT／固定教师几何配对；不改旧GT或推断类别。

共同支持为processed known与teacher正ID交集，两臂都ignore教师零缝。
编号不同不代表预测不同；只比较实际八通道同/异实例关系。
本模块不授予教师人工确认、不核模型身份，调用者必须封存源/权重回执。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.affinity_loss import build_affinity_targets_torch
from utils.offset_letterbox import letterbox_instance_geometry


def _source_stem(value):
    if not isinstance(value, (str, Path)):
        raise ValueError('Training source name must be a filename/stem')
    text = str(value)
    if Path(text).name != text or '/' in text or '\\' in text:
        raise ValueError('Source must be a bare training filename/stem')
    stem = Path(text).stem
    if re.fullmatch(r'train_[0-9]+', stem) is None:
        raise ValueError('Only train_ sources are permitted; test sources forbidden')
    return stem


def _source_gate(source_name, allowed_training_sources):
    if not isinstance(allowed_training_sources, (list, tuple, set, frozenset)):
        raise ValueError('Explicit fixed 32-source allowlist required')
    allowed = [_source_stem(value) for value in allowed_training_sources]
    if len(allowed) != 32 or len(set(allowed)) != 32:
        raise ValueError('Exactly 32 unique approved training source names required')
    source = _source_stem(source_name)
    if source not in allowed:
        raise ValueError('Source outside approved 32 training source allowlist')
    digest = hashlib.sha256(('\n'.join(sorted(allowed))+'\n').encode('utf8')).hexdigest()
    return source, digest


def _ids(value, name, *, teacher=False):
    value = np.asarray(value)
    if (value.ndim != 2 or not value.size or not np.issubdtype(value.dtype, np.integer)
            or value.dtype == np.bool_ or np.any(value < 0) or int(value.max()) > 65535):
        raise ValueError(name+' must be nonempty integer HW IDs in 0..65535')
    if teacher and value.dtype != np.uint16:
        raise ValueError('teacher must retain single-channel uint16 PNG IDs')
    return value


def _binary(value, shape, name):
    value = np.asarray(value)
    if (value.shape != shape or not (value.dtype == np.bool_ or np.issubdtype(value.dtype, np.integer))
            or np.any((value != 0) & (value != 1))):
        raise ValueError(name+' must be matching binary HW provenance')
    return value.astype(bool, copy=True)


def _sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    digest = hashlib.sha256(str(array.dtype).encode('ascii')+str(array.shape).encode('ascii'))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_teacher_pair(manual_ids, teacher_ids, original_covered, *, source_name,
                       allowed_training_sources, filled=None, residual_unknown=None):
    """返回 ``pair, audit``：numpy原生图配对与可JSON写入的来源/覆盖统计。

    manual_original只读副本；manual_masked、teacher_masked各自保留原编号，
    仅将共同支持外置0。类别不参加该配对，不从教师JSON提供语义标签。
    """
    source, allowlist_sha = _source_gate(source_name, allowed_training_sources)
    manual, teacher = _ids(manual_ids, 'manual'), _ids(teacher_ids, 'teacher', teacher=True)
    if teacher.shape != manual.shape:
        raise ValueError('Manual and teacher must share exact native HW shape')
    original = _binary(original_covered, manual.shape, 'original_covered')
    known = manual > 0
    if np.any(original & ~known):
        raise ValueError('original_covered cannot contain processed GT unknown')
    expected_filled = known & ~original
    expected_unknown = ~known
    if filled is not None and not np.array_equal(_binary(filled, manual.shape, 'filled'), expected_filled):
        raise ValueError('filled differs from processed-known outside original_covered')
    if residual_unknown is not None and not np.array_equal(
            _binary(residual_unknown, manual.shape, 'residual_unknown'), expected_unknown):
        raise ValueError('residual_unknown differs from manual==0')
    common = known & (teacher > 0)
    masked_manual = np.where(common, manual, 0).astype(np.uint16)
    masked_teacher = np.where(common, teacher, 0).astype(np.uint16)
    pair = dict(source_name=source, allowlist_sha256=allowlist_sha,
                manual_original=manual.astype(np.uint16, copy=True),
                manual_masked=masked_manual, teacher_masked=masked_teacher,
                common_known=common.copy(), original_covered=original,
                filled=expected_filled.copy())
    lost = known & ~common
    audit = dict(format='affinity_teacher_pair_v1', source_name=source,
                 source_allowlist_count=32, source_allowlist_sha256=allowlist_sha,
                 native_shape=list(manual.shape), source_gate_passed=True,
                 support_rule='(processed manual IDs>0) & (teacher uint16 IDs>0), identical in both arms',
                 pixels=dict(total=manual.size, processed_known=int(known.sum()),
                             original_covered=int(original.sum()), filled=int(expected_filled.sum()),
                             residual_unknown=int(expected_unknown.sum()), common_known=int(common.sum()),
                             common_original=int((common & original).sum()),
                             common_filled=int((common & expected_filled).sum()),
                             teacher_zero_on_manual_known=int(lost.sum()),
                             teacher_zero_on_original=int((lost & original).sum()),
                             teacher_zero_on_filled=int((lost & expected_filled).sum()),
                             ignored_teacher_positive_outside_manual=int(((teacher > 0) & ~known).sum())),
                 hashes={name:_sha(pair[name]) for name in
                         ('manual_original','manual_masked','teacher_masked','common_known','original_covered','filled')},
                 teacher_input_sha256=_sha(teacher),
                 uncovered_as_boundary=False, original_and_filled_equal_weight=True,
                 teacher_semantic_labels_used=False, teacher_human_confirmation_granted=False,
                 teacher_model_and_source_hash_identity_verified_here=False,
                 source_files_modified=False,
                 caveat='Hard self-distillation candidate, not corrected manual GT; source-name gate is not source-image hash verification')
    return pair, audit


def load_teacher_pair(manual_npz, teacher_png, *, source_name, allowed_training_sources):
    """只读canonical npz provenance和uint16 PNG；不覆盖GT或读取教师类别JSON。"""
    source, _ = _source_gate(source_name, allowed_training_sources)
    manual_path, teacher_path = Path(manual_npz), Path(teacher_png)
    if manual_path.name != source+'_gt.npz':
        raise ValueError('Manual filename must identify the declared training source')
    if teacher_path.suffix.lower() != '.png':
        raise ValueError('Teacher geometry must be uint16 PNG')
    with np.load(manual_path, allow_pickle=False) as payload:
        required = {'instance_map','original_covered','filled','residual_unknown'}
        if not required.issubset(payload.files):
            raise ValueError('Canonical processed GT provenance incomplete')
        values = {key:payload[key].copy() for key in required}
    teacher = cv2.imread(str(teacher_path), cv2.IMREAD_UNCHANGED)
    if teacher is None:
        raise FileNotFoundError(teacher_path)
    pair, audit = build_teacher_pair(values['instance_map'], teacher, values['original_covered'],
        source_name=source, allowed_training_sources=allowed_training_sources,
        filled=values['filled'], residual_unknown=values['residual_unknown'])
    audit['input_files'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (manual_path, teacher_path)}
    audit['teacher_file_source_binding_verified_here'] = False
    return pair, audit


def _relations(manual_targets, teacher_targets, selected):
    per_channel = []
    for channel, offset in enumerate(DEFAULT_AFFINITY_OFFSETS):
        mask = selected[0, channel]
        manual_same, teacher_same = manual_targets[0, channel] == 1, teacher_targets[0, channel] == 1
        counts = dict(manual_same_teacher_same=int((mask & manual_same & teacher_same).sum()),
                      manual_same_teacher_different=int((mask & manual_same & ~teacher_same).sum()),
                      manual_different_teacher_same=int((mask & ~manual_same & teacher_same).sum()),
                      manual_different_teacher_different=int((mask & ~manual_same & ~teacher_same).sum()))
        per_channel.append(dict(channel=channel, offset=list(offset), selected_edges=int(mask.sum()), **counts))
    totals = {key:sum(row[key] for row in per_channel) for key in
              ('selected_edges','manual_same_teacher_same','manual_same_teacher_different',
               'manual_different_teacher_same','manual_different_teacher_different')}
    conflicts = totals['manual_same_teacher_different']+totals['manual_different_teacher_same']
    return dict(**totals, conflict_edges=conflicts,
                conflict_fraction=conflicts/totals['selected_edges'] if totals['selected_edges'] else None,
                per_channel=per_channel,
                directional_meaning='same->different: teacher splits manual; different->same: teacher joins manual; neither source is assumed physically correct')


def resize_teacher_pair(pair, *, input_size=1024, output_grid=512):
    """复用原nearest等比letterbox、八通道双端规则；返回numpy网格/CPU torch targets。

    两臂edge_valid逐值相同。统计按original-original/涉及filled分层，并报告
    相对原processed manual合法边因教师零缝丢弃的数量；不偷偷恢复零缝。
    不新增原生整路径门禁，这是原完整GT合同的端点监督，不是物理边界真值。
    """
    manual = _ids(pair['manual_masked'], 'manual_masked')
    teacher = _ids(pair['teacher_masked'], 'teacher_masked', teacher=True)
    original_manual = _ids(pair['manual_original'], 'manual_original')
    if manual.shape != teacher.shape or manual.shape != original_manual.shape:
        raise ValueError('Paired native maps must share shape')
    common = _binary(pair['common_known'], manual.shape, 'common_known')
    original = _binary(pair['original_covered'], manual.shape, 'original_covered')
    filled = _binary(pair['filled'], manual.shape, 'filled')
    if (not np.array_equal(manual > 0, common) or not np.array_equal(teacher > 0, common)
            or np.any(common & (original_manual == 0)) or np.any(original & filled)
            or not np.array_equal(original | filled, original_manual > 0)
            or not np.array_equal(manual[common], original_manual[common])):
        raise ValueError('Common support/provenance or original manual identity inconsistent')
    mgrid, valid, metadata = letterbox_instance_geometry(manual, input_size, output_grid)
    tgrid, tvalid, tmeta = letterbox_instance_geometry(teacher, input_size, output_grid)
    oldgrid, oldvalid, oldmeta = letterbox_instance_geometry(original_manual, input_size, output_grid)
    original_grid, _, _ = letterbox_instance_geometry(original.astype(np.uint8), input_size, output_grid)
    filled_grid, _, _ = letterbox_instance_geometry(filled.astype(np.uint8), input_size, output_grid)
    if (not np.array_equal(valid, tvalid) or not np.array_equal(valid, oldvalid)
            or metadata != tmeta or metadata != oldmeta or not np.array_equal(mgrid > 0, tgrid > 0)):
        raise AssertionError('Original resize/letterbox did not preserve common support')
    content = torch.from_numpy(valid.copy())[None, None]
    mt, legal = build_affinity_targets_torch(torch.from_numpy(mgrid.astype(np.int64))[None], content)
    tt, teacher_legal = build_affinity_targets_torch(torch.from_numpy(tgrid.astype(np.int64))[None], content)
    _, oldlegal = build_affinity_targets_torch(torch.from_numpy(oldgrid.astype(np.int64))[None], content)
    if not torch.equal(legal, teacher_legal) or torch.any(legal & ~oldlegal):
        raise AssertionError('Teacher must not alter common edge mask or expand old manual coverage')
    original_edges = torch.zeros_like(legal)
    height, width = mgrid.shape
    original_nodes = original_grid > 0
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        if height <= abs(dy) or width <= abs(dx):
            continue
        source, dest = _edge_slices(height, width, dy, dx)
        original_edges[0, channel][source] = (legal[0, channel][source]
            & torch.from_numpy(original_nodes[source] & original_nodes[dest]))
    filled_edges = legal & ~original_edges
    lost = oldlegal & ~legal
    geometry = dict(manual_grid=mgrid, teacher_grid=tgrid, valid_content=valid,
                    common_known=(mgrid > 0), original_covered=original_nodes,
                    filled=filled_grid > 0, manual_targets=mt, teacher_targets=tt,
                    edge_valid=legal, metadata=metadata.to_dict())
    stats = dict(format='affinity_teacher_relations_v1', source_name=pair['source_name'],
                 native_shape=list(manual.shape), resize=metadata.to_dict(),
                 common_edge_mask_equal=True, common_edge_mask_sha256=_sha(legal),
                 targets_sha256=dict(manual=_sha(mt),teacher=_sha(tt)),
                 prior_processed_manual_edges=int(oldlegal.sum()), common_edges=int(legal.sum()),
                 lost_manual_edges_due_teacher_zero=int(lost.sum()),
                 lost_manual_edges_per_channel=[int(lost[0, c].sum()) for c in range(8)],
                 lost_manual_edge_fraction=float(lost.sum())/int(oldlegal.sum()) if oldlegal.any() else None,
                 strata=dict(all=_relations(mt,tt,legal),
                             original_original=_relations(mt,tt,original_edges),
                             involving_filled=_relations(mt,tt,filled_edges)),
                 unknown_padding_and_outside_pairs_ignored=True, uncovered_as_boundary=False,
                 support_is_processed_known_including_filled=True,
                 rule='original eight offsets, nearest letterbox IDs, both endpoints known; no added full-native-path restriction',
                 confidence_filtering_applied=False, labels_compared_by_ID_equality=False,
                 teacher_human_confirmation_granted=False)
    return geometry, stats
