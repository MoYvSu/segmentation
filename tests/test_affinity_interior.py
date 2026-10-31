# -*- coding: utf-8 -*-
"""可信完整内部关系、长关系穿越和实际融合重放的虚构CPU检查。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.probe_affinity_interior as probe
from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_fusion import affinity_boundary_probability
from utils.offset_letterbox import letterbox_instance_geometry


def tile_for(gt, grid=11):
    labels, valid, geometry = letterbox_instance_geometry(gt, 1024, grid)
    return dict(box=[0, 0, *gt.shape], grid_gt=labels, valid=valid,
                geometry=geometry.to_dict())


@pytest.mark.parametrize('obstacle', ['unknown', 'third_GT', 'filled'])
def test_short_and_long_cannot_jump_a_thin_native_obstacle(obstacle):
    gt = np.full((41, 41), 40, np.uint16)
    original = np.ones(gt.shape, bool)
    if obstacle == 'unknown':
        gt[:, 6] = 0
        original[:, 6] = False
    elif obstacle == 'third_GT':
        gt[:, 6] = 99
    else:
        original[:, 6] = False
    tile = tile_for(gt)
    # nearest网格没有采中col6，align_corners中心4和8也都还是GT40。
    assert tile['grid_gt'][5, 1] == tile['grid_gt'][5, 2] == 40
    mask, audit = probe.positive_relation_mask(tile, gt, original, channels='all', margin=0)
    assert not mask[0, 0, 5, 1]  # short: native4 -> 8，穿过col6。
    assert not mask[0, 4, 5, 0]  # radius2: native0 -> 8，同样被拒。
    assert not mask[0, 6, 5, 0]  # radius4: native0 -> 16，同样被拒。
    assert mask[0, 0, 5, 4]  # 障碍另一侧仍有合法关系。
    assert audit['domain_provenance']['filled_pixels'] == 0


def test_native_four_pixel_guard_and_short_all_isolation_keep_inputs():
    gt = np.full((41, 41), 40, np.uint16)
    gt[:, 32:] = 41
    original = np.ones(gt.shape, bool)
    tile = tile_for(gt)
    saved = deepcopy(tile)
    before_gt, before_original = gt.copy(), original.copy()
    short, _ = probe.positive_relation_mask(tile, gt, original, channels='short')
    all_channels, _ = probe.positive_relation_mask(tile, gt, original, channels='all')
    assert short.any() and all_channels[:, 4:].any()
    assert np.array_equal(short[:, :4], all_channels[:, :4])
    assert not short[:, 4:].any()
    assert not all_channels[:, :, :, 0].any()  # native x0处于4px外周保护。
    assert not all_channels[:, :, :, 8:].any()  # GT41及其跨界关系不修改。
    assert np.array_equal(gt, before_gt) and np.array_equal(original, before_original)
    assert np.array_equal(tile['grid_gt'], saved['grid_gt'])
    assert np.array_equal(tile['valid'], saved['valid']) and tile['geometry'] == saved['geometry']


def test_relation_replacement_preserves_unselected_long_and_noncontiguous_input():
    logits = torch.randn((1, 8, 12, 12)).contiguous(memory_format=torch.channels_last)
    before = logits.clone()
    mask = np.zeros(tuple(logits.shape), bool)
    mask[0, :4, 3:9, 3:9] = True
    corrected = probe.replace_positive(logits, mask)
    chosen = torch.from_numpy(mask)
    assert torch.equal(corrected[~chosen], logits[~chosen])
    assert torch.equal(corrected[:, 4:], logits[:, 4:])
    assert torch.equal(logits, before)
    assert torch.all(corrected[chosen] == 16.)
    assert float(corrected[chosen].sigmoid().min()) > .999999


def test_actual_fp32_empty_fusion_native_and_blend_then_short_positive_replay():
    config = dict(affinity_deployment=dict(fusion_mode='gated', short_reduction='top2',
        distance2_weight=.5, distance4_weight=.25))
    logits = torch.full((1, 8, 16, 16), -3.)
    mode, options = probe.fusion_options(config)
    grid = affinity_boundary_probability(logits, mode=mode, **options)
    native = crop_letterbox_output(grid, 1024, 0, 0, (32, 32))
    tile = dict(index=0, box=[0, 0, 32, 32], logits=logits,
        logits_stride=list(logits.stride()), grid=grid.numpy().copy(),
        native=native.numpy().copy(), pad_h=0, pad_w=0)
    views = SimpleNamespace(crop_letterbox_output=crop_letterbox_output)
    empty = np.zeros(tuple(logits.shape), bool)
    baseline, audit, tiles = probe.recompute_positive(views, [tile], [empty], config,
                                                    (32, 32), empty_exact=True)
    assert audit['tiles'] == 1 and tiles[0]['native_changed_pixels'] == 0
    assert tiles[0]['grid_changed_pixels'] == 0
    mask = empty.copy()
    mask[0, :4, 4:12, 4:12] = True
    edited, _, actual = probe.recompute_positive(views, [tile], [mask], config, (32, 32))
    assert edited[16, 16] < baseline[16, 16]
    assert actual[0]['grid_changed_pixels'] > 0 and actual[0]['unselected_logits_exact']
    assert torch.all(logits == -3.)
    with pytest.raises(ValueError, match='Every actual tile'):
        probe.recompute_positive(views, [tile], [], config, (32, 32))


def test_existing_output_rejected_without_altering_protected_status(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    output = tmp_path/'outputs/geometry_interior'
    output.mkdir(parents=True)
    status = output/'status.json'
    status.write_bytes(b'{"status":"complete"}')
    with pytest.raises(FileExistsError, match='Fresh ignored outputs'):
        probe.run(SimpleNamespace(output='outputs/geometry_interior'))
    assert status.read_bytes() == b'{"status":"complete"}'
    assert sorted(p.name for p in output.iterdir()) == ['status.json']


def test_historical_source_is_verified_separately_and_only_safety_fix_can_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    relative = 'tools/probe_affinity_structure.py'
    current = tmp_path/relative
    saved = tmp_path/'outputs/old/source'/relative
    current.parent.mkdir(parents=True)
    saved.parent.mkdir(parents=True)
    saved.write_text('def safety_metrics():\n    return "old"\ndef decode_reflect():\n    return 42\n', encoding='utf8')
    digest = probe.sha(saved)
    current.write_text('def safety_metrics():\n    return "known_matched_phase_only_v2"\ndef decode_reflect():\n    return 42\n', encoding='utf8')
    protected, revisions = probe.historical_source_gate(tmp_path/'outputs/old/report.json', {relative:digest})
    assert protected[str(saved)] == digest
    assert revisions[0]['historical_sha256'] == digest
    assert revisions[0]['current_sha256'] == probe.sha(current)
    # 任一真实解码语句变化不能借“只修统计”名义通过。
    current.write_text(current.read_text().replace('return 42', 'return 43'), encoding='utf8')
    with pytest.raises(RuntimeError, match='outside safety_metrics'):
        probe.historical_source_gate(tmp_path/'outputs/old/report.json', {relative:digest})
