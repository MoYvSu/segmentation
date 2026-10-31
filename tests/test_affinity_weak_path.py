# -*- coding: utf-8 -*-
"""弱界只读捕获的合成保护测试；不读取赛题图片/权重。"""
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tools import probe_affinity_weak_path as subject
from utils.affinity_deployment import crop_letterbox_output, probability_to_logit
from utils.affinity_fusion import affinity_boundary_probability
from utils.patch_diagnostic import BlendMap, tile_boxes


def test_contact_mask_does_not_fill_unknown_or_other_gt():
    gt = np.array([[69, 70, 0], [69, 0, 70], [69, 71, 70]], np.int32)
    before = gt.copy()
    contact = subject.contact_mask(gt)
    assert np.array_equal(contact, np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], bool))
    assert not contact[gt == 0].any() and not contact[gt == 71].any()
    assert np.array_equal(gt, before)


def test_statistics_use_actual_cpu_roundtrip_and_strict_greater():
    values = np.array([[.2, .65, .9]], np.float32)
    before = values.copy(); selected = np.ones(values.shape, bool)
    result = subject.threshold_stats(values, selected, .65)
    actual = torch.sigmoid(probability_to_logit(torch.from_numpy(values))).numpy()
    assert result['strict_above'] == int((actual > .65).sum())
    assert result['equal'] == int((actual == .65).sum())
    assert result['historical_raw_greater_equal'] == 2
    assert result['strict_above'] + result['equal'] + result['below'] == 3
    assert np.array_equal(values, before)


def test_transparent_hooks_one_formal_pass_original_stride_and_zero_stitch(tmp_path):
    """虚构原49调用顺序，验证hook不多前向、恢复wrapper且空干预逐值exact。"""
    old_threads = torch.get_num_threads(); torch.set_num_threads(4)
    try:
        _exercise_capture(tmp_path)
    finally:
        torch.set_num_threads(old_threads)


def _exercise_capture(tmp_path):
    shape = (1936, 2584)
    gt = np.ones(shape, np.int32)
    gt[750:800, 450:465] = 69; gt[750:800, 465:480] = 70
    # Six tiles see at least one focal GT; only two contain legal 69/70 relations.
    # An isolated one-sided GT patch must not trigger unnecessary logits capture.
    gt[1000:1100, 800:850] = 69
    config = dict(affinity_native=dict(overlap=.25), affinity_deployment=dict(fusion_mode='gated',
        short_reduction='top2', distance2_weight=.5, distance4_weight=.25))
    calls = dict(head=0, predict=0)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('logits', torch.full((1, 8, 512, 512), 8.).contiguous(memory_format=torch.channels_last))

        def forward(self, _input):
            calls['head'] += 1
            return dict(affinity_logits=self.logits)

    model = SimpleNamespace(affinity_decoder=Head())
    before = model.affinity_decoder.logits.clone()
    views = SimpleNamespace()

    def original_maps(*_args, **_kwargs):
        return np.zeros((*shape, 3), np.uint8), torch.zeros((1, 1, *shape)), None

    def original_grid(actual_model, image, cfg):
        mode, kwargs = subject.fusion_options(cfg)
        return affinity_boundary_probability(actual_model.affinity_decoder(image)['affinity_logits'], mode=mode, **kwargs)

    views.prediction_maps = original_maps
    views.boundary_grid = original_grid
    views.crop_letterbox_output = crop_letterbox_output

    def formal_predict(actual_model, restorer, path, cfg, device, seed, sizes, *, fixed_audit):
        calls['predict'] += 1
        assert sizes == [1024] and seed == subject.SEED
        rgb, semantic, _ = views.prediction_maps(actual_model, path, cfg, device, restorer, seed)
        raw = views.crop_letterbox_output(torch.zeros((1, 1, 512, 512)), 1024,
            1024-round(shape[0]*1024/max(shape)), 0, shape)
        fixed_audit.update(geometry_semantic_sha256=subject.array_sha(semantic.numpy()),
            raw_probability_sha256=subject.array_sha(raw.cpu().sigmoid()[0, 0].numpy()))
        blend = BlendMap(shape)
        for box in tile_boxes(shape, 1024, .25):
            y, x, y1, x1 = box
            grid = views.boundary_grid(actual_model, None, cfg)
            native = views.crop_letterbox_output(grid, 1024, 0, 0, (y1-y, x1-x))
            blend.add(box, native[0, 0].numpy())
        boundary, audit = blend.finish(); audit['tiles'] = 12
        markers = np.zeros(shape, np.int32); markers[500, 500] = 1
        cv2.watershed(np.zeros((*shape, 3), np.uint8), markers)
        ids = np.maximum(markers, 0).astype(np.uint16)
        return rgb, dict(patch1024=dict(instances=ids, classes={'1': 1}, boundary=boundary, blend=audit))

    core = SimpleNamespace(coverage_original_views=views, predict_views=formal_predict)
    value, record = subject.capture_once(core, model, None, 'fictional.jpg', config, gt, tmp_path, 'synthetic')
    assert calls == dict(head=12, predict=1)
    assert torch.equal(model.affinity_decoder.logits, before)
    assert views.prediction_maps is original_maps and views.boundary_grid is original_grid
    assert views.crop_letterbox_output is crop_letterbox_output
    assert record['empty_mask_grid_native_and_stitch_exact']
    assert record['actual_watershed_calls'] == 1 and sum(r['relevant'] for r in record['tiles']) == 2
    assert sum(np.isin(gt[y:y1, x:x1], subject.PAIR).any()
               for y, x, y1, x1 in tile_boxes(shape, 1024, .25)) == 6
    assert [r['index'] for r in record['tiles'] if r['relevant']] == [0, 4]
    assert all(r['logits_stride'] == list(before.stride()) for r in record['tiles'])
    with np.load(tmp_path / 'synthetic/maps.npz', allow_pickle=False) as maps:
        assert np.array_equal(maps['baseline'], value['boundary'])
        assert maps['markers'][500, 500] == 1
        assert not np.array_equal(maps['baseline'], maps['short_correct'])
    for row in record['tiles']:
        with np.load(tmp_path / row['map_file'], allow_pickle=False) as tile:
            assert ('logits' in tile.files) == row['relevant']
            assert np.array_equal(tile['native'], tile['native_empty'])
            if row['relevant']:
                assert row['short_correct']['selected_edges'] > 0
                assert row['all_correct']['selected_edges'] > row['short_correct']['selected_edges']
                assert row['short_correct']['per_channel'][4:] == [0, 0, 0, 0]
