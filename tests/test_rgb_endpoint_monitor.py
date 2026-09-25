# -*- coding: utf-8 -*-
"""新过程图使用同一训练配对，真实图不读答案且不移动训练随机流。"""
import copy

import cv2
import numpy as np
import torch

from models.rgb_restoration import build_rgb_restorer
from tools.rgb_endpoint_monitor import (build_endpoint_fixed_samples, build_real_fixed_samples,
                                        monitor_endpoint, monitor_real)
from train_rgb_diffusion_restoration import fixed_training_samples
from train_rgb_restoration import build_datasets
from utils.config import load_config


def test_paired_fixed_monitors_preserve_training_rng(tmp_path):
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = tmp_path/'sources'
        source.mkdir()
        yy, xx = np.mgrid[:48, :64]
        for i in range(2):
            array = np.stack(((xx*4+i*19)%256, yy*5, ((xx//4+yy//5)%2)*210), axis=-1).astype(np.uint8)
            assert cv2.imwrite(str(source/f'image_{i}.png'), array)
        config = load_config('config/train/rgb_restoration_diffusion_d4_spatial_all60_monitored.yaml')
        cfg = config['rgb_restoration']
        cfg.update(data_dir=str(source), image_size=48, crop_size=32, masked_pretraining=None,
                   expected_train_sources=2)
        cfg['model'].update(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                            time_dim=16, num_steps=3)
        cfg['monitor'].update(samples=2, thumbnail_size=64, roi_size=16,
            endpoint_blur=dict(enabled=True, samples=2, weak_sigma=[.6, 1.5], strong_sigma=[4., 5.2],
                               transition_fraction=[.2, .5], min_region_fraction=.05),
            real=dict(enabled=True, data_dir=str(source), images=['image_0', 'image_1']))
        dataset, heldout = build_datasets(config)
        assert heldout is None
        _, original = fixed_training_samples(config, 2)
        fixed, manifest = build_endpoint_fixed_samples(config, dataset, original)
        altered = copy.deepcopy(config)
        altered['rgb_restoration']['degradation']['spatial_blur']['endpoint_transition'] = {
            **cfg['monitor']['endpoint_blur'], 'probability': .5}
        repeated, second = build_endpoint_fixed_samples(altered, dataset, original)
        assert manifest == second
        for key in ('input', 'target', 'sigma', 'valid'):
            assert torch.equal(fixed[key], repeated[key])
        real, real_manifest = build_real_fixed_samples(config)
        assert 'target' not in real and real_manifest['has_ground_truth'] is False
        model = build_rgb_restorer(cfg['model']).train()
        with torch.no_grad():
            model.correction.weight.add_(.001)
        before_rng = torch.get_rng_state().clone()
        endpoint_report = monitor_endpoint(model, fixed, tmp_path, device=torch.device('cpu'), epoch=1,
            updates=2, monitor_cfg=cfg['monitor'], use_amp=False, amp_dtype=torch.bfloat16)
        real_report = monitor_real(model, real, tmp_path, device=torch.device('cpu'), epoch=1,
            updates=2, monitor_cfg=cfg['monitor'], use_amp=False, amp_dtype=torch.bfloat16)
        assert model.training and torch.equal(before_rng, torch.get_rng_state())
        assert len(endpoint_report['records']) == 6
        assert all(row['regions']['weak']['pixels'] >= 32*32*.05 and
                   row['regions']['strong']['pixels'] >= 32*32*.05 for row in endpoint_report['records'])
        assert real_report['has_ground_truth'] is False
        assert len(real_report['records']) == 2
        paths = list((tmp_path/'endpoint_monitor').rglob('*.png')) + list((tmp_path/'real_monitor').rglob('*.png'))
        assert len(paths) == 4 and all(cv2.imread(str(path)) is not None for path in paths)
    finally:
        torch.set_num_threads(before_threads)
