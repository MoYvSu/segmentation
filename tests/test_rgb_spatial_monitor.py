# -*- coding: utf-8 -*-
"""空间模糊过程观察不得移动训练随机流或改变原采样轨迹。"""
import copy

import cv2
import numpy as np
import torch

from models.rgb_restoration import build_rgb_restorer
from tools.rgb_spatial_monitor import build_spatial_fixed_samples, clean_prediction_trajectory, monitor_spatial
from train_rgb_restoration import build_datasets
from train_rgb_diffusion_restoration import fixed_training_samples
from utils.config import load_config


def test_spatial_monitor_fixed_sources_rng_and_deployed_trajectory(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = tmp_path/'sources'
        source.mkdir()
        yy, xx = np.mgrid[:48,:64]
        for index in range(2):
            a = np.stack(((xx*4+index*19)%256,yy*5,((xx//4+yy//5)%2)*210),axis=-1).astype(np.uint8)
            assert cv2.imwrite(str(source/f'image_{index}.png'),a)
        config = load_config('config/train/rgb_restoration_diffusion_d4_spatial_all60_monitored.yaml')
        cfg = config['rgb_restoration']
        cfg.update(data_dir=str(source),image_size=48,crop_size=32,masked_pretraining=None,expected_train_sources=2)
        cfg['model'].update(width=4,encoder_blocks=[1],middle_blocks=1,decoder_blocks=[1],time_dim=16,num_steps=3)
        cfg['monitor'].update(samples=2,thumbnail_size=64,roi_size=16)
        cfg['monitor']['spatial_blur']['samples'] = 2
        train_set, holdout = build_datasets(config)
        assert holdout is None
        original_config = copy.deepcopy(config)
        _, original_manifest = fixed_training_samples(config,2)
        fixed, manifest = build_spatial_fixed_samples(config,train_set,original_manifest)
        duplicate, second_manifest = build_spatial_fixed_samples(config,train_set,original_manifest)
        assert manifest == second_manifest and config == original_config
        assert manifest['sources'] == original_manifest['sources']
        for key in ('input','target','valid','sigma'):
            assert torch.equal(fixed[key],duplicate[key])
        model = build_rgb_restorer(cfg['model']).train()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.add_(.0001)
            chain = list(clean_prediction_trajectory(model,fixed['input'],torch.Generator().manual_seed(17)))
            deployed = model(fixed['input'],generator=torch.Generator().manual_seed(17))
        assert len(chain) == 3 and torch.equal(chain[-1][1],deployed)
        before = torch.get_rng_state().clone()
        report = monitor_spatial(model,fixed,tmp_path/'monitor',device=torch.device('cpu'),epoch=20,
                                 updates=5000,monitor_cfg=cfg['monitor'],use_amp=False,amp_dtype=torch.bfloat16)
        assert torch.equal(before,torch.get_rng_state()) and model.training
        assert len(report['records']) == 6 and len(report['clean_input_drift']) == 4
        assert all(r['regions']['weak']['pixels'] > 0 and r['regions']['strong']['pixels'] > 0
                   for r in report['records'])
        previews = list((tmp_path/'monitor').rglob('*.png'))
        assert len(previews) == 5
        assert all(cv2.imread(str(p)) is not None for p in previews)
    finally:
        torch.set_num_threads(old_threads)
