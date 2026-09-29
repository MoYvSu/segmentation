# -*- coding: utf-8 -*-
"""只预测缓变明度增益和全局gamma；不直接生成RGB纹理。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


FORMAT = 'illumination_calibration_v1'


def luminance(image):
    return (image * image.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)


def apply_curve(image, log_gain, gamma):
    """同位置RGB共用正增益，截断前保持色度比例；零参数逐值恒等。"""
    light = luminance(image)
    scale = torch.exp(log_gain + (gamma - 1) * light.clamp_min(1e-5).log())
    return image * scale


class IlluminationCalibrator(nn.Module):
    def __init__(self, width=24, context_size=256, field_size=8,
                 max_log_gain=.6, max_log_gamma=.3):
        super().__init__()
        self.config = dict(width=width, context_size=context_size, field_size=field_size,
                           max_log_gain=max_log_gain, max_log_gamma=max_log_gamma)
        self.features = nn.Sequential(
            nn.Conv2d(4, width, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.GELU())
        self.gain_head = nn.Conv2d(width * 2, 1, 1)
        self.gamma_head = nn.Linear(width * 2, 1)
        for layer in (self.gain_head, self.gamma_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, image, valid=None, return_fields=False):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected BCHW RGB')
        if valid is None:
            valid = torch.ones_like(image[:, :1])
        if valid.shape != image[:, :1].shape:
            raise ValueError('valid must be B1HW')
        image, valid = image.float(), valid.float()
        # 原图已经等比letterbox；缩小正方形不挤压内容，mask明确标出padding。
        context = F.interpolate(torch.cat((image, valid), 1),
                                size=(self.config['context_size'],) * 2, mode='area')
        features = self.features(context)
        mask = F.interpolate(valid, size=features.shape[-2:], mode='area')
        pooled = (features * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        gamma = (self.config['max_log_gamma'] * self.gamma_head(pooled).tanh()).exp()[:, :, None, None]
        coarse = self.gain_head(F.adaptive_avg_pool2d(features, self.config['field_size'])).tanh()
        log_gain = F.interpolate(coarse, image.shape[-2:], mode='bilinear', align_corners=False)
        log_gain = self.config['max_log_gain'] * log_gain
        raw = apply_curve(image, log_gain, gamma)
        output = raw.clamp(0, 1)
        if return_fields:
            return output, dict(log_gain=log_gain, gamma=gamma, raw=raw, coarse=coarse)
        return output


def calibration_errors(prediction, target, valid):
    count = valid.sum().clamp_min(1)
    rgb = ((prediction - target).abs() * valid).sum() / (3 * count)
    delta = luminance(prediction) - luminance(target)
    # 带有效域归一的低频误差，避免把反射padding当监督。
    weight = F.avg_pool2d(valid, 32, 32)
    low = F.avg_pool2d(delta * valid, 32, 32) / weight.clamp_min(1e-5)
    low = (low.abs() * weight).sum() / weight.sum().clamp_min(1)
    gradient = prediction.new_zeros(())
    for dim in (2, 3):
        p, t = prediction.diff(dim=dim), target.diff(dim=dim)
        m = valid.narrow(dim, 0, valid.shape[dim] - 1) * valid.narrow(dim, 1, valid.shape[dim] - 1)
        gradient = gradient + ((p - t).abs() * m).sum() / (3 * m.sum().clamp_min(1))
    return dict(rgb_l1=rgb, low_l1=low, gradient_l1=gradient / 2)


def calibration_loss(prediction, target, valid, fields):
    errors = calibration_errors(prediction, target, valid)
    # Smooth-L1在小误差处平滑：避免恒等样本的L1尖点压住小幅校光学习。
    rgb = (F.smooth_l1_loss(prediction, target, reduction='none', beta=.01) * valid).sum() / (3 * valid.sum().clamp_min(1))
    weight = F.avg_pool2d(valid, 32, 32)
    low = F.avg_pool2d((luminance(prediction) - luminance(target)) * valid, 32, 32) / weight.clamp_min(1e-5)
    low_loss = (F.smooth_l1_loss(low, torch.zeros_like(low), reduction='none', beta=.01) * weight).sum() / weight.sum().clamp_min(1)
    coarse = fields['coarse']
    smooth = coarse.diff(dim=2).square().mean() + coarse.diff(dim=3).square().mean()
    return rgb + low_loss + .1 * errors['gradient_l1'] + .001 * smooth, errors


def load_calibrator(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT:
        raise ValueError('Unsupported calibration checkpoint')
    model = IlluminationCalibrator(**saved['model_config']).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.eval(), saved
