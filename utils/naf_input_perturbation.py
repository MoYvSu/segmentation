# -*- coding: utf-8 -*-
"""NAF训练输入的整图原生空间模糊：不读GT，不改变图像几何或源文件。

必须先对完整有效RGB调用一次，再裁D5a/affinity所需窗口；不能在每个窗口
重新生成模糊场。σ是本合成器的名义原生像素参数，不是测试图的光学测量。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Integral, Real

import cv2
import numpy as np


FORMAT = 'naf_native_spatial_blur_v1'
SEED_NAMESPACE = 2026103011
PROFILES = {
    'moderate': dict(weak_sigma=1.0, strong_sigma=4.5, transition_fraction=.35,
                     min_region_fraction=.15, levels=8),
    # strong先用于预检；是否进入正式训练由冻结前端的实际响应检查决定。
    'strong': dict(weak_sigma=1.0, strong_sigma=7.0, transition_fraction=.35,
                   min_region_fraction=.15, levels=8),
}


def _make_sigma_map(shape, settings, rng):
    """内置D5a完整图端点算法，避免向封存前端导入新data命名空间。"""
    h, w = shape
    weak = float(rng.uniform(settings['weak_sigma'], settings['weak_sigma']))
    strong = float(rng.uniform(settings['strong_sigma'], settings['strong_sigma']))
    angle = float(rng.uniform(0, 2*np.pi))
    dy, dx = np.sin(angle), np.cos(angle)
    projection = np.arange(h)[:, None]*dy+np.arange(w)[None, :]*dx
    fraction = settings['min_region_fraction']
    low, high = np.quantile(projection, [fraction, 1-fraction])
    span = float(np.ptp(projection))
    transition = settings['transition_fraction']
    width = min(float(rng.uniform(transition, transition))*span, float(high-low))
    if width <= 0:
        raise ValueError('Native spatial field has degenerate projected extent')
    start = float(rng.uniform(low, high-width))
    phase = np.clip((projection-start)/width, 0., 1.)
    blend = phase**3*(phase*(6*phase-15)+10)
    variance = weak**2+(strong**2-weak**2)*blend
    return np.ascontiguousarray(np.sqrt(np.clip(variance, weak**2, strong**2)), dtype=np.float32)


def _apply_sigma_map_blur(clean, sigma_map, bounds, levels):
    """与data.rgb_spatial_blur的σ²分段插值逐值相同，自包含且只保留一个级别。"""
    sigma_levels = np.linspace(*bounds, levels, dtype=np.float64)
    variance_levels = sigma_levels**2
    variance = sigma_map*sigma_map
    image = np.zeros_like(clean, dtype=np.float32)
    for index, sigma in enumerate(sigma_levels):
        if index == 0:
            weight = (variance_levels[1]-variance)/(variance_levels[1]-variance_levels[0])
        elif index == len(sigma_levels)-1:
            weight = (variance-variance_levels[-2])/(variance_levels[-1]-variance_levels[-2])
        else:
            left = (variance-variance_levels[index-1])/(variance_levels[index]-variance_levels[index-1])
            right = (variance_levels[index+1]-variance)/(variance_levels[index+1]-variance_levels[index])
            weight = np.minimum(left, right)
        weight = np.clip(weight, 0, 1).astype(np.float32, copy=False)
        if not np.any(weight):
            continue
        blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT)
        np.multiply(blurred, weight[..., None], out=blurred)
        np.add(image, blurred, out=image)
    return np.ascontiguousarray(image, dtype=np.float32)


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(tuple(value.shape)).encode('ascii'))
    digest.update(value.tobytes())
    return digest.hexdigest()


def profile_settings(profile, config=None):
    """冻结小型候选配方；可显式覆盖幅度，不能暗中添加噪声/照明/重采样。"""
    if profile not in ('clean', *PROFILES):
        raise ValueError('Unknown NAF input perturbation profile')
    if config is not None and (not isinstance(config, dict) or set(config)-set(PROFILES['moderate'])):
        raise ValueError('Only native blur profile parameters may be overridden')
    if profile == 'clean':
        if config:
            raise ValueError('Clean reference does not accept blur settings')
        return None
    result = deepcopy(PROFILES[profile])
    result.update(config or {})
    for name in ('weak_sigma', 'strong_sigma', 'transition_fraction', 'min_region_fraction'):
        value = result[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError('Blur parameters must be finite real numbers')
        result[name] = float(value)
    if not 0 < result['weak_sigma'] <= 1.5 or not 4 <= result['strong_sigma'] <= 16:
        raise ValueError('Expected weak sigma in (0,1.5], strong sigma in [4,16] native pixels')
    if not 0 < result['transition_fraction'] < 1 or not 0 < result['min_region_fraction'] < .5:
        raise ValueError('Invalid smooth transition or endpoint coverage fraction')
    levels = result['levels']
    if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, Integral) or not 2 <= int(levels) <= 32:
        raise ValueError('Blur levels must be an integer in [2,32]')
    result['levels'] = int(levels)
    return result


def perturb_native_rgb(rgb, *, profile='moderate', seed=0, config=None):
    """返回 ``(RGB_uint8, sigma_float32, receipt)``，从不原地修改输入。

    输入为尚未letterbox/padding的完整原生有效图，至少2×2像素。当前局部
    D5a的原生1024窗不缩放，因此这里的σ也是它实际输入网格上的像素单位。
    同一seed的moderate/strong共享焦面方向与位置，只改变端点强度。

    退化复用D5a的σ²分段插值：多个固定σ高斯结果按位置凸组合，BORDER_REFLECT。
    这种近似不是每个位置的单一高斯PSF；一次uint8量化保证正式输入契约。
    不增加噪声、曝光、颜色增益、下采样、局部遮块或测试统计拟合。
    """
    if (not isinstance(rgb, np.ndarray) or np.ma.isMaskedArray(rgb) or rgb.dtype != np.uint8
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2):
        raise ValueError('Expected complete native uint8 RGB HxWx3, both dimensions >=2')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**64:
        raise ValueError('Expected nonnegative integer seed <2**64')
    settings = profile_settings(profile, config)
    before = array_sha256(rgb)
    h, w = rgb.shape[:2]
    if settings is None:
        output = np.array(rgb, copy=True, order='C')
        sigma = np.zeros((h, w), np.float32)
    else:
        weak, strong = settings['weak_sigma'], settings['strong_sigma']
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), SEED_NAMESPACE]))
        sigma = _make_sigma_map((h, w), settings, rng)
        blurred = _apply_sigma_map_blur(rgb.astype(np.float32), sigma, [weak, strong], settings['levels'])
        if not np.isfinite(blurred).all():
            raise RuntimeError('Nonfinite spatial blur output')
        output = np.ascontiguousarray(np.rint(np.clip(blurred, 0, 255)).astype(np.uint8))
    if output.shape != rgb.shape or output.dtype != np.uint8 or array_sha256(rgb) != before:
        raise RuntimeError('RGB shape/dtype/immutable-input contract failed')
    variance = sigma.astype(np.float64)**2
    stats = dict(min=float(sigma.min()), max=float(sigma.max()), mean_square=float(variance.mean()),
                 quantiles={str(q): float(np.quantile(sigma, q)) for q in (0, .05, .5, .95, 1)})
    if settings is not None:
        stats.update(weak_endpoint_fraction=float(np.mean(np.isclose(sigma, settings['weak_sigma'], rtol=0, atol=1e-6))),
                     strong_endpoint_fraction=float(np.mean(np.isclose(sigma, settings['strong_sigma'], rtol=0, atol=1e-6))))
    receipt = dict(format=FORMAT, profile=profile, config=settings, seed=int(seed),
                   seed_namespace=SEED_NAMESPACE, native_shape=[h, w],
                   input_sha256=before, output_sha256=array_sha256(output), sigma_sha256=array_sha256(sigma),
                   sigma_units='native pixels; scale1 in current native1024 local D5a input',
                   sigma_statistics=stats, border='BORDER_REFLECT', full_image_applications=1,
                   order='complete valid native RGB -> spatial blur -> uint8 once -> caller crops/reflect-pads',
                   padding_is_not_input=True, per_tile_field_regeneration=False,
                   geometry_changed=False, noise_or_illumination_or_resampling=False,
                   source_unchanged=True, GT_is_not_an_input=True,
                   caveat='Nominal sigma from variance-interpolated Gaussian outputs; not measured test PSF or affinity width')
    return output, np.ascontiguousarray(sigma, dtype=np.float32), receipt
