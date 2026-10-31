# -*- coding: utf-8 -*-
"""只读比较同图 GT 渲染/真实响应及跨域真实热图；不推理、不训练、不拟合阈值。

manifest.rows 每行提供 source、split、cache_path 与 source_shape；缓存至少包含
rgb(HWC uint8)、real_boundary(HW float)、valid(HW bool)。train_gt 另需完整
original_known/filled_known/unknown、truth_boundary、known_interior。所有空间量
均保持缓存的原生像素单位；仅使用固定 .45/.65，不把响应带宽当物理晶界宽。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize


THRESHOLDS = (.45, .65)
MAP_FIELDS = ('real_boundary', 'local_boundary', 'gt_nominal', 'recipe_target', 'recipe_input', 'grid_control')
GT_FIELDS = ('truth_boundary', 'known_interior', 'original_known', 'filled_known', 'unknown')


def array_statistics(values):
    """每个源图自己的统计；空区域保持 None，绝不当成零响应。"""
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size:
        return dict(count=0, mean=None, median=None, q25=None, q75=None,
                    p05=None, p95=None, near_zero_fraction=None)
    if not np.isfinite(values).all():
        raise ValueError('Statistics require finite values')
    return dict(count=int(values.size), mean=float(values.mean()), median=float(np.median(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)),
                near_zero_fraction=float(np.mean(values <= .01)))


def _probability(value, shape=None):
    value = np.asarray(value)
    if (value.ndim != 2 or not value.size or value.dtype.kind != 'f'
            or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
            or (shape is not None and value.shape != shape)):
        raise ValueError('Response must be finite floating HW values in [0,1] matching RGB')
    return value.astype(np.float32, copy=False)


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or not np.isin(value, [0, 1]).all():
        raise ValueError(f'{name}: binary mask matching response required')
    return value.astype(bool, copy=False)


def _inside_distance(mask):
    # 显式填零使全有效图的四条外边界也参与距离，而非依赖 EDT 的隐式边界。
    return distance_transform_edt(np.pad(mask, 1, mode='constant'))[1:-1, 1:-1]


def _nonjunction_skeleton(mask):
    skeleton = skeletonize(mask)
    adjacent = cv2.filter2D(skeleton.astype(np.float32), -1, np.ones((3, 3), np.float32),
                           borderType=cv2.BORDER_CONSTANT) - skeleton
    junction = skeleton & (adjacent > 2)
    excluded = cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return skeleton, skeleton & ~excluded


def response_statistics(probability, valid, center_mask=None, edge_margin=4):
    """带宽=骨架上的2×EDT；剔除分叉及被无效域截断的响应带。

    center_mask 只限定统计中心，不重新切开响应mask。否则 original/fill 分界
    或标注范围边缘会制造虚假的窄响应带。半径+margin 的距离门槛进一步排除截断。
    """
    probability = _probability(probability)
    valid = _mask(valid, probability.shape, 'valid')
    center = valid if center_mask is None else valid & _mask(center_mask, valid.shape, 'center_mask')
    if int(edge_margin) != edge_margin or edge_margin < 0:
        raise ValueError('edge_margin must be a nonnegative native pixel count')
    room = _inside_distance(valid)
    result = dict(values=array_statistics(probability[center]), thresholds={})
    for threshold in THRESHOLDS:
        foreground = valid & (probability >= threshold)
        skeleton, nonjunction = _nonjunction_skeleton(foreground)
        radius = distance_transform_edt(foreground)
        keep = nonjunction & center & (room > radius + edge_margin)
        width = 2 * radius[keep]
        result['thresholds'][str(threshold)] = dict(
            occupancy=float(np.mean(probability[center] >= threshold)) if center.any() else None,
            response_pixels=int((foreground & center).sum()), skeleton_pixels=int((skeleton & center).sum()),
            reliable_skeleton_pixels=int(keep.sum()), response_band_width=array_statistics(width))
    return result


def gt_distance_statistics(maps, valid, truth_boundary, known_interior, strata, deep_distance=16):
    """同一 GT 的固定距离带与相同骨架坐标峰值；不估计复杂交点的法向。

    局部峰值只是半径4/8邻域最大值，不宣称一一配准。中心需距无效区>radius+4；
    original/filled 表示中心来源，不宣称峰值邻域全由同一来源构成。
    """
    truth_boundary = np.asarray(truth_boundary, bool)
    if truth_boundary.any():
        distance = distance_transform_edt(~truth_boundary)
    else:
        # 没有已知边界不代表整张图都能被证明为深内部。
        distance = np.full(valid.shape, np.inf)
    _, seed = _nonjunction_skeleton(truth_boundary)
    seed &= valid
    room = _inside_distance(valid)
    bands = [('on_boundary', 0., .5), ('distance_0p5_2', .5, 2.), ('distance_2_4', 2., 4.),
             ('distance_4_8', 4., 8.), ('distance_8_16', 8., 16.), ('distance_16_32', 16., 32.)]
    result = {}
    for name, probability in maps.items():
        peaks = {}
        for radius in (4, 8):
            yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
            peaks[radius] = maximum_filter(probability, footprint=yy*yy+xx*xx <= radius*radius,
                                           mode='constant', cval=0.)
        result[name] = {}
        for stratum, support in strata.items():
            region = valid & support
            profiles = {label: array_statistics(probability[region & (distance >= lo) & (distance < hi)])
                        for label, lo, hi in bands}
            deep = region & known_interior & (distance > deep_distance) & np.isfinite(distance)
            centers = seed & support & (room > 4)
            measured = dict(distance_profile=profiles, known_deep_interior=array_statistics(probability[deep]),
                            gt_skeleton_center=array_statistics(probability[centers]), neighborhood_peaks={})
            for radius in (4, 8):
                selected = centers & (room > radius + 4)
                measured['neighborhood_peaks'][str(radius)] = array_statistics(peaks[radius][selected])
            result[name][stratum] = measured
    return result


def image_covariates(rgb, valid):
    """亮度及纹理/模糊混合代理量；不把 Laplacian 方差直接解释为光学模糊。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
    reliable = valid & (_inside_distance(valid) > 2)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    values = gray[valid]
    return dict(brightness=array_statistics(values),
                brightness_std=float(values.std()) if values.size else None,
                laplacian_variance_texture_blur_proxy=float(laplacian[reliable].var()) if reliable.any() else None,
                mean_squared_gradient_texture_blur_proxy=float((gx*gx+gy*gy)[reliable].mean()) if reliable.any() else None)


def analyze_cache(arrays, metadata, map_fields=MAP_FIELDS):
    """单源独立测量。无测试GT；train_gt强制核验原覆盖/填充/未知来源。"""
    rgb = np.asarray(arrays['rgb'])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('rgb must be uint8 HWC RGB')
    shape = rgb.shape[:2]
    valid = _mask(arrays['valid'], shape, 'valid')
    if not valid.any():
        raise ValueError('No valid pixels')
    split = metadata.get('split', metadata.get('domain'))
    if not isinstance(split, str) or not split:
        raise ValueError('Explicit split/domain required')
    has_gt = 'truth_boundary' in arrays
    if has_gt and split != 'train_gt':
        raise ValueError('GT diagnostics are restricted to train_gt; test/unlabeled GT forbidden')
    if split == 'train_gt' and not has_gt:
        raise ValueError('train_gt requires processed GT provenance')
    maps = {name: _probability(arrays[name], shape) for name in map_fields if name in arrays}
    if 'real_boundary' not in maps:
        raise ValueError('real_boundary is required')
    strata = {'all_valid': valid}
    result = dict(source=str(metadata['source']), split=split, shape=list(shape),
                  source_shape=metadata.get('source_shape'), native_crop=metadata.get('native_crop'),
                  recipe_meta=metadata.get('recipe_meta'), covariates=image_covariates(rgb, valid),
                  coverage=dict(total_pixels=int(valid.size), valid_pixels=int(valid.sum()),
                                valid_fraction=float(valid.mean())), maps={})
    if has_gt:
        masks = {name: _mask(arrays[name], shape, name) for name in GT_FIELDS}
        original, filled, unknown = (masks[k] for k in ('original_known', 'filled_known', 'unknown'))
        if np.any(original & filled) or not np.array_equal(original | filled, ~unknown):
            raise ValueError('Original/filled/unknown must form a disjoint complete partition')
        if np.any(valid & unknown) or np.any(masks['truth_boundary'] & unknown):
            raise ValueError('Unknown pixels cannot become reliable GT or boundaries')
        if np.any(masks['known_interior'] & unknown):
            raise ValueError('Known interior cannot include unknown pixels')
        strata.update(original_known=valid & original, filled_known=valid & filled)
        result['coverage'].update(original_known_pixels=int(original.sum()), filled_known_pixels=int(filled.sum()),
                                  unknown_pixels=int(unknown.sum()), unknown_fraction=float(unknown.mean()))
    for name, probability in maps.items():
        result['maps'][name] = {stratum: response_statistics(probability, valid, support)
                               for stratum, support in strata.items()}
    if has_gt:
        result['gt_matched'] = gt_distance_statistics(maps, valid, masks['truth_boundary'],
                                                      masks['known_interior'], strata)
    if len(maps) > 1:
        # 先作同一图、同一位置的差，再作源间汇总；不以两个组中位之差冒充配对效应。
        # 无GT时也能比较 primary 窗 local 与真实 stitched 输入，但不产生真值诊断。
        result['paired_difference_from_real'] = {}
        reference = dict(response=result['maps']['real_boundary'])
        if has_gt:
            reference['gt_matched'] = result['gt_matched']['real_boundary']
        real_leaves = _numeric_leaves(reference)
        for name in maps:
            if name == 'real_boundary':
                continue
            candidate = dict(response=result['maps'][name])
            if has_gt:
                candidate['gt_matched'] = result['gt_matched'][name]
            compared = _numeric_leaves(candidate)
            result['paired_difference_from_real'][name] = {
                key: value-real_leaves[key] for key, value in compared.items() if key in real_leaves}
    return result


def _numeric_leaves(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_numeric_leaves(item, f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
        result[prefix] = float(value)
    return result


def equal_source_summary(rows):
    """先对同源多视图取中位，再对源等权取中位/IQR；绝不按像素数加权。"""
    sources = defaultdict(lambda: defaultdict(list))
    for row in rows:
        source = (row['split'], row['source'])
        selected = {k: row[k] for k in ('maps', 'gt_matched', 'paired_difference_from_real', 'covariates', 'coverage') if k in row}
        for key, value in _numeric_leaves(selected).items():
            sources[source][key].append(value)
    per_source = {source: {key: float(np.median(values)) for key, values in metrics.items()}
                  for source, metrics in sources.items()}
    metric_values = defaultdict(list)
    for metrics in per_source.values():
        for key, value in metrics.items():
            metric_values[key].append(value)
    return dict(sources=len(sources), views=len(rows), weighting='within-source median, then equal-source median/IQR',
                metrics={key: dict(sources=len(values), median=float(np.median(values)),
                                   q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))
                         for key, values in sorted(metric_values.items())})


def analyze_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = document.get('rows', document.get('files', []))
    if not entries:
        raise ValueError('Nonempty manifest rows/files required')
    if document.get('complete') is False:
        raise ValueError('Capture/analysis manifest is incomplete')
    rows = []
    for item in entries:
        path = Path(item.get('cache_path', item.get('path', '')))
        if not path.is_absolute():
            path = manifest_path.parent / path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = item.get('sha256', item.get('cache_sha256'))
        if expected and digest != expected:
            raise ValueError(f'Cache SHA256 differs: {item["source"]}')
        with np.load(path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        row = analyze_cache(arrays, item, document.get('map_fields', MAP_FIELDS))
        row['cache_sha256'] = digest
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row['split']].append(row)
        resolution = 'x'.join(map(str, row['source_shape'])) if row['source_shape'] else 'unspecified'
        groups[row['split'] + '/resolution_' + resolution].append(row)
    return dict(format='boundary_domain_v1', manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                thresholds=list(THRESHOLDS), native_pixel_units=True,
                method=dict(width='2*EDT at nonjunction skeleton; require distance-to-invalid > halfwidth+4',
                            optional_local='local_boundary is the primary-window center; real_boundary is actual stitched input',
                            gt_profile='same-pixel distance bands and radius4/8 neighborhood peaks; not exact normal registration',
                            provenance='original/filled strata refer to measurement centers, not a pure-provenance neighborhood',
                            deep_interior='provided known_interior AND distance from any known GT boundary >16 AND valid',
                            aggregation='per source median/IQR, not pooled pixel weighting'),
                limitations=['Response-band widths are not physical boundary widths or a scale calibration.',
                             'Test/unlabeled low-response values are lower-tail response statistics, not known background accuracy.',
                             'Aggregate groups describe the sampled sources; equal counts per resolution do not represent population frequencies.',
                             'Training GT images are seen sources; GT omissions and filled labels are not independent ground truth.',
                             'Brightness and derivative covariates mix texture and blur; no inferred optical blur or magnification ratio.',
                             'A manually resampled grid control cannot by itself explain the measured native response width.'],
                images=rows, groups={key: equal_source_summary(values) for key, values in sorted(groups.items())})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Ignored private artifact directory')
    args = parser.parse_args(argv)
    cv2.setNumThreads(2)
    result = analyze_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'boundary_domain.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(path=str(path), sources=len({(r['split'], r['source']) for r in result['images']}),
                          views=len(result['images']), trained=False)))


if __name__ == '__main__':
    main()
