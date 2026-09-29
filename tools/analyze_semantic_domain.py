# -*- coding: utf-8 -*-
"""一次性检查原图与 D5a 后的训练/测试外观覆盖；不训练、不读取测试标签。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.backend_adaptation import PairedDegradationDataset
from data.mim_dataset import list_images
from data.rgb_restoration_dataset import prepare_rgb, read_rgb
from utils.config import load_config, project_path


METRICS = (
    'brightness_mean', 'brightness_p10', 'brightness_p90', 'contrast_std',
    'local_contrast_rms', 'detail_l1', 'gradient_rms', 'gradient_normalized',
    'blur_scale_ratio', 'clipped_fraction', 'color_spread',
)
# 重复相关的分位数不进入多变量距离；无量纲的模糊代理仍会受组织纹理影响。
DISTANCE_METRICS = ('brightness_mean', 'contrast_std', 'local_contrast_rms',
                    'detail_l1', 'gradient_normalized', 'blur_scale_ratio', 'color_spread')
QUANTILES = (.05, .25, .5, .75, .95)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def content_box(valid):
    """从经过翻转/旋转的有效域取矩形；绝不把 reflect padding 纳入统计。"""
    valid = np.asarray(valid).squeeze().astype(bool)
    if valid.ndim != 2 or not valid.any():
        raise ValueError('valid content must be a nonempty 2D rectangle')
    yy, xx = np.where(valid)
    y0, y1, x0, x1 = int(yy.min()), int(yy.max() + 1), int(xx.min()), int(xx.max() + 1)
    if valid.sum() != (y1 - y0) * (x1 - x0):
        raise ValueError('valid content is not rectangular')
    return y0, y1, x0, x1


def appearance_rows(rgb, valid, grid=4):
    """在同一 1024 尺度计算滤波图，再汇总整图及规则有效网格。

    blur_scale_ratio = RMS(梯度(G4))/RMS(梯度(G1))，较大通常表示更平滑；
    它与对比缩放基本无关，但不是光学模糊的真值，也受晶粒/纹理尺度影响。
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError('expected finite H x W x 3 RGB')
    y0, y1, x0, x1 = content_box(valid)
    rgb = rgb[y0:y1, x0:x1]
    if min(rgb.shape[:2]) < grid * 8:
        raise ValueError('valid content too small for the requested grid')
    gray = rgb @ np.array([.299, .587, .114], dtype=np.float32)
    smooth1 = cv2.GaussianBlur(gray, (0, 0), 1., borderType=cv2.BORDER_REFLECT)
    smooth4 = cv2.GaussianBlur(gray, (0, 0), 4., borderType=cv2.BORDER_REFLECT)
    smooth8 = cv2.GaussianBlur(gray, (0, 0), 8., borderType=cv2.BORDER_REFLECT)

    def gradient_square(image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8, borderType=cv2.BORDER_REFLECT)
        return gx * gx + gy * gy

    g1, g4 = gradient_square(smooth1), gradient_square(smooth4)
    detail = np.abs(gray - smooth1)
    local_square = (gray - smooth8) ** 2
    color = rgb.max(axis=-1) - rgb.min(axis=-1)
    clipped = ((rgb <= 1 / 255).any(axis=-1) | (rgb >= 254 / 255).any(axis=-1)).astype(np.float32)
    h, w = gray.shape
    regions = [('image', -1, -1, 0, h, 0, w)]
    ys, xs = np.linspace(0, h, grid + 1, dtype=int), np.linspace(0, w, grid + 1, dtype=int)
    regions.extend(('patch', i, j, ys[i], ys[i + 1], xs[j], xs[j + 1])
                   for i in range(grid) for j in range(grid))
    rows = []
    for scope, gy, gx, a, b, c, d in regions:
        region = np.s_[a:b, c:d]
        values = gray[region]
        sd = float(values.std())
        grad1, grad4 = float(np.sqrt(g1[region].mean())), float(np.sqrt(g4[region].mean()))
        rows.append({
            'scope': scope, 'grid_y': gy, 'grid_x': gx, 'pixels': int(values.size),
            'brightness_mean': float(values.mean()),
            'brightness_p10': float(np.quantile(values, .1)),
            'brightness_p90': float(np.quantile(values, .9)),
            'contrast_std': sd, 'local_contrast_rms': float(np.sqrt(local_square[region].mean())),
            'detail_l1': float(detail[region].mean()), 'gradient_rms': grad1,
            'gradient_normalized': grad1 / max(sd, 1e-6),
            'blur_scale_ratio': grad4 / max(grad1, 1e-6),
            'clipped_fraction': float(clipped[region].mean()), 'color_spread': float(color[region].mean()),
            'texture_informative': bool(sd >= .003 and grad1 >= 1e-5),
        })
    return rows


class ImagePool:
    """只读原图及有效域；供现有 PairedDegradationDataset 重放真实增强代码。"""
    def __init__(self, files, image_size=1024):
        self.files, self.image_size = list(files), int(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image, valid = prepare_rgb(read_rgb(str(path)), self.image_size)
        _, h, _, w = content_box(valid)
        return {'image': torch.from_numpy(image).permute(2, 0, 1),
                'valid_content': torch.from_numpy(valid).unsqueeze(0),
                'input_content_shape': torch.tensor([h, w], dtype=torch.int32),
                # 兼容训练服务器旧版 PairedDegradation 的旋转尺寸更新；
                # 本诊断没有 affinity 低分辨率网格，二者都代表输入有效域。
                'content_shape': torch.tensor([h, w], dtype=torch.int32), 'name': path.name}


def verify_cohorts(pool_files, manual_dir, expected_manual):
    """人工目录曾重新编号；只依据文件 SHA 确认来源，禁止以同名认定同图。"""
    pool = {path.stem: path for path in pool_files}
    if len(pool) != len(pool_files):
        raise ValueError('training pool contains duplicated stems')
    manual = [Path(path) for path in list_images(str(manual_dir)) if Path(path).with_suffix('.json').is_file()]
    if len(manual) != expected_manual or len({path.stem for path in manual}) != expected_manual:
        raise ValueError(f'expected {expected_manual} distinct manual sources, found {len(manual)}')
    by_sha = {}
    for path in pool_files:
        by_sha.setdefault(file_sha(path), []).append(path)
    mapping, used = [], set()
    for path in manual:
        digest = file_sha(path)
        matches = by_sha.get(digest, [])
        if len(matches) != 1:
            raise ValueError(f'no unique SHA-identical training source for {path.name}: {len(matches)} matches')
        match = matches[0]
        if match.stem in used:
            raise ValueError(f'multiple manual names map to the same pool source: {match.name}')
        used.add(match.stem)
        mapping.append({'manual_name': path.name, 'pool_name': match.name, 'sha256': digest})
    return mapping


def feature_matrix(rows, keys=DISTANCE_METRICS):
    return np.asarray([[row[key] for key in keys] for row in rows], dtype=np.float64)


def nearest_coverage(reference, query, calibration=None):
    """稳健缩放后的最近邻；阈值来自同池跨源图距离，不是模型验证集。

    每个源图的所有网格/变体均排除出该源的参照最近邻，避免相邻块/同图泄漏。
    """
    x, y = feature_matrix(reference), feature_matrix(query)
    owners = np.asarray([row['name'] for row in reference])
    _, owner_counts = np.unique(owners, return_counts=True)
    if len(owner_counts) < 2:
        raise ValueError('nearest-neighbor reference requires at least two source images')
    # manual 与 all_train 必须使用相同 all_train 标尺和阈值，否则参照样本
    # 增多本身会缩小自然 NN 半径，制造“扩大数据反而覆盖更差”的表象。
    if calibration is None:
        center = np.median(x, axis=0)
        scale = np.maximum(np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0), .001)
    else:
        center = np.asarray([calibration['center'][key] for key in DISTANCE_METRICS])
        scale = np.asarray([calibration['scale_iqr_floor_0001'][key] for key in DISTANCE_METRICS])
    xx, yy = (x - center) / scale, (y - center) / scale
    tree = cKDTree(xx)
    # k 比单源最大行数多 1，必定至少包括一张其他源图。
    k = min(len(xx), int(owner_counts.max()) + 1)
    cross_source = []
    if calibration is None:
        for start in range(0, len(xx), 2048):
            distance, index = tree.query(xx[start:start + 2048], k=k, workers=1)
            distance = np.where(owners[index] != owners[start:start + 2048, None], distance, np.inf)
            cross_source.extend(distance.min(axis=1).tolist())
    d, _ = tree.query(yy, k=1, workers=1)
    # RMS 标准化维度距离，便于理解；仍不具备准确率含义。
    divisor = len(DISTANCE_METRICS) ** .5
    d = np.asarray(d) / divisor
    threshold = (float(np.quantile(np.asarray(cross_source) / divisor, .95)) if calibration is None
                 else float(calibration['reference_cross_source_nn_q95']))
    lo, hi = np.quantile(x, .05, axis=0), np.quantile(x, .95, axis=0)
    envelope = (y >= lo) & (y <= hi)
    return {
        'reference_rows': len(reference), 'query_rows': len(query),
        'reference_sources': int(len(owner_counts)), 'dimensions': list(DISTANCE_METRICS),
        'normalization_reference_sources': (int(len(owner_counts)) if calibration is None
                                            else calibration['normalization_reference_sources']),
        'normalization_and_nn_threshold': 'shared_all_train_same_variant_and_scope',
        'center': dict(zip(DISTANCE_METRICS, center.tolist())),
        'scale_iqr_floor_0001': dict(zip(DISTANCE_METRICS, scale.tolist())),
        'reference_cross_source_nn_q95': threshold,
        'query_nn_quantiles': dict(zip(map(str, QUANTILES), np.quantile(d, QUANTILES).tolist())),
        'query_within_reference_nn_q95': float((d <= threshold + 1e-12).mean()),
        'query_inside_all_marginal_05_95': float(envelope.all(axis=1).mean()),
        'marginal_coverage': {key: {'inside': float(envelope[:, i].mean()),
                                   'below': float((y[:, i] < lo[i]).mean()),
                                   'above': float((y[:, i] > hi[i]).mean())}
                              for i, key in enumerate(DISTANCE_METRICS)},
    }


def summarize(rows):
    def select(cohort, variant, scope):
        return [row for row in rows if row['scope'] == scope and row['variant'] == variant
                and (row['cohort'] == cohort or cohort == 'all_train' and row['cohort'] != 'test')]
    distributions, coverage = {}, {}
    for scope in ('image', 'patch'):
        for cohort in ('manual', 'other_train', 'all_train', 'test'):
            for variant in ('raw', 'degraded', 'd5a', 'degraded_d5a'):
                selected = select(cohort, variant, scope)
                if not selected:
                    continue
                distributions[f'{cohort}/{variant}/{scope}'] = {
                    'rows': len(selected), 'sources': len({row['name'] for row in selected}),
                    'uninformative_texture_rows': sum(not row['texture_informative'] for row in selected),
                    'quantiles': {key: dict(zip(map(str, QUANTILES),
                                               np.quantile([row[key] for row in selected], QUANTILES).tolist()))
                                  for key in METRICS},
                }
        for test_variant in ('raw', 'd5a'):
            query = select('test', test_variant, scope)
            ref_variants = ('raw', 'degraded') if test_variant == 'raw' else ('d5a', 'degraded_d5a')
            for variant in ref_variants:
                shared = nearest_coverage(select('all_train', variant, scope), query)
                coverage[f'all_train/{variant}->test/{test_variant}/{scope}'] = shared
                coverage[f'manual/{variant}->test/{test_variant}/{scope}'] = nearest_coverage(
                    select('manual', variant, scope), query, calibration=shared)
        # 接近实际训练的单次退化抽样；不混入额外人为配比。
    return {'distributions': distributions, 'coverage': coverage}


def plot_distributions(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [('manual', 'raw'), ('other_train', 'raw'), ('test', 'raw'),
              ('manual', 'd5a'), ('other_train', 'd5a'), ('test', 'd5a'),
              ('manual', 'degraded_d5a'), ('other_train', 'degraded_d5a')]
    names = [f'{cohort}\n{variant}' for cohort, variant in groups]
    for scope in ('image', 'patch'):
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, metric in zip(axes.flat, ('brightness_mean', 'contrast_std', 'local_contrast_rms',
                                         'detail_l1', 'gradient_normalized', 'blur_scale_ratio')):
            values = [[row[metric] for row in rows if row['cohort'] == cohort and
                       row['variant'] == variant and row['scope'] == scope] for cohort, variant in groups]
            ax.boxplot(values, labels=names, showfliers=False, whis=(5, 95))
            ax.set_title(metric)
            ax.tick_params(axis='x', labelsize=7)
            ax.grid(axis='y', alpha=.2)
        figure.suptitle(f'Appearance only / {scope} / box 25-75%, whisker 5-95% / NOT accuracy')
        figure.tight_layout()
        figure.savefig(directory / f'distribution_{scope}.png', dpi=140)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_d5a.yaml')
    parser.add_argument('--train-dir', default='data/unlabeled')
    parser.add_argument('--output-dir', default='outputs/semantic_domain')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--epoch', type=int, default=10003)
    parser.add_argument('--grid', type=int, default=4)
    parser.add_argument('--expected-train', type=int, default=1000)
    parser.add_argument('--expected-test', type=int, default=100)
    args = parser.parse_args()
    if args.grid < 1 or args.seed < 0 or args.epoch < 0:
        raise ValueError('grid must be positive and seed/epoch nonnegative')
    config = load_config(args.config)
    cfg = config['semantic_adaptation']
    directory = Path(project_path(config, args.output_dir)).resolve()
    # 私有产物必须放在仓库约定的忽略目录下，不接受 docs 或随意改名绕过。
    root = Path(project_path(config)).resolve()
    if not any(directory.is_relative_to(root / folder) for folder in ('output', 'outputs')):
        raise ValueError('diagnostic outputs must remain in private output/ or outputs/')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'metrics.csv').exists():
        raise FileExistsError('existing diagnostics; select a new output directory')
    train_files = [Path(path) for path in list_images(project_path(config, args.train_dir))]
    test_files = [Path(path) for path in list_images(project_path(config, config['inference']['test_dir']))]
    if (len(train_files), len(test_files)) != (args.expected_train, args.expected_test):
        raise ValueError(f'unexpected source counts: {len(train_files)} train, {len(test_files)} test')
    source_mapping = verify_cohorts(train_files, Path(project_path(config, config['paths']['raw_data_dir'])),
                                    cfg['expected_manual_sources'])
    manual = {Path(pair['pool_name']).stem for pair in source_mapping}
    if {p.stem for p in train_files} & {p.stem for p in test_files}:
        raise ValueError('training and test source stems overlap')
    restoration = cfg['restoration']
    checkpoint = Path(project_path(config, restoration['checkpoint']))
    checkpoint_sha = file_sha(checkpoint)
    if checkpoint_sha != restoration['checkpoint_sha256'] or restoration['mode'] != 'first_x0':
        raise ValueError('diagnostic must use the designated frozen D5a first_x0 checkpoint')
    degradation_path = project_path(config, restoration['degradation_config'])
    degradation = load_config(degradation_path)['rgb_restoration']['degradation']
    base = ImagePool(train_files)
    degraded = PairedDegradationDataset(base, degradation, cfg['noise'], args.seed, repeats=1, augment=True)
    degraded.set_epoch(args.epoch)
    manifest = {
        'purpose': 'one_time_unlabeled_appearance_diagnostic_not_validation_or_checkpoint_selection',
        'seed': args.seed, 'degradation_epoch': args.epoch, 'degradation_draws_per_source': 1,
        'image_size': 1024, 'valid_grid': args.grid, 'training_sources': len(train_files),
        'manual_sources': len(manual), 'other_training_sources': len(train_files) - len(manual),
        'test_sources': len(test_files), 'checkpoint': restoration['checkpoint'], 'checkpoint_sha256': checkpoint_sha,
        'restoration_mode': 'FP32_first_x0', 'restoration_seed_rule': 'inference_seed + original sorted pool index; paired raw/degraded use same seed',
        'degradation': degradation, 'noise': cfg['noise'], 'config': config,
        'train_names': [p.name for p in train_files], 'test_names': [p.name for p in test_files],
        'manual_pool_stems': sorted(manual), 'manual_source_mapping': source_mapping,
        'manual_source_files_sha_equal_pool': True,
        'no_GT_values_read': True, 'no_test_training': True, 'no_training_holdout': True,
        'limitations': [
            'Appearance overlap is not segmentation accuracy and may reflect phase proportions or grain scale.',
            'Blur scale ratio is only a contrast-normalized texture proxy, not optical blur ground truth.',
            'One synthetic draw per source checks broad coverage; it does not enumerate every online training perturbation.',
            'Nearest-neighbor reference threshold excludes the same source but is not a model validation metric.',
            'Patch rows within an image are correlated and are not independent statistical samples.',
        ],
    }
    write_json(directory / 'manifest.json', manifest)
    from models.rgb_restoration import load_rgb_restorer
    from models.backend_adaptation import restore_first
    restorer = load_rgb_restorer(str(checkpoint), args.device).float().eval().requires_grad_(False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    started, rows, draws = time.time(), [], []
    stream = (directory / 'metrics.csv').open('w', newline='', encoding='utf-8')
    writer = None

    def collect(image, valid, cohort, name, variant):
        nonlocal writer
        rgb = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        mask = valid.detach().cpu().numpy()
        values = [{'cohort': cohort, 'name': name, 'variant': variant, **row}
                  for row in appearance_rows(rgb, mask, args.grid)]
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
        writer.writerows(values)
        stream.flush()
        rows.extend(values)

    try:
        with torch.inference_mode():
            for cohort_name, files in (('train', train_files), ('test', test_files)):
                pool = base if cohort_name == 'train' else ImagePool(files)
                for index, path in enumerate(files):
                    sample = pool[index]
                    cohort = ('manual' if path.stem in manual else 'other_train') if cohort_name == 'train' else 'test'
                    seed = int(restoration['inference_seed']) + index
                    image = sample['image'].unsqueeze(0).to(args.device)
                    collect(image, sample['valid_content'], cohort, path.name, 'raw')
                    restored = restore_first(restorer, image, seed)
                    collect(restored, sample['valid_content'], cohort, path.name, 'd5a')
                    del restored
                    if cohort_name == 'train':
                        item = degraded[index]
                        image = item['image'].unsqueeze(0).to(args.device)
                        collect(image, item['valid_content'], cohort, path.name, 'degraded')
                        restored = restore_first(restorer, image, seed)
                        collect(restored, item['valid_content'], cohort, path.name, 'degraded_d5a')
                        draws.append({key: item[key] for key in (
                            'name', 'profile', 'is_spatial', 'endpoint_applied', 'noise_sigma', 'base_sigma',
                            'draw_index', 'source_index', 'horizontal_flip', 'vertical_flip', 'rotation_k')})
                        del restored
                    if (index + 1) % 25 == 0 or index + 1 == len(files):
                        print(json.dumps({'cohort': cohort_name, 'completed': index + 1,
                                          'sources': len(files), 'seconds': round(time.time() - started, 1)}), flush=True)
    finally:
        stream.close()
    write_json(directory / 'degradation_draws.json', draws)
    report = summarize(rows)
    report.update(elapsed_seconds=time.time() - started, row_count=len(rows),
                  synthetic_profiles={profile: sum(row['profile'] == profile for row in draws)
                                      for profile in sorted({row['profile'] for row in draws})},
                  synthetic_spatial_sources=sum(row['is_spatial'] for row in draws),
                  complete=True, checkpoint_sha256=checkpoint_sha)
    write_json(directory / 'report.json', report)
    plot_distributions(rows, directory)
    print(json.dumps({'completed': True, 'output': str(directory), 'seconds': round(time.time() - started, 1)}), flush=True)


if __name__ == '__main__':
    main()
