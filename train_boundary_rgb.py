# -*- coding: utf-8 -*-
"""RGB条件NAF边界纠错：容差截面监督＋可信内部抑制＋小权重PatchGAN。

独立实验入口。冻结N1缓存为输入、处理后32张GT为目标，未加载分割后端训练。
随机初始化、可严格续训；只在固定小样本重新生成种子/障碍，不自动全150图。
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.boundary_rgb import BoundaryRGBDataset, collate_boundary_rgb
from models.boundary_naf import BoundaryDiscriminator
from models.boundary_rgb import BoundaryRGBNAFNet
from models.boundary_restore import state_sha256
from train_boundary_restore import capture_rng, restore_rng, canonical_sha, file_sha, write_json, reconcile_log
from utils.boundary_rgb_training import (
    build_tolerant_targets, coherent3_coverage, directional_coverage_logits,
    rgb_gan_update, tolerant_boundary_loss,
)
from utils.config import load_config, project_path

FORMAT = 'boundary_rgb_training_v1'


def erasure_enabled(cfg):
    """旧配方缺省保持关闭；擦断需显式声明，不能借旧gap字段静默开启。"""
    settings = cfg.get('erasure')
    return isinstance(settings, dict) and settings.get('enabled') is True


def shift_consistency_enabled(cfg):
    """新约束只由显式配置启用；旧控制不导入或改变额外行为。"""
    settings = cfg.get('shift_consistency')
    return isinstance(settings, dict) and settings.get('enabled') is True


def ridge_supervision_enabled(cfg):
    """新增形状监督仅由明确开关启用，旧配方和回执保持原样。"""
    settings = cfg.get('ridge_supervision')
    return isinstance(settings, dict) and settings.get('enabled') is True


def ridge_reconstruction_metrics(logits, batch, cfg, *, prepared=None):
    """固定批形状诊断；只对脱离G的logits测独立梯度，不改模型或参数梯度。"""
    from utils.boundary_ridge_loss import (
        build_ridge_targets, ridge_supervision_loss, ridge_supervision_settings,
    )
    if 'ridge_instances' not in batch:
        raise ValueError('法向单连通诊断缺少同坐标的ridge_instances')
    settings = ridge_supervision_settings(cfg['ridge_supervision'])
    if prepared is None:
        prepared = build_tolerant_targets(batch['target'], batch['valid'],
                                         tolerance=cfg['rgb_loss']['tolerance'])
    targets = build_ridge_targets(prepared, batch['ridge_instances'], settings)
    # reconstruction_metrics及其调用者处于no_grad中，局部开启仅用于此项对输出的导数。
    with torch.enable_grad():
        independent = logits.detach().requires_grad_(True)
        loss, audit = ridge_supervision_loss(independent, targets)
        gradient = torch.autograd.grad(settings['weight'] * loss, independent)[0]
    norm = float(gradient.detach().norm())
    if not math.isfinite(norm):
        raise FloatingPointError('法向单连通监督的独立logits梯度非有限')
    result = scalar_audit(audit)
    result.update(ridge_loss=float(loss.detach()),
                  ridge_weighted=settings['weight'] * float(loss.detach()),
                  ridge_logit_grad_norm=norm, ridge_supervision_enabled=True,
                  ridge_gradient_scope='detached_logits_only_not_G_parameter_proof')
    return result


def ridge_smoke_checks(after, nonzero_objective_updates, independent_gradient_max):
    """新项确实可用且曾产生监督；终态为零本身可能是已经满足约束。"""
    return dict(ridge_eligible_present=after.get('ridge_eligible_points', 0) > 0,
        ridge_nonzero_objective=nonzero_objective_updates > 0,
        ridge_independent_logit_gradient=(math.isfinite(independent_gradient_max)
                                         and independent_gradient_max > 0))


def erasure_batch_audit(batch):
    """只汇总输入任务回执，不参与loss或改变其分母。"""
    metadata = batch.get('meta', [])
    if not metadata or not any('erasure' in item for item in metadata):
        return {}
    if not all('erasure' in item for item in metadata):
        raise ValueError('同批擦断回执缺失，不能把未记录的样本算作未擦')
    records = [item['erasure'] for item in metadata]
    reasons = Counter(record['reason'] for record in records)
    failed = Counter(record['reason'] for record in records if record['requested'] and not record['applied'])
    provenance = Counter(segment['provenance'] for record in records for segment in record['segments'])
    return dict(erasure_requested_samples=sum(bool(record['requested']) for record in records),
        erasure_applied_samples=sum(bool(record['applied']) for record in records),
        erasure_deleted_input_pixels=sum(int(record['deleted_input_pixels']) for record in records),
        erasure_artificial_gap_gt_points=sum(int(record['artificial_gap_gt_points']) for record in records),
        erasure_requested_segments=sum(int(record['requested_count']) for record in records),
        erasure_applied_segments=sum(int(record['applied_count']) for record in records),
        erasure_reason_counts=dict(sorted(reasons.items())),
        erasure_failed_request_reasons=dict(sorted(failed.items())),
        erasure_segments_by_provenance=dict(sorted(provenance.items())))


def scalar_audit(stats):
    """loss诊断含可反传的标量分项；写回执时显式脱离计算图。"""
    return {key: float(value.detach()) if torch.is_tensor(value) else value
            for key, value in stats.items()}


@torch.no_grad()
def reconstruction_metrics(logits, batch, cfg, *, clean_logits=None):
    """同一固定批的无更新验收，不允许只靠背景loss下降放行。"""
    tolerance = cfg['rgb_loss']['tolerance']
    mode = cfg['rgb_loss'].get('coverage_mode', 'pointwise')
    prepared = build_tolerant_targets(batch['target'], batch['valid'], tolerance=tolerance)
    loss, stats = tolerant_boundary_loss(logits, batch['target'], batch['valid'], prepared=prepared,
                                         coverage_mode=mode)
    peaks = directional_coverage_logits(logits, prepared['normal_index'], prepared['valid'],
                                        tolerance, prepared['fallback']).sigmoid()
    coherent_peaks = None
    if mode == 'coherent3':
        common = coherent3_coverage(logits, prepared)
        coherent_peaks = torch.where(common['eligible'], common['probabilities'], peaks)
    binary = batch['input'][:, -1:]
    near = torch.nn.functional.max_pool2d(binary, 2 * tolerance + 1, stride=1, padding=tolerance) >= .5
    missing = prepared['positive'] & ~near
    supported = prepared['positive'] & near
    recorded_interior_input = prepared['negative'] & (binary >= .5)
    result = dict(loss=float(loss), **scalar_audit(stats))
    for name, mask, values in (('missing', missing, peaks), ('supported', supported, peaks),
                               ('recorded_interior_input', recorded_interior_input, logits.sigmoid())):
        count = int(mask.sum())
        result[name + '_points'] = count
        result[name + '_probability'] = float(values[mask].mean()) if count else None
        result[name + '_above065'] = float((values[mask] > .65).float().mean()) if count else None
        if mode == 'coherent3' and name != 'recorded_interior_input':
            result['coherent3_' + name + '_probability'] = float(coherent_peaks[mask].mean()) if count else None
            result['coherent3_' + name + '_above065'] = float((coherent_peaks[mask] > .65).float().mean()) if count else None
    if erasure_enabled(cfg):
        clean = batch['clean_input_boundary']
        if tuple(clean.shape) != tuple(binary.shape):
            raise ValueError('擦断前输入必须与实际输入同坐标同shape')
        clean_near = torch.nn.functional.max_pool2d(clean, 2 * tolerance + 1,
                                                   stride=1, padding=tolerance) >= .5
        erased = batch['erasure_mask'].bool()
        artificial = batch['artificial_gap_mask'].bool() & prepared['positive']
        cohorts = {
            'natural_missing': prepared['positive'] & ~clean_near,
            'artificial_missing': artificial,
            'original_supported': prepared['positive'] & clean_near,
            'untouched_supported': prepared['positive'] & clean_near & ~(
                torch.nn.functional.max_pool2d(erased.float(), 2 * tolerance + 1,
                                               stride=1, padding=tolerance) > 0),
        }
        clean_peaks = None
        if clean_logits is not None:
            if tuple(clean_logits.shape) != tuple(logits.shape):
                raise ValueError('干净输入预测shape与擦断预测不同')
            clean_peaks = directional_coverage_logits(clean_logits, prepared['normal_index'],
                prepared['valid'], tolerance, prepared['fallback']).sigmoid()
        for name, mask in cohorts.items():
            count = int(mask.sum())
            result[name + '_points'] = count
            result[name + '_probability'] = float(peaks[mask].mean()) if count else None
            result[name + '_above065'] = float((peaks[mask] > .65).float().mean()) if count else None
            if clean_peaks is not None:
                result['clean_output_' + name + '_probability'] = float(clean_peaks[mask].mean()) if count else None
                result['clean_output_' + name + '_above065'] = float((clean_peaks[mask] > .65).float().mean()) if count else None
        result.update(erasure_enabled=True, missing_definition='actual input missing; includes artificial and pre-existing gaps',
                      natural_missing_definition='same clean T8 cohort before erasure', **erasure_batch_audit(batch))
    if ridge_supervision_enabled(cfg):
        result.update(ridge_reconstruction_metrics(logits, batch, cfg, prepared=prepared))
    return result


def smoke_checks(before, after, exact_reload, discriminator_updates, *, adversarial_weight=None,
                 discriminator_state_unchanged=False, discriminator_optimizer_unchanged=False,
                 discriminator_forward_calls=None, generator_updates=None):
    """只验固定批任务可学，不把这组条件当泛化/官方收益。"""
    checks = dict(exact_reload=bool(exact_reload), discriminator_updated=discriminator_updates > 0,
        reconstruction_reduced=math.isfinite(after['loss']) and after['loss'] < before['loss'] * .9,
        natural_missing_present=before['missing_points'] > 0,
        supported_present=before['supported_points'] > 0)
    if type(adversarial_weight) in (int, float) and adversarial_weight == 0:
        # 只为显式零GAN配方替换D门槛：不更新不是放弃核验，而是要验证D全程未参与。
        checks.pop('discriminator_updated')
        checks.update(discriminator_not_updated=discriminator_updates == 0,
            discriminator_state_unchanged=bool(discriminator_state_unchanged),
            discriminator_optimizer_unchanged=bool(discriminator_optimizer_unchanged),
            discriminator_never_forwarded=discriminator_forward_calls == 0,
            generator_updated=generator_updates is not None and generator_updates > 0)
    for name, increasing in (('missing', True), ('supported', True), ('recorded_interior_input', False)):
        # 原字段保留pointwise含义；新实验的放行条件显式使用对应三点目标。
        prefix = 'coherent3_' if after.get('coverage_mode') == 'coherent3' and name != 'recorded_interior_input' else ''
        a, b = before[prefix + name + '_probability'], after[prefix + name + '_probability']
        if a is None or b is None:
            checks[name + '_progress'] = name == 'recorded_interior_input'
        elif name == 'missing':
            checks[name + '_progress'] = b >= .65 and (b >= a + .05 or b >= .95)
        elif increasing:
            checks[name + '_progress'] = b >= a - .02 or b >= .95
        else:
            checks[name + '_progress'] = b <= a - .05 or b <= .05
    if after.get('erasure_enabled'):
        checks['natural_missing_present'] = before['natural_missing_points'] > 0
        checks['artificial_missing_present'] = before['artificial_missing_points'] > 0
        a, b = before['artificial_missing_probability'], after['artificial_missing_probability']
        checks['artificial_missing_progress'] = (a is not None and b is not None
            and b >= .65 and (b >= a + .05 or b >= .95))
        a, b = before['clean_output_original_supported_probability'], after['clean_output_original_supported_probability']
        checks['clean_original_support_preserved'] = a is not None and b is not None and (b >= a - .02 or b >= .95)
    return checks


@torch.no_grad()
def shift_reconstruction_metrics(generator, batch, cfg, *, base_probability=None):
    """四向移位后仍需恢复真线和人工缺口；常量/模糊稳定解不能靠低MAE放行。"""
    from utils.boundary_shift_consistency import (
        build_shift_masks, shift_consistency_settings, shift_integer_reflect, translation_probe,
    )
    settings = shift_consistency_settings(cfg['shift_consistency'])
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1),
              (16, 0), (-16, 0), (0, 16), (0, -16))
    phase = translation_probe(generator, batch['input'], batch['target'], batch['valid'],
                              settings=settings, shifts=shifts, base_probability=base_probability)
    prepared = build_tolerant_targets(batch['target'], batch['valid'],
        tolerance=cfg['rgb_loss']['tolerance'])
    tolerance = cfg['rgb_loss']['tolerance']
    near_clean = torch.nn.functional.max_pool2d(batch['clean_input_boundary'],
                                                2 * tolerance + 1, 1, tolerance) > .5
    flags = [(module, module.training) for module in generator.modules()]
    generator.eval()
    try:
        rows = []
        for dy, dx in shifts[:4]:
            moved = generator(shift_integer_reflect(batch['input'], dy, dx))
            aligned = shift_integer_reflect(moved, -dy, -dx)
            masks = build_shift_masks(prepared, settings, dy, dx)
            inner = masks['inner']
            peaks = torch.nn.functional.max_pool2d(aligned.masked_fill(~prepared['valid'], 0.),
                                                    2 * tolerance + 1, 1, tolerance)
            chosen = dict(original_supported=prepared['positive'] & near_clean & inner,
                artificial_missing=batch['artificial_gap_mask'].bool() & prepared['positive'] & inner,
                trusted_negative=prepared['negative'] & inner)
            row = dict(dy=dy, dx=dx)
            for name, mask in chosen.items():
                n = int(mask.sum())
                values = aligned if name == 'trusted_negative' else peaks
                row[name + '_points'] = n
                row[name + '_probability'] = float(values[mask].mean()) if n else None
                row[name + '_above065'] = float((values[mask] > .65).float().mean()) if n else None
            base_area = int(((batch['clean_input_boundary'] > .5) & masks['band']).sum())
            output_area = int(((aligned > .65) & masks['band']).sum())
            ratio = output_area / base_area if base_area else None
            row.update(output_strong_band_pixels=output_area, clean_T8_band_pixels=base_area,
                       strong_band_area_ratio_to_clean_T8=ratio,
                       strong_band_area_warning=ratio is not None and ratio > 1.25)
            rows.append(row)
        return dict(phase=phase, shifted_reconstruction=rows,
            input='same_damaged_T8_and_unchanged_GT',
            denominator='common central valid field; original support/erasure masks unchanged',
            area_ratio_interpretation='fixed diagnostic warning only, not a physical boundary-width or accuracy gate')
    finally:
        for module, training in flags:
            module.training = training


def shift_smoke_checks(after, nonzero_objective_updates, nonzero_gradient_updates, continued):
    """原任务门槛与一致性分开；不要求比随机常量初态更一致。"""
    rows = after['shift_probe']['shifted_reconstruction']
    return dict(
        shift_consistency_nonzero_objective=nonzero_objective_updates > 0,
        shift_consistency_independent_gradient=nonzero_gradient_updates > 0,
        shift_supported_output_not_collapsed=(after['supported_probability'] is not None
            and after['supported_probability'] >= .95),
        shift_recomputation_matches_first_branch=(
            math.isfinite(continued['shift_recompute_max_error'])
            and continued['shift_recompute_max_error'] <= 1e-6),
        shift_all_directions_keep_original_support=all(
            r['original_supported_points'] > 0 and r['original_supported_above065'] >= .95 for r in rows),
        shift_all_directions_recover_artificial_gap=all(
            r['artificial_missing_points'] > 0 and r['artificial_missing_above065'] >= .95 for r in rows),
        shift_all_directions_protect_trusted_interior=all(
            r['trusted_negative_points'] > 0 and r['trusted_negative_probability'] <= .05 for r in rows),
    )


def validate_config(config):
    cfg = config['boundary_naf']
    t, loss = cfg['training'], cfg['rgb_loss']
    if loss.get('coverage_mode', 'pointwise') not in ('pointwise', 'coherent3'):
        raise ValueError('coverage_mode必须是pointwise或coherent3')
    if cfg.get('input_mode') not in ('boundary', 'rgb_boundary'):
        raise ValueError('必须明确选择boundary或rgb_boundary')
    if cfg.get('input_source') != 'real_n1' or cfg['expected_sources'] != 32:
        raise ValueError('本实验固定全部32份新GT与封存真实N1输入')
    if t['precision'] != 'float32' or t['topology_updates'] != 0:
        raise ValueError('本轮明确FP32，不叠加PH阶段')
    for key in ('main_updates', 'main_schedule_updates', 'segment_updates', 'effective_batch_size',
                'micro_batch_size', 'checkpoint_every_updates', 'log_every_updates'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError('无效训练整数: ' + key)
    if t['main_updates'] > t['main_schedule_updates']:
        raise ValueError('短训不得超过预定学习率日程')
    if (t['effective_batch_size'] % t['micro_batch_size'] or
            t['segment_updates'] * t['effective_batch_size'] != cfg['data']['samples_per_epoch']):
        raise ValueError('有效batch、微批与采样段不一致')
    if type(t['adversarial_warmup_updates']) is not int or t['adversarial_warmup_updates'] < 0:
        raise ValueError('无效GAN热身更新数')
    if (type(t['adversarial_weight']) not in (int, float)
            or not math.isfinite(t['adversarial_weight']) or t['adversarial_weight'] < 0):
        raise ValueError('GAN权重必须是非负有限数；显式0表示不训练或调用判别器')
    for value in (t['learning_rate'], t['discriminator_learning_rate'], t['minimum_free_gib']):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('学习率和保留磁盘空间必须为正')
    if t['bce_weight'] != 0 or t['topology_weight'] != 0:
        raise ValueError('不允许暗中启用旧exact BCE或PH')
    if type(loss['tolerance']) is not int or loss['tolerance'] < 1:
        raise ValueError('位置容差必须明确')
    if (cfg['inference']['input_mode'] != cfg['input_mode'] or
            cfg['inference'].get('boundary_representation') != 'binary' or
            cfg['inference']['input_threshold'] != .5):
        raise ValueError('真实输入固定N1>=0.5二值图及明确的输入模式')
    from utils.boundary_input import input_normalization_settings
    train_normalization = input_normalization_settings(cfg.get('input_normalization'))
    inference_normalization = input_normalization_settings(cfg['inference'].get('input_normalization'))
    if train_normalization != inference_normalization:
        raise ValueError('训练和推理的输入线宽标准化必须相同')
    if 'erasure' in cfg and not isinstance(cfg['erasure'], dict):
        raise ValueError('erasure必须是显式配置对象')
    if 'erasure' in cfg:
        from utils.boundary_erasure import erasure_settings
        erasure_settings(cfg['erasure'])
    if erasure_enabled(cfg):
        if (train_normalization != {'mode': 'skeleton_dilate', 'kernel_size': 8} or
                loss.get('coverage_mode', 'pointwise') != 'pointwise'):
            raise ValueError('显式擦断仅叠加于原8核/pointwise配方')
        if cfg['erasure'].get('version') != 'native_supported_v1':
            raise ValueError('擦断需显式native_supported_v1版本')
    if cfg.get('restored_rgb_manifest') is not None:
        if (cfg['input_mode'] != 'rgb_boundary' or
                not isinstance(cfg['restored_rgb_manifest'], str) or
                not cfg['restored_rgb_manifest'].strip()):
            raise ValueError('restored_rgb_manifest必须是RGB模型的显式非空路径')
        monitor_manifest = cfg['monitor'].get('restored_rgb_manifest')
        if not isinstance(monitor_manifest, str) or not monitor_manifest.strip():
            raise ValueError('修复RGB训练需显式独立monitor.restored_rgb_manifest，禁止测试回退raw RGB')
        if monitor_manifest == cfg['restored_rgb_manifest']:
            raise ValueError('训练和monitor修复RGB清单必须隔离')
        if cfg['inference'].get('rgb_input_format') != 'float32_unit':
            raise ValueError('无损修复RGB推理需显式rgb_input_format=float32_unit')
    elif cfg['monitor'].get('restored_rgb_manifest') is not None:
        raise ValueError('修复RGB的训练/monitor清单必须成对声明')
    if 'shift_consistency' in cfg:
        from utils.boundary_shift_consistency import shift_consistency_settings
        shift = shift_consistency_settings(cfg['shift_consistency'])
        if shift['enabled']:
            if (cfg['input_mode'] != 'boundary' or not erasure_enabled(cfg)
                    or cfg.get('restored_rgb_manifest') is not None):
                raise ValueError('本轮平移一致性仅用于无RGB的单通道擦断控制')
            probe_indices = cfg['monitor'].get('shift_pair_indices', [])
            if (not isinstance(probe_indices, list) or not probe_indices
                    or any(type(index) is not int or index < 0 for index in probe_indices)
                    or len(set(probe_indices)) != len(probe_indices)
                    or not set(probe_indices).issubset(cfg['monitor']['pair_indices'])):
                raise ValueError('位移monitor仅使用已有固定pair中的明确draw')
    if 'ridge_supervision' in cfg:
        from utils.boundary_ridge_loss import ridge_supervision_settings
        ridge = ridge_supervision_settings(cfg['ridge_supervision'])
        if ridge['enabled']:
            if (not erasure_enabled(cfg)
                    or (cfg['input_mode'] == 'rgb_boundary'
                        and (cfg.get('restored_rgb_manifest') is None
                             or cfg['model'].get('upsampling') != 'project_bilinear_v1'))
                    or loss.get('coverage_mode', 'pointwise') != 'pointwise'):
                raise ValueError('法向单连通监督仅追加于bilinear pointwise擦断控制；RGB须显式冻结恢复清单')
            if shift_consistency_enabled(cfg):
                raise ValueError('法向单连通与shift_consistency不得同时启用')
    if cfg['inference'].get('output_stitch') != 'probability':
        raise ValueError('新读出必须直接拼接输出概率，不能混用旧残差接口')
    if cfg['inference']['iterations'] != 1 or cfg['inference']['revert_unconnected']:
        raise ValueError('不额外加入迭代和未连接回退')
    required = {0, 1, t['main_updates']}
    if not required.issubset(set(t['monitor_updates'])):
        raise ValueError('至少保留初态、首步和终态过程图')
    return cfg


def dataset_for(config, epoch):
    cfg = copy.deepcopy(config['boundary_naf'])
    cfg['real_input_manifest'] = str(project_path(config, cfg['real_input_manifest']))
    if cfg.get('restored_rgb_manifest'):
        cfg['restored_rgb_manifest'] = str(project_path(config, cfg['restored_rgb_manifest']))
    return BoundaryRGBDataset(project_path(config, cfg['manifest']), cfg, epoch=epoch)


def build_models(cfg, device):
    g = BoundaryRGBNAFNet(**cfg['model'], input_mode=cfg['input_mode'], seed=int(cfg['seed'])).to(device)
    d = BoundaryDiscriminator(**cfg['discriminator'], seed=int(cfg['seed']) + 1).to(device)
    return g, d


def optimizers(g, d, cfg):
    t = cfg['training']
    common = dict(betas=tuple(t['adam_betas']), eps=float(t['adam_eps']))
    return (torch.optim.Adam(g.parameters(), lr=float(t['learning_rate']), **common),
            torch.optim.Adam(d.parameters(), lr=float(t['discriminator_learning_rate']), **common))


def rates(step, cfg):
    t = cfg['training']
    fraction = min(step / max(1, t['main_schedule_updates'] - 1), 1.)
    ratio = t['minimum_lr_ratio'] + (1 - t['minimum_lr_ratio']) * .5 * (1 + math.cos(math.pi * fraction))
    return float(t['learning_rate'] * ratio), float(t['discriminator_learning_rate'])


def source_identity(config, dataset, g, d):
    root = Path(__file__).resolve().parent
    files = ['train_boundary_rgb.py', 'data/boundary_rgb.py', 'models/boundary_rgb.py',
             'utils/boundary_rgb_training.py', 'tools/boundary_rgb_monitor.py',
             'utils/boundary_input.py',
             'data/boundary_naf.py', 'data/boundary_naf_real.py', 'data/boundary_restore.py',
             'models/boundary_naf.py', 'models/boundary_restore.py',
             'tools/boundary_naf_monitor.py', 'tools/boundary_restore_monitor.py',
             'tools/boundary_restore_readout.py', 'train_boundary_restore.py', 'utils/config.py']
    cfg = config['boundary_naf']
    if erasure_enabled(cfg):
        files.extend(['utils/boundary_erasure.py', 'utils/boundary_core_supervision.py',
                      'utils/boundary_rgb_inputs.py'])
    if shift_consistency_enabled(cfg):
        files.append('utils/boundary_shift_consistency.py')
    if ridge_supervision_enabled(cfg):
        files.append('utils/boundary_ridge_loss.py')
    assets = {}
    names = [cfg['manifest'], cfg['real_input_manifest'], *cfg['monitor'].get('control_manifests', [])]
    if cfg['monitor'].get('roi_spec'):
        names.append(cfg['monitor']['roi_spec'])
    if cfg.get('restored_rgb_manifest'):
        names.append(cfg['restored_rgb_manifest'])
        names.append(cfg['monitor']['restored_rgb_manifest'])
    for name in names:
        p = str(project_path(config, name))
        assets[p] = file_sha(p)
    return dict(format=FORMAT, recipe=copy.deepcopy(cfg), dataset=copy.deepcopy(dataset.identity),
                architecture=copy.deepcopy(g.architecture), discriminator_architecture=copy.deepcopy(d.architecture),
                initial_model_sha256=state_sha256(g), initial_discriminator_sha256=state_sha256(d),
                source_sha256={name: file_sha(root / name) for name in files}, assets_sha256=assets,
                target='processed_GT32_tolerant_boundary_sections_not_N1_copy',
                initialization='random_no_old_restorer_or_supervised_weights', no_test_training=True,
                frozen_backend_not_loaded_for_training=True,
                numerical_contract=dict(precision='float32', cudnn_tf32=False, matmul_tf32=False,
                                        cudnn_benchmark=False, cudnn_deterministic=True),
                versions=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__))


def verify_inputs(config, identity, dataset):
    dataset.verify_sources()
    root = Path(__file__).resolve().parent
    for name, sha in identity['source_sha256'].items():
        if file_sha(root / name) != sha:
            raise RuntimeError('运行期间源码改变: ' + name)
    for path, sha in identity['assets_sha256'].items():
        if file_sha(path) != sha:
            raise RuntimeError('运行期间监督/缓存/控制清单改变: ' + path)


def check_space(output, cfg):
    free = shutil.disk_usage(output).free / 1024 ** 3
    if free < float(cfg['training']['minimum_free_gib']):
        raise RuntimeError(f'磁盘安全停止：剩余{free:.2f}GiB，保留latest，不写大文件或删除旧资产')
    return free


def save_checkpoint(path, g, d, go, do, cfg, identity, step, monitors):
    payload = dict(format=FORMAT, architecture=copy.deepcopy(g.architecture),
                   model_state_dict={k: v.detach().cpu().contiguous() for k, v in g.state_dict().items()},
                   discriminator_architecture=copy.deepcopy(d.architecture),
                   discriminator_state_dict={k: v.detach().cpu().contiguous() for k, v in d.state_dict().items()},
                   model_sha256=state_sha256(g), discriminator_sha256=state_sha256(d),
                   optimizer=go.state_dict(), discriminator_optimizer=do.state_dict(),
                   identity=identity, identity_sha256=canonical_sha(identity), global_step=int(step),
                   next_rates=rates(step, cfg), monitor_steps=list(monitors), rng=capture_rng(),
                   config=copy.deepcopy(cfg), input_contract=copy.deepcopy(cfg['inference']))
    path = Path(path)
    check_space(path.parent, cfg)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, g, d, go, do, cfg, identity):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('format') != FORMAT or saved.get('identity') != identity or
            saved.get('identity_sha256') != canonical_sha(identity)):
        raise ValueError('checkpoint配方、来源或源码不一致')
    step = saved['global_step']
    if (type(step) is not int or not 0 <= step <= cfg['training']['main_updates'] or
            saved['architecture'] != g.architecture or saved['discriminator_architecture'] != d.architecture or
            tuple(saved['next_rates']) != rates(step, cfg) or saved['input_contract'] != cfg['inference']):
        raise ValueError('checkpoint架构、日程或输入合同不一致')
    g.load_state_dict(saved['model_state_dict'], strict=True)
    d.load_state_dict(saved['discriminator_state_dict'], strict=True)
    if state_sha256(g) != saved['model_sha256'] or state_sha256(d) != saved['discriminator_sha256']:
        raise RuntimeError('模型严格重载哈希不一致')
    go.load_state_dict(saved['optimizer'])
    do.load_state_dict(saved['discriminator_optimizer'])
    restore_rng(saved['rng'])
    return step, list(saved['monitor_steps'])


def monitor(g, config, output, step, dataset, **kwargs):
    from tools.boundary_rgb_monitor import save_monitor
    before, rng, mode = state_sha256(g), capture_rng(), g.training
    try:
        g.eval()
        with torch.no_grad():
            result = save_monitor(g, config, output, step, dataset, **kwargs)
        if state_sha256(g) != before:
            raise RuntimeError('monitor修改模型状态')
        return result
    finally:
        restore_rng(rng)
        g.train(mode)


def loader(dataset, start, cfg):
    t = cfg['training']
    workers = int(t['num_workers'])
    args = dict(batch_size=t['effective_batch_size'], shuffle=False, collate_fn=collate_boundary_rgb,
                num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=True,
                generator=torch.Generator().manual_seed(cfg['seed'] + dataset.epoch + 117))
    if workers:
        args.update(prefetch_factor=int(t['prefetch_factor']), persistent_workers=False)
    return DataLoader(Subset(dataset, range(start, len(dataset))), **args)


def update(g, d, go, do, batch, cfg, step):
    t, loss = cfg['training'], cfg['rgb_loss']
    glr, dlr = rates(step, cfg)
    for optimizer, rate in ((go, glr), (do, dlr)):
        for group in optimizer.param_groups:
            group['lr'] = rate
    adv = t['adversarial_weight'] if step >= t['adversarial_warmup_updates'] else 0.
    additional = {}
    if shift_consistency_enabled(cfg):
        additional = dict(shift_consistency=cfg['shift_consistency'], consistency_step=step)
    if ridge_supervision_enabled(cfg):
        additional['ridge_supervision'] = cfg['ridge_supervision']
    stats = rgb_gan_update(g, d, go, do, batch, micro_batch_size=t['micro_batch_size'],
        coverage_weight=loss['coverage_weight'], interior_weight=loss['interior_weight'],
        adversarial_weight=adv, tolerance=loss['tolerance'], grad_clip=t['grad_clip'],
        coverage_mode=loss.get('coverage_mode', 'pointwise'), **additional)
    if erasure_enabled(cfg) and cfg['input_mode'] == 'rgb_boundary':
        # 只读更新后的梯度/权重，证明新增条件通道确有学习；不改loss或optimizer。
        gradient = g.intro.weight.grad
        norm = float(gradient[:, :3].detach().norm()) if gradient is not None else 0.
        if not math.isfinite(norm):
            raise FloatingPointError('RGB入口梯度非有限')
        stats.update(rgb_intro_grad_norm=norm,
                     rgb_intro_weights_l2=float(g.intro.weight[:, :3].detach().norm()))
    return dict(learning_rate=glr, discriminator_learning_rate=dlr, effective_adversarial_weight=adv,
                **stats, **erasure_batch_audit(batch))


def preflight(config, output, device='cpu'):
    cfg = validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = dataset_for(config, 1)
    g, d = build_models(cfg, device)
    identity = source_identity(config, dataset, g, d)
    batch = collate_boundary_rgb([dataset[i] for i in range(32)])
    if not batch['valid'].bool().any() or not batch['gap_mask'].bool().any():
        raise RuntimeError('真实数据预检缺少可靠监督或天然缺失GT')
    # 无参数更新，只核对真实输入/目标与新loss可用性。
    with torch.no_grad():
        logits = g.forward_logits(batch['input'][:2].to(device))
        _, stats = tolerant_boundary_loss(logits, batch['target'][:2].to(device),
            batch['valid'][:2].to(device), tolerance=cfg['rgb_loss']['tolerance'],
            coverage_mode=cfg['rgb_loss'].get('coverage_mode', 'pointwise'))
        if shift_consistency_enabled(cfg):
            from utils.boundary_shift_consistency import translation_probe
            shift_diagnostic = translation_probe(g, batch['input'][:2].to(device),
                batch['target'][:2].to(device), batch['valid'][:2].to(device),
                settings=cfg['shift_consistency'], base_probability=logits.sigmoid())
        if ridge_supervision_enabled(cfg):
            small = {key: value[:2] if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            small = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in small.items()}
            ridge_diagnostic = ridge_reconstruction_metrics(logits, small, cfg)
            from utils.boundary_ridge_loss import build_ridge_targets, ridge_supervision_settings
            prepared = build_tolerant_targets(batch['target'], batch['valid'],
                                              tolerance=cfg['rgb_loss']['tolerance'])
            all_ridge = build_ridge_targets(prepared, batch['ridge_instances'],
                ridge_supervision_settings(cfg['ridge_supervision']))
            ridge_batch_eligible = int(all_ridge['eligible'].sum())
            if ridge_batch_eligible < 1:
                raise RuntimeError('真实首批32样本没有可信法向横截面，不能空监督启动')
    if cfg['rgb_loss'].get('coverage_mode', 'pointwise') == 'coherent3' and not stats['coherent3_eligible_gt_points']:
        raise RuntimeError('真实配对预检没有可用的可信同向三点，不能启动coherent3实验')
    patches = d.valid_patch_mask(batch['valid'].to(device)).sum().item()
    if patches < 1:
        raise RuntimeError('没有未知域安全的判别器感受野')
    verify_inputs(config, identity, dataset)
    result = dict(complete=True, no_parameter_updates=True, initialization_identity=identity,
                  plan=dataset.plan_receipt(), samples=32, diagnostic=scalar_audit(stats),
                  gap_pixels=int(batch['gap_mask'].sum()), adversarial_patches=int(patches),
                  generator_parameters=sum(p.numel() for p in g.parameters()),
                  discriminator_parameters=sum(p.numel() for p in d.parameters()),
                  free_gib=check_space(output, cfg))
    if erasure_enabled(cfg):
        erasure = erasure_batch_audit(batch)
        if erasure['erasure_applied_samples'] < 1 or erasure['erasure_artificial_gap_gt_points'] < 1:
            raise RuntimeError('首批32样本没有实际擦断/新增缺口，不能以legacy gap配置假启动')
        result.update(erasure_training_enabled=True, erasure=erasure,
                      erasure_definition='only input T8 changed; target/valid/source-box-D4 plan fixed',
                      legacy_identity_and_gap_settings_applied=False,
                      corruption_switch='boundary_naf.erasure.enabled; inherited legacy gap settings remain ignored')
    if shift_consistency_enabled(cfg):
        result.update(shift_consistency=copy.deepcopy(cfg['shift_consistency']),
                      shift_diagnostic=shift_diagnostic)
    if ridge_supervision_enabled(cfg):
        result.update(ridge_supervision=copy.deepcopy(cfg['ridge_supervision']),
                      ridge_diagnostic=ridge_diagnostic,
                      ridge_batch_eligible_points=ridge_batch_eligible,
                      ridge_batch_positions=ridge_batch_eligible * (2 * cfg['rgb_loss']['tolerance'] + 1))
    write_json(output / 'preflight.json', result)
    return result


def smoke(config, output, device='cuda', updates=128):
    cfg = validate_config(config)
    if updates < cfg['training']['adversarial_warmup_updates'] + 2:
        raise ValueError('真实短测必须跨过GAN热身并验证实际D更新')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'latest.pt').exists():
        raise ValueError('不覆盖已有短测权重')
    dataset = dataset_for(config, 1)
    g, d = build_models(cfg, device)
    go, do = optimizers(g, d, cfg)
    identity = source_identity(config, dataset, g, d)
    gan_disabled = cfg['training']['adversarial_weight'] == 0
    if gan_disabled:
        g_initial_sha = state_sha256(g)
        d_initial_sha = state_sha256(d)
        d_initial_optimizer = copy.deepcopy(do.state_dict())
        d_forward_counter = [0]
        def count_d_forward(module, args):
            d_forward_counter[0] += 1
        d_forward_handle = d.register_forward_pre_hook(count_d_forward)
        generator_updates = 0
    samples = [dataset[i] for i in range(cfg['training']['effective_batch_size'])]
    batch = {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in collate_boundary_rgb(samples).items()}
    ridge_gradient_max = 0.
    def measure():
        nonlocal ridge_gradient_max
        g.eval()
        with torch.no_grad():
            logits = g.forward_logits(batch['input'])
            if erasure_enabled(cfg):
                from utils.boundary_rgb_inputs import clean_model_input
                clean_input = clean_model_input(batch, input_mode=cfg['input_mode'])
                clean_logits = g.forward_logits(clean_input)
            else:
                clean_logits = None
            stat = reconstruction_metrics(logits, batch, cfg, clean_logits=clean_logits)
            if ridge_supervision_enabled(cfg):
                ridge_gradient_max = max(ridge_gradient_max, stat['ridge_logit_grad_norm'])
            if shift_consistency_enabled(cfg):
                stat['shift_probe'] = shift_reconstruction_metrics(g, batch, cfg,
                    base_probability=logits.sigmoid())
        return stat
    before = measure()
    began = time.perf_counter()
    d_updates = 0
    rgb_grad_updates = 0
    consistency_nonzero_updates = 0
    consistency_gradient_updates = 0
    ridge_nonzero_updates = 0
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for step in range(updates):
        stat = update(g, d, go, do, batch, cfg, step)
        if gan_disabled:
            generator_updates += int(stat.get('generator_updated', False))
        d_updates += int(stat.get('discriminator_updated', False))
        rgb_grad_updates += int(stat.get('rgb_intro_grad_norm', 0.) > 0)
        consistency_nonzero_updates += int(stat.get('shift_consistency_weighted', 0.) > 0)
        consistency_gradient_updates += int(stat.get('shift_second_branch_probe_grad_norm', 0.) > 0)
        ridge_nonzero_updates += int(stat.get('ridge_weighted', 0.) > 0)
        with (output / 'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(global_step=step + 1, **stat), allow_nan=False) + '\n')
        if step in (0, 15, 63, updates - 1):
            print(json.dumps(dict(event='smoke_progress', step=step + 1, metrics=measure()), allow_nan=False), flush=True)
    after = measure()
    save_checkpoint(output / 'latest.pt', g, d, go, do, cfg, identity, updates, [])
    # 真正重建模型/优化器，严格重载及下一步训练；短测权重不进入正式初始化。
    fresh_g, fresh_d = build_models(cfg, device)
    fresh_go, fresh_do = optimizers(fresh_g, fresh_d, cfg)
    loaded, _ = load_checkpoint(output / 'latest.pt', fresh_g, fresh_d, fresh_go, fresh_do, cfg, identity)
    with torch.no_grad():
        g.eval()
        fresh_g.eval()
        exact = torch.equal(g(batch['input'][:1]), fresh_g(batch['input'][:1]))
    if gan_disabled:
        g_state_changed = state_sha256(g) != g_initial_sha
        d_after_smoke_sha = state_sha256(d)
        d_optimizer_unchanged = (not do.state and do.state_dict() == d_initial_optimizer
                                 and not fresh_do.state and fresh_do.state_dict() == d_initial_optimizer)
        d_forward_handle.remove()
        resumed_forward_handle = fresh_d.register_forward_pre_hook(count_d_forward)
    del g, d, go, do
    continued = update(fresh_g, fresh_d, fresh_go, fresh_do, batch, cfg, loaded)
    if gan_disabled:
        d_after_resume_sha = state_sha256(fresh_d)
        resumed_forward_handle.remove()
        d_state_unchanged = d_initial_sha == d_after_smoke_sha == d_after_resume_sha
        d_optimizer_unchanged = (d_optimizer_unchanged and not fresh_do.state
                                 and fresh_do.state_dict() == d_initial_optimizer)
        checks = smoke_checks(before, after, exact, d_updates,
            adversarial_weight=0, discriminator_state_unchanged=d_state_unchanged,
            discriminator_optimizer_unchanged=d_optimizer_unchanged,
            discriminator_forward_calls=d_forward_counter[0], generator_updates=generator_updates)
        checks['all_generator_updates_present'] = generator_updates == updates and continued['generator_updated']
        checks['generator_state_changed'] = g_state_changed
        checks['resumed_discriminator_not_updated'] = not continued['discriminator_updated']
    else:
        checks = smoke_checks(before, after, exact, d_updates)
    if erasure_enabled(cfg) and cfg['input_mode'] == 'rgb_boundary':
        checks['RGB_intro_received_nonzero_gradients'] = rgb_grad_updates > 0
        checks['RGB_intro_weights_updated_from_zero'] = continued['rgb_intro_weights_l2'] > 0
    if shift_consistency_enabled(cfg):
        # 随机初态近常数，本身就可能低MAE，不能以比随机初态更小为放行条件。
        checks.update(shift_smoke_checks(after, consistency_nonzero_updates,
                                         consistency_gradient_updates, continued))
    if ridge_supervision_enabled(cfg):
        checks.update(ridge_smoke_checks(after, ridge_nonzero_updates, ridge_gradient_max))
    passed = all(checks.values())
    result = dict(complete=True, passed=passed, purpose='fixed_training_batch_not_generalization',
                  updates=updates, before=before, after=after, exact_reload=exact,
                  checks=checks, discriminator_updates=d_updates, resumed_update=continued,
                  elapsed_seconds=time.perf_counter() - began,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3 if str(device).startswith('cuda') else None,
                  initialization_identity=identity, no_formal_weight_reuse=True)
    if gan_disabled:
        result.update(gan_disabled=True, discriminator_state_unchanged=d_state_unchanged,
            discriminator_optimizer_unchanged=d_optimizer_unchanged,
            discriminator_forward_calls=d_forward_counter[0], generator_updates=generator_updates,
            generator_state_changed=g_state_changed,
            discriminator_initial_sha256=d_initial_sha, discriminator_after_smoke_sha256=d_after_smoke_sha,
            discriminator_after_resume_sha256=d_after_resume_sha)
    if erasure_enabled(cfg) and cfg['input_mode'] == 'rgb_boundary':
        result.update(rgb_intro_nonzero_gradient_updates=rgb_grad_updates)
    if shift_consistency_enabled(cfg):
        result.update(shift_consistency_nonzero_objective_updates=consistency_nonzero_updates,
            shift_consistency_independent_gradient_updates=consistency_gradient_updates,
            shift_probe_interpretation='Single-window diagnostic, not accuracy; low MAE alone can be constant-output collapse')
    if ridge_supervision_enabled(cfg):
        result.update(ridge_supervision=copy.deepcopy(cfg['ridge_supervision']),
            ridge_nonzero_objective_updates=ridge_nonzero_updates,
            ridge_independent_logit_gradient_max=ridge_gradient_max,
            ridge_gradient_scope='detached_logits_only_not_G_parameter_proof')
    verify_inputs(config, identity, dataset)
    write_json(output / 'smoke.json', result)
    if not passed:
        raise RuntimeError('短测未通过真实更新/重载/固定批恢复损失门槛，不启动正式短训')
    return result


def train(config, output, device='cuda', resume=None):
    cfg = validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset, fixed = dataset_for(config, 1), dataset_for(config, 0)
    g, d = build_models(cfg, device)
    go, do = optimizers(g, d, cfg)
    identity = source_identity(config, dataset, g, d)
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('输出目录属于不同实验')
    if not resume and ((output / 'latest.pt').exists() or (output / 'steps.jsonl').exists()):
        raise ValueError('存在已有训练，须显式续训')
    write_json(output / 'identity.json', identity)
    write_json(output / 'config.json', config)
    step, monitors = load_checkpoint(resume, g, d, go, do, cfg, identity) if resume else (0, [])
    if (output / 'steps.jsonl').exists():
        reconcile_log(output, step)
    total, segment, batch_size = (cfg['training'][k] for k in ('main_updates', 'segment_updates', 'effective_batch_size'))
    began = time.perf_counter()
    if step == 0 and 0 not in monitors:
        monitor(g, config, output, 0, fixed)
        monitors.append(0)
        save_checkpoint(output / 'initial.pt', g, d, go, do, cfg, identity, 0, monitors)
    print(json.dumps(dict(event='formal_started', step=step, total_updates=total,
        input_mode=cfg['input_mode'], initial_model_sha256=identity['initial_model_sha256'])), flush=True)
    while step < total:
        epoch = step // segment + 1
        dataset.set_epoch(epoch)
        write_json(output / f'plan_{epoch:03d}.json', dataset.plan_receipt())
        start = (step % segment) * batch_size
        data_started = time.perf_counter()
        for batch in loader(dataset, start, cfg):
            if step >= total or step // segment + 1 != epoch:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            data_seconds = time.perf_counter() - data_started
            then = time.perf_counter()
            stat = update(g, d, go, do, batch, cfg, step)
            step += 1
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            row = dict(global_step=step, input_mode=cfg['input_mode'], sources=batch['source'], draws=batch['draw'],
                       data_seconds=data_seconds, update_seconds=time.perf_counter() - then,
                       elapsed_seconds=time.perf_counter() - began, **stat)
            with (output / 'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            if step % cfg['training']['log_every_updates'] == 0:
                print(json.dumps(dict(event='training_update', step=step, loss_g=stat.get('loss_g'),
                      update_seconds=row['update_seconds'], data_seconds=data_seconds), allow_nan=False), flush=True)
            if step in cfg['training']['monitor_updates'] and step not in monitors:
                monitor(g, config, output, step, fixed)
                monitors.append(step)
            if step % cfg['training']['checkpoint_every_updates'] == 0 or step == 1 or step == total:
                check_space(output, cfg)
                save_checkpoint(output / 'latest.pt', g, d, go, do, cfg, identity, step, monitors)
                write_json(output / 'status.json', dict(complete=False, global_step=step, monitor_steps=monitors))
            data_started = time.perf_counter()
    verify_inputs(config, identity, dataset)
    result = dict(complete=True, global_step=step, model_sha256=state_sha256(g),
                  discriminator_sha256=state_sha256(d), input_mode=cfg['input_mode'],
                  checkpoint='latest.pt', monitor_steps=monitors, elapsed_seconds=time.perf_counter() - began,
                  no_full_test_inference=True, no_test_training=True, deployment_replaced=False,
                  identity_sha256=canonical_sha(identity), free_gib=check_space(output, cfg))
    write_json(output / 'complete.json', result)
    write_json(output / 'status.json', result)
    print(json.dumps(dict(event='formal_complete', **result)), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', choices=['preflight', 'smoke', 'train'], default='train')
    parser.add_argument('--resume')
    parser.add_argument('--smoke-updates', type=int, default=128)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cfg = validate_config(config)
    torch.set_num_threads(int(cfg['training']['torch_threads']))
    cv2.setNumThreads(int(cfg['training']['opencv_threads']))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    output = args.output or project_path(config, cfg['output_dir'])
    if args.mode == 'preflight':
        result = preflight(config, output, args.device)
    elif args.mode == 'smoke':
        result = smoke(config, output, args.device, args.smoke_updates)
    else:
        result = train(config, output, args.device, args.resume)
    print(json.dumps(dict(mode=args.mode, complete=result.get('complete'))), flush=True)


if __name__ == '__main__':
    main()
