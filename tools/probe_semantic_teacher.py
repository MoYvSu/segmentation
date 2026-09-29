# -*- coding: utf-8 -*-
"""原图S-align与D5a/simple的固定训练源诊断；无优化器、无测试图、无新标签。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import find_objects
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def select_unlabelled(files, mapping, count, seed):
    """映射须由内容身份核验产生；人工文件名不能直接用作训练池排除项。"""
    files = sorted(map(Path, files))
    names = [p.name for p in files]
    excluded = {r['pool_name'] for r in mapping}
    if len(set(names)) != len(names) or not excluded.issubset(names):
        raise ValueError('Pool source identity mismatch')
    candidates = [p for p in files if p.name not in excluded]
    if not 0 < count <= len(candidates):
        raise ValueError('Invalid other-training-source count')
    indices = np.random.default_rng(seed).choice(len(candidates), count, replace=False)
    return sorted(candidates[int(i)] for i in indices)


def probability_grid(logits, grid=256):
    """比较两头时先sigmoid再面积平均到学生网格，不平均logits。"""
    p = logits.float().sigmoid()
    if p.ndim != 4 or p.shape[:2] != (1, 1) or min(p.shape[-2:]) < grid:
        raise ValueError('Expected single B1HW prediction at or above comparison grid')
    return F.interpolate(p, (grid, grid), mode='area') if p.shape[-2:] != (grid, grid) else p


def reliable_target(views, valid, confidence=.9, gap=.05, erosion=1):
    """原图、轻微外观变化、镜像复原均高置信同类；只在有效内容内向内收缩。"""
    if len(views) < 2 or any(v.shape != views[0].shape for v in views):
        raise ValueError('Aligned probability views required')
    if not .5 < confidence < 1 or not 0 <= gap <= 1 or erosion < 0:
        raise ValueError('Invalid reliable-target settings')
    values = torch.stack(views)
    valid = F.interpolate(valid.float(), views[0].shape[-2:], mode='area') >= 1
    same = ((values > .5) == (values[0] > .5)).all(0)
    confident = torch.maximum(values, 1 - values).amin(0) >= confidence
    stable = values.amax(0) - values.amin(0) <= gap
    candidate = valid & same & confident & stable
    target = values.mean(0)
    accepted = torch.zeros_like(candidate)
    for cls in [False, True]:
        part = candidate & ((target > .5) == cls)
        if erosion:
            part = -F.max_pool2d(-F.pad(part.float(), (erosion,) * 4), 2 * erosion + 1, 1) > .5
        accepted |= part
    return target, accepted, {'valid': int(valid.sum()), 'candidate': int(candidate.sum()),
        'accepted': int(accepted.sum()), 'weak_class_disagreement': int((valid & ~same).sum()),
        'weak_unstable': int((valid & ~stable).sum()),
        'weak_gap_sum': float((values.amax(0) - values.amin(0))[valid].sum()),
        'accepted_ferrite': int((accepted & (target > .5)).sum()),
        'accepted_pearlite': int((accepted & (target <= .5)).sum())}


def pure_grid_gt(target, grid=256):
    """含未知或跨类别的格点保持ignore；不以多数类伪造标签。"""
    known = ((target == 0) | (target == 1)).float()
    known = F.interpolate(known, (grid, grid), mode='area') >= 1
    positive = F.interpolate((target == 1).float(), (grid, grid), mode='area')
    truth = torch.full_like(positive, -1)
    truth[known & (positive == 0)] = 0
    truth[known & (positive == 1)] = 1
    return truth


def pair_stats(teacher, student, mask, truth=None):
    """差异诊断；仅在确有GT的位置判断谁对谁错。"""
    if not (teacher.shape == student.shape == mask.shape):
        raise ValueError('Pair shapes differ')
    n = int(mask.sum())
    d = (teacher - student).abs()
    tp, sp = teacher > .5, student > .5
    result = {'n': n, 'absolute_gap_sum': float(d[mask].sum()),
        'class_disagreement': int(((tp != sp) & mask).sum()),
        'gap_above_005': int(((d > .05) & mask).sum()),
        'gap_above_01': int(((d > .1) & mask).sum()),
        'teacher_f_student_p': int((tp & ~sp & mask).sum()),
        'teacher_p_student_f': int((~tp & sp & mask).sum()),
        'teacher_confidence_sum': float(torch.maximum(teacher, 1 - teacher)[mask].sum())}
    if truth is not None:
        known = mask & ((truth == 0) | (truth == 1))
        tc, sc = tp == truth, sp == truth
        result['gt'] = {'n': int(known.sum()), 'teacher_wrong': int((known & ~tc).sum()),
            'student_wrong': int((known & ~sc).sum()),
            'teacher_correct_student_wrong': int((known & tc & ~sc).sum()),
            'teacher_wrong_student_correct': int((known & ~tc & sc).sum()),
            'both_wrong': int((known & ~tc & ~sc).sum())}
    return result


def sum_records(records):
    if not records:
        return {}
    result = {}
    for key in records[0]:
        if isinstance(records[0][key], dict):
            result[key] = sum_records([r[key] for r in records])
        else:
            result[key] = sum(r[key] for r in records)
    return result


def summarize(rows):
    result = {}
    for cohort in ['manual', 'other']:
        selected = [r for r in rows if r['cohort'] == cohort]
        if not selected:
            continue
        data = {'images': len(selected), 'teachers': {}}
        common = sum_records([r['teacher_difference_on_common_mask'] for r in selected])
        common['mean_absolute_gap'] = common['absolute_gap_sum'] / max(1, common['n'])
        common['class_disagreement_fraction'] = common['class_disagreement'] / max(1, common['n'])
        data['teacher_difference_on_common_mask'] = common
        for name in ['raw', 'self']:
            stable = sum_records([r['reliability'][name] for r in selected])
            stable['accepted_fraction'] = stable['accepted'] / max(1, stable['valid'])
            stable['weak_class_disagreement_fraction'] = stable['weak_class_disagreement'] / max(1, stable['valid'])
            stable['mean_weak_probability_range'] = stable['weak_gap_sum'] / max(1, stable['valid'])
            pairs = {}
            for condition in selected[0]['pairs'][name]:
                item = sum_records([r['pairs'][name][condition] for r in selected])
                item['mean_absolute_gap'] = item['absolute_gap_sum'] / max(1, item['n'])
                item['class_disagreement_fraction'] = item['class_disagreement'] / max(1, item['n'])
                item['gap_above_01_fraction'] = item['gap_above_01'] / max(1, item['n'])
                pairs[condition] = item
            data['teachers'][name] = {'reliability': stable, 'pairs': pairs}
        if cohort == 'manual':
            data['native_gt_votes'] = sum_records([r['native_gt_votes']['counts'] for r in selected])
        result[cohort] = data
    return result


def native_gt_votes(probabilities, instances, lookup):
    """固定GT实例的原尺寸float32均值投票，严格>0.5；无预测实例或面积代理。"""
    rows = []
    for value, region in enumerate(find_objects(instances.astype(np.int32)), start=1):
        if region is None or lookup[value] not in (0, 1):
            continue
        mask = instances[region] == value
        scores = {name: float(p[region][mask].mean()) for name, p in probabilities.items()}
        rows.append({'id': value, 'gt': int(lookup[value]), 'area': int(mask.sum()), 'scores': scores})
    counts = {'instances': len(rows), 'models': {}, 'raw_vs_student': {}}
    for name in probabilities:
        counts['models'][name] = {'wrong': sum(int(r['scores'][name] > .5) != r['gt'] for r in rows)}
    for name in probabilities:
        if not name.startswith('student_'):
            continue
        counts['raw_vs_student'][name] = {
            'teacher_correct_student_wrong': sum(int(r['scores']['raw'] > .5) == r['gt'] and int(r['scores'][name] > .5) != r['gt'] for r in rows),
            'teacher_wrong_student_correct': sum(int(r['scores']['raw'] > .5) != r['gt'] and int(r['scores'][name] > .5) == r['gt'] for r in rows)}
    return {'counts': counts, 'instances': rows}


def preview(path, raw, restored, teacher, clean_student, strong_student, mask, target):
    h, w = raw.shape[:2]
    size = (224, max(1, round(h * 224 / w)))
    bodies = [('Raw input', raw), ('Strong input -> D5a', restored)]
    for name, p in [('Raw teacher P(F)', teacher), ('Simple clean P(F)', clean_student), ('Simple strong P(F)', strong_student)]:
        heat = cv2.applyColorMap(np.rint(np.clip(p, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        bodies.append((name, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)))
    change = np.full((*mask.shape, 3), 70, dtype=np.uint8)
    change[mask] = (235, 235, 235)
    change[mask & (np.abs(target - strong_student) > .1)] = (245, 165, 35)
    change[mask & ((target > .5) != (strong_student > .5))] = (230, 45, 65)
    bodies.append(('Reliable: red=class differs', change))
    panels = []
    for title, body in bodies:
        scaled = cv2.resize(body, size, interpolation=cv2.INTER_AREA)
        canvas = cv2.copyMakeBorder(scaled, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(canvas, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(canvas)
    cv2.imencode('.png', cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))[1].tofile(path)


@torch.inference_mode()
def run(config_path, output_override=None, smoke=False):
    from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
    from data.mim_dataset import list_images
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_consistency import apply_acquisition_appearance
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import build_backend, load_backend, restore_first
    from tools.analyze_semantic_domain import ImagePool, verify_cohorts
    from tools.labelled_light_probe import inverse_geometry
    from train_backend_adaptation import sha, tensor_digest
    from train_semantic_d5a import frozen_digest, get_restorer
    from utils.affinity_deployment import crop_letterbox_output
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Use GPU server sam2_env')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    config = load_config(config_path)
    cfg, ucfg, acfg = (config[k] for k in ['semantic_teacher_probe', 'semantic_consistency', 'semantic_adaptation'])
    output = Path(project_path(config, output_override or cfg['output_dir']))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'previews').mkdir()
    started = time.time()
    report = {'status': 'preflight', 'training_started': False, 'smoke': smoke, 'images': [],
        'protocol': {'selection_seed': cfg['seed'], 'manual_count': 32, 'other_count': cfg['unlabeled_sources'],
            'source_selection': 'all_manual_and_seeded_uniform_without_replacement_from_other_training_sources',
            'test_images_used': False, 'holdout_created': False, 'affinity_forward': False,
            'comparison_grid': 256, 'grid_resampling': 'sigmoid_then_area_average',
            'weak_views': ['identity', f"{cfg['weak_contrast']}*RGB+{cfg['weak_offset']}", 'horizontal_flip_then_unflip_logits'],
            'weak_diagnostic_is_not_replay': 'Uses a nonclipping contrast perturbation and mirror check for both teachers, not prior training random-offset acceptance.',
            'teacher_domains': {'raw': 'S-align on original/weak/mirrored original', 'self': 'simple e60 on D5a clean/weak/mirrored D5a clean'},
            'strong_draws': cfg['strong_draws'], 'strong_recipe': 'current online degradation/noise/acquisition appearance, D5a first_x0',
            'precision': 'FP32; evaluation only', 'gt': 'native completed GT votes and fully-known single-class grid cells',
            'scope': 'training-source diagnostic; not held-out or official accuracy',
            'automatic_training': False, 'reliability': {k: cfg[k] for k in ['confidence', 'max_probability_gap', 'erosion_radius']}},
        'config': config}
    status_path = output / 'report.json'
    write_json(status_path, report)
    try:
        pool = [Path(p) for p in list_images(project_path(config, ucfg['unlabeled_dir']))]
        if len(pool) != ucfg['expected_sources']:
            raise ValueError('Training pool source count changed')
        manual_dir = project_path(config, config['paths']['raw_data_dir'])
        mapping = verify_cohorts(pool, manual_dir, acfg['expected_manual_sources'])
        other = select_unlabelled(pool, mapping, cfg['unlabeled_sources'], cfg['seed'])
        manual = CanonicalBackendDataset(manual_dir, project_path(config, acfg['completed_gt_dir']))
        entries = [{'path': p, 'cohort': 'manual', 'manual_index': i} for i, p in enumerate(manual.samples)]
        entries += [{'path': p, 'cohort': 'other'} for p in other]
        if smoke:
            entries = [e for e in entries if e['cohort'] == 'manual'][:1] + [e for e in entries if e['cohort'] == 'other'][:2]
        for e in entries:
            e['sha256'] = sha(e['path'])
        report['sources'] = {'manual_mapping': mapping, 'selected_other': [p.name for p in other],
            'processed': [{**e, 'path': str(e['path'])} for e in entries]}
        write_json(status_path, report)

        model, teacher_info = build_backend(config, device)
        if teacher_info['sources']['semantic']['sha256'] != cfg['teacher_sha256']:
            raise ValueError('Original raw semantic teacher changed')
        teacher, encoder = model.semantic_decoder, model.encoder
        frozen = frozen_digest(model)
        del model
        student_path = project_path(config, ucfg['initial_checkpoint'])
        if sha(student_path) != ucfg['initial_sha256']:
            raise ValueError('Original simple e60 checkpoint changed')
        model, bundle = load_backend(student_path, config, device)
        if frozen_digest(model) != frozen or bundle['epoch'] != 60:
            raise ValueError('Teacher/student frozen features or student epoch differ')
        if model.semantic_decoder.semantic_residual is not None:
            raise ValueError('Student must be the simple head')
        student = model.semantic_decoder
        del model, bundle
        gc.collect()
        torch.cuda.empty_cache()
        restorer = get_restorer(config, device)
        modules = {'encoder': encoder, 'teacher': teacher, 'student': student, 'D5a': restorer}
        for module in modules.values():
            module.eval().requires_grad_(False)
        before = {k: tensor_digest(v.state_dict().items()) for k, v in modules.items()}
        report['models'] = {'raw_teacher': teacher_info, 'student': {'checkpoint': ucfg['initial_checkpoint'],
            'sha256': ucfg['initial_sha256'], 'epoch': 60}, 'D5a': acfg['restoration'],
            'shared_nonsemantic_state': frozen, 'state_digests_before': before,
            'unique_resident_parameters': sum(sum(p.numel() for p in m.parameters()) for m in modules.values())}
        if report['models']['unique_resident_parameters'] >= 500_000_000:
            raise ValueError('Parameter limit violated')
        source = ImagePool([e['path'] for e in entries], 1024)
        degradation = load_config(project_path(config, acfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
        paired = PairedDegradationDataset(source, degradation, acfg['noise'], seed=cfg['seed'])
        report['status'] = 'running'
        write_json(status_path, report)

        def forward(head, image):
            return head(encoder(image), image).float()

        for index, entry in enumerate(entries):
            sample = source[index]
            image = sample['image'][None].to(device)
            valid = sample['valid_content'][None].to(device)
            h, w = map(int, sample['input_content_shape'].tolist())
            rgb = read_rgb(str(entry['path']))
            seed = int(acfg['restoration']['inference_seed']) + index
            weak = cfg['weak_contrast'] * image + cfg['weak_offset']
            if not (0 <= float(weak.min()) <= float(weak.max()) <= 1):
                raise ValueError('Weak diagnostic unexpectedly clips')
            t_logits = forward(teacher, image)
            tw_logits = forward(teacher, weak)
            tf_logits = forward(teacher, image.flip(-1)).flip(-1)
            restored = restore_first(restorer, image, seed)
            weak_restored = restore_first(restorer, weak, seed)
            s_logits = forward(student, restored)
            sw_logits = forward(student, weak_restored)
            sf_logits = forward(student, restored.flip(-1)).flip(-1)
            raw_views = [probability_grid(v) for v in [t_logits, tw_logits, tf_logits]]
            self_views = [probability_grid(v) for v in [s_logits, sw_logits, sf_logits]]
            teacher_sets = {}
            row = {'name': entry['path'].name, 'cohort': entry['cohort'], 'reliability': {}, 'pairs': {},
                'strong_inputs': [], 'native_shape': list(rgb.shape[:2]), 'weak_max_input_change': float((weak-image).abs().max())}
            gt_grid, gt_native, lookup = None, None, None
            if entry['cohort'] == 'manual':
                gt_sample = manual[entry['manual_index']]
                if not torch.equal(gt_sample['image'], sample['image']):
                    raise RuntimeError('Manual GT and image preprocessing differ')
                gt_grid = pure_grid_gt(gt_sample['semantic_target'][None].to(device))
                gt_native, lookup = load_completed_semantic_source(manual.completed_gt_dir / (entry['path'].stem + '_gt.npz'), rgb.shape[:2])

            def native(logits):
                return crop_letterbox_output(logits, 1024, 1024-h, 1024-w, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()

            native_predictions = {'raw': native(t_logits), 'raw_weak': native(tw_logits),
                                  'raw_flip': native(tf_logits), 'student_clean': native(s_logits)} if gt_native is not None else {}
            for name, views in [('raw', raw_views), ('self', self_views)]:
                target, mask, info = reliable_target(views, valid, cfg['confidence'], cfg['max_probability_gap'], cfg['erosion_radius'])
                teacher_sets[name] = (target, mask)
                row['reliability'][name] = info
                row['pairs'][name] = {'student_clean': pair_stats(target, self_views[0], mask, gt_grid)}
            common = teacher_sets['raw'][1] & teacher_sets['self'][1]
            row['teacher_difference_on_common_mask'] = pair_stats(teacher_sets['raw'][0], teacher_sets['self'][0], common, gt_grid)
            for draw in range(cfg['strong_draws']):
                paired.set_epoch(100 + draw)
                degraded = paired[index]
                if not torch.equal(inverse_geometry(degraded['valid_content'], degraded), sample['valid_content']):
                    raise RuntimeError('Strong-input geometry alignment failed')
                strong_input = inverse_geometry(degraded['image'], degraded)[None].to(device)
                strong_input, appearance = apply_acquisition_appearance(strong_input, valid, ucfg['appearance'], cfg['seed'] + index + draw * 10000)
                strong = restore_first(restorer, strong_input, seed + (draw+1)*10000)
                logits = forward(student, strong)
                probability = probability_grid(logits)
                row['strong_inputs'].append({'draw': draw, 'appearance': appearance,
                    **{k: degraded[k] for k in ['profile', 'is_spatial', 'endpoint_applied', 'base_sigma', 'noise_sigma']}})
                for name, (target, mask) in teacher_sets.items():
                    row['pairs'][name][f'strong_{draw}'] = pair_stats(target, probability, mask, gt_grid)
                    row['pairs'][name][f'strong_{draw}_common'] = pair_stats(target, probability, common, gt_grid)
                if gt_native is not None:
                    native_predictions[f'student_strong_{draw}'] = native(logits)
                if draw == 0:
                    ch, cw = int(round(h/4)), int(round(w/4))
                    def array(t):
                        return t[0,0,:ch,:cw].cpu().numpy()
                    srgb = (strong[0,:,:h,:w].permute(1,2,0).cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
                    preview(output/'previews'/f"{entry['cohort']}_{entry['path'].stem}.png", rgb, srgb,
                            array(raw_views[0]), array(self_views[0]), array(probability),
                            array(teacher_sets['raw'][1]), array(teacher_sets['raw'][0]))
            if gt_native is not None:
                row['native_gt_votes'] = native_gt_votes(native_predictions, gt_native, lookup)
            report['images'].append(row)
            report['summary'] = summarize(report['images'])
            report['elapsed_seconds'] = time.time() - started
            write_json(status_path, report)
            print(f"probe {index+1}/{len(entries)} {entry['cohort']}/{entry['path'].name}", flush=True)
        after = {k: tensor_digest(v.state_dict().items()) for k, v in modules.items()}
        if before != after:
            raise RuntimeError('A frozen module changed during diagnostic')
        report.update(status='completed', frozen_states_unchanged=True, elapsed_seconds=time.time()-started)
        write_json(status_path, report)
        write_json(output/'summary.json', {'protocol': report['protocol'], 'models': report['models'],
            'summary': report['summary'], 'status': report['status'], 'elapsed_seconds': report['elapsed_seconds']})
    except BaseException as error:
        report.update(status='failed', error=repr(error), elapsed_seconds=time.time()-started)
        write_json(status_path, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiments/semantic_teacher_probe.yaml')
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output_dir, args.smoke)
