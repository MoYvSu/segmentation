# -*- coding: utf-8 -*-
"""固定347/GT40的真实affinity内部表示检查；不训练或制作提交。

short只替换四短关系，all替换八通道的合法同GT关系；两臂复用同一
实际12窗捕获。未选logits逐值不变，gated支持与插值的标量传播另报。
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import gc
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.semantic_targets import load_completed_semantic_source
from tools.affinity_bottleneck_capture import capture_native_tiles
from tools.analyze_affinity_bottleneck import source_name
from tools.analyze_affinity_structure import case_summary
from tools.analyze_marker_stages import verify_hashes
from tools.probe_affinity_bottleneck import measure, project_native_support, validate_case
from tools.probe_affinity_structure import (R1_SHA, STAGE_SHA, decode_reflect,
    safety_metrics, target_result)
from tools.probe_marker_stages import compare_actual, connectivity_for_variant, trace_cases
from tools.run_marker_reflect import (array_sha, config_gate, precision_record,
    read, replay_candidates, save_prediction, sha, source_snapshot, write)
from train_backend_adaptation import fusion_options
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_structure import interior_mask
from utils.patch_diagnostic import BlendMap, tile_boxes

SOURCE = 'train_347'
SEED = 314161
TARGET = 40
MARGIN = 4
RESTORER_SHA = '048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a'


def positive_relation_mask(tile, gt, original, *, channels, gid=TARGET, margin=MARGIN):
    """双端和整条原生像素路径均在原覆盖的同GT保护内部。

    nearest GT与实际align_corners插值中心一致才有效；long也逐点检查，
    不允许端点同GT却跨过未知、filled、第三GT或外周保护带。
    """
    if channels not in ('short', 'all'):
        raise ValueError('channels must be short or all')
    proposal = interior_mask(gt, original, gid, margin=margin)
    domain = proposal['strict_mask']
    nodes, short_edges, yy, xx = project_native_support(tile, gt, domain)
    grid_gt = tile['grid_gt']
    nodes &= grid_gt == gid
    h, w = grid_gt.shape
    mask = np.zeros((1, 8, h, w), bool)
    mask[0, :4] = short_edges
    ay, ax = np.indices((h, w))
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[4:], 4):
        if channels == 'short':
            break
        by, bx = ay+dy, ax+dx
        inside = (by >= 0) & (by < h) & (bx >= 0) & (bx < w)
        sy, sx = ay[inside], ax[inside]
        ty, tx = by[inside], bx[inside]
        keep = nodes[sy, sx] & nodes[ty, tx]
        sy, sx, ty, tx = sy[keep], sx[keep], ty[keep], tx[keep]
        if not len(sy):
            continue
        p0y, p0x, p1y, p1x = yy[sy, sx], xx[sy, sx], yy[ty, tx], xx[ty, tx]
        steps = np.maximum(np.abs(p1y-p0y), np.abs(p1x-p0x))
        legal = np.ones(len(sy), bool)
        for step in range(int(steps.max())+1):
            fraction = np.minimum(step, steps)/np.maximum(1, steps)
            py = np.rint(p0y+fraction*(p1y-p0y)).astype(np.int32)
            px = np.rint(p0x+fraction*(p1x-p0x)).astype(np.int32)
            legal &= domain[py, px] & original[py, px] & (gt[py, px] == gid)
        mask[0, channel, sy[legal], sx[legal]] = True
    if channels == 'short' and mask[:, 4:].any():
        raise AssertionError('Short arm changed long channels')
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        y, x = np.nonzero(mask[0, channel])
        if len(y) and not (np.all(nodes[y, x]) and np.all(nodes[y+dy, x+dx])
                           and np.all(grid_gt[y, x] == gid) and np.all(grid_gt[y+dy, x+dx] == gid)):
            raise AssertionError('Selected relation is not trusted same-GT')
    return mask, dict(channels=channels, gid=gid, margin_native_pixels=margin,
        per_channel=[int(mask[0, c].sum()) for c in range(8)], selected=int(mask.sum()),
        mask_sha256=array_sha(mask), selected_nodes=int(nodes.sum()),
        native_domain_sha256=array_sha(domain),
        domain_provenance=proposal['strict_provenance'],
        both_endpoints_and_complete_native_path_same_GT_original_covered=True,
        nearest_and_interpolation_center_GT_agreement_required=True)


def replace_positive(logits, mask):
    if (logits.ndim != 4 or logits.shape[1] != 8 or logits.dtype != torch.float32
            or not bool(torch.isfinite(logits).all()) or mask.dtype != np.bool_
            or tuple(mask.shape) != tuple(logits.shape)):
        raise ValueError('Finite FP32 eight-channel logits and matching bool mask required')
    selected = torch.zeros_like(logits, dtype=torch.bool)
    selected.copy_(torch.from_numpy(mask))
    # 与之前冻结关系纠正一致，有限+16对应FP32 same概率约0.99999988。
    result = torch.where(selected, logits.new_tensor(16.), logits)
    if not torch.equal(result[~selected], logits[~selected]):
        raise AssertionError('Unselected logits changed')
    return result


def recompute_positive(views, tiles, masks, config, shape, *, empty_exact=False):
    if len(tiles) != len(masks):
        raise ValueError('Every actual tile needs exactly one relation mask')
    mode, options = fusion_options(config)
    blend = BlendMap(shape)
    audits = []
    with torch.no_grad():
        for tile, mask in zip(tiles, masks):
            logits = replace_positive(tile['logits'], mask)
            if mask.any() or empty_exact:
                grid = affinity_boundary_probability(logits.float(), mode=mode, **options)
                y, x, y1, x1 = tile['box']
                native = views.crop_letterbox_output(grid, 1024, tile['pad_h'], tile['pad_w'],
                    (y1-y, x1-x)).cpu()[0, 0].numpy()
                if empty_exact and (not np.array_equal(grid.cpu().numpy(), tile['grid'])
                                    or not np.array_equal(native, tile['native'][0, 0])):
                    raise RuntimeError('Actual empty positive CUDA grid/native replay differs')
                grid_changes = int(np.count_nonzero(grid.cpu().numpy() != tile['grid']))
            else:
                native = tile['native'][0, 0]
                grid_changes = 0
            blend.add(tile['box'], native)
            audits.append(dict(index=tile['index'], box=tile['box'],
                selected=int(mask.sum()), per_channel=[int(mask[0, c].sum()) for c in range(8)],
                original_logits_stride=tile['logits_stride'], grid_changed_pixels=grid_changes,
                native_changed_pixels=int(np.count_nonzero(native != tile['native'][0, 0])),
                unselected_logits_exact=True))
    boundary, audit = blend.finish()
    audit['tiles'] = len(tiles)
    return boundary, audit, audits


def boundary_propagation(before, after, gt, original, domain):
    changed = before != after
    true_interface = np.zeros(gt.shape, bool)
    for a, b in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        contact = (gt[a] > 0) & (gt[b] > 0) & (gt[a] != gt[b]) & original[a] & original[b]
        true_interface[a] |= contact
        true_interface[b] |= contact
    delta = np.abs(after-before)
    return dict(changed_pixels=int(changed.sum()), outside_trusted_interior=int((changed & ~domain).sum()),
        outside_target_GT=int((changed & (gt != TARGET)).sum()),
        filled_changed_pixels=int((changed & (gt > 0) & ~original).sum()),
        unknown_changed_pixels=int((changed & (gt == 0)).sum()),
        trusted_true_interface_changed_pixels=int((changed & true_interface).sum()),
        trusted_true_interface_max_absolute_change=float(delta[true_interface].max(initial=0)),
        caveat='Only unselected logits are exact. Gated max-pool support and bilinear interpolation can propagate scalar changes outside selected nodes.')


def case_outcomes(decoded, gt, original, cases, metrics, classes, votes):
    values = [target_result(decoded, gt, original, c, metrics, classes, votes) for c in cases]
    return values, [case_summary(value) for value in values]


def cpu_runtime():
    import skimage
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        opencv=cv2.__version__, skimage=skimage.__version__, torch_threads=torch.get_num_threads(),
        opencv_thinning_available=hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'))


def historical_source_gate(structure_path, hashes):
    """核验历史封存源码本身；当前只接受明确的safety分类口径修正。

    旧产物的运行身份不因修复None分类统计而改写。允许差异必须仅限
    safety_metrics函数的AST，解码、模型和其他函数变化一律拒绝。
    """
    protected, revisions = {}, []
    for relative, digest in hashes.items():
        saved = structure_path.parent/'source'/relative
        if sha(saved) != digest:
            raise RuntimeError('Sealed historical structure source differs: '+relative)
        protected[str(saved)] = digest
        current = ROOT/relative
        actual = sha(current)
        if actual == digest:
            continue
        if relative != 'tools/probe_affinity_structure.py':
            raise RuntimeError('Unapproved change from structure source: '+relative)
        def other_statements(path):
            tree = ast.parse(path.read_text(encoding='utf8'))
            tree.body = [node for node in tree.body
                         if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         or node.name != 'safety_metrics']
            return ast.dump(tree, include_attributes=False)
        if other_statements(saved) != other_statements(current):
            raise RuntimeError('Structure source changed outside safety_metrics')
        revisions.append(dict(file=relative, historical_sha256=digest, current_sha256=actual,
            accepted_change='known_matched_phase_only_v2: unmatched/None is not F-to-P; no inference change'))
    return protected, revisions


def all_case_regression(structure, current, selected_name=SOURCE):
    rows = []
    for source in structure['sources']:
        name = source_name(source['source'])
        outcomes = current if name == selected_name else source['reference_cases']
        rows.extend(dict(source=name, source_recomputed=name == selected_name,
                         outcome=case_summary(c)) for c in outcomes)
    if len(rows) != 17:
        raise RuntimeError('Fixed seventeen-case regression cohort differs')
    return rows


def run(args):
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from train_backend_adaptation import tensor_digest
    from tools.run_affinity_sweep import runtime_contract
    from utils.config import load_config, project_path

    output = (ROOT/args.output).resolve()
    if output.exists():
        raise FileExistsError('Fresh ignored outputs directory required')
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Ignored outputs directory required')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual sam2_env CUDA required; no CPU proxy for source capture')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    stage_path, structure_path, receipt_path = [ROOT/p for p in (args.stage, args.structure, args.receipt)]
    if sha(stage_path) != STAGE_SHA:
        raise RuntimeError('Fixed original seven-source stage receipt differs')
    stage, structure, receipt = read(stage_path), read(structure_path), read(receipt_path)
    if (stage.get('complete') is not True or structure.get('complete') is not True
            or structure.get('format') != 'affinity_structure_probe_v1'
            or structure.get('reference_cohort') != dict(matched_F=689, safe_area=135)):
        raise RuntimeError('Complete fixed seven-source structure/reflection receipt required')
    verify_hashes(structure['protected_inputs'])
    historical_code, compatibility_revisions = historical_source_gate(structure_path, structure['protected_code'])
    if cpu_runtime() != structure['current_CPU_runtime']:
        raise RuntimeError('Actual CPU readout runtime differs from structure probe')
    r1 = receipt['arms']['r1']
    config = deepcopy(r1['config'])
    config_gate(config, load_config(str(ROOT/'config/train/affinity_balance_mixed.yaml')))
    if precision_record() != receipt['precision'] or runtime_contract(config) != receipt['current_runtime']:
        raise RuntimeError('Pinned actual CUDA/SAM2 runtime or precision differs')
    record = next(s for s in stage['sources'] if source_name(s['source']) == SOURCE)
    verify_hashes(record['input_hashes'])
    source = record['source']
    structured = next(s for s in structure['sources'] if source_name(s['source']) == SOURCE)
    if source != structured['source'] or source['seed'] != SEED:
        raise RuntimeError('Fixed train347 source/seed identity differs')
    cases = source['cases']
    target_case = next(c for c in cases if c['direction'] == 'positive' and c['gt_ids'] == [TARGET])
    if target_case.get('censored') or target_case.get('previous_gt_structure_ambiguity'):
        raise RuntimeError('GT40 primary target must remain uncensored and unambiguous')
    image, gt_path = ROOT/source['source_path'], ROOT/source['gt_path']
    class_path = gt_path.with_name(gt_path.name.removesuffix('_gt.npz')+'_class.json')
    rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    gt, lookup = load_completed_semantic_source(gt_path, rgb.shape[:2])
    classes = {int(g):int(lookup[g]) for g in np.unique(gt) if g > 0}
    with np.load(gt_path, allow_pickle=False) as payload:
        original = payload['original_covered'].astype(bool)
        filled, unknown = payload['filled'].astype(bool), payload['residual_unknown'].astype(bool)
    if (not np.array_equal(filled, (gt > 0) & ~original)
            or not np.array_equal(unknown, gt == 0) or np.any(original & (gt == 0))):
        raise RuntimeError('Processed newGT provenance differs')
    for case in cases:
        validate_case(case, gt)
    if classes[TARGET] != 1 or len(tile_boxes(gt.shape, 1024, .25)) != 12:
        raise RuntimeError('Fixed ferrite GT40 or actual twelve-window geometry differs')
    checkpoint = Path(r1['checkpoint'])
    restore_path = Path(project_path(config, config['backend_adaptation']['restoration']['checkpoint']))
    if sha(checkpoint) != R1_SHA or sha(restore_path) != RESTORER_SHA:
        raise RuntimeError('Frozen mixed-R1/D5a weight hash differs')
    protected = {**structure['protected_inputs'], **record['input_hashes'], **historical_code}
    for path in (stage_path, structure_path, receipt_path, image, gt_path, class_path, checkpoint, restore_path):
        protected[str(path)] = sha(path)
    refdir, prefix = ROOT/source['reference_dir'], source.get('reference_prefix', 'r1')
    for suffix in ('_maps.npz', '_inst.png', '_class.json'):
        path = refdir/(prefix+suffix)
        protected[str(path)] = sha(path)
    reflect_dir = ROOT/'outputs/marker_stages'/SOURCE
    for suffix in ('_maps.npz', '_inst.png', '_class.json'):
        path = reflect_dir/('reflect'+suffix)
        protected[str(path)] = sha(path)
    verify_hashes(protected)
    core = load_original_core(config)
    views = core.coverage_original_views
    model, bundle = load_backend(checkpoint, config, 'cuda')
    model.eval().requires_grad_(False)
    restorer = get_restorer(config, 'cuda').eval().requires_grad_(False)
    states = (tensor_digest(model.state_dict().items()), tensor_digest(restorer.state_dict().items()))
    if states != (r1['model_state_sha256'], receipt['restorer_state_sha256']) or bundle['epoch'] != 20:
        raise RuntimeError('Pinned frozen R1/D5a state or epoch differs')
    if any(m.training for m in list(model.modules())+list(restorer.modules())):
        raise RuntimeError('Every module must remain in eval mode')
    parameters = sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    if parameters >= 500_000_000:
        raise RuntimeError('Competition parameter budget exceeded')
    output.mkdir(parents=True)
    started = time.time()
    report = dict(format='affinity_interior_probe_v1', complete=False, training=False,
        no_optimizer_steps=True, no_submission=True, source=source,
        checkpoint_sha256=R1_SHA, restorer_checkpoint_sha256=RESTORER_SHA,
        model_state_sha256=states[0], restorer_state_sha256=states[1], parameter_count=parameters,
        config=config, CPU_runtime=cpu_runtime(), runtime=runtime_contract(config), precision=precision_record(),
        protected_inputs=protected, policy=dict(target_GT=TARGET, margin_native_pixels=MARGIN,
            source_seed=SEED, actual_windows=12, input_size=1024, output_grid=512, overlap=.25,
            same_instance_logit=16., changed_layers='none; frozen output relation intervention',
            scopes=['original_covered only'], channels=['short4', 'all8'], fusion_changed=False),
        historical_structure_artifact_source_exact=True, compatibility_revisions=compatibility_revisions,
        safety_policy_version='known_matched_phase_only_v2',
        arms={}, caveat='Seen training source frozen relation counterfactual, relative to processed GT. GT40 contains a visible dark arc yet GT says same grain; no physically verified false-boundary claim. Not model learnability, generalization, test truth, or official score; tiny/zero contributions retained.')
    write(output/'status.json', dict(status='running', stage='capture', training=False))
    code, virtual = source_snapshot(output)
    # 主入口执行时__main__也纳入快照，明确核验自己的源文件。
    report.update(protected_code=code, virtual_code_paths=virtual)
    write(output/'preflight.json', report)
    try:
        _, captured, tiles, capture_audit = capture_native_tiles(
            views, model, restorer, image, config, gt, 'cuda', SEED)
        if len(tiles) != 12:
            raise RuntimeError('Actual capture did not cover all twelve windows')
        baseline, variants, replay = replay_candidates(captured, config)
        with np.load(refdir/(prefix+'_maps.npz'), allow_pickle=False) as payload:
            actual = {k:np.squeeze(payload[k]).copy() for k in ('boundary','markers','watershed')}
            for k, value in [('semantic', captured['semantic'].numpy()), ('raw_probability', captured['raw_probability'])]:
                if not np.array_equal(np.squeeze(payload[k]), np.squeeze(value)):
                    raise RuntimeError('Captured frozen whole semantics/raw differs: '+k)
        boundary = captured['baseline']['boundary']
        raw = captured['raw_probability']
        compare_actual(baseline, raw, actual, refdir, prefix, boundary)
        empty = [np.zeros(tuple(tile['logits'].shape), bool) for tile in tiles]
        empty_boundary, empty_blend, _ = recompute_positive(views, tiles, empty, config, gt.shape, empty_exact=True)
        if not np.array_equal(empty_boundary, boundary) or empty_blend != captured['baseline']['blend']:
            raise RuntimeError('Actual empty positive full stitch differs')
        reference = variants['32']
        from tools.analyze_affinity_bottleneck import load_prediction
        cached = load_prediction(reflect_dir, 'reflect')
        if any(not np.array_equal(cached[k], value) for k, value in (
            ('instances', reference['instances']), ('markers', reference['actual_markers']),
            ('watershed', reference['watershed']), ('boundary', boundary))) or cached['classes'] != reference['classes']:
            raise RuntimeError('Actual reflect complete output differs from cached scored-chain reference')
        reference_metrics, reference_votes = measure(reference, raw, gt, classes)
        cohort = dict(matched_F=sum(r['matched_pred'] is not None for r in reference_metrics['full_ferrite']['gt_rows']),
                      safe_area=reference_metrics['fixed_area']['original_fixed_count'])
        if (cohort != dict(matched_F=104, safe_area=28)
                or json.loads(json.dumps(reference_metrics, allow_nan=False)) != structured['reference_metrics']):
            raise RuntimeError('Actual reflect 104/28 full-source reference differs')
        reference_outcomes, reference_case_summaries = case_outcomes(
            reference, gt, original, cases, reference_metrics, classes, reference_votes)
        report.update(actual_capture=capture_audit, original_replay=replay,
            empty_positive_grid_native_and_full_stitch_exact=True, actual_reflect_cohort=cohort,
            reference_metrics=reference_metrics, reference_cases=reference_outcomes,
            reference_case_summaries=reference_case_summaries,
            frozen_maps=dict(semantic_sha256=array_sha(captured['semantic'].numpy()), raw_sha256=array_sha(raw),
                original_boundary_sha256=array_sha(boundary)),
            global_reference_cohort=structure['reference_cohort'])
        domain = interior_mask(gt, original, TARGET, margin=MARGIN)['strict_mask']
        panels, labels = [(reference['instances'], reference_votes)], ['reflect reference']
        for arm in ('short', 'all'):
            masks, mask_audits = [], []
            for tile in tiles:
                mask, audit = positive_relation_mask(tile, gt, original, channels=arm)
                masks.append(mask)
                mask_audits.append(dict(index=tile['index'], box=tile['box'], seed=tile['seed'], **audit))
            if not sum(m.sum() for m in masks):
                raise RuntimeError('No legal complete interior relations selected')
            edited, blend, tile_audits = recompute_positive(views, tiles, masks, config, gt.shape)
            decoded, stages, variant = decode_reflect(captured['semantic'], edited, config)
            metrics, votes = measure(decoded, raw, gt, classes, reference_metrics)
            safety = safety_metrics(reference_metrics, metrics)
            if safety.get('class_change_semantics') != 'known_matched_phase_only_v2':
                raise RuntimeError('Explicit matched-phase-only safety v2 required')
            outcomes, summaries = case_outcomes(decoded, gt, original, cases, metrics, classes, votes)
            maps = connectivity_for_variant(stages, variant, 'reflect')
            output_hashes = save_prediction(output, arm, decoded['instances'], votes)
            y, x = np.nonzero(edited != boundary)
            np.savez_compressed(output/(arm+'_native_delta.npz'), y=y.astype(np.int32), x=x.astype(np.int32),
                                before=boundary[y, x], after=edited[y, x])
            sparse = [(t['index'], c, y, x) for t, m in zip(tiles, masks)
                      for _, c, y, x in zip(*np.nonzero(m))]
            np.savez_compressed(output/(arm+'_relations.npz'), canonical=np.asarray(sparse, np.int32).reshape(-1, 4))
            signal = next(c for c in summaries if c['case_id'] == target_case['case_id'])
            report['arms'][arm] = dict(mask_audits=mask_audits, tile_replay=tile_audits,
                original_affinity_channels_and_fusion_function_preserved=True,
                unselected_logits_exact=True, selected_same_instance_probability=float(torch.sigmoid(torch.tensor(16.)).item()),
                boundary_sha256=array_sha(edited), blend=blend, boundary_propagation=boundary_propagation(
                    boundary, edited, gt, original, domain), output=output_hashes,
                metrics=metrics, safety=safety, target_results=outcomes, residual_summaries=summaries,
                stage_traces=trace_cases(maps, decoded, gt, cases),
                seventeen_case_regression=all_case_regression(structure, outcomes),
                representation_signal=bool(signal['thresholded_diagnostic_pass']
                    and signal['processed_known_residuals']['all_deep_in_target_main']
                    and signal['original_covered_residuals']['all_deep_in_target_main']
                    and safety['no_new_failures'] and safety['no_average_regression']),
                training_permission=False, other_six_sources_recomputed=False,
                all_unmodified_source_outputs_reused='Original fixed cached reflect; inputs and all weights protected')
            panels.append((decoded['instances'], votes)); labels.append(arm+' trusted same-GT')
            write(output/'report.json', report)
            print('INTERIOR', arm, 'selected', len(sparse), 'representation_signal',
                  report['arms'][arm]['representation_signal'], flush=True)
            del maps, decoded, metrics, stages, variant, masks
            gc.collect()
        from tools.analyze_affinity_connectivity import render
        render(image, panels, labels, output/'compare.png', '347 GT40 frozen affinity mechanism only', width=420)
        render(image, panels, labels, output/'detail.png', '347 GT40; actual full-image readout',
               crop=[0., 0., .15, .15], width=420)
        if states != (tensor_digest(model.state_dict().items()), tensor_digest(restorer.state_dict().items())):
            raise RuntimeError('Frozen model/restorer state changed')
        if cpu_runtime() != report['CPU_runtime'] or precision_record() != report['precision'] or runtime_contract(config) != report['runtime']:
            raise RuntimeError('Pinned CPU/CUDA runtime changed')
        verify_hashes(protected); verify_hashes(code, code=True)
        report.update(complete=True, frozen_states_exact=True, protected_inputs_exact=True,
                      protected_code_exact=True, elapsed_seconds=time.time()-started)
        write(output/'report.json', report)
        write(output/'status.json', dict(status='complete', training=False, elapsed_seconds=report['elapsed_seconds']))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>347 frozen affinity interior</h1>'
            '<p>Trusted training GT counterfactual; no learning or submission.</p>'
            '<img style="max-width:100%" src="compare.png"><img style="max-width:100%" src="detail.png">', encoding='utf8')
        print('INTERIOR COMPLETE', cohort, flush=True)
    except BaseException as exc:
        write(output/'status.json', dict(status='failed', training=False, error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='outputs/marker_stages/report.json')
    parser.add_argument('--structure', default='outputs/geometry_structure/report.json')
    parser.add_argument('--receipt', default='outputs/affinity_weak/capture.json')
    parser.add_argument('--output', default='outputs/geometry_interior')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
