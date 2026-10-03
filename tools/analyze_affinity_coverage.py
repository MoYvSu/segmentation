# -*- coding: utf-8 -*-
"""Coverage固定e20审计：复用100图输出，以及32张已见newGT的机制诊断。

CPU不重推测试集、不重训控制。GPU只读前向，必须通过两个原R1缓存样本的
精确部署等价门禁；复用完整32图原生R1缓存，不选择测试epoch或调后处理。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.affinity_coverage import (CoveragePlan, FIXED_KEYS, FORMAT, METADATA_SHA,
    OPTIONS, REFERENCE_SHA, REFERENCE_STATUS_SHA, REFERENCE_STEPS_SHA)
from tools.analyze_affinity_connectivity import relation
from utils.patch_diagnostic import phase_description

EXPECTED_CANDIDATE = '0f828a71cc8550ef3b00958f88bc3b24e2b9db70d8f79861ec8b949500ad9af0'
CAPTURE_SHA = '7fc4c1a23f609322a44cfb12f70f29110cf0b16afbeb4649563cafc82217f611'
REPORT_SHA = 'c47b46c1d08a2b54fd5f6d9c230b82fac41b75051906cbee5be57d85deb3957d'
VISUAL_FIXED = ('test_101', 'test_116')
VISUAL_RANDOM = ('test_100', 'test_130', 'test_154', 'test_094', 'test_124', 'test_102')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def safe_member(value):
    path = Path(value)
    if (not value or value.startswith('/') or path.is_absolute() or '..' in path.parts
            or '\\' in value or ':' in value):
        raise RuntimeError('Unsafe relative member: ' + value)
    return path


def normalized_config(config, *, candidate=False):
    result = deepcopy(config)
    result['paths'].pop('project_root', None)
    result['backend_adaptation'].pop('output_dir', None)
    if candidate:
        if result.pop('affinity_coverage', None) != OPTIONS:
            raise RuntimeError('Coverage recipe changed')
    return result


def capture_config_projection(config, registered_tail):
    """仅剥离实际封存TAIL的已登记训练附加项；不忽略任何推理配置。"""
    result = deepcopy(config)
    if ('affinity_fused' not in result or result['affinity_fused'] != registered_tail['affinity_fused']):
        raise RuntimeError('Cached training-only affinity_fused metadata differs')
    result.pop('affinity_fused')
    return normalized_config(result)


def fixed_area_cohort(old, new):
    """逐个追踪原R1安全面积名单，候选失配/错类/删失必须显式保留。"""
    before = {r['gt']: r for r in old['gt_rows'] if r['safe_one_to_one_area_eligible']}
    after = {r['gt']: r for r in new['gt_rows']}
    if not set(before).issubset(after):
        raise RuntimeError('Candidate canonical GT cohort changed')
    records = []
    for value, prior in before.items():
        row = after[value]
        reasons = []
        if row['matched_pred'] is None: reasons.append('unmatched')
        if row['matched_pred'] is not None and row['predicted_phase'] != 1: reasons.append('wrong_class')
        if row['censored'] or (row['matched_pred'] is not None and row['matched_pred'] in new['censored_pred_ids']):
            reasons.append('unknown_or_frame_censored')
        if row['significant_geometry_split']: reasons.append('significant_split')
        if row['matched_pred_significant_geometry_merge']: reasons.append('significant_merge')
        records.append(dict(gt=value, GT_pixels=prior['area'], r1_relative_error=prior['area_relative_error'],
            candidate_comparable=row['safe_one_to_one_area_eligible'], candidate_relative_error=row['area_relative_error'],
            candidate_matched_pred=row['matched_pred'], candidate_predicted_phase=row['predicted_phase'],
            excluded_reasons=reasons))
    newly_eligible = sorted(r['gt'] for r in new['gt_rows'] if r['safe_one_to_one_area_eligible'] and r['gt'] not in before)
    return dict(original_fixed_count=len(before), rows=records, newly_eligible_not_in_original_cohort=newly_eligible,
        comparable_count=sum(r['candidate_comparable'] for r in records),
        failure_counts={reason: sum(reason in r['excluded_reasons'] for r in records)
                        for reason in ('unmatched', 'wrong_class', 'unknown_or_frame_censored', 'significant_split', 'significant_merge')},
        caveat='Failures retained in original cohort; reasons may overlap. New eligible objects never replace denominator.')


def fixed_contact_trace(diagnostic, gt_ids=(69, 70)):
    """固定GT对的所有贡献，不靠双10%阈值的进入/退出判定界面修复。"""
    selected = {r['gt']: r for r in diagnostic['gt_rows'] if r['gt'] in gt_ids}
    if set(selected) != set(gt_ids):
        raise RuntimeError('Fixed weak-interface GT pair missing')
    contribution = {g: {c['pred']: c for c in selected[g]['contributions']} for g in gt_ids}
    dominant = {g: max(contribution[g], key=lambda p: contribution[g][p]['pixels']) if contribution[g] else None for g in gt_ids}
    shared = sorted(set(contribution[gt_ids[0]]) & set(contribution[gt_ids[1]]))
    return dict(gt_ids=list(gt_ids), dominant_pred=dominant,
        same_dominant_prediction=dominant[gt_ids[0]] is not None and dominant[gt_ids[0]] == dominant[gt_ids[1]],
        all_contributions={g: selected[g]['contributions'] for g in gt_ids},
        shared_prediction_rows=[dict(pred=p, gt_pixels={g: contribution[g][p]['pixels'] for g in gt_ids},
            gt_fractions={g: contribution[g][p]['pixels']/selected[g]['area'] for g in gt_ids}) for p in shared],
        caveat='Fixed GT IDs, all nonzero overlaps; IDs differ across predictions. No threshold-based repair claim.')


def weak_case(out, rgb, gt, before, after, old_boundary, new_boundary, old_diagnostic, new_diagnostic, threshold):
    """既有train659/GT69,70的完整贡献与原生界面响应，局部仅展示。"""
    from tools.analyze_affinity_fused_gt import gallery_panel
    chosen = np.isin(gt, [69, 70]); yy, xx = np.nonzero(chosen)
    if not len(yy): raise RuntimeError('Fixed weak interface absent')
    edge = np.zeros(gt.shape, bool)
    for a, b, target_a, target_b in ((gt[:, :-1], gt[:, 1:], edge[:, :-1], edge[:, 1:]),
                                    (gt[:-1], gt[1:], edge[:-1], edge[1:])):
        contact = ((a == 69) & (b == 70)) | ((a == 70) & (b == 69))
        target_a |= contact; target_b |= contact
    if not edge.any(): raise RuntimeError('Fixed GT69/70 original cardinal contact absent')
    trace = dict(source='train_659.jpg', gt_ids=[69,70], threshold=threshold, native_contact_pixels=int(edge.sum()),
        arms={arm: dict(contributions=fixed_contact_trace(diagnostic), native_contact_boundary_quantiles=quantiles(boundary[edge]),
              above_fixed_threshold=int((boundary[edge] >= threshold).sum()))
              for arm, diagnostic, boundary in [('r1', old_diagnostic, old_boundary), ('coverage', new_diagnostic, new_boundary)]},
        scope='Fixed existing weak-interface case; response and unthresholded contribution trace, not physical GT quality proof.')
    write(out / 'weak659.json', trace)
    y,x,y1,x1=max(0,int(yy.min())-30),max(0,int(xx.min())-30),min(gt.shape[0],int(yy.max())+31),min(gt.shape[1],int(xx.max())+31)
    panels=[('Original',rgb),('Processed GT',gallery_panel(rgb,gt,gt>0)),
            ('R1 e20',gallery_panel(rgb,before[0],gt>0,gt)),('Coverage e20',gallery_panel(rgb,after[0],gt>0,gt))]
    for title,boundary in [('R1 boundary',old_boundary),('Coverage boundary',new_boundary)]:
        heat=cv2.applyColorMap(np.rint(np.clip(boundary,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
        panels.append((title,cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)))
    strips=[]
    for title,body in panels:
        body=cv2.resize(body[y:y1,x:x1],(300,max(1,round(300*(y1-y)/(x1-x)))),interpolation=cv2.INTER_AREA)
        body=cv2.copyMakeBorder(body,26,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(body,title,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        strips.append(body)
    if not cv2.imwrite(str(out/'weak659.png'),cv2.cvtColor(np.concatenate(strips,1),cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Fixed weak case figure write failed')


def audit_steps(reference, actual, declared):
    plan = CoveragePlan(reference, declared)
    if len(actual) != 1280:
        raise RuntimeError('Exactly1280 executed updates required')
    changed = 0
    strata = {'whole': [], 'native_uniform': [], 'native_anchor': []}
    for old, new, expected in zip(reference, actual, plan.candidate):
        if any(new[k] != expected[k] for k in FIXED_KEYS):
            raise RuntimeError('Executed source/order/augmentation/crop differs from sealed plan')
        selected = bool(expected.get('counterfactual_anchor', {}).get('selected', False))
        tag = new.get('coverage')
        if (tag != dict(format=FORMAT, anchor=selected, actual_update=new['updates'], executed=True)
                or 'counterfactual_not_executed' in new):
            raise RuntimeError('Expected actual execution receipt')
        if new['manual_crop'] != old['manual_crop']:
            changed += 1
        if any(not np.isfinite(new[k]) or (k == 'grad_norm' and new[k] <= 0)
               for k in ('manual_bce', 'pseudo_bce', 'grad_norm')):
            raise RuntimeError('Nonfinite loss or missing parameter gradient')
        group = ('whole' if new['training_view'] == 'whole' else
                 'native_anchor' if selected else 'native_uniform')
        strata[group].append((old, new))
    if {k: len(v) for k, v in strata.items()} != dict(whole=640, native_uniform=320, native_anchor=320):
        raise RuntimeError('Exact640whole/320uniform/320anchor required')
    # 相同源不等于相同裁窗；anchor仅分别描述训练拟合，不作同输入因果比较。
    summary = {}
    for key, pairs in strata.items():
        summary[key] = dict(updates=len(pairs), same_input_recipe=key != 'native_anchor', periods={})
        for label, lo, hi in [('e1_5', 1, 5), ('e11_15', 11, 15), ('e16_20', 16, 20)]:
            segment = [(a, b) for a, b in pairs if lo <= b['epoch'] <= hi]
            values = {}
            for index, arm in enumerate(('r1', 'coverage')):
                values[arm] = {metric: float(np.mean([pair[index][metric] for pair in segment]))
                               for metric in ('manual_bce', 'pseudo_bce', 'grad_norm')}
                for source_index, source in enumerate(('manual', 'pseudo')):
                    for metric in ('recall', 'specificity'):
                        values[arm][source + '_' + metric] = float(np.mean(
                            [pair[index]['supervision'][source_index][metric] for pair in segment]))
            summary[key]['periods'][label] = values
    return dict(passed=True, updates=1280, actual_changed_manual_coordinates=changed,
                unchanged_whole_updates=640, unchanged_native_uniform_updates=320,
                changed_route_updates=320, strata=summary,
                caveat='Training fit only; changed crops are not same-input loss comparisons.')


def training_audit(args):
    run, ref = Path(args.run_root), Path(args.reference)
    pipeline = read(run / 'pipeline_status.json')
    state, old = read(run / 'candidate/status.json'), read(ref / 'status.json')
    if (pipeline.get('status') != 'complete' or pipeline.get('stage') != 'done'
            or pipeline.get('smoke') is not False or pipeline.get('control_retrained') is not False
            or pipeline.get('completed') != ['training', 'inference', 'render']):
        raise RuntimeError('Complete isolated coverage20 training/inference/render required')
    for value, digest in [(state, args.expected_checkpoint), (old, REFERENCE_SHA)]:
        if (value.get('status') != 'complete' or value.get('epoch') != 20 or value.get('updates') != 1280
                or value.get('smoke') is not False or value.get('size') != 1024
                or value.get('precision') != 'FP32' or value.get('final_checkpoint_sha256') != digest
                or any(value.get(k) is not True for k in
                       ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'))):
            raise RuntimeError('Fixed e20/1280/FP32 identity or state checks differ')
    if (sha(ref / 'status.json') != REFERENCE_STATUS_SHA or sha(ref / 'steps.jsonl') != REFERENCE_STEPS_SHA
            or sha(args.plan) != METADATA_SHA):
        raise RuntimeError('Original R1 or predeclared crop plan identity differs')
    if normalized_config(state['config'], candidate=True) != normalized_config(old['config']):
        raise RuntimeError('A variable besides coverage coordinates changed')
    if state['config'] != pipeline['config']:
        raise RuntimeError('Pipeline and actual training configuration differ')
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256'):
        if state[key] != old[key]:
            raise RuntimeError('Common initialization or frozen state changed: ' + key)
    data = deepcopy(state['data'])
    if 'coverage' not in data:
        raise RuntimeError('Actual coverage data receipt absent')
    data.pop('coverage')
    if data != old['data']:
        raise RuntimeError('Underlying source/newGT/data changed')
    sources = pipeline['sources']
    if len(sources) != 157 or sources != pipeline['contract']['sources']:
        raise RuntimeError('Registered157 training snapshot differs')
    for name, digest in sources.items():
        if sha(run / 'source' / safe_member(name)) != digest:
            raise RuntimeError('Archived training source differs: ' + name)
    executed = audit_steps(rows(ref / 'steps.jsonl'), rows(run / 'candidate/steps.jsonl'), rows(args.plan))
    epochs = rows(run / 'candidate/epochs.jsonl')
    if (len(epochs) != 20 or [r['epoch'] for r in epochs] != list(range(1, 21))
            or epochs[-1]['updates'] != 1280):
        raise RuntimeError('Actual epoch budget differs')
    images = state['config']['backend_adaptation']['monitor']['images']
    expected = [run / 'candidate/monitor' / f'epoch_{e:03d}' / (name + '_' + suffix + '.png')
                for e in (0, 1, 5, 10, 15, 20) for name in images for suffix in ('full', 'detail', 'boundary')]
    if len(expected) != 108 or not all(path.is_file() for path in expected):
        raise RuntimeError('Six-epoch108 fixed process thumbnails incomplete')
    checkpoints = {}
    for arm, path, digest in [('r1', ref / 'final.pt', REFERENCE_SHA),
                               ('coverage', run / 'candidate/final.pt', args.expected_checkpoint)]:
        if path.exists() and sha(path) != digest:
            raise RuntimeError('Actual checkpoint file differs: ' + arm)
        checkpoints[arm] = dict(sha256=digest, file=str(path), actual_file_sha_verified=path.is_file())
    recorded = pipeline['audit']
    if (recorded.get('passed') is not True or recorded.get('updates') != 1280
            or recorded.get('epoch') != 20 or recorded.get('final_checkpoint_sha256') != args.expected_checkpoint):
        raise RuntimeError('Recorded independent final weight/resume audit differs')
    return dict(complete=True, epoch=20, updates=1280, checkpoints=checkpoints,
                configuration=state['config'], initial_state_sha256=state['initial_state_sha256'],
                frozen_state_sha256=state['frozen_state_sha256'], restorer_state_sha256=state['restorer_state_sha256'],
                recorded_weight_reload_resume_audit=recorded, actual_log_audit=executed,
                archived_sources_verified=157, current_training_sources_checked=False,
                monitor_images=108, pipeline_sha256=sha(run / 'pipeline_status.json'),
                state_sha256=sha(run / 'candidate/status.json'), newgt=pipeline['contract']['newgt'],
                caveat='Missing large weights are explicitly recorded-only; no local tensor reload claim.')


class Predictions:
    """读取目录或既有提交ZIP；不解包，不改写预测，不接受额外产物。"""
    def __init__(self, location):
        self.path = Path(location)
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.archive:
            members = [entry.filename for entry in self.archive.infolist() if not entry.is_dir()]
            if len(members) != len(set(members)):
                raise RuntimeError('Duplicate ZIP member')
            for name in members:
                safe_member(name)
            self.members = set(members)
        else:
            self.members = {p.name for p in self.path.iterdir() if p.is_file()}

    def bytes(self, name):
        return self.archive.read(name) if self.archive else (self.path / name).read_bytes()

    def require(self, names):
        expected = {stem + suffix for stem in names for suffix in ('_inst.png', '_class.json')}
        if self.members != expected:
            raise RuntimeError('Exactly100 final PNG/JSON pairs required')

    def load(self, stem, shape=None):
        ids = cv2.imdecode(np.frombuffer(self.bytes(stem + '_inst.png'), np.uint8), cv2.IMREAD_UNCHANGED)
        classes = json.loads(self.bytes(stem + '_class.json'))
        if (ids is None or ids.dtype != np.uint16 or ids.ndim != 2 or int(ids.max()) > 65535
                or (shape is not None and ids.shape != tuple(shape))
                or {str(int(i)) for i in np.unique(ids) if i > 0} != set(classes)
                or any(type(c) is not int or c not in (0, 1) for c in classes.values())):
            raise RuntimeError('Final uint16/class/shape contract differs: ' + stem)
        return ids, classes

    def close(self):
        if self.archive:
            self.archive.close()


def validate_manifest(manifest, digest, *, candidate):
    selected = [r for r in manifest.get('images', []) if r.get('view') == 'patch1024']
    names = [r.get('source') for r in selected]
    if (manifest.get('complete') is not True or manifest.get('epoch') != 20
            or manifest.get('precision') != 'FP32' or manifest.get('checkpoint_sha256') != digest
            or 1024 not in manifest.get('sizes', []) or (candidate and manifest['sizes'] != [1024])
            or len(names) != 100 or len(set(names)) != 100 or any(not n for n in names)):
        raise RuntimeError('Complete fixed-e20 native1024 manifest differs')
    return {r['source']: r for r in selected}


def area_values(ids, classes):
    counts = np.bincount(ids.ravel())
    return np.asarray([counts[int(k)] for k, value in classes.items() if value == 1], dtype=np.int64)


def quantiles(values):
    return {str(q): float(np.percentile(values, q)) if len(values) else None for q in (0, 10, 25, 50, 75, 90, 100)}


def output_audit(args):
    run = Path(args.run_root)
    manifest = read(run / 'candidate/deployment/manifest.json')
    selected = validate_manifest(manifest, args.expected_checkpoint, candidate=True)
    names = list(selected)
    ref = Predictions(args.reference_predictions)
    candidate = Predictions(run / 'candidate/deployment/patch1024')
    ref.require(names); candidate.require(names)
    records = []
    try:
        for name in names:
            old = ref.load(name)
            new = candidate.load(name, old[0].shape)
            phases = phase_description(*new)
            if phases != selected[name]['phases']:
                raise RuntimeError('Actual final output differs from recorded manifest: ' + name)
            before, after = phase_description(*old), phases
            areas = dict(r1=area_values(*old), coverage=area_values(*new))
            median = float(np.median(areas['r1'])) if len(areas['r1']) else None
            stats = {}
            for arm, values in areas.items():
                stats[arm] = dict(instances=len(values), pixels=int(values.sum()), area_quantiles=quantiles(values),
                    tiny_under200=int((values < 200).sum()), tiny_under200_pixels=int(values[values < 200].sum()),
                    fixed_r1_median_bins=None if median is None else dict(
                        under_quarter=int((values < median / 4).sum()),
                        quarter_to_half=int(((values >= median / 4) & (values < median / 2)).sum()),
                        half_to_two=int(((values >= median / 2) & (values < median * 2)).sum()),
                        at_least_two=int((values >= median * 2).sum())))
            if before['ferrite']['mean_area'] and after['ferrite']['mean_area']:
                delta = 100 * (after['ferrite']['mean_area'] / before['ferrite']['mean_area'] - 1)
            else:
                delta = None
            records.append(dict(source=name, shape=list(old[0].shape), phases=dict(r1=before, coverage=after),
                ferrite_stats=stats, ferrite_mean_area_change_percent=delta,
                relation=relation(old[0], new[0], old[1], new[1]),
                final_ids_exact=bool(np.array_equal(old[0], new[0])), class_json_exact=old[1] == new[1],
                sha256={arm: {suffix: hashlib.sha256(reader.bytes(name + suffix)).hexdigest()
                             for suffix in ('_inst.png', '_class.json')}
                        for arm, reader in [('r1', ref), ('coverage', candidate)]}))
    finally:
        ref.close(); candidate.close()
    totals = {arm: {key: sum(r['ferrite_stats'][arm][key] for r in records)
                        for key in ('instances', 'pixels', 'tiny_under200', 'tiny_under200_pixels')}
              for arm in ('r1', 'coverage')}
    for value in totals.values():
        value['pooled_mean_area'] = value['pixels'] / value['instances'] if value['instances'] else None
    keys = ('split_relations', 'merge_relations', 'matched_iou50', 'matched_iou95',
            'class_changed_matched_iou50', 'class_changed_pixels', 'common_covered_pixels')
    comparisons = dict(relations={k: sum(r['relation'][k] for r in records) for k in keys},
                       exact_final_images=sum(r['final_ids_exact'] and r['class_json_exact'] for r in records))
    comparisons['by_shape'] = {}
    for group, chosen in [('all', records), ('large', [r for r in records if max(r['shape']) > 2048]),
                          ('other', [r for r in records if max(r['shape']) <= 2048])]:
        values = [r['ferrite_mean_area_change_percent'] for r in chosen if r['ferrite_mean_area_change_percent'] is not None]
        comparisons['by_shape'][group] = dict(images=len(chosen), mean_area_change_percent=quantiles(values),
                                             area_decreased=sum(v < 0 for v in values))
    return dict(complete=True, images=records, totals=totals, comparisons=comparisons,
                reference_location=str(args.reference_predictions), manifest_sha256=sha(run / 'candidate/deployment/manifest.json'),
                scope='100 unlabeled final outputs; prediction-to-prediction matches, not GT correctness or area score.',
                no_inference_rerun=True, no_threshold_search=True,
                caveat='Small instances are not labelled errors; semantic vote may change when affinity changes instances.')


def pin_files(folder, manifest):
    if manifest.get('complete') is not True:
        raise RuntimeError('Complete cache manifest required')
    for name, digest in manifest['files'].items():
        if sha(Path(folder) / safe_member(name)) != digest:
            raise RuntimeError('Cached artifact differs: ' + name)


def gallery(args):
    """复用原训练完成后图册；样本先固定，不按此次变化挑图。"""
    run, out = Path(args.run_root), Path(args.output)
    monitor = set(read(run / 'candidate/status.json')['config']['backend_adaptation']['monitor']['images'])
    if set(VISUAL_RANDOM) & monitor:
        raise RuntimeError('Preselected random images overlap process monitor')
    target = out / 'gallery'; target.mkdir()
    selected = []
    for name in (*VISUAL_FIXED, *VISUAL_RANDOM):
        path = run / 'gallery' / (name + '.png')
        if not path.is_file():
            raise RuntimeError('Original final paired gallery missing: ' + name)
        shutil.copyfile(path, target / path.name)
        selected.append(dict(source=name, original_gallery_sha256=sha(path)))
    write(out / 'selection.json', dict(fixed=list(VISUAL_FIXED), random=list(VISUAL_RANDOM),
          random_seed=20261003, selection_before_output_inspection=True, rows=selected))
    body = ['<!doctype html><meta charset="utf-8"><title>Coverage e20</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<p>原图 / 固定R1 mixed balance e20 / coverage e20。两个固定样本与预先随机六图；无测试GT。</p>',
            '<p><a href="training.json">训练审计</a> | <a href="test.json">100图变化</a></p>']
    for record in selected:
        name = record['source']
        body.append(f'<p>{name}<br><a href="gallery/{name}.png"><img src="gallery/{name}.png"></a></p>')
    (out / 'index.html').write_text('\n'.join(body), encoding='utf8')


def candidate_gt(args):
    """实际GPU只推候选32图；旧R1仅两图精确门禁，之后复用全部缓存。"""
    import torch
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend
    from tools.analyze_ferrite_native import analyze_ferrite_native
    from tools.run_affinity_sweep import runtime_contract
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_affinity_coverage import load_original_core, reference_config
    from train_backend_adaptation import tensor_digest
    from utils.config import project_path, load_config
    if not torch.cuda.is_available():
        raise RuntimeError('sam2_env actual GPU required')
    run, cache = Path(args.run_root), Path(args.reference_cache)
    out = Path(args.output) / 'gt'
    if out.exists():
        raise RuntimeError('Fresh GT analysis required')
    audit = training_audit(args)
    if not all(r['actual_file_sha_verified'] for r in audit['checkpoints'].values()):
        raise RuntimeError('GPU must verify both actual checkpoint files')
    pipeline = read(run / 'pipeline_status.json')
    for name, digest in pipeline['sources'].items():
        if sha(ROOT / safe_member(name)) != digest:
            raise RuntimeError('GPU current source differs from sealed157 snapshot: ' + name)
    pin_files(cache, read(cache / 'capture_manifest.json'))
    if sha(cache / 'capture.json') != CAPTURE_SHA or sha(cache / 'report.json') != REPORT_SHA:
        raise RuntimeError('Approved original32 R1 cache identity differs')
    capture, old_report = read(cache / 'capture.json'), read(cache / 'report.json')
    config = audit['configuration']
    base = reference_config(config)
    if (not capture.get('complete') or capture.get('checkpoint_sha256') != REFERENCE_SHA
            or len(capture['rows']) != 32 or len(capture['seed_map']) != 32
            or capture_config_projection(capture['config'], load_config(str(ROOT/'config/train/affinity_tail.yaml')))
               != normalized_config(base)):
        raise RuntimeError('Full32 native R1 cache contract differs')
    gt_audit = audit_new_gt(config)
    if gt_audit != capture['newgt'] or gt_audit != pipeline['contract']['newgt']:
        raise RuntimeError('Canonical processed32 newGT including filled differs')
    runtime = runtime_contract(config)
    # 迁移环境差异显式报告；不能谎称旧运行环境等价。数值部署等价另由两图门禁核验。
    runtime_delta = {k: dict(old=capture['sealed_runtime'].get(k), current=v)
                     for k, v in runtime.items() if v != capture['sealed_runtime'].get(k)}
    core = load_original_core(config)
    restorer = get_restorer(config, 'cuda')
    d5a = tensor_digest(restorer.state_dict().items())
    if d5a != audit['restorer_state_sha256']:
        raise RuntimeError('Actual D5a state differs')
    old_model = load_backend(Path(args.reference) / 'final.pt', base, 'cuda')[0].eval().requires_grad_(False)
    reference_state = tensor_digest(old_model.state_dict().items())
    if reference_state != capture['frozen_states_sha256']['model']:
        raise RuntimeError('Actual R1 weights differ from cache capture')
    parity = []
    torch.set_num_threads(4); cv2.setNumThreads(2)
    with torch.no_grad():
        for row in capture['rows'][:2]:
            path = Path(project_path(config, row['source_path']))
            fixed = {}
            _, views = core.predict_views(old_model, restorer, path, base, 'cuda', row['seed'], [1024], fixed_audit=fixed)
            value = views['patch1024']
            cached = Predictions(cache / 'masks')
            before = cached.load(path.stem, row['shape']); cached.close()
            with np.load(cache / row['map_file'], allow_pickle=False) as blob:
                exact = np.array_equal(value['boundary'], blob['boundary'])
            if (not np.array_equal(value['instances'], before[0]) or value['classes'] != before[1]
                    or not exact or fixed != row['fixed_whole_semantics'] or value['blend'] != row['blend']):
                raise RuntimeError('Migrated original49 exact R1 deployment parity failed: ' + row['source'])
            parity.append(dict(source=row['source'], seed=row['seed'], exact_final_and_boundary_and_semantic=True))
    del old_model; torch.cuda.empty_cache()
    model = load_backend(run / 'candidate/final.pt', config, 'cuda')[0].eval().requires_grad_(False)
    state = tensor_digest(model.state_dict().items())
    if frozen_digest(model) != audit['frozen_state_sha256']:
        raise RuntimeError('Actual frozen candidate encoder/LoRA/semantic differs')
    out.mkdir(parents=True); (out / 'masks').mkdir()
    previous = {r['source']: r['diagnostic'] for r in old_report['rows']}
    records = []
    with torch.no_grad():
        for row in capture['rows']:
            if row['seed'] != capture['seed_map'][row['source']]:
                raise RuntimeError('Full32 cached source seed map differs')
            path, gt_path = (Path(project_path(config, row[key])) for key in ('source_path', 'gt_path'))
            if sha(path) != row['source_file_sha256'] or sha(gt_path) != row['gt_file_sha256']:
                raise RuntimeError('Source or processed GT differs: ' + row['source'])
            rgb = read_rgb(str(path))
            gt, lookup = load_completed_semantic_source(gt_path, rgb.shape[:2])
            classes = {int(i): int(lookup[i]) for i in np.unique(gt) if i > 0}
            cached = Predictions(cache / 'masks'); before = cached.load(path.stem, gt.shape); cached.close()
            old = analyze_ferrite_native(gt, before[0], classes, before[1])
            if old != previous[row['source']]:
                raise RuntimeError('Full32 cached R1 diagnostic no longer reproduces canonical newGT')
            fixed = {}
            _, views = core.predict_views(model, restorer, path, config, 'cuda', row['seed'], [1024], fixed_audit=fixed)
            value = views['patch1024']
            core.verify_prediction(value['instances'], value['classes'], gt.shape)
            if fixed != row['fixed_whole_semantics']:
                raise RuntimeError('Frozen whole-image semantic paths changed')
            diagnostic = analyze_ferrite_native(gt, value['instances'], classes, value['classes'])
            if not cv2.imwrite(str(out / 'masks' / (path.stem + '_inst.png')), value['instances']):
                raise RuntimeError('Cannot save uint16 candidate output')
            write(out / 'masks' / (path.stem + '_class.json'), value['classes'])
            if row['source']=='train_659.jpg':
                with np.load(cache/row['map_file'],allow_pickle=False) as blob:
                    weak_case(out,rgb,gt,before,(value['instances'],value['classes']),blob['boundary'],
                              value['boundary'],old,diagnostic,config['inference']['boundary_threshold'])
            fixed_area=fixed_area_cohort(old,diagnostic)
            left, right = ({r['gt']: r for r in d['gt_rows']} for d in (old, diagnostic))
            common = sorted(g for g in left.keys() & right.keys()
                            if left[g]['matched_iou'] is not None and right[g]['matched_iou'] is not None)
            area_common = [g for g in common if left[g]['safe_one_to_one_area_eligible']
                           and right[g]['safe_one_to_one_area_eligible']]
            records.append(dict(source=row['source'], seed=row['seed'], fixed_whole_semantics=fixed, fixed_original_area=fixed_area,
                r1=old, coverage=diagnostic, common_matched_F=[dict(gt=g, r1_iou=left[g]['matched_iou'],
                    coverage_iou=right[g]['matched_iou'], delta=right[g]['matched_iou']-left[g]['matched_iou']) for g in common],
                common_safe_one_to_one_F_area=[dict(gt=g, r1_relative_error=left[g]['area_relative_error'],
                    coverage_relative_error=right[g]['area_relative_error']) for g in area_common]))
            write(out / 'progress.json', dict(complete=False, sources=len(records), expected=32))
            print('GT', len(records), row['source'], flush=True)
    if (tensor_digest(model.state_dict().items()) != state or tensor_digest(restorer.state_dict().items()) != d5a
            or runtime_contract(config) != runtime or audit_new_gt(config) != gt_audit):
        raise RuntimeError('Read-only analysis mutated model/restorer/runtime/newGT')
    totals = {arm: {key: sum(r[arm]['summary']['mechanism_counts'][key] for r in records)
                   for key in records[0][arm]['summary']['mechanism_counts']} for arm in ('r1', 'coverage')}
    common_rows = [r for row in records for r in row['common_matched_F']]
    original_area=[r for row in records for r in row['fixed_original_area']['rows']]
    if len(original_area)!=467:
        raise RuntimeError('Original fixed R1 safe-area cohort467 differs')
    report = dict(complete=True, sources=32, epoch=20, no_training=True, no_intervention=True,
        checkpoints=audit['checkpoints'], original49_binding=True, exact_two_source_parity=parity,
        current_runtime=runtime, recorded_training_runtime=pipeline['contract']['runtime'],
        migrated_runtime_delta=runtime_delta, runtime_exact_equal=not runtime_delta,
        newgt=gt_audit, rows=records, mechanism_totals=totals,
        common_matched_F=dict(count=len(common_rows), iou_delta=quantiles([r['delta'] for r in common_rows])),
        fixed_original_F_area=dict(original_count=467,comparable_count=sum(r['candidate_comparable'] for r in original_area),
            failure_counts={key:sum(row['fixed_original_area']['failure_counts'][key] for row in records)
                for key in records[0]['fixed_original_area']['failure_counts']},
            original_r1_relative_error=quantiles([r['r1_relative_error'] for r in original_area]),
            candidate_common_comparable_relative_error=quantiles([r['candidate_relative_error'] for r in original_area if r['candidate_comparable']]),
            new_eligible_excluded_from_original_cohort=sum(len(row['fixed_original_area']['newly_eligible_not_in_original_cohort']) for row in records)),
        scope='32 seen labelled sources; full processed newGT including filled; unknown/frame censoring unchanged. Not validation or official scores.')
    write(out / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('cpu', 'gt', 'all'), default='cpu')
    p.add_argument('--run-root', default='outputs/affinity_coverage')
    p.add_argument('--reference', default='outputs/affinity_balance/mixed')
    p.add_argument('--reference-predictions', default='outputs/affinity_balance/mixed/deployment/patch1024')
    p.add_argument('--reference-cache', default='outputs/ferrite_native/analysis')
    p.add_argument('--plan', default='outputs/affinity_crop_trial/metadata_receipts.jsonl')
    p.add_argument('--expected-checkpoint', default=EXPECTED_CANDIDATE)
    p.add_argument('--output', default='outputs/affinity_coverage/analysis')
    args = p.parse_args(); out = Path(args.output)
    if args.mode in ('cpu', 'all'):
        if out.exists():
            raise RuntimeError('Fresh private CPU analysis output required')
        training = training_audit(args); test = output_audit(args)
        out.mkdir(parents=True)
        write(out / 'training.json', training); write(out / 'test.json', test)
        write(out / 'summary.json', dict(complete=True, training=training,
            test={k: v for k, v in test.items() if k != 'images'}))
        (out / 'sources').mkdir(); shutil.copyfile(__file__, out / 'sources' / Path(__file__).name)
        gallery(args)
        print(json.dumps(dict(stage='cpu_complete', comparisons=test['comparisons'], totals=test['totals']), ensure_ascii=False))
    if args.mode in ('gt', 'all'):
        candidate_gt(args)


if __name__ == '__main__':
    main()
