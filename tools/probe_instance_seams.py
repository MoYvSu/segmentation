# -*- coding: utf-8 -*-
"""一次预设的同相细零缝归属：32已见训练图CPU诊断，不是官方打分器。

归属核心只接最终掩码和原类别；完整processed GT仅用于事后分析。
原467完整F面积对固定GT/预测ID直接比较，候选触unknown也不从名单删除。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.analyze_ferrite_native import (
    _classes, _color_panel, _labels, _unsafe_ids, analyze_ferrite_native,
)
from tools.assign_instance_seams import assign_instance_seams

FORMAT = 'instance_seams_seen_gt_v1'
PARENT_REPORT_SHA = 'c47b46c1d08a2b54fd5f6d9c230b82fac41b75051906cbee5be57d85deb3957d'
PARENT_CAPTURE_SHA = '7fc4c1a23f609322a44cfb12f70f29110cf0b16afbeb4649563cafc82217f611'
PARENT_MANIFEST_SHA = 'ad0a933980152ef6e6895c961e5f29c08e6baa8c72f0566f0ff888d7849a23e9'
R1_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
CORE_SHA = '9a6b5f16d82f0afcd75dbea812789816c24cc46d4200d45f82790b3cdb7db0b0'
FERRITE_CORE_SHA = 'e59e95f8813d3fa68a9ae641467bce7b817b33e5651728857105f760018746b2'
CAVEATS = [
    'All labelled sources were seen in training; these are mechanism diagnostics, not independent generalization or official scores.',
    'Full processed newGT including filled is used only after the GT-free assignment; GT0/invalid is ignored in known-domain counts.',
    'Original positive pixels, class table and nonzero ID set remain fixed; connected components and contacts may nevertheless change.',
    'Final0 alone does not identify watershed boundary versus filtering or another zero source, especially on unlabelled test images.',
    'Fixed parent safe-F area pairs remain in the denominator even if the candidate prediction becomes censored; newly censored pairs are explicit.',
    'Geometry matching uses IoU>=0.5 on the same known domain. Counts, conditional matching IoU and seen-GT area errors are not contest scores.',
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def _phase(labels, classes):
    lookup = np.full(65536, -1, np.int8)
    for iid, phase in classes.items():
        lookup[iid] = phase
    return lookup[labels]


def _matrix(gt_phase, pred_phase, known):
    # 两相加未分配三列闭合；包括完全不接触F的P预测，不能只读F贡献表。
    rows = [[int(np.count_nonzero(known & (gt_phase == truth) & (pred_phase == guess)))
             for guess in (0, 1, -1)] for truth in (0, 1)]
    if sum(sum(row) for row in rows) != int(known.sum()):
        raise RuntimeError('Known phase allocation does not close')
    return rows


def validate_assignment(baseline, candidate, pred_classes):
    """独立检查原8邻最短距离集合，不依赖核心EDT平局次序或GT。"""
    a, b = _labels(baseline, 'baseline'), _labels(candidate, 'candidate')
    if a.dtype != np.uint16 or b.dtype != np.uint16 or a.shape != b.shape:
        raise ValueError('Same native uint16 final-mask grids required')
    pc = _classes(pred_classes, a, 'pred_classes')
    if _classes(pred_classes, b, 'candidate_classes') != pc:
        raise ValueError('Nonzero IDs/classes changed')
    positive, zero = a > 0, a == 0
    if not np.array_equal(a[positive], b[positive]):
        raise ValueError('Original positive pixels changed')
    h, w = a.shape
    padded = np.pad(a, 1)
    phases = _phase(a, pc)
    padded_phase = np.pad(phases, 1, constant_values=-1)
    has_p, has_f = np.zeros(a.shape, bool), np.zeros(a.shape, bool)
    cardinal = np.zeros(a.shape, bool)
    chosen_cardinal, chosen_diagonal = np.zeros(a.shape, bool), np.zeros(a.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            neighbor = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
            phase = padded_phase[1+dy:1+dy+h, 1+dx:1+dx+w]
            has_p |= phase == 0
            has_f |= phase == 1
            if abs(dx)+abs(dy) == 1:
                cardinal |= neighbor > 0
                chosen_cardinal |= (neighbor > 0) & (neighbor == b)
            else:
                chosen_diagonal |= (neighbor > 0) & (neighbor == b)
    frame = np.zeros(a.shape, bool)
    frame[[0, -1], :] = True; frame[:, [0, -1]] = True
    eligible = zero & ~frame & (has_p | has_f) & ~(has_p & has_f)
    added = zero & (b > 0)
    if not np.array_equal(added, eligible):
        raise ValueError('Assignment is not exactly the original single-pass eligibility set')
    if np.any(added & ~np.where(cardinal, chosen_cardinal, chosen_diagonal)):
        raise ValueError('Chosen label is not an original nearest Euclidean neighbor')
    return dict(passed=True, eligible_pixels=int(eligible.sum()), assigned_pixels=int(added.sum()),
                original_positive_pixels_unchanged=True, instance_ids_and_classes_unchanged=True,
                nearest_original_neighbor_verified=True, no_propagation_verified=True,
                outer_frame_unchanged=True, topology_unchanged_claim=False)


def _area_stats(rows):
    if not rows:
        return dict(n=0, gt_pixels=0, before_pred_pixels=0, after_pred_pixels=0,
                    before_mean_signed=None, after_mean_signed=None,
                    before_mean_absolute=None, after_mean_absolute=None,
                    improved=0, worsened=0, tied=0, newly_censored=0)
    before = np.asarray([r['before_relative_error'] for r in rows])
    after = np.asarray([r['after_relative_error'] for r in rows])
    delta = np.abs(after)-np.abs(before)
    ga = sum(r['gt_pixels'] for r in rows)
    ba, ca = (sum(r[k] for r in rows) for k in ('before_pred_pixels', 'after_pred_pixels'))
    return dict(n=len(rows), gt_pixels=ga, before_pred_pixels=ba, after_pred_pixels=ca,
        before_mean_signed=float(before.mean()), after_mean_signed=float(after.mean()),
        before_mean_absolute=float(np.abs(before).mean()), after_mean_absolute=float(np.abs(after).mean()),
        before_median_signed=float(np.median(before)), after_median_signed=float(np.median(after)),
        before_pooled_signed=(ba-ga)/ga, after_pooled_signed=(ca-ga)/ga,
        improved=int(np.count_nonzero(delta < -1e-12)), worsened=int(np.count_nonzero(delta > 1e-12)),
        tied=int(np.count_nonzero(np.abs(delta) <= 1e-12)),
        newly_censored=sum(r['candidate_pred_censored'] for r in rows),
        before_shrinking=int(np.count_nonzero(before < 0)), after_shrinking=int(np.count_nonzero(after < 0)),
        before_expanding=int(np.count_nonzero(before > 0)), after_expanding=int(np.count_nonzero(after > 0)))


def _pair_changes(before, after, gc, safe_gt_ids=None):
    a, b = {r['gt']: r for r in before}, {r['gt']: r for r in after}
    if safe_gt_ids is not None:
        a = {k: v for k, v in a.items() if k in safe_gt_ids}
        b = {k: v for k, v in b.items() if k in safe_gt_ids}
    rows = []
    for gid in sorted(set(a) & set(b)):
        rows.append(dict(gt=gid, phase=gc[gid], before_pred=a[gid]['pred'], after_pred=b[gid]['pred'],
            before_iou=a[gid]['iou'], after_iou=b[gid]['iou'], delta_iou=b[gid]['iou']-a[gid]['iou'],
            pred_identity_unchanged=a[gid]['pred'] == b[gid]['pred']))
    def stats(values):
        n = len(values)
        return dict(n=n, before_mean_iou=sum(r['before_iou'] for r in values)/n if n else None,
            after_mean_iou=sum(r['after_iou'] for r in values)/n if n else None,
            mean_delta_iou=sum(r['delta_iou'] for r in values)/n if n else None,
            improved=sum(r['delta_iou'] > 1e-12 for r in values),
            worsened=sum(r['delta_iou'] < -1e-12 for r in values),
            tied=sum(abs(r['delta_iou']) <= 1e-12 for r in values),
            changed_pred_identity=sum(not r['pred_identity_unchanged'] for r in values))
    return dict(before_n=len(a), after_n=len(b), common_rows=rows,
        gained_gt=sorted(set(b)-set(a)), lost_gt=sorted(set(a)-set(b)),
        common_all=stats(rows), common_F=stats([r for r in rows if r['phase'] == 1]))


def analyze_instance_seams(gt, baseline, candidate, gt_classes, pred_classes,
                           parentdiagnostic=None, valid_content=None):
    """同网格事后诊断。GT不会传入归属核心；原完整F名单不重选。"""
    validation = validate_assignment(baseline, candidate, pred_classes)
    gt = _labels(gt, 'gt')
    if gt.shape != baseline.shape:
        raise ValueError('GT must match the exact native prediction grid')
    gc, pc = _classes(gt_classes, gt, 'gt_classes'), _classes(pred_classes, baseline, 'pred_classes')
    content = np.ones(gt.shape, bool) if valid_content is None else np.asarray(valid_content)
    if content.shape != gt.shape or content.dtype != np.bool_:
        raise ValueError('valid_content must be same-grid boolean')
    known = (gt > 0) & content
    old = analyze_ferrite_native(gt, baseline, gc, pc, content)
    if parentdiagnostic is not None and old != parentdiagnostic:
        raise ValueError('Fixed parent diagnostic differs from the actual baseline arrays')
    new = analyze_ferrite_native(gt, candidate, gc, pc, content)
    gp, bp, cp = _phase(gt, gc), _phase(baseline, pc), _phase(candidate, pc)
    added = (baseline == 0) & (candidate > 0)
    before_matrix, after_matrix = _matrix(gp, bp, known), _matrix(gp, cp, known)
    gt_area = np.bincount(gt[known], minlength=65536)
    before_area = np.bincount(baseline.ravel(), minlength=65536)
    after_area = np.bincount(candidate.ravel(), minlength=65536)
    unsafe = _unsafe_ids(candidate, known)
    fixed = []
    # 一次交叉计数，只计算被填像素，避免逐个GT扫描整图。
    encoded, counts = np.unique(gt[added].astype(np.int64)*65536 + candidate[added], return_counts=True)
    added_pairs = {int(code): int(n) for code, n in zip(encoded, counts)}
    for r in old['gt_rows']:
        if not r['safe_one_to_one_area_eligible']:
            continue
        gid, pid = r['gt'], r['matched_pred']
        ga, ba, ca = int(gt_area[gid]), int(before_area[pid]), int(after_area[pid])
        if ga != r['area'] or ba != r['matched_pred_total_pixels']:
            raise ValueError('Fixed parent complete-F area denominator differs')
        gain = ca-ba
        gain_in_gt = added_pairs.get(gid*65536+pid, 0)
        fixed.append(dict(gt=gid, matched_pred=pid, gt_pixels=ga,
            before_pred_pixels=ba, after_pred_pixels=ca, added_pred_pixels=gain,
            added_pred_inside_own_GT=gain_in_gt, added_pred_outside_own_GT=gain-gain_in_gt,
            before_relative_error=(ba-ga)/ga, after_relative_error=(ca-ga)/ga,
            candidate_pred_censored=pid in unsafe,
            candidate_still_reselected_safe=any(nr['gt'] == gid and nr['safe_one_to_one_area_eligible']
                                                for nr in new['gt_rows'])))
    safe_gt = set(gc)-set(old['censored_gt_ids'])-set(new['censored_gt_ids'])
    # 限定两路配对的预测均不censor；不替代所有共同对象的主结果。
    safe_pair_gids = {r['gt'] for r in old['pairs'] if r['pred'] not in old['censored_pred_ids']}
    safe_pair_gids &= {r['gt'] for r in new['pairs'] if r['pred'] not in new['censored_pred_ids']}
    safe_pair_gids &= safe_gt
    fixed_ids = {r['gt'] for r in fixed}
    new_fixed_ids = {r['gt'] for r in new['gt_rows'] if r['safe_one_to_one_area_eligible']}
    result = dict(format=FORMAT, shape=list(gt.shape), assignment_validation=validation,
        known_phase_before=before_matrix, known_phase_after=after_matrix,
        known_phase_delta=[[b-a for a,b in zip(x,y)] for x,y in zip(before_matrix,after_matrix)],
        matrix_rows=['GT_P', 'GT_F'], matrix_columns=['pred_P', 'pred_F', 'pred0'],
        known_phase_allocation_closed=True,
        added_pixels=dict(total=int(added.sum()), known=int((added & known).sum()),
            unknown_or_invalid=int((added & ~known).sum()),
            known_correct_phase=int(np.count_nonzero(added & known & (gp == cp))),
            known_wrong_phase=int(np.count_nonzero(added & known & (gp != cp))),
            by_pred_phase={label:int(np.count_nonzero(added & (cp == phase))) for phase,label in ((0,'P'),(1,'F'))}),
        matched_change=_pair_changes(old['pairs'], new['pairs'], gc),
        conservative_common_matched_change=_pair_changes(old['pairs'], new['pairs'], gc, safe_pair_gids),
        fixed_parent_safe_F_rows=fixed, fixed_parent_safe_F_summary=_area_stats(fixed),
        fixed_parent_safe_F_remaining_uncensored_summary=_area_stats([r for r in fixed if not r['candidate_pred_censored']]),
        candidate_reselected_safe_F=dict(n=len(new_fixed_ids), new_gt_ids=sorted(new_fixed_ids-fixed_ids),
            lost_parent_gt_ids=sorted(fixed_ids-new_fixed_ids), secondary_only=True),
        counts=dict(instances=len(pc), F_instances=sum(v==1 for v in pc.values()),
            P_instances=sum(v==0 for v in pc.values()), ID_set_and_classes_fixed=True,
            F_pixels_before=int(np.count_nonzero(bp == 1)), F_pixels_after=int(np.count_nonzero(cp == 1)),
            P_pixels_before=int(np.count_nonzero(bp == 0)), P_pixels_after=int(np.count_nonzero(cp == 0))),
        before_mechanisms=old['summary']['mechanism_counts'], after_mechanisms=new['summary']['mechanism_counts'],
        candidate_diagnostic=new, caveats=list(CAVEATS))
    if result['added_pixels']['known_correct_phase']+result['added_pixels']['known_wrong_phase'] != result['added_pixels']['known']:
        raise RuntimeError('Added known phase correctness does not close')
    if any(after_matrix[i][j] < before_matrix[i][j] for i in range(2) for j in range(2)):
        raise RuntimeError('Original known positive phase allocation unexpectedly decreased')
    json.dumps(result, allow_nan=False)
    return result


def summarize(rows):
    """逐源与固定对象宏平均；不把不完整GT均面积当比赛面积。"""
    values = [r['diagnostic'] for r in rows]
    fixed = [dict(source=row['source'], **area) for row in rows
             for area in row['diagnostic']['fixed_parent_safe_F_rows']]
    paired = [dict(source=row['source'], **p) for row in rows
              for p in row['diagnostic']['matched_change']['common_rows']]
    common_f = [p for p in paired if p['phase'] == 1]
    before = np.asarray([v['known_phase_before'] for v in values], np.int64).sum(0) if values else np.zeros((2,3),np.int64)
    after = np.asarray([v['known_phase_after'] for v in values], np.int64).sum(0) if values else np.zeros((2,3),np.int64)
    return dict(sources=len(rows), fixed_parent_safe_F=_area_stats(fixed),
        known_phase_before=before.tolist(), known_phase_after=after.tolist(), known_phase_delta=(after-before).tolist(),
        added_pixels={key:sum(v['added_pixels'][key] for v in values)
                      for key in ('total','known','unknown_or_invalid','known_correct_phase','known_wrong_phase')},
        common_matched_F=dict(n=len(common_f), before_mean_iou=np.mean([p['before_iou'] for p in common_f]).item() if common_f else None,
            after_mean_iou=np.mean([p['after_iou'] for p in common_f]).item() if common_f else None,
            improved=sum(p['delta_iou']>1e-12 for p in common_f), worsened=sum(p['delta_iou'] < -1e-12 for p in common_f)),
        source_rows=[dict(source=r['source'], fixed_F=r['diagnostic']['fixed_parent_safe_F_summary'],
            common_F=r['diagnostic']['matched_change']['common_F'], added=r['diagnostic']['added_pixels']) for r in rows],
        original_fixed_pair_list=[dict(source=r['source'],gt=a['gt'],pred=a['matched_pred'])
                                  for r in rows for a in r['diagnostic']['fixed_parent_safe_F_rows']],
        caveats=list(CAVEATS))


def _render(folder, stem, rgb, gt, baseline, candidate, case):
    focus = gt == case['gt']
    y,x = np.nonzero(focus)
    box = [max(0,int(y.min())-20),max(0,int(x.min())-20),min(gt.shape[0],int(y.max())+21),min(gt.shape[1],int(x.max())+21)]
    known = gt > 0
    panels = [('Seen original RGB',rgb), ('Full processed newGT',_color_panel(rgb,gt,gt,known)),
              ('Best R1 final',_color_panel(rgb,baseline,gt,known)),
              ('Same-phase assignment',_color_panel(rgb,candidate,gt,known))]
    crops=[]
    for title,panel in panels:
        crop=panel[box[0]:box[2],box[1]:box[3]]
        crop=cv2.resize(crop,(320,max(1,round(320*crop.shape[0]/crop.shape[1]))),interpolation=cv2.INTER_AREA)
        crop=cv2.copyMakeBorder(crop,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(crop,title,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        crops.append(crop)
    folder.mkdir(parents=True,exist_ok=True)
    name=stem+'_fixed_F_'+str(case['gt'])+'.png'
    if not cv2.imwrite(str(folder/name),cv2.cvtColor(np.concatenate(crops,1),cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Gallery write failed')
    return dict(source=stem,gt=case['gt'],pred=case['matched_pred'],native_box_yxyx=box,file=name,
        rule='first source sorted and lowest original safe-F GT ID with any assigned pixels; no ranking by improvement')


def run(output):
    """服务器既有32缓存；只CPU、固定父回执和全部来源，不重复模型推理。"""
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from tools.analyze_affinity_tail_gt import runtime_files_contract
    from tools.probe_ferrite_native import protect_manifest, verify_protected
    from tools.render_backend_ablation import read_prediction
    from tools.run_affinity_tail import verify_newgt, verify_snapshot
    from tools.verify_affinity_newgt import audit_new_gt
    from utils.config import project_path

    output=Path(output).resolve()
    if output != ROOT/'outputs/affinity_seams/seen' or output.exists():
        raise RuntimeError('Fresh outputs/affinity_seams/seen required')
    parent=ROOT/'outputs/ferrite_native/analysis'
    for name,digest in [('report.json',PARENT_REPORT_SHA),('capture.json',PARENT_CAPTURE_SHA),('download_manifest.json',PARENT_MANIFEST_SHA)]:
        if _sha(parent/name)!=digest:
            raise RuntimeError('Accepted parent identity differs: '+name)
    report,capture=_read(parent/'report.json'),_read(parent/'capture.json')
    if (report.get('complete') is not True or capture.get('complete') is not True
            or len(report['rows'])!=32 or len(capture['rows'])!=32
            or len(capture['sealed_sources'])!=148 or capture['checkpoint_sha256']!=R1_SHA
            or _sha(ROOT/capture['checkpoint'])!=R1_SHA
            or _sha(ROOT/'tools/assign_instance_seams.py')!=CORE_SHA
            or _sha(ROOT/'tools/analyze_ferrite_native.py')!=FERRITE_CORE_SHA):
        raise RuntimeError('Fixed R1/full32/source148/new-core contract differs')
    protected=dict(capture['protected_sha256'])
    protect_manifest(parent,protected)
    protected.update({str(Path(__file__)):_sha(__file__),str(ROOT/'tools/assign_instance_seams.py'):CORE_SHA})
    def gates():
        verify_protected(protected)
        verify_snapshot(ROOT/'outputs/affinity_tail',capture['sealed_sources'])
        gt_audit=audit_new_gt(capture['config']); verify_newgt(capture['config'],gt_audit)
        runtime_audit=runtime_files_contract(capture['sealed_runtime'],capture['config'])
        if gt_audit!=capture['newgt'] or runtime_audit!=capture['runtime_audit']:
            raise RuntimeError('NewGT or sealed runtime differs')
        return dict(passed=True, protected_files=len(protected), source_files=148,
                    newGT=gt_audit, runtime=runtime_audit)
    start_gates=gates()
    old_rows={r['source']:r for r in report['rows']}
    names=[r['source'] for r in capture['rows']]
    seed_base=int(capture['config']['backend_adaptation']['restoration']['inference_seed'])
    if (len(set(names))!=32 or set(names)!=set(old_rows)
            or capture['seed_map']!={name:seed_base+i for i,name in enumerate(names)}
            or any(r['seed']!=capture['seed_map'][r['source']] or old_rows[r['source']]['seed']!=r['seed'] for r in capture['rows'])
            or sum(sum(g['safe_one_to_one_area_eligible'] for g in r['diagnostic']['gt_rows']) for r in report['rows'])!=467):
        raise RuntimeError('Full32 seeds or original467 cohort differs')
    output.mkdir(parents=True)
    (output/'sources').mkdir(); (output/'masks').mkdir()
    for name in ('tools/probe_instance_seams.py','tools/assign_instance_seams.py'):
        saved=output/'sources'/Path(name).name
        shutil.copyfile(ROOT/name,saved)
        protected[str(saved)]=_sha(saved)
    cv2.setNumThreads(2)
    rows=[]; gallery=[]
    selected_sources=sorted(names)[:2]
    for item in capture['rows']:
        name=item['source']; stem=Path(name).stem
        gt_path=Path(project_path(capture['config'],item['gt_path']))
        if _sha(gt_path)!=item['gt_file_sha256']:
            raise RuntimeError('GT file differs: '+name)
        gt,lookup=load_completed_semantic_source(gt_path,tuple(item['shape']))
        gc={int(i):int(lookup[i]) for i in np.unique(gt) if i>0}
        if {str(i):v for i,v in gc.items()}!=item['gt_classes']:
            raise RuntimeError('Full processed canonical classes differ')
        baseline,pc,_=read_prediction(parent/'masks',stem,tuple(item['shape']))
        # 核心永远不接GT；GT只在下一行进入分析。
        candidate,receipt=assign_instance_seams(baseline,pc)
        diagnostic=analyze_instance_seams(gt,baseline,candidate,gc,pc,old_rows[name]['diagnostic'])
        if diagnostic['assignment_validation']['assigned_pixels']!=receipt['assigned_pixels']:
            raise RuntimeError('Independent eligible-set audit differs')
        if not cv2.imwrite(str(output/'masks'/f'{stem}_inst.png'),candidate):
            raise RuntimeError('Candidate uint16 write failed')
        shutil.copyfile(parent/item['class_file'],output/'masks'/f'{stem}_class.json')
        roundtrip,roundtrip_classes,_=read_prediction(output/'masks',stem,tuple(item['shape']))
        if (not np.array_equal(roundtrip,candidate) or roundtrip_classes!=pc
                or _sha(parent/item['class_file'])!=_sha(output/'masks'/f'{stem}_class.json')):
            raise RuntimeError('Final PNG/class byte and identity roundtrip differs')
        row=dict(source=name,seed=item['seed'],assignment=receipt,diagnostic=diagnostic,
                 baseline_mask_sha256=_sha(parent/item['mask_file']),
                 candidate_mask_sha256=_sha(output/'masks'/f'{stem}_inst.png'),
                 unchanged_class_file_sha256=_sha(output/'masks'/f'{stem}_class.json'))
        rows.append(row)
        if name in selected_sources:
            cases=[r for r in diagnostic['fixed_parent_safe_F_rows'] if r['added_pred_pixels']>0]
            if cases:
                case=min(cases,key=lambda r:r['gt'])
                rgb=read_rgb(str(project_path(capture['config'],item['source_path'])))
                gallery.append(_render(output/'gallery',stem,rgb,gt,baseline,candidate,case))
        _write(output/'progress.json',dict(complete=False,completed=len(rows),total=32,last_source=name))
        print(f'{len(rows)}/32 {name}',flush=True)
    end_gates=gates()
    summary=summarize(rows)
    if summary['fixed_parent_safe_F']['n']!=467:
        raise RuntimeError('Fixed original467 area list was not preserved')
    result=dict(format=FORMAT,complete=True,CPU_only=True,no_training=True,no_model_forward=True,
        baseline_checkpoint_sha256=R1_SHA,parent_report_sha256=PARENT_REPORT_SHA,
        parent_capture_sha256=PARENT_CAPTURE_SHA,parent_manifest_sha256=PARENT_MANIFEST_SHA,
        source148_unchanged=True,newGT_unchanged=True,sealed_runtime_unchanged=True,
        assignment_core_sha256=CORE_SHA,analysis_script_sha256=_sha(__file__),
        protection_receipt=dict(start=start_gates,end=end_gates,
            immutable_sources_sha256=capture['sealed_sources'],
            protected_sha256=protected, parent_report_sha256=PARENT_REPORT_SHA,
            parent_capture_sha256=PARENT_CAPTURE_SHA,parent_manifest_sha256=PARENT_MANIFEST_SHA,
            original_fixed_F_count=467,candidate_PNG_class_roundtrip_passed=True),
        seed_map=capture['seed_map'],rows=rows,summary=summary,gallery=gallery,caveats=list(CAVEATS))
    _write(output/'report.json',result); _write(output/'summary.json',summary)
    _write(output/'progress.json',dict(complete=True,completed=32,total=32))
    (output/'index.html').write_text('<meta charset="utf-8"><h1>Seen training / single-pass same-phase seams</h1>'+
        ''.join(f'<p>{g["source"]} GT{g["gt"]}</p><img style="max-width:100%" src="gallery/{g["file"]}">' for g in gallery),encoding='utf8')
    return summary


def selftest():
    gt=np.full((40,40),2,np.uint16); gt[10:30,10:30]=1
    baseline=gt.copy(); baseline[15:25,20]=0
    candidate,_=assign_instance_seams(baseline,{1:1,2:0})
    result=analyze_instance_seams(gt,baseline,candidate,{1:1,2:0},{1:1,2:0})
    assert result['fixed_parent_safe_F_summary']['n']==1
    assert result['fixed_parent_safe_F_summary']['after_mean_absolute']==0
    assert result['added_pixels']['known_correct_phase']==10
    assert result['matched_change']['common_F']['mean_delta_iou']>0
    return dict(passed=True,format=FORMAT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--output',default='outputs/affinity_seams/seen')
    args=parser.parse_args()
    print(json.dumps(selftest() if args.selftest else run(args.output),ensure_ascii=False))
