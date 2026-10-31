# -*- coding: utf-8 -*-
"""同源GT两臂固定e5完整推理；与已评分reflect32输出比较，不提交或晋级。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_marker_reflect import (array_sha, capture_original, config_gate,
    load_prediction, precision_record, read, replay_candidates, save_prediction,
    sha, source_snapshot, verify_prediction, write)
from tools.semantic_crossover import _read_prediction
from tools.verify_marker_submission import report_rows
from train_affinity_connectivity import frozen_digest, get_restorer
from train_affinity_gt_source import R1_SHA, D5A_SHA, canonical_sha, validate_config
from train_backend_adaptation import tensor_digest
from utils.config import load_config, project_path


def _phase(ids, classes):
    verify_prediction(ids, classes, ids.shape)
    if any(not k.isascii() or not k.isdecimal() or str(int(k)) != k
           or not 1 <= int(k) <= 65535 for k in classes):
        raise ValueError('Canonical positive class ID keys required')
    if any(type(value) is not int for value in classes.values()):
        raise ValueError('Classes must be integer 0/1')
    lookup = np.full(65536, -1, np.int8)
    for key, value in classes.items():
        lookup[int(key)] = value
    return lookup[ids]


def _areas(ids, classes):
    counts = np.bincount(ids.ravel(), minlength=65536)
    values = np.asarray([counts[int(k)] for k, c in classes.items() if c == 1], np.int64)
    return dict(instances=len(values), pixels=int(values.sum()),
        mean_area=float(values.mean()) if len(values) else None,
        median_area=float(np.median(values)) if len(values) else None,
        q25_area=float(np.quantile(values, .25)) if len(values) else None,
        q75_area=float(np.quantile(values, .75)) if len(values) else None,
        minimum_area=int(values.min()) if len(values) else None,
        maximum_area=int(values.max()) if len(values) else None,
        tiny_under200=int((values < 200).sum()))


def _choose2(counts):
    return sum(int(n)*(int(n)-1)//2 for n in counts)


def compare_predictions(reference_ids, reference_classes, candidate_ids, candidate_classes):
    """无GT的相对划分检查；Rand成对变化与显著交叉均不依赖实例编号。

    Rand仅在双方正ID交集计算，零覆盖变化独立列出。显著split/merge沿现有
    诊断门槛max(50px,该来源实例面积10%)，不作为推理过滤或准确率。
    """
    if reference_ids.shape != candidate_ids.shape:
        raise ValueError('Predictions must have the same native shape')
    rp, cp = _phase(reference_ids, reference_classes), _phase(candidate_ids, candidate_classes)
    common = (reference_ids > 0) & (candidate_ids > 0)
    r, ri = np.unique(reference_ids[common], return_inverse=True)
    c, ci = np.unique(candidate_ids[common], return_inverse=True)
    if common.any():
        # uint16有界；仅保存真正相交的pair，不创建65536x65536矩阵。
        keys, intersections = np.unique(ri.astype(np.int64)*len(c)+ci, return_counts=True)
        rows, cols = keys//len(c), keys%len(c)
        ra = np.bincount(ri, minlength=len(r)); ca = np.bincount(ci, minlength=len(c))
        same_both = _choose2(intersections)
        changed_pairs = _choose2(ra)+_choose2(ca)-2*same_both
        row_degree = np.bincount(rows, minlength=len(r))
        col_degree = np.bincount(cols, minlength=len(c))
        common_partition_identical = bool(np.all(row_degree == 1) and np.all(col_degree == 1))
        rfull, cfull = np.bincount(reference_ids.ravel()), np.bincount(candidate_ids.ravel())
        significant_r = intersections >= np.maximum(50, .1*rfull[r[rows]])
        significant_c = intersections >= np.maximum(50, .1*cfull[c[cols]])
        split_ids = r[np.bincount(rows[significant_r], minlength=len(r)) >= 2]
        merge_ids = c[np.bincount(cols[significant_c], minlength=len(c)) >= 2]
    else:
        changed_pairs = 0; common_partition_identical = True
        split_ids = merge_ids = np.empty(0, np.uint16)
    total_pairs = _choose2([int(common.sum())])
    coverage_changed = int(((reference_ids > 0) != (candidate_ids > 0)).sum())
    rf, cf = _areas(reference_ids, reference_classes), _areas(candidate_ids, candidate_classes)
    return dict(pixels=int(reference_ids.size), common_positive_pixels=int(common.sum()),
        reference_zero_pixels=int((reference_ids == 0).sum()), candidate_zero_pixels=int((candidate_ids == 0).sum()),
        coverage_changed_pixels=coverage_changed,
        phase_changed_pixels=int((rp != cp).sum()),
        common_phase_changed_pixels=int((common & (rp != cp)).sum()),
        ferrite_to_pearlite_pixels=int(((rp == 1) & (cp == 0)).sum()),
        pearlite_to_ferrite_pixels=int(((rp == 0) & (cp == 1)).sum()),
        ferrite_to_zero_pixels=int(((rp == 1) & (cp < 0)).sum()),
        zero_to_ferrite_pixels=int(((rp < 0) & (cp == 1)).sum()),
        reference_ferrite=rf, candidate_ferrite=cf,
        ferrite_instance_delta=cf['instances']-rf['instances'],
        ferrite_mean_area_delta=(cf['mean_area']-rf['mean_area']) if rf['mean_area'] is not None and cf['mean_area'] is not None else None,
        partition=dict(common_positive_partition_identical=common_partition_identical,
            complete_partition_identical=common_partition_identical and coverage_changed == 0,
            pair_disagreement_count=changed_pairs, common_pair_denominator=total_pairs,
            pair_disagreement_fraction=changed_pairs/total_pairs if total_pairs else None,
            significant_split_reference_ids=[int(i) for i in split_ids],
            significant_merge_candidate_ids=[int(i) for i in merge_ids],
            ferrite_significant_splits=sum(reference_classes[str(int(i))] == 1 for i in split_ids),
            ferrite_significant_merges=sum(candidate_classes[str(int(i))] == 1 for i in merge_ids),
            threshold='intersection >= max(50 pixels, 0.1 * full area of the object being tested)',
            interpretation='Relative to the already scored R1 prediction; not GT accuracy or corrected topology'),
        official_accuracy=False, test_GT_used=False)


def validate_training_status(status, config, arm, checkpoint_sha):
    """固定e5/320更新身份；不得把未完成或续训其他epoch静默当本轮末态。"""
    validate_config(config)
    contract = status.get('contract', {})
    if (status.get('format') != 'affinity_gt_source_v1' or status.get('arm') != arm
            or status.get('status') != 'complete' or status.get('epoch') != 5
            or status.get('updates') != 320 or status.get('semantic_supervision') is not False
            or any(status.get(key) is not True for key in ('frozen_unchanged','restorer_unchanged','strict_reload_equal'))
            or status.get('final_checkpoint_sha256') != checkpoint_sha
            or contract.get('checkpoint_sha256') != R1_SHA
            or contract.get('restorer_checkpoint_sha256') != D5A_SHA
            or contract.get('recipe_sha256') != canonical_sha(config)
            or status.get('config') != json.loads(json.dumps(config))):
        raise RuntimeError('Completed paired e5/320 training checkpoint/recipe identity differs')
    required = ('frozen_state_sha256','restorer_state_sha256','manifest_sha256')
    if any(not isinstance(contract.get(key), str) or len(contract[key]) != 64 for key in required):
        raise RuntimeError('Training frozen/source contract incomplete')
    return contract


def _edges(ids):
    result = np.zeros(ids.shape, bool)
    result[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    result[1:] |= ids[1:] != ids[:-1]
    return result


def save_gallery(path, rgb, reference_ids, reference_classes, candidate_ids, candidate_classes, source):
    panels = [rgb.copy()]
    titles = ['Original', 'R1 reflect32 edges', 'Candidate reflect32 edges', 'R1 phase', 'Candidate phase']
    for ids in (reference_ids, candidate_ids):
        image = rgb.copy(); image[_edges(ids)] = (255, 90, 0); panels.append(image)
    for ids, classes in ((reference_ids,reference_classes),(candidate_ids,candidate_classes)):
        phase = _phase(ids, classes); image = rgb.copy()
        for value, color in ((0,(220,100,50)),(1,(40,190,230))):
            mask = phase == value
            image[mask] = np.rint(.45*image[mask]+.55*np.asarray(color)).astype(np.uint8)
        image[_edges(ids)] = 255; panels.append(image)
    boards = []
    h,w = rgb.shape[:2]
    crops = [np.s_[:,:], np.s_[max(0,h//2-256):min(h,h//2+256),max(0,w//2-256):min(w,w//2+256)]]
    for crop in crops:
        tiles = []
        for image, title in zip(panels,titles):
            image = image[crop]; ih,iw = image.shape[:2]
            image = cv2.resize(image,(250,max(1,round(ih*250/iw))),interpolation=cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image,25,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(image,title,(4,17),cv2.FONT_HERSHEY_SIMPLEX,.37,(0,0,0),1,cv2.LINE_AA)
            tiles.append(image)
        boards.append(np.concatenate(tiles,axis=1))
    board = np.concatenate(boards,axis=0)
    if not cv2.imwrite(str(path),cv2.cvtColor(board,cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Gallery write failed: '+source)


def _summary(rows):
    totals = {key:sum(row['comparison'][key] for row in rows) for key in (
        'pixels','common_positive_pixels','coverage_changed_pixels','phase_changed_pixels',
        'common_phase_changed_pixels','ferrite_to_pearlite_pixels','pearlite_to_ferrite_pixels',
        'ferrite_to_zero_pixels','zero_to_ferrite_pixels')}
    pairs = sum(row['comparison']['partition']['common_pair_denominator'] for row in rows)
    changed = sum(row['comparison']['partition']['pair_disagreement_count'] for row in rows)
    totals.update(images=len(rows), phase_changed_fraction=totals['phase_changed_pixels']/max(1,totals['pixels']),
        complete_partition_identical_images=sum(row['comparison']['partition']['complete_partition_identical'] for row in rows),
        partition_pair_disagreement_count=changed, partition_pair_denominator=pairs,
        partition_pair_disagreement_fraction=changed/pairs if pairs else None,
        ferrite_significant_splits=sum(row['comparison']['partition']['ferrite_significant_splits'] for row in rows),
        ferrite_significant_merges=sum(row['comparison']['partition']['ferrite_significant_merges'] for row in rows))
    for kind in ('reference','candidate'):
        entries = [row['comparison'][kind+'_ferrite'] for row in rows]
        count, pixels = sum(r['instances'] for r in entries),sum(r['pixels'] for r in entries)
        means = [r['mean_area'] for r in entries if r['mean_area'] is not None]
        totals[kind+'_ferrite'] = dict(instances=count,pixels=pixels,
            pooled_mean_area=pixels/count if count else None,
            image_mean_area_mean=float(np.mean(means)) if means else None,
            images_without_ferrite=len(rows)-len(means),tiny_under200=sum(r['tiny_under200'] for r in entries))
    return totals


def run(args):
    from data.mim_dataset import list_images
    from models.backend_adaptation import load_backend
    from train_affinity_coverage import load_original_core
    from tools.run_affinity_sweep import runtime_contract

    config_path = Path(args.config).resolve(); config = load_config(str(config_path)); validate_config(config)
    opts = config['gt_source']; base = deepcopy(config); base.pop('gt_source')
    arm_dir = (ROOT/'outputs/gt_source'/args.arm).resolve()
    checkpoint = Path(project_path(config,args.checkpoint)).resolve()
    if checkpoint != arm_dir/'final.pt' or Path(project_path(config)).resolve() != ROOT:
        raise ValueError('Only the current arm project-local final.pt may be evaluated')
    output = arm_dir/'deployment'
    if output.exists():
        raise FileExistsError('Fresh private deployment required; original outputs cannot be overwritten')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual sam2_env CUDA required; CPU comparison tests are not full inference')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    status_path = arm_dir/'status.json'; status = read(status_path)
    contract = validate_training_status(status,config,args.arm,sha(checkpoint))
    reference_report_path = Path(project_path(config,opts['monitor_reference_report'])).resolve()
    if sha(reference_report_path) != opts['monitor_reference_sha256']:
        raise RuntimeError('Already scored reflect32 reference report differs')
    reference = read(reference_report_path); reference_rows = report_rows(reference)
    config_gate(base,reference['config'])
    if reference['checkpoint_sha256'] != R1_SHA or reference['restorer_checkpoint_sha256'] != D5A_SHA:
        raise RuntimeError('Reference R1/D5a identity differs')
    baseline_zip = Path(project_path(config,args.reference_zip)).resolve()
    files = [Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))]
    if len(files) != 100 or len({p.stem for p in files}) != 100 or [p.stem for p in files] != [r['source'] for r in reference_rows]:
        raise RuntimeError('Complete sorted100 test cohort differs from scored reference')
    manifest_path = Path(project_path(config,opts['manifest'])).resolve(); manifest = read(manifest_path)
    if sha(manifest_path) != contract['manifest_sha256']:
        raise RuntimeError('Training source manifest differs from completed arm')
    precision,runtime = precision_record(),runtime_contract(config)
    if precision != manifest['precision'] or runtime != manifest['runtime'] or precision != reference['precision']:
        raise RuntimeError('Validated FP32 runtime or numerical settings differ')
    core = load_original_core(base); views = core.coverage_original_views
    model,bundle = load_backend(checkpoint,base,'cuda');model.eval().requires_grad_(False)
    restorer = get_restorer(base,'cuda')
    before,restored = tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items())
    extra = bundle.get('extra', {})
    if (bundle['epoch'] != 5 or extra.get('arm') != args.arm or extra.get('updates') != 320
            or extra.get('format') != 'affinity_gt_source_v1' or extra.get('contract') != contract
            or before != status['final_state_sha256'] or frozen_digest(model) != contract['frozen_state_sha256']
            or restored != contract['restorer_state_sha256'] or restored != reference['restorer_state_sha256']):
        raise RuntimeError('Strict candidate/upstream/D5a loaded state differs')
    if any(m.training for m in list(model.modules())+list(restorer.modules())) or any(p.requires_grad for p in list(model.parameters())+list(restorer.parameters())):
        raise RuntimeError('Complete inference models must be frozen/eval')
    parameters = sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    if parameters >= 500_000_000:
        raise RuntimeError('Model parameter budget exceeded')
    protected = {str(p):sha(p) for p in (checkpoint,status_path,config_path,reference_report_path,baseline_zip,manifest_path)}
    restore_path = Path(project_path(config,base['backend_adaptation']['restoration']['checkpoint']))
    protected[str(restore_path)] = D5A_SHA
    output.mkdir(); predictions = output/'patch1024';predictions.mkdir();(output/'gallery6').mkdir()
    sources,virtual = source_snapshot(output)
    report = dict(format='affinity_gt_source_evaluation_v1',complete=False,arm=args.arm,epoch=5,
        training=False,no_optimizer_steps=True,official_accuracy=False,test_GT_used=False,
        best_deployment_changed=False,submission_package_created=False,platform_uploaded=False,
        only_variable='affinity head from processed-manual versus model geometry on identical support; fixed oldD5a/encoder/semantic/fusion/reflect32',
        checkpoint_sha256=sha(checkpoint),model_state_sha256=before,frozen_state_sha256=contract['frozen_state_sha256'],
        restorer_checkpoint_sha256=D5A_SHA,restorer_state_sha256=restored,reference_zip_sha256=sha(baseline_zip),
        reference_report_sha256=sha(reference_report_path),protected_inputs=protected,protected_code=sources,
        virtual_code_paths=virtual,config=config,precision=precision,runtime=runtime,parameter_count=parameters,
        reflection_padding=32,first6_rule='fixed sorted full-cohort indices0..5; no output-based selection',images=[],
        interpretation='Unlabeled output differences against scored R1; splitting, merging and ferrite area are diagnostics, not accuracy or score improvement')
    started = time.time();write(output/'report.json',report)
    try:
        for index,(path,old) in enumerate(zip(files,reference_rows)):
            seed = 314159+index
            if old['seed'] != seed or sha(path) != old['source_sha256']:
                raise RuntimeError('Source/seed identity differs: '+path.stem)
            protected[str(path)] = old['source_sha256']
            with torch.no_grad():
                _,captured = capture_original(views,model,restorer,path,base,'cuda',seed)
            maps = dict(semantic_sha256=array_sha(captured['semantic'].numpy()),
                        raw_probability_sha256=array_sha(captured['raw_probability']))
            if any(maps[key] != old['fixed_maps'][key] for key in maps):
                raise RuntimeError('Frozen semantic/restoration input changed: '+path.stem)
            original,variants,gates = replay_candidates(captured,base)
            candidate = variants['32']
            if list(candidate['instances'].shape) != old['shape']:
                raise RuntimeError('Candidate native shape differs')
            saved = save_prediction(predictions,path.stem,candidate['instances'],candidate['classes'])
            report['images'].append(dict(source=path.stem,full_source_index=index,seed=seed,shape=old['shape'],
                source_sha256=old['source_sha256'],view='patch1024',output=saved,replay=gates,
                fixed_maps=maps,frozen_semantic_rawvote_exact=True,
                candidate_boundary_sha256=array_sha(captured['boundary'].numpy()),
                elevation_sha256=candidate['elevation_sha256'],real_WS_elevation_exact=candidate['elevation_sha256']==original['elevation_sha256']))
            write(output/'status.json',dict(status='running',stage='single_full100_inference',completed_images=index+1))
            print('GT_SOURCE_EVAL',args.arm,index+1,path.stem,flush=True)
            del captured,original,variants,candidate;gc.collect()
        # 一次完整推理结束后再比较现有ZIP；不重复R1或候选forward。
        with zipfile.ZipFile(baseline_zip) as archive:
            expected = {p.stem+suffix for p in files for suffix in ('_inst.png','_class.json')}
            if len(archive.namelist()) != 200 or set(archive.namelist()) != expected:
                raise RuntimeError('Scored reference ZIP must contain exact200 unique flat files')
            for index,(path,old,row) in enumerate(zip(files,reference_rows,report['images'])):
                for suffix,key in (('_inst.png','png_sha256'),('_class.json','classes_sha256')):
                    if hashlib.sha256(archive.read(path.stem+suffix)).hexdigest() != old['output'][key]:
                        raise RuntimeError('Scored reference ZIP/report member differs: '+path.stem)
                ref_ids,ref_classes,_ = _read_prediction(archive,path.stem)
                cand_ids,cand_classes = load_prediction(predictions,path.stem,old['shape'])
                row['comparison'] = compare_predictions(ref_ids,ref_classes,cand_ids,cand_classes)
                if index < 6:
                    rgb = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
                    save_gallery(output/'gallery6'/(path.stem+'.png'),rgb,ref_ids,ref_classes,cand_ids,cand_classes,path.stem)
        if (tensor_digest(model.state_dict().items()) != before or frozen_digest(model) != contract['frozen_state_sha256']
                or tensor_digest(restorer.state_dict().items()) != restored
                or precision_record() != precision or runtime_contract(config) != runtime):
            raise RuntimeError('Frozen model/runtime changed during full inference')
        for path,digest in protected.items():
            if sha(path) != digest:
                raise RuntimeError('Protected input changed: '+path)
        for name,digest in sources.items():
            if sha(ROOT/name) != digest or sha(output/'source'/name) != digest:
                raise RuntimeError('Inference source changed: '+name)
        if {p.name for p in predictions.iterdir()} != expected or len(report['images']) != 100:
            raise RuntimeError('Final100 prediction cohort incomplete')
        report.update(complete=True,full100_complete=True,model_unchanged=True,restorer_unchanged=True,
            strict_full_bundle_load=True,training_strict_reload_verified=True,baseline_member_sha256_exact=True,
            summary=_summary(report['images']),elapsed_seconds=time.time()-started)
        write(output/'report.json',report)
        write(output/'status.json',dict(status='complete',full100_complete=True,training=False,
                                      completed_images=100,report_sha256=sha(output/'report.json')))
        print('GT_SOURCE_EVALUATION_COMPLETE',args.arm,output,flush=True)
    except Exception as exc:
        write(output/'report.json',report)
        write(output/'status.json',dict(status='failed',error=repr(exc),training=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['manual','teacher'],required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',default='config/train/affinity_gt_source.yaml')
    parser.add_argument('--reference-zip',default='outputs/marker_trial/reflect1024.zip')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
