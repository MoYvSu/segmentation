# -*- coding: utf-8 -*-
"""预封存两张非monitor测试图的三头诊断；实际六次reflect32划分，不训练。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
import time
from unittest.mock import patch
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.probe_teacher_seams import (ARMS,CONFIGS,EXPECTED,FOLDERS,SOURCES,
    array_sha,probability_summary,read,response_drift,save_gallery,save_final_gallery,
    sha,validate_candidate_status,write)

def run(args):
    from data.mim_dataset import list_images
    from data.rgb_restoration_dataset import read_rgb
    from models.backend_adaptation import load_backend
    from tools import affinity_bottleneck_capture
    from tools.affinity_feature_capture import capture_frozen_features, replay_head_boundary
    from tools.evaluate_affinity_gt_source import compare_predictions
    from tools.run_affinity_sweep import runtime_contract
    from tools.run_marker_reflect import load_prediction, precision_record, source_snapshot
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_affinity_coverage import load_original_core
    from train_affinity_gt_source import D5A_SHA, relative_input, validate_pair
    from train_backend_adaptation import tensor_digest
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA sam2_env required; no real-data CPU fallback')
    output = (ROOT / args.output).resolve()
    if output != ROOT / 'outputs/teacher_seams/extra_probe' or output.exists():
        raise ValueError('Fresh fixed private outputs/teacher_seams/extra_probe required')
    configs = {arm: load_config(str(ROOT / path)) for arm, path in CONFIGS.items()}
    config = configs['before']
    if tuple(config['gt_source']['train_monitors']) + tuple(config['gt_source']['monitor_images']) != SOURCES:
        raise ValueError('Original fixed process sources differ')
    if any(Path(project_path(value)).resolve() != ROOT for value in configs.values()):
        raise ValueError('Actual project root differs')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    precision = precision_record(); runtime = runtime_contract(config)
    if runtime_contract(configs['after']) != runtime:
        raise RuntimeError('Repair inference runtime contract changed')
    manifest_path = relative_input(config, config['gt_source']['manifest'])
    manifest = read(manifest_path)
    if (manifest.get('status') != 'complete' or manifest.get('teacher_checkpoint_sha256') != EXPECTED['r1']
            or manifest.get('restorer_checkpoint_sha256') != D5A_SHA
            or manifest['precision'] != precision or manifest['runtime'] != runtime):
        raise RuntimeError('Source manifest/R1/D5a/FP32 runtime differs')
    protected = {str(ROOT / value): sha(ROOT / value) for value in CONFIGS.values()}
    protected[str(manifest_path)] = sha(manifest_path)
    statuses = {}; contracts = {}; receipts = {}; checkpoints = {}
    for arm in ('before', 'after'):
        folder = ROOT / FOLDERS[arm]
        checkpoints[arm] = folder / 'final.pt'
        protected[str(checkpoints[arm])] = EXPECTED[arm]
        status_path = folder / 'status.json'; statuses[arm] = read(status_path)
        protected[str(status_path)] = sha(status_path)
        contracts[arm] = validate_candidate_status(statuses[arm], configs[arm], arm)
        for epoch in (0, 5):
            path = folder / 'monitor' / f'e{epoch:02d}' / 'receipt.json'
            receipt = read(path); protected[str(path)] = sha(path)
            if receipt.get('epoch') != epoch or len(receipt.get('rows', [])) != 4 or {r['source'] for r in receipt['rows']} != set(SOURCES):
                raise RuntimeError('Exact four-source monitor receipt required')
            receipts[(arm, epoch)] = {row['source']: row for row in receipt['rows']}
    for key in ('checkpoint_sha256','restorer_checkpoint_sha256','manifest_sha256',
                'frozen_state_sha256','restorer_state_sha256','initial_head_sha256'):
        if contracts['before'][key] != contracts['after'][key]:
            raise RuntimeError('Candidate isolation contract differs: ' + key)
    if sha(manifest_path) != contracts['before']['manifest_sha256']:
        raise RuntimeError('Original common manifest SHA differs')
    repair_manifest = relative_input(configs['after'], configs['after']['gt_source']['seam_repair']['manifest'])
    protected[str(repair_manifest)] = contracts['after']['seam_manifest_sha256']
    for source in SOURCES:
        if receipts[('before',0)][source] != receipts[('after',0)][source]:
            raise RuntimeError('Original and repaired e0 source receipts differ')
    checkpoints['r1'] = relative_input(config, config['gt_source']['checkpoint'])
    protected[str(checkpoints['r1'])] = EXPECTED['r1']
    restorer_path = relative_input(config, config['backend_adaptation']['restoration']['checkpoint'])
    protected[str(restorer_path)] = D5A_SHA
    for path, digest in protected.items():
        if sha(path) != digest:
            raise RuntimeError('Protected identity differs: ' + path)
    core = load_original_core(config); views = core.coverage_original_views
    model, bundle = load_backend(checkpoints['r1'], config, 'cuda')
    model.eval().requires_grad_(False)
    frozen = frozen_digest(model); state = tensor_digest(model.state_dict().items())
    if frozen != contracts['before']['frozen_state_sha256']:
        raise RuntimeError('Actual frozen R1 state differs')
    base_state = bundle.get('model_state_dict')
    if not isinstance(base_state, dict) or not base_state:
        raise RuntimeError('Complete R1 model_state_dict required')
    heads = dict(r1=model.affinity_decoder); head_records = {}
    for arm in ('before','after'):
        candidate = torch.load(checkpoints[arm],map_location='cpu',weights_only=False)
        values = candidate.get('model_state_dict')
        if (not isinstance(values,dict) or set(values)!=set(base_state) or candidate.get('epoch')!=5
                or candidate.get('extra',{}).get('arm')!='teacher' or candidate['extra'].get('updates')!=320
                or candidate['extra'].get('contract')!=contracts[arm]
                or tensor_digest(values.items())!=statuses[arm]['final_state_sha256']):
            raise RuntimeError('Complete candidate state/epoch/identity differs: '+arm)
        if any(not torch.equal(values[key],base_state[key]) for key in base_state if not key.startswith('affinity_decoder.')):
            raise RuntimeError('Candidate non-head values differ from R1')
        head_state={key[len('affinity_decoder.'):]:value for key,value in values.items() if key.startswith('affinity_decoder.')}
        head=deepcopy(model.affinity_decoder).cpu();head.load_state_dict(head_state,strict=True)
        heads[arm]=head.to('cuda').eval().requires_grad_(False)
        head_records[arm]=dict(checkpoint_sha256=EXPECTED[arm],training_arm='teacher',
            diagnostic_arm=arm,head_state_sha256=tensor_digest(head_state.items()),
            full_state_sha256=statuses[arm]['final_state_sha256'],all_nonhead_values_equal_R1=True)
        del candidate,values,head_state;gc.collect()
    for arm,head in heads.items():
        modules=list(head.modules())
        if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in modules):
            raise RuntimeError('Actual affinity head unexpectedly contains BatchNorm/Dropout')
        head_records.setdefault(arm,dict(checkpoint_sha256=EXPECTED[arm],head_state_sha256=tensor_digest(head.state_dict().items())))
        head_records[arm].update(groupnorm_modules=sum(isinstance(m,torch.nn.GroupNorm) for m in modules),
                                 batchnorm_modules=0,dropout_modules=0)
    restorer=get_restorer(config,'cuda').eval().requires_grad_(False)
    restored=tensor_digest(restorer.state_dict().items())
    if restored!=contracts['before']['restorer_state_sha256']:
        raise RuntimeError('Frozen D5a state differs')
    tests={Path(p).stem:Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))}
    train_rows={row['source']:row for row in manifest['rows'] if row['source'] in SOURCES}
    if set(train_rows)!=set(SOURCES[:2]) or not set(SOURCES[2:]).issubset(tests):
        raise RuntimeError('Fixed source cohort missing')
    from tools.probe_affinity_structure import decode_reflect
    from tools.semantic_crossover import revote_instances
    from tools.run_marker_reflect import save_prediction
    cohort_path=ROOT/'outputs/teacher_seams/nonmonitor_cohort.json';cohort=read(cohort_path)
    protected[str(cohort_path)]=sha(cohort_path)
    if (cohort.get('seed')!=20261004 or cohort.get('test_GT_used') is not False
        or [r['source'] for r in cohort['sources']]!=['test_091','test_075']):
        raise RuntimeError('Exact preselected two-source cohort required')
    refpath=relative_input(config,config['gt_source']['monitor_reference_report'])
    protected[str(refpath)]=config['gt_source']['monitor_reference_sha256']
    if sha(refpath)!=protected[str(refpath)]:
        raise RuntimeError('R1 full reference report SHA differs')
    reference=read(refpath);refrows={r['source']:r for r in reference['images']}
    if not reference.get('full100_complete') or reference['checkpoint_sha256']!=EXPECTED['r1']:
        raise RuntimeError('Existing best full reference required')
    output.mkdir(parents=True)
    for sub in ('maps','gallery','final_gallery',*ARMS):
        (output/sub).mkdir()
    code,virtual=source_snapshot(output)
    protected.update({str(ROOT/name):digest for name,digest in code.items()})
    protected[str(Path(__file__).resolve())]=sha(__file__)
    protected[str(ROOT/'tools/probe_teacher_seams.py')]=sha(ROOT/'tools/probe_teacher_seams.py')
    report=dict(format='teacher_seams_extra_probe_v1',complete=False,training=False,no_optimizer_steps=True,
        test_GT_used=False,source_order=[r['source'] for r in cohort['sources']],cohort=cohort,
        checkpoints=head_records,epochs=dict(r1=20,before=5,after=5),updates=dict(before=320,after=320),
        frozen_state_sha256=frozen,restorer_checkpoint_sha256=D5A_SHA,restorer_state_sha256=restored,
        protected_inputs=protected,protected_code=code,virtual_code_paths=virtual,config=config,
        precision=precision,runtime=runtime,images=[],watershed_forward_calls=0,
        interpretation='Preselected two nonmonitor unlabeled test sources, shared frozen features and whole-image semantic probability; relative output only, no accuracy or official score.')
    write(output/'report.json',report);started=time.time()
    for source_row in cohort['sources']:
        source=source_row['source'];seed=source_row['inference_seed'];path=ROOT/source_row['image']
        protected[str(path)]=source_row['image_sha256']
        old=refrows[source]
        if sha(path)!=source_row['image_sha256'] or old['seed']!=seed or old['source_sha256']!=source_row['image_sha256']:
            raise RuntimeError('Actual source/seed differs from preselected and R1 reference')
        gt=np.zeros(read_rgb(str(path)).shape[:2],np.uint16)
        refdir=ROOT/'outputs/marker_trial/deployment/patch1024'
        ids,classes=load_prediction(refdir,source,gt.shape)
        for suffix,key in (('_inst.png','png_sha256'),('_class.json','classes_sha256')):
            p=refdir/(source+suffix);protected[str(p)]=old['output'][key]
            if sha(p)!=protected[str(p)]:
                raise RuntimeError('Existing R1 final output SHA differs')
        def capture_boundary_only(actual_views,actual_model,actual_restorer,actual_path,actual_config,device,actual_seed):
            observed={};calls=[]
            def record_decode(semantic,boundary,_config):
                if calls:
                    raise RuntimeError('Exactly one semantic/boundary capture required')
                calls.append(True);observed.update(semantic=semantic.detach().cpu().clone(),boundary=boundary.detach().cpu().clone())
                return ids.copy(),deepcopy(classes)
            def reuse_vote(actual_ids,probability):
                observed['raw_probability']=np.asarray(probability).copy();return deepcopy(classes),{}
            with patch.object(actual_views,'decode_native',side_effect=record_decode),patch.object(actual_views,'revote_instances',side_effect=reuse_vote):
                rgb,results=actual_views.predict_views(actual_model,actual_restorer,actual_path,actual_config,device,actual_seed,[1024])
            if len(calls)!=1 or set(results)!={'patch1024'}:
                raise RuntimeError('Single native1024 capture required')
            observed['baseline']=results['patch1024'];return rgb,observed
        with patch.object(affinity_bottleneck_capture,'capture_original',side_effect=capture_boundary_only):
            rgb,captured,tiles,capture_audit=capture_frozen_features(views,model,restorer,path,config,gt,'cuda',seed)
        maps={};replays={};predictions={};hashes={};decode_records={}
        for arm,head in heads.items():
            maps[arm],replays[arm]=replay_head_boundary(views,tiles,head,config,gt.shape,empty_exact=arm=='r1')
            value,_,_=decode_reflect(captured['semantic'],maps[arm],config)
            report['watershed_forward_calls']+=1
            voted,_=revote_instances(value['instances'],captured['raw_probability'])
            predictions[arm]=(value['instances'],voted)
            hashes[arm]=save_prediction(output/arm,source,*predictions[arm])
            decode_records[arm]=dict(reflect_padding_native=32,semantic_probability_sha256=array_sha(captured['raw_probability']),
                raw_vote_after_geometry=True)
        if array_sha(maps['r1'])!=old['fixed_maps']['boundary_sha256'] or hashes['r1']!=old['output']:
            raise RuntimeError('Actual R1 new float/reflect/semantic vote does not exactly reproduce full reference')
        beforedir=ROOT/'outputs/gt_source/teacher/deployment/patch1024'
        before_ids,before_classes=load_prediction(beforedir,source,gt.shape)
        if not np.array_equal(predictions['before'][0],before_ids) or predictions['before'][1]!=before_classes:
            raise RuntimeError('Actual before final prediction differs from saved partial75')
        for suffix,key in (('_inst.png','png_sha256'),('_class.json','classes_sha256')):
            p=beforedir/(source+suffix);protected[str(p)]=sha(p)
            if sha(p)!=hashes['before'][key]:
                raise RuntimeError('Actual before file SHA differs from saved partial75')
        image=dict(source=source,kind='test_nonmonitor',seed=seed,shape=list(gt.shape),source_sha256=sha(path),
            boundary_sha256={arm:array_sha(value) for arm,value in maps.items()},
            R1_existing_full_reference_exact=True,before_existing_partial75_exact=True,
            frozen_features=capture_audit,head_replays=replays,decode_records=decode_records,
            probability_summary={arm:probability_summary(value) for arm,value in maps.items()},
            response_strata={reference:{name:response_drift(maps[reference],maps['after'],mask)
                for name,mask in dict(high_ge065=maps[reference]>=.65,low_lt045=maps[reference]<.45,
                middle_045_to065=(maps[reference]>=.45)&(maps[reference]<.65)).items()}
                for reference in ('r1','before')},
            final_output_hashes=hashes,
            final_comparisons={a+'_to_'+b:compare_predictions(*predictions[a],*predictions[b])
                for a,b in (('r1','before'),('r1','after'),('before','after'))},test_GT_used=False)
        mw=384;mh=max(1,round(gt.shape[0]*mw/gt.shape[1]))
        np.savez_compressed(output/'maps'/(source+'.npz'),**{arm:cv2.resize(value,(mw,mh),interpolation=cv2.INTER_AREA).astype(np.float32) for arm,value in maps.items()})
        save_gallery(output/'gallery'/(source+'.png'),rgb,maps,source)
        save_final_gallery(output/'final_gallery'/(source+'.png'),rgb,*predictions['before'],*predictions['after'],source)
        report['images'].append(image);write(output/'report.json',report)
        write(output/'status.json',dict(status='running',completed_images=len(report['images']),maximum_images=2,training=False))
        print('TEACHER_SEAMS_EXTRA',len(report['images']),source,{arm:round(image['probability_summary'][arm]['mean'],6) for arm in ARMS},flush=True)
        del rgb,captured,tiles,maps;gc.collect();torch.cuda.empty_cache()
    for path,digest in protected.items():
        if sha(path)!=digest:
            raise RuntimeError('Protected input/code changed: '+path)
    if (tensor_digest(model.state_dict().items())!=state or frozen_digest(model)!=frozen
        or tensor_digest(restorer.state_dict().items())!=restored
        or precision_record()!=precision or runtime_contract(config)!=runtime):
        raise RuntimeError('Frozen state/runtime changed')
    for arm,head in heads.items():
        if tensor_digest(head.state_dict().items())!=head_records[arm]['head_state_sha256']:
            raise RuntimeError('Read-only head state changed')
    report.update(complete=True,protected_before_after_exact=True,elapsed_seconds=time.time()-started)
    write(output/'report.json',report);write(output/'status.json',dict(status='complete',completed_images=2,maximum_images=2,training=False))
    write(output/'artifact_manifest.json',dict(files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/teacher_seams/extra_probe')
    run(parser.parse_args())


if __name__=='__main__':
    main()
