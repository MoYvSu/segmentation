# -*- coding: utf-8 -*-
"""两张已见训练原图的正式native1024划分检查；不训练、不干预、不选参数。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_affinity_fused_gt import diagnose_partition, gallery_panel
from tools.analyze_affinity_tail_gt import CHECKPOINTS, existing_project_sources, pinned_files, runtime_files_contract
from train_backend_adaptation import sha, tensor_digest, write_json

PARENT_SHA = '93edc44f1b69051c0d9271b85c43dfb2d46a066cc985eb716ae65cd4eb62a634'
SOURCES = ('train_686.jpg', 'train_707.jpg')
PRESETS = {'train_686.jpg': [('split', [109,109]), ('split', [154,154])],
           'train_707.jpg': [('split', [46,46]), ('merge', [21,22]), ('merge', [110,131])]}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def object_record(case, gt, classes, diagnostic):
    """固定GT身份，区分raw错误、保守错误和censor；不把不匹配视作零面积。"""
    kind, ids = case
    raw, safe = diagnostic['geometry'], diagnostic['conservative']
    if any(not bool((gt == g).any()) for g in ids):
        raise RuntimeError('Preset newGT identity missing from original source')
    objects = []
    for value in sorted(set(ids)):
        pair = next((p for p in diagnostic['pairs'] if p['gt'] == value), None)
        objects.append(dict(gt=value,phase=classes[value],known_pixels=int((gt==value).sum()),
            matched_pair=pair,gt_unknown_or_frame_censored=value in safe['censored_gt_ids'],
            raw_split=value in raw['split_gt_ids'],conservative_split=value in safe['split_gt_ids']))
    key = sorted(ids)
    return dict(kind=kind,gt_ids=ids,objects=objects,
        raw_error=(ids[0] in raw['split_gt_ids'] if kind=='split' else key in raw['merged_gt_pairs']),
        conservative_error=(ids[0] in safe['split_gt_ids'] if kind=='split' else key in safe['merged_gt_pairs']),
        scope='same fixed GT IDs on full original image; not the degraded whole training view')


def render(folder, name, rgb, gt, predictions, box=None, suffix='full'):
    panels=[('original training RGB',rgb),('processed GT / unknown gray',gallery_panel(rgb,gt,gt>0))]
    panels += [(arm+' native1024 final',gallery_panel(rgb,v['instances'],gt>0,gt)) for arm,v in predictions.items()]
    strips=[]
    for title,body in panels:
        if box is not None:
            y,x,y1,x1=box;body=body[y:y1,x:x1]
        body=cv2.resize(body,(360,max(1,round(360*body.shape[0]/body.shape[1]))),interpolation=cv2.INTER_AREA)
        body=cv2.copyMakeBorder(body,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(body,title,(4,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv2.LINE_AA);strips.append(body)
    if not cv2.imwrite(str(folder/(name+'_'+suffix+'.png')),cv2.cvtColor(np.concatenate(strips,1),cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Gallery write failed')


def run(output):
    from data.affinity_mixed import build_datasets
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views, verify_prediction
    from tools.run_affinity_tail import verify_newgt, verify_snapshot
    from tools.run_affinity_short import precision_record
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_affinity_tail import CONFIG_PATH, REFERENCE
    from utils.config import load_config, project_path

    root=ROOT/'outputs/affinity_tail';output=Path(output).resolve()
    if output!=root/'analysis/partition_native' or output.exists():
        raise RuntimeError('Fresh outputs/affinity_tail/analysis/partition_native required')
    if not torch.cuda.is_available():raise RuntimeError('Actual server sam2_env GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    parent=root/'analysis/four_gt';old=read(parent/'report.json')
    if sha(parent/'report.json')!=PARENT_SHA or old['complete'] is not True:
        raise RuntimeError('Accepted fixed four-draw report required')
    manifest=read(parent/'download_manifest.json')
    if manifest['complete'] is not True:raise RuntimeError('Complete accepted cache required')
    pinned_files(parent,manifest['files'])
    config=load_config(CONFIG_PATH);pipeline=read(root/'pipeline_status.json')
    if (pipeline['status']!='complete' or pipeline['stage']!='done'
            or pipeline['completed']!=['training','inference','spatial_after','render']
            or len(pipeline['sources'])!=148 or pipeline.get('control_retrained') is not False
            or pipeline.get('ranking_parent_retrained') is not False
            or json.loads(json.dumps(config))!=pipeline['config']):
        raise RuntimeError('Completed source148 fixed TAIL config required')
    verify_snapshot(root,pipeline['sources']);runtime=runtime_files_contract(pipeline['runtime'],config)
    precision=precision_record(torch)
    if precision!=pipeline['precision'] or precision!=old['precision']:raise RuntimeError('Fixed precision differs')
    gt_audit=audit_new_gt(config);verify_newgt(config,gt_audit)
    if gt_audit!=old['newgt'] or config['inference']!=old['inference']:
        raise RuntimeError('Full processed newGT or deployed postprocess differs')
    if (config['inference'].get('marker_partition_restore',{}).get('enabled',False)
            or config['affinity_native']['overlap']!=pipeline['config']['affinity_native']['overlap']):
        raise RuntimeError('Unexpected marker restoration/window overlap')
    paths=dict(r1=Path(project_path(config,REFERENCE,'final.pt')),tail=root/'candidate/final.pt')
    if any(sha(p)!=CHECKPOINTS[n] for n,p in paths.items()):raise RuntimeError('Checkpoint differs')
    status=read(root/'candidate/status.json')
    if status['epoch']!=20 or status['smoke'] is not False:raise RuntimeError('Formal e20 required')
    protected={str(p):sha(p) for p in [Path(__file__),root/'pipeline_status.json',root/'candidate/status.json',
        parent/'download_manifest.json',*paths.values()]}
    protected.update({str(parent/n):h for n,h in manifest['files'].items()})
    imports=existing_project_sources(list(sys.modules.values()),ROOT)
    protected.update({str(ROOT/n):h for n,h in imports['files'].items()})
    models={n:load_backend(p,config,'cuda')[0] for n,p in paths.items()};restorer=get_restorer(config,'cuda')
    if (frozen_digest(models['r1'])!=frozen_digest(models['tail'])
            or frozen_digest(models['r1'])!=status['frozen_state_sha256']
            or tensor_digest(restorer.state_dict().items())!=status['restorer_state_sha256']):
        raise RuntimeError('Frozen encoder/semantic/D5a identity differs')
    states={n:tensor_digest(m.state_dict().items()) for n,m in models.items()}
    states['restorer']=tensor_digest(restorer.state_dict().items())
    manual,_,_=build_datasets(config,1024)
    source_map={Path(p).name:Path(p) for p in manual.whole.base_dataset.base.samples}
    if not set(SOURCES).issubset(source_map):raise RuntimeError('Fixed allowed training sources missing')
    output.mkdir(parents=True);(output/'masks').mkdir();(output/'gallery').mkdir();(output/'markers').mkdir()
    rows=[];seed_base=int(config['backend_adaptation']['restoration']['inference_seed'])
    original=cv2.watershed
    for index,name in enumerate(SOURCES):
        source=source_map[name];rgb=read_rgb(str(source));protected[str(source)]=sha(source)
        gt_path=Path(project_path(config,config['backend_adaptation']['completed_gt_dir'],source.stem+'_gt.npz'))
        gt,lookup=load_completed_semantic_source(gt_path,rgb.shape[:2])
        classes={int(g):int(lookup[g]) for g in np.unique(gt) if g>0}
        protected[str(gt_path)]=sha(gt_path)
        protected[str(gt_path.with_name(source.stem+'_class.json'))]=sha(gt_path.with_name(source.stem+'_class.json'))
        arms={};predictions={};fixed={};seed=seed_base+index
        for arm,model in models.items():
            captures=[]
            def capture(image,markers):
                before=markers.copy();result=original(image,markers)
                captures.append(dict(markers=before,watershed=np.maximum(result,0).copy()))
                return result
            audit={}
            with patch('utils.post_process.cv2.watershed',side_effect=capture):
                actual_rgb,views=predict_views(model,restorer,source,config,'cuda',seed,[1024],fixed_audit=audit)
            if set(views)!= {'patch1024'} or not np.array_equal(actual_rgb,rgb) or len(captures)!=1:
                raise RuntimeError('Actual one-view native1024 RGB/watershed capture differs')
            if set(audit)!= {'geometry_semantic_sha256','raw_probability_sha256'}:
                raise RuntimeError('Both whole-image semantic audit routes required')
            fixed[arm]=audit
            if arm!='r1' and audit!=fixed['r1']:raise RuntimeError('Whole-image geometry/raw semantics differ across arms')
            value=views['patch1024'];verify_prediction(value['instances'],value['classes'],gt.shape)
            diagnostic=diagnose_partition(gt,value['instances'],classes,value['classes'])
            arms[arm]=dict(diagnostic=diagnostic,presets=[object_record(c,gt,classes,diagnostic) for c in PRESETS[name]],
                blend=value['blend'],boundary_sha256=array_sha(value['boundary']),
                actual_marker_sha256=array_sha(captures[0]['markers']),actual_watershed_sha256=array_sha(captures[0]['watershed']),
                marker_count=int(len(np.unique(captures[0]['markers'][captures[0]['markers']>0]))),actual_watershed_calls=1)
            predictions[arm]=value
            if not cv2.imwrite(str(output/'masks'/(source.stem+'_'+arm+'_inst.png')),value['instances']):raise RuntimeError('Mask write failed')
            write_json(output/'masks'/(source.stem+'_'+arm+'_class.json'),value['classes'])
            np.savez_compressed(output/'markers'/(source.stem+'_'+arm+'.npz'),markers=captures[0]['markers'],watershed=captures[0]['watershed'])
        render(output/'gallery',source.stem,rgb,gt,predictions)
        for ci,case in enumerate(PRESETS[name]):
            yy,xx=np.nonzero(np.isin(gt,case[1]));box=(max(0,int(yy.min())-30),max(0,int(xx.min())-30),
                min(gt.shape[0],int(yy.max())+31),min(gt.shape[1],int(xx.max())+31))
            render(output/'gallery',source.stem,rgb,gt,predictions,box,'case'+str(ci))
        rows.append(dict(source=name,seed=seed,shape=list(gt.shape),gt_classes=classes,
            original_rgb_sha256=array_sha(rgb),source_file_sha256=sha(source),gt_file_sha256=sha(gt_path),
            fixed_whole_semantics=fixed,whole_semantics_exact_across_arms=True,arms=arms))
        write_json(output/'progress.json',dict(complete=False,rows=rows))
        print(json.dumps(dict(source=name,seed=seed,stage='native_complete',
            presets={n:[(p['kind'],p['gt_ids'],p['raw_error'],p['conservative_error']) for p in v['presets']] for n,v in arms.items()})),flush=True)
    final={n:tensor_digest(m.state_dict().items()) for n,m in models.items()}
    final['restorer']=tensor_digest(restorer.state_dict().items())
    if states!=final or any(sha(Path(p))!=h for p,h in protected.items()):raise RuntimeError('Protected file or frozen state changed')
    verify_snapshot(root,pipeline['sources'])
    if runtime_files_contract(pipeline['runtime'],config)!=runtime or precision_record(torch)!=precision:
        raise RuntimeError('Sealed runtime/precision changed')
    if audit_new_gt(config)!=gt_audit:raise RuntimeError('Full processed GT changed')
    report=dict(complete=True,no_training=True,no_intervention=True,epoch=20,sizes=[1024],sources=list(SOURCES),
        seed_rule='inference_seed + sorted fixed two-source index [0,1]; same across arms',rows=rows,
        checkpoints={n:CHECKPOINTS[n] for n in models},parent_report_sha256=PARENT_SHA,newgt=gt_audit,
        runtime_audit=runtime,precision=precision,config=config,sealed_sources=pipeline['sources'],
        frozen_states_sha256=states,source148_unchanged=True,formal_pipeline_unchanged=True,protected_sha256=protected,
        imported_sources=existing_project_sources(list(sys.modules.values()),ROOT),
        scope='two seen labelled original training images; actual native1024 sliding-window boundary and frozen two whole-image semantic routes',
        caveats=['Full processed newGT including filled; unknown ignored and whole-object frame/unknown censor separately reported.',
            'Original RGB differs from artificial degraded whole training draws; changes are not a single-variable scale comparison.',
            'Preset IDs locate mechanism; this is neither independent validation nor official mIoU/area.',
            'No output intervention, parameter search, new holdout, training, blackbox, or deployment change.'])
    write_json(output/'report.json',report)
    pages=['<!doctype html><meta charset="utf-8"><title>Native1024 seen-GT check</title><p>Original training images; same full semantics, GT colors; diagnostic only.</p>']
    for p in sorted((output/'gallery').glob('*.png')):pages.append('<p>'+p.stem+'</p><img style="max-width:100%" src="gallery/'+p.name+'">')
    (output/'index.html').write_text('\n'.join(pages),encoding='utf8')
    write_json(output/'download_manifest.json',dict(complete=True,files={p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file() and p.name!='download_manifest.json'}))
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/affinity_tail/analysis/partition_native')
    args=parser.parse_args();run(ROOT/args.output)
