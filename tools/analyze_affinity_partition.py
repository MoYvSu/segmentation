# -*- coding: utf-8 -*-
"""固定R1/TAIL四个已见新GT视图，补缓存实际关系干预并追踪完整划分；不训练。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_affinity_tail_gt import (
    CHECKPOINTS, cache_contract, existing_project_sources, pinned_files,
    runtime_files_contract,
)
from tools.analyze_affinity_fused_gt import diagnose_partition, gallery_panel
from train_backend_adaptation import sha, tensor_digest, write_json

PARENT_SHA = '93edc44f1b69051c0d9271b85c43dfb2d46a066cc985eb716ae65cd4eb62a634'
PRIMARY = {(24, 'split', (109, 109)), (24, 'split', (154, 154)),
           (25, 'split', (46, 46)), (25, 'merge', (21, 22)),
           (25, 'merge', (110, 131))}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def render_trace(folder, stem, rgb, gt, case, base, changed):
    """裁图仅展示；全部连通性和分水岭始终在完整原尺寸计算。"""
    chosen = np.isin(gt, case['gt_ids'])
    yy, xx = np.nonzero(chosen)
    y, x = max(0, int(yy.min())-30), max(0, int(xx.min())-30)
    y1, x1 = min(gt.shape[0], int(yy.max())+31), min(gt.shape[1], int(xx.max())+31)
    panels = [('degraded input', rgb), ('processed GT', gallery_panel(rgb, gt, gt>0))]
    for title, value in [('base final', base['instances']), ('corrected final', changed['instances'])]:
        panels.append((title, gallery_panel(rgb, value, gt>0, gt)))
    for title, body in [('base boundary', base['stages']['boundary_probability']),
                        ('corrected boundary', changed['stages']['boundary_probability'])]:
        heat = cv2.applyColorMap(np.rint(body*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        panels.append((title, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)))
    for title, body in [('base markers', base['actual_markers']), ('corrected markers', changed['actual_markers'])]:
        panels.append((title, gallery_panel(rgb, body, gt>0, gt)))
    strips=[]
    for title, body in panels:
        crop=body[y:y1,x:x1]
        crop=cv2.resize(crop,(300,max(1,round(300*crop.shape[0]/crop.shape[1]))),interpolation=cv2.INTER_AREA)
        crop=cv2.copyMakeBorder(crop,26,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(crop,title,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        strips.append(crop)
    if not cv2.imwrite(str(folder/(stem+'.png')),cv2.cvtColor(np.concatenate(strips,1),cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Trace gallery write failed')


def capture(output):
    """GPU只做冻结前向和关系替换后的实际融合；不创建优化器。"""
    from data.affinity_connectivity import undo_spatial
    from data.affinity_mixed import build_datasets
    from data.direct_dual_head_dataset import _spatial_transform
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend, restore_first
    from tools.affinity_connectivity_views import decode_native
    from tools.probe_affinity_partition import correct_local_relations, select_cases
    from tools.run_affinity_tail import verify_snapshot, verify_newgt
    from tools.run_affinity_short import precision_record
    from tools.semantic_crossover import revote_instances
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import draw_receipt, frozen_digest, get_restorer
    from train_affinity_tail import CONFIG_PATH, REFERENCE
    from train_backend_adaptation import fusion_options
    from train_direct_semantic_affinity import move_batch, set_seed
    from torch.utils.data import default_collate
    from utils.affinity_deployment import crop_letterbox_output
    from utils.affinity_fusion import affinity_boundary_probability
    from utils.config import load_config, project_path
    from utils.offset_letterbox import letterbox_instance_geometry

    root=ROOT/'outputs/affinity_tail';output=Path(output).resolve()
    if output.parent!=root/'analysis' or output.exists():
        raise RuntimeError('Fresh output child of TAIL analysis required')
    if not torch.cuda.is_available():raise RuntimeError('Actual sam2_env GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    parent=root/'analysis/four_gt';parent_report=read(parent/'report.json')
    if sha(parent/'report.json')!=PARENT_SHA or not parent_report['complete']:
        raise RuntimeError('Accepted TAIL four-draw report required')
    pinned_files(parent,read(parent/'download_manifest.json')['files'])
    old,old_manifest=cache_contract(ROOT/'outputs/affinity_fused/analysis/four_gt_v2')
    config=load_config(CONFIG_PATH);pipeline=read(root/'pipeline_status.json')
    if (pipeline['status']!='complete' or pipeline['stage']!='done' or len(pipeline['sources'])!=148
            or json.loads(json.dumps(config))!=pipeline['config']):
        raise RuntimeError('Formal source148/config pipeline differs')
    verify_snapshot(root,pipeline['sources']);runtime=runtime_files_contract(pipeline['runtime'],config)
    precision=precision_record(torch)
    if precision!=pipeline['precision'] or precision!=parent_report['precision']:
        raise RuntimeError('Precision changed')
    gt_audit=audit_new_gt(config);verify_newgt(config,gt_audit)
    if gt_audit!=parent_report['newgt'] or config['inference']!=parent_report['inference']:
        raise RuntimeError('Processed newGT or postprocess changed')
    paths=dict(r1=Path(project_path(config,REFERENCE,'final.pt')),tail=root/'candidate/final.pt')
    if any(sha(p)!=CHECKPOINTS[n] for n,p in paths.items()):raise RuntimeError('Checkpoint differs')
    protected={str(p):sha(p) for p in [Path(__file__),ROOT/'tools/probe_affinity_partition.py',
        root/'pipeline_status.json',root/'candidate/status.json',parent/'report.json',*paths.values()]}
    protected.update({str(parent/n):h for n,h in read(parent/'download_manifest.json')['files'].items()})
    protected.update({str(ROOT/'outputs/affinity_fused/analysis/four_gt_v2'/n):h for n,h in old_manifest['files'].items()})
    output.mkdir(parents=True);(output/'inputs').mkdir();(output/'boundaries').mkdir()
    models={name:load_backend(path,config,'cuda')[0] for name,path in paths.items()}
    restorer=get_restorer(config,'cuda');status=read(root/'candidate/status.json')
    if (frozen_digest(models['r1'])!=frozen_digest(models['tail'])
            or frozen_digest(models['r1'])!=status['frozen_state_sha256']
            or tensor_digest(restorer.state_dict().items())!=status['restorer_state_sha256']):
        raise RuntimeError('Frozen encoder/semantic/restorer changed')
    states={n:tensor_digest(m.state_dict().items()) for n,m in models.items()}
    states['restorer']=tensor_digest(restorer.state_dict().items())
    manual,_,_=build_datasets(config,1024);manual.set_epoch(1)
    sources={Path(p).name:Path(p) for p in manual.whole.base_dataset.base.samples}
    receipts=[json.loads(s) for s in (paths['r1'].parent/'steps.jsonl').read_text().splitlines()]
    mode,kwargs=fusion_options(config);rows=[];primary_seen=set()
    if mode!='gated':raise RuntimeError('Fixed gated fusion required')
    with torch.no_grad():
        for saved in parent_report['rows']:
            index=saved['draw'];raw=default_collate([manual[index]])
            receipt=next(r for r in receipts[:64] if int(r['manual_draw']['draw_index'][0])==index)
            if raw['crop_box'].tolist()!=receipt['manual_crop'] or draw_receipt(raw)!=receipt['manual_draw']:
                raise RuntimeError('Draw recipe changed')
            batch=move_batch(raw,'cuda');set_seed(saved['seed'])
            restored=restore_first(restorer,batch['image'],saved['seed']+1000000)
            if (tensor_digest([('input',batch['image'])])!=saved['input_sha256']
                    or tensor_digest([('restored',restored)])!=saved['restored_sha256']):
                raise RuntimeError('Input/restoration changed')
            ids=batch['affinity_instance_map'][0].cpu().numpy()
            content=batch['affinity_valid_content'][0,0].cpu().numpy().astype(bool)
            features=models['r1'].encoder(restored)
            sem=models['r1'].semantic_decoder(features,restored)
            raw_sem=models['r1'].semantic_decoder(models['r1'].encoder(batch['image']),batch['image'])
            if isinstance(sem,dict):sem=sem['semantic_logits']
            if isinstance(raw_sem,dict):raw_sem=raw_sem['semantic_logits']
            source=sources[saved['source'][0]];rgb=read_rgb(str(source))
            gt_path=Path(project_path(config,config['backend_adaptation']['completed_gt_dir'],source.stem+'_gt.npz'))
            full_gt,lookup=load_completed_semantic_source(gt_path,rgb.shape[:2])
            protected[str(gt_path)]=sha(gt_path)
            protected[str(gt_path.with_name(source.stem+'_class.json'))]=sha(gt_path.with_name(source.stem+'_class.json'))
            y,x,y1,x1=saved['native_crop'];gt=full_gt[y:y1,x:x1]
            classes={int(i):int(lookup[i]) for i in np.unique(gt) if i>0}
            hf,vf,k=bool(raw['horizontal_flip'].item()),bool(raw['vertical_flip'].item()),int(raw['rotation_k'].item())
            grid,valid,meta=letterbox_instance_geometry(gt,1024,512)
            if (not np.array_equal(_spatial_transform(torch.from_numpy(grid),hf,vf,k).numpy(),ids)
                    or not np.array_equal(_spatial_transform(torch.from_numpy(valid),hf,vf,k).numpy(),content)):
                raise RuntimeError('Native/grid GT alignment changed')
            def native(value):
                return crop_letterbox_output(undo_spatial(value.float(),hf,vf,k),1024,
                    1024-meta.resized_height,1024-meta.resized_width,gt.shape).cpu()
            native_sem=native(sem);probability=native(raw_sem).sigmoid()[0,0].numpy()
            if (tensor_digest([('semantic',native_sem)])!=saved['shared_native_restored_semantic_sha256']
                    or tensor_digest([('raw',torch.from_numpy(probability))])!=saved['shared_raw_probability_sha256']):
                raise RuntimeError('Shared semantic context changed')
            input_rgb=np.rint(native(batch['image'])[0].permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
            predictions={};diagnostics={};logits={};base_files={}
            for name,model in models.items():
                value=model.affinity_decoder(features)['affinity_logits'].float();logits[name]=value.cpu()
                boundary=native(affinity_boundary_probability(value,mode=mode,**kwargs)).clamp(0,1)
                mask,_=decode_native(native_sem,boundary,config);labels,_=revote_instances(mask,probability)
                old_mask=cv2.imread(str(parent/'masks'/f'draw_{index:03d}_{name}_inst.png'),cv2.IMREAD_UNCHANGED)
                old_labels=read(parent/'masks'/f'draw_{index:03d}_{name}_class.json')
                if not np.array_equal(mask,old_mask) or labels!=old_labels:raise RuntimeError('Accepted final parity failed')
                diagnostics[name]=diagnose_partition(gt,mask,classes,labels)
                if diagnostics[name]!=saved['arms'][name]:raise RuntimeError('Accepted GT diagnostic changed')
                predictions[name]=dict(instances=mask,classes=labels)
                filename=f'draw_{index:03d}_{name}_base.npz';base_files[name]='boundaries/'+filename
                np.savez_compressed(output/base_files[name],boundary=boundary[0,0].numpy())
            cases=select_cases(gt,classes,predictions,diagnostics)
            for case in cases:
                a,b=[int(predictions[case['reference_arm']]['instances'][py,px]) for py,px in case['points']]
                if min(a,b)<=0 or (case['kind']=='merge')!=(a==b):
                    raise RuntimeError('Fixed case anchors do not reproduce reference error')
            variants=[]
            for ci,case in enumerate(cases):
                key=(index,case['kind'],tuple(sorted(case['gt_ids'])))
                if key not in PRIMARY:continue
                primary_seen.add(key)
                partition=predictions[case['reference_arm']]['instances']
                partition,_,_=letterbox_instance_geometry(partition,1024,512)
                partition=_spatial_transform(torch.from_numpy(partition),hf,vf,k).numpy()
                for channels in ('short','all'):
                    chosen_reference=None
                    for name in models:
                        changed,chosen,counts=correct_local_relations(logits[name],ids,content,case,
                            channels=channels,partition=partition)
                        if chosen_reference is None:chosen_reference=chosen
                        elif not torch.equal(chosen,chosen_reference):raise RuntimeError('Same case selection must be shared')
                        if not torch.equal(changed[~chosen],logits[name][~chosen]):raise RuntimeError('Unselected logits changed')
                        boundary=native(affinity_boundary_probability(changed.to('cuda'),mode=mode,**kwargs)).clamp(0,1)
                        filename=f'draw_{index:03d}_{name}_case{ci:02d}_{channels}.npz'
                        np.savez_compressed(output/'boundaries'/filename,boundary=boundary[0,0].numpy())
                        variants.append(dict(arm=name,case_index=ci,channels=channels,counts=counts,
                            boundary_file='boundaries/'+filename,selection_sha256=tensor_digest([('chosen',chosen)])))
            np.savez_compressed(output/'inputs'/f'draw_{index:03d}.npz',gt=gt,grid_gt=ids,content=content,
                semantic=native_sem.numpy(),raw_probability=probability,rgb=input_rgb,
                logits_r1=logits['r1'].numpy(),logits_tail=logits['tail'].numpy())
            rows.append(dict(draw=index,source=saved['source'],view=saved['view'],cases=cases,variants=variants,
                base_files=base_files,input_file=f'inputs/draw_{index:03d}.npz',gt_classes=classes,
                input_sha256=saved['input_sha256'],restored_sha256=saved['restored_sha256'],base_final_exact=True))
            write_json(output/'capture_progress.json',dict(complete=False,rows=rows))
            print(json.dumps(dict(stage='captured',draw=index,cases=len(cases),variants=len(variants))),flush=True)
    if primary_seen!=PRIMARY or sum(len(r['variants']) for r in rows)!=20:
        raise RuntimeError('All five preset interventions / twenty arm variants required')
    final_states={n:tensor_digest(m.state_dict().items()) for n,m in models.items()}
    final_states['restorer']=tensor_digest(restorer.state_dict().items())
    if states!=final_states or any(sha(Path(p))!=h for p,h in protected.items()):raise RuntimeError('Protected state/artifact changed')
    verify_snapshot(root,pipeline['sources'])
    if runtime_files_contract(pipeline['runtime'],config)!=runtime:raise RuntimeError('Environment changed')
    report=dict(complete=True,config=config,checkpoints={n:CHECKPOINTS[n] for n in models},epoch=20,
        parent_report_sha256=PARENT_SHA,rows=rows,precision=precision,newgt=gt_audit,
        frozen_states=states,protected_sha256=protected,source148_unchanged=True,no_training=True,
        imported_sources=existing_project_sources(list(sys.modules.values()),ROOT),sealed_sources=pipeline['sources'],
        sealed_runtime=pipeline['runtime'],runtime_audit=runtime,
        scope='four seen augmented processed-newGT draws; relation interventions are diagnostic and not attainable bounds')
    write_json(output/'capture.json',report)
    write_json(output/'capture_manifest.json',dict(complete=True,files={p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file()}))


def analyze(output):
    """仅使用封存实际GPU前向／融合图，CPU重放完整后处理并同时检查新错。"""
    from tools.probe_affinity_partition import decode_capture, trace_relations
    from tools.semantic_crossover import revote_instances
    from tools.affinity_native_views import verify_prediction
    from tools.run_affinity_tail import verify_snapshot
    from tools.run_affinity_short import precision_record
    output=Path(output).resolve();manifest=read(output/'capture_manifest.json')
    pinned_files(output,manifest['files']);record=read(output/'capture.json')
    if not record['complete'] or not record['no_training']:raise RuntimeError('Completed frozen capture required')
    if (output/'report.json').exists():raise RuntimeError('Do not overwrite finished analysis')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    (output/'gallery').mkdir(exist_ok=True);rows=[];pages=[]
    for item in record['rows']:
        with np.load(output/item['input_file'],allow_pickle=False) as blob:
            gt=blob['gt'].copy();sem=torch.from_numpy(blob['semantic'].copy())
            raw=blob['raw_probability'].copy();rgb=blob['rgb'].copy()
        classes={int(k):int(v) for k,v in item['gt_classes'].items()};decoded={};baselines={}
        for arm,path in item['base_files'].items():
            with np.load(output/path,allow_pickle=False) as blob:boundary=torch.from_numpy(blob['boundary'].copy())[None,None]
            value=decode_capture(sem,boundary,record['config']);decoded[arm]=value
            labels,_=revote_instances(value['instances'],raw);verify_prediction(value['instances'],labels,gt.shape)
            old=ROOT/'outputs/affinity_tail/analysis/four_gt/masks'/f"draw_{item['draw']:03d}_{arm}_inst.png"
            if not np.array_equal(value['instances'],cv2.imread(str(old),cv2.IMREAD_UNCHANGED)):
                raise RuntimeError('CPU replay differs from accepted final')
            diagnostic=diagnose_partition(gt,value['instances'],classes,labels)
            baselines[arm]=dict(diagnostic=diagnostic,traces=trace_relations(value,gt,item['cases']),
                capture_audit=value['capture_audit'],marker_count=value['stages']['marker_count'])
        variants=[]
        for variant in item['variants']:
            with np.load(output/variant['boundary_file'],allow_pickle=False) as blob:boundary=torch.from_numpy(blob['boundary'].copy())[None,None]
            value=decode_capture(sem,boundary,record['config']);labels,_=revote_instances(value['instances'],raw)
            verify_prediction(value['instances'],labels,gt.shape)
            diagnostic=diagnose_partition(gt,value['instances'],classes,labels)
            arm=variant['arm'];base=baselines[arm]['diagnostic']
            changes={}
            for domain in ('geometry','conservative'):
                changes[domain]=dict(new_split_gt=sorted(set(diagnostic[domain]['split_gt_ids'])-set(base[domain]['split_gt_ids'])),
                    removed_split_gt=sorted(set(base[domain]['split_gt_ids'])-set(diagnostic[domain]['split_gt_ids'])),
                    new_merge_pairs=sorted(set(map(tuple,diagnostic[domain]['merged_gt_pairs']))-set(map(tuple,base[domain]['merged_gt_pairs']))),
                    removed_merge_pairs=sorted(set(map(tuple,base[domain]['merged_gt_pairs']))-set(map(tuple,diagnostic[domain]['merged_gt_pairs']))))
            changes['conservative']['removed_split_still_raw_error']=[g for g in changes['conservative']['removed_split_gt']
                if g in diagnostic['geometry']['split_gt_ids']]
            changes['conservative']['removed_merge_still_raw_error']=[pair for pair in changes['conservative']['removed_merge_pairs']
                if pair in set(map(tuple,diagnostic['geometry']['merged_gt_pairs']))]
            traces=trace_relations(value,gt,item['cases']);ci=variant['case_index'];case=item['cases'][ci]
            stem=f"draw_{item['draw']:03d}_{arm}_case{ci:02d}_{variant['channels']}"
            render_trace(output/'gallery',stem,rgb,gt,case,decoded[arm],value)
            pages.append(stem+'.png')
            variants.append(dict(**variant,diagnostic=diagnostic,traces=traces,changes=changes,
                capture_audit=value['capture_audit'],marker_count=value['stages']['marker_count'],gallery='gallery/'+stem+'.png'))
            print(json.dumps(dict(stage='analyzed',draw=item['draw'],arm=arm,case=ci,channels=variant['channels'])),flush=True)
        rows.append(dict(draw=item['draw'],source=item['source'],view=item['view'],cases=item['cases'],
            baselines=baselines,variants=variants));write_json(output/'analysis_progress.json',dict(complete=False,rows=rows))
    if any(sha(Path(p))!=h for p,h in record['protected_sha256'].items()):raise RuntimeError('Protected artifact changed during analysis')
    verify_snapshot(ROOT/'outputs/affinity_tail',record['sealed_sources'])
    if (runtime_files_contract(record['sealed_runtime'],record['config'])!=record['runtime_audit']
            or precision_record(torch)!=record['precision']):
        raise RuntimeError('Sealed runtime or precision changed during CPU replay')
    report=dict(complete=True,rows=rows,checkpoints=record['checkpoints'],epoch=20,no_training=True,
        parent_report_sha256=PARENT_SHA,capture_sha256=sha(output/'capture.json'),
        scope=record['scope'],caveats=['point-pair errors are not independent objects or official accuracy',
        'correcting output relations is not training and is not a generalization bound',
        'full original draw connectivity; cropped gallery only displays',
        'all processed known GT including filled; unknown/padding never corrected'])
    write_json(output/'report.json',report)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed partition interventions</title>'
        '<p>Seen processed-newGT draws; frozen output interventions, not official scores.</p>'+
        ''.join(f'<p>{p}</p><img src="gallery/{p}" style="max-width:100%">' for p in pages),encoding='utf-8')
    write_json(output/'download_manifest.json',dict(complete=True,files={p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file() and p.name!='download_manifest.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/affinity_tail/analysis/partition')
    parser.add_argument('--mode',choices=('capture','analyze','both'),default='both');args=parser.parse_args()
    target=ROOT/args.output
    if args.mode in ('capture','both'):capture(target)
    if args.mode in ('analyze','both'):analyze(target)
