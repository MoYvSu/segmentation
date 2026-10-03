# -*- coding: utf-8 -*-
"""固定四draw完整新GT诊断：复用R1/FUSED缓存，仅计算TAIL必要路径。"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_affinity_fused_gt import (
    common_safe_change, diagnose_partition, gallery_panel, read,
    spatial_partition_links,
)
from tools.analyze_affinity_sweep_gt import paired_changes
from train_backend_adaptation import sha, write_json

CHECKPOINTS = dict(r1='43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434',
    fused='efb16cbeed5ba07d5104ec8f768bc64aecf2dfea200e667264a5e18f6f7eb6d2',
    tail='03518f14a01bcfd9c467714b60ba78e66cb71270ef239b4998c8a4384b237083')
CACHE_REPORT_SHA256 = 'd981971df36abdc99054f8cf1223ff3e1f0804b375566c9e408edae943d7110e'
ARMS = tuple(CHECKPOINTS)


def pinned_files(directory, expected):
    """验证已封存真实文件清单，不从动态Torch模块重新枚举文件。"""
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(Path(directory)/relative) != digest:
            raise RuntimeError('Pinned file differs: '+name)


def existing_project_sources(modules, root):
    """仅不存在的虚拟__file__被跳过；真实项目文件的读取异常必须显露。"""
    result, virtual = {}, []
    for module in modules:
        value = getattr(module, '__file__', None)
        if not value or Path(value).suffix != '.py':
            continue
        path = Path(value)
        if not path.is_file():
            virtual.append(str(path)); continue
        absolute = path.resolve()
        if absolute.is_relative_to(root):
            result[absolute.relative_to(root).as_posix()] = sha(absolute)
    return dict(files=result, skipped_nonexistent_virtual_paths=sorted(set(virtual)))


def cache_contract(folder):
    folder = Path(folder)
    manifest = read(folder/'download_manifest.json')
    if manifest.get('complete') is not True:
        raise RuntimeError('Accepted private cache manifest required')
    pinned_files(folder, manifest['files'])
    if manifest['files'].get('report.json') != CACHE_REPORT_SHA256:
        raise RuntimeError('Only the accepted four-draw FUSED report may be reused')
    report = read(folder/'report.json')
    if (report.get('complete') is not True or len(report.get('rows', [])) != 4
            or report.get('epoch') != 20 or report.get('updates') != 1280
            or report.get('checkpoints') != dict(control=CHECKPOINTS['r1'],candidate=CHECKPOINTS['fused'])
            or sha(folder/'gpu_execution.py') != report['metadata_recovery']['original_execution_script_sha256']
            or report['newgt'].get('original_covered_is_supervision_mask') is not False):
        raise RuntimeError('Cached original execution/support/checkpoint contract differs')
    return report, manifest


def phase_diagnostics(gt, pred, gt_classes, pred_classes, geometry):
    """同已知域的类别诊断与真实F几何拆粘；不是官方mIoU或物理面积。"""
    keep = gt > 0
    result = {}
    if keep.any():
        g, gi = np.unique(gt[keep], return_inverse=True)
        p, pi = np.unique(pred[keep], return_inverse=True)
        matrix = np.bincount(gi*len(p)+pi, minlength=len(g)*len(p)).reshape(len(g),len(p))
        ga, pa = matrix.sum(1), matrix.sum(0)
    else:
        g = p = np.array([], np.int32); matrix = np.zeros((0,0), np.int64)
        ga = pa = np.array([], np.int64)
    for label, name in ((0,'pearlite'), (1,'ferrite')):
        ig = np.array([i for i,value in enumerate(g) if gt_classes[int(value)] == label], int)
        ip = np.array([i for i,value in enumerate(p) if value > 0 and pred_classes[str(int(value))] == label], int)
        intersection = matrix[np.ix_(ig,ip)]
        iou = intersection/np.maximum(1,ga[ig,None]+pa[None,ip]-intersection)
        qualified = iou >= .5
        pairs=[]
        if len(ig) and len(ip):
            score = qualified.astype(float)*(min(iou.shape)+1)+np.where(qualified,iou,0)
            left,right = linear_sum_assignment(-score)
            pairs=[dict(gt=int(g[ig[i]]),pred=int(p[ip[j]]),iou=float(iou[i,j]))
                   for i,j in zip(left,right) if qualified[i,j]]
        total=sum(row['iou'] for row in pairs)
        result[name]=dict(gt_instances=len(ig),pred_instances_in_known_domain=len(ip),
            class_aware_matches=len(pairs),iou_sum=total,
            seen_gt_penalized_instance_iou=total/max(1,len(ig)),pairs=pairs)
    values=[v['seen_gt_penalized_instance_iou'] for v in result.values() if v['gt_instances']]
    result['seen_class_aware_mean_instance_iou']=float(np.mean(values)) if values else None
    def ferrite_topology(group):
        ff_pairs=sorted({tuple(pair) for row in group['merged_predictions']
            for pair in combinations(sorted(g for g in row['gt_ids'] if gt_classes[g]==1),2)})
        return dict(F_split_gt_ids=[g for g in group['split_gt_ids'] if gt_classes[g]==1],
            FF_merged_gt_pairs=[list(pair) for pair in ff_pairs],
            F_P_mixed_merged_predictions=sum(any(gt_classes[g]==1 for g in r['gt_ids'])
                and any(gt_classes[g]==0 for g in r['gt_ids']) for r in group['merged_predictions']))
    result['true_F_geometry']=ferrite_topology(geometry['geometry'])
    result['true_F_geometry_conservative']=ferrite_topology(geometry['conservative'])
    return result


def pairwise_diagnostic(a, b, classes, ca, cb):
    raw=paired_changes(a,b,classes,ca,cb)
    safe=common_safe_change(a,b)
    phases={}
    for scope,gain,loss in [('known',raw['candidate_only_matched_gt'],raw['reference_only_matched_gt']),
            ('conservative',safe['candidate_only_safe_matched_gt'],safe['control_only_safe_matched_gt'])]:
        phases[scope]={name:dict(gained_gt=[g for g in gain if classes[g]==label],
            lost_gt=[g for g in loss if classes[g]==label]) for label,name in ((0,'P'),(1,'F'))}
    raw['common_iou_delta_mean']=raw['common_iou_delta_sum']/raw['common_matches'] if raw['common_matches'] else None
    return dict(known=raw,conservative=safe,phase_gained_lost=phases)


def summarize_rows(rows):
    """按GT共同集配对汇总，避免将三个不同匹配集直接相减。"""
    totals={}
    for name in ARMS:
        totals[name]=dict(gt_instances=sum(r['arms'][name]['geometry']['gt_instances'] for r in rows),
            matched=sum(r['arms'][name]['geometry']['matched'] for r in rows),
            iou_sum=sum(r['arms'][name]['geometry']['iou_sum'] for r in rows),
            split_gt=sum(r['arms'][name]['geometry']['split_gt'] for r in rows),
            merged_pred=sum(r['arms'][name]['geometry']['merged_pred'] for r in rows),
            safe_matched=sum(r['arms'][name]['conservative']['matched'] for r in rows),
            safe_split_gt=sum(len(r['arms'][name]['conservative']['split_gt_ids']) for r in rows),
            safe_merged_pred=sum(len(r['arms'][name]['conservative']['merged_predictions']) for r in rows))
        totals[name]['phases']={}
        for phase in ('ferrite','pearlite'):
            n=sum(r['phase_diagnostics'][name][phase]['gt_instances'] for r in rows)
            total=sum(r['phase_diagnostics'][name][phase]['iou_sum'] for r in rows)
            totals[name]['phases'][phase]=dict(gt_instances=n,
                class_aware_matches=sum(r['phase_diagnostics'][name][phase]['class_aware_matches'] for r in rows),
                iou_sum=total,seen_gt_penalized_instance_iou=total/max(1,n))
        totals[name]['true_F_geometry']={scope:dict(
            split_gt=sum(len(r['phase_diagnostics'][name][scope]['F_split_gt_ids']) for r in rows),
            FF_merged_gt_pairs=sum(len(r['phase_diagnostics'][name][scope]['FF_merged_gt_pairs']) for r in rows),
            F_P_mixed_merged_pred=sum(r['phase_diagnostics'][name][scope]['F_P_mixed_merged_predictions'] for r in rows))
            for scope in ('true_F_geometry','true_F_geometry_conservative')}
    paired={}
    for key in ('r1_to_fused','r1_to_tail','fused_to_tail'):
        paired[key]={}
        for scope in ('known','conservative'):
            count=sum(r['pairwise'][key][scope]['common_matches'] for r in rows)
            delta=sum(r['pairwise'][key][scope]['common_iou_delta_sum'] for r in rows)
            paired[key][scope]=dict(common_matches=count,common_iou_delta_sum=delta,
                common_iou_delta_mean=delta/count if count else None)
        paired[key]['phase_gained_lost']={scope:{phase:dict(
            gained=sum(len(r['pairwise'][key]['phase_gained_lost'][scope][phase]['gained_gt']) for r in rows),
            lost=sum(len(r['pairwise'][key]['phase_gained_lost'][scope][phase]['lost_gt']) for r in rows))
            for phase in ('F','P')} for scope in ('known','conservative')}
    triple={}
    for scope in ('known','conservative'):
        n=0;values={name:0. for name in ARMS}
        for row in rows:
            pools={name:{p['gt']:p for p in (row['arms'][name]['pairs'] if scope=='known'
                else row['arms'][name]['conservative']['safe_pairs'])} for name in ARMS}
            common=set.intersection(*(set(p) for p in pools.values()));n+=len(common)
            for name in ARMS:values[name]+=sum(pools[name][g]['iou'] for g in common)
        triple[scope]=dict(common_matches=n,common_iou_sum=values,
            common_iou_mean={name:v/n if n else None for name,v in values.items()})
    return dict(totals=totals,paired=paired,triple_common=triple,
        scope='seen augmented newGT diagnostic; no official accuracy or crop physical mean area')


def runtime_files_contract(expected, config):
    """复核封存环境与SAM2实际文件；不会访问Torch虚拟模块路径。"""
    import importlib
    import subprocess
    from utils.config import project_path
    actual=dict(python_executable=sys.executable,python_version=sys.version,torch=torch.__version__,
                cuda=torch.version.cuda,numpy=np.__version__,cv2=cv2.__version__)
    if any(actual[k]!=expected[k] for k in actual):
        raise RuntimeError('Sealed environment scalar differs')
    repository=Path(project_path(config,config['sam2']['sam2_repo_path'])).resolve()
    sys.path.insert(0,str(repository)); package=Path(importlib.import_module('sam2').__file__).resolve().parent
    if str(package)!=expected['sam2_package'] or sha(expected['sam2_config'])!=expected['sam2_config_sha256']:
        raise RuntimeError('Actual SAM2 package/config differs')
    pinned_files(package,expected['sam2_python_sha256'])
    commit=subprocess.run(['git','-C',str(repository),'rev-parse','HEAD'],capture_output=True,text=True,check=False)
    if (commit.stdout.strip() if commit.returncode==0 else None)!=expected['sam2_repo_head']:
        raise RuntimeError('SAM2 repository revision differs')
    return dict(passed=True,method='sealed real file inventory plus actual environment scalars',
                files=len(expected['sam2_python_sha256']),sam2_config_sha256=expected['sam2_config_sha256'])


def render(folder, stem, rgb, gt, predictions, crop=None, suffix='full'):
    panels=[('Degraded seen draw',rgb),('Processed newGT / unknown gray',gallery_panel(rgb,gt,gt>0))]
    panels.extend((name+' final',gallery_panel(rgb,predictions[name]['instances'],gt>0,gt)) for name in ARMS)
    strips=[]
    for title,image in panels:
        if crop is not None:
            y,x,y1,x1=crop;image=image[y:y1,x:x1]
        image=cv2.resize(image,(360,round(360*image.shape[0]/image.shape[1])),interpolation=cv2.INTER_AREA)
        image=cv2.copyMakeBorder(image,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(image,title,(4,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv2.LINE_AA);strips.append(image)
    if not cv2.imwrite(str(folder/f'{stem}_{suffix}.png'),cv2.cvtColor(np.concatenate(strips,1),cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Gallery write failed')


def run(cache, output):
    from data.affinity_connectivity import undo_spatial
    from data.affinity_mixed import build_datasets
    from data.direct_dual_head_dataset import _spatial_transform
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend,restore_first
    from tools.affinity_connectivity_views import decode_native
    from tools.affinity_native_views import verify_prediction
    from tools.probe_affinity_spatial import compare_spatial_boundaries
    from tools.run_affinity_tail import verify_snapshot,verify_newgt
    from tools.run_affinity_short import precision_record
    from tools.semantic_crossover import revote_instances
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import draw_receipt,frozen_digest,get_restorer
    from train_affinity_tail import CONFIG_PATH,REFERENCE,verify_run
    from train_backend_adaptation import fusion_options,tensor_digest
    from train_direct_semantic_affinity import move_batch,set_seed
    from torch.utils.data import default_collate
    from utils.affinity_deployment import crop_letterbox_output
    from utils.affinity_fusion import affinity_boundary_probability
    from utils.config import load_config,project_path
    from utils.offset_letterbox import letterbox_instance_geometry

    root=ROOT/'outputs/affinity_tail';cache=Path(cache).resolve();output=Path(output).resolve()
    if output.parent!=root/'analysis' or output.exists():
        raise RuntimeError('Fresh output child of outputs/affinity_tail/analysis required')
    if not torch.cuda.is_available():raise RuntimeError('Actual four draws require server sam2_env GPU')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    old,manifest=cache_contract(cache);config=load_config(CONFIG_PATH)
    pipeline=read(root/'pipeline_status.json')
    if (pipeline.get('status')!='complete' or pipeline.get('stage')!='done'
            or pipeline.get('completed')!=['training','inference','spatial_after','render']
            or len(pipeline.get('sources',{}))!=148 or pipeline.get('control_retrained') is not False
            or pipeline.get('ranking_parent_retrained') is not False
            or json.loads(json.dumps(config))!=pipeline['config']):
        raise RuntimeError('Completed source148 TAIL formal pipeline required')
    verify_snapshot(root,pipeline['sources'])
    runtime=runtime_files_contract(pipeline['runtime'],config)
    precision=precision_record(torch)
    if precision!=pipeline['precision'] or precision!=old['precision']:
        raise RuntimeError('Original and TAIL precision differ')
    gt_audit=audit_new_gt(config);verify_newgt(config,gt_audit)
    if gt_audit!=old['newgt'] or config['inference']!=old['inference']:
        raise RuntimeError('NewGT/support or full final postprocess differs')
    run_audit=verify_run(config,root/'candidate',1280);status=read(root/'candidate/status.json')
    if status['epoch']!=20 or status['smoke'] is not False:raise RuntimeError('Formal e20 required')
    paths=dict(r1=Path(project_path(config,REFERENCE,'final.pt')),
        fused=ROOT/'outputs/affinity_fused/candidate/final.pt',tail=root/'candidate/final.pt')
    if any(sha(paths[name])!=digest for name,digest in CHECKPOINTS.items()):
        raise RuntimeError('Registered checkpoint identity differs')
    spatial_after=read(root/'spatial_after/report.json');spatial_before=read(root/'spatial_before/report.json')
    if any(not r['complete'] or len(r['rows'])!=4 for r in (spatial_after,spatial_before)):
        raise RuntimeError('Four sealed spatial rows required')
    protected={str(p):sha(p) for p in [Path(__file__),root/'pipeline_status.json',root/'candidate/status.json',
        root/'spatial_before/report.json',root/'spatial_after/report.json',
        ROOT/'outputs/affinity_fused/pipeline_status.json',*paths.values(),cache/'download_manifest.json']}
    protected.update({str(cache/name):digest for name,digest in manifest['files'].items()})
    initial_imports=existing_project_sources(list(sys.modules.values()),ROOT)
    protected.update({str(ROOT/name):digest for name,digest in initial_imports['files'].items()})
    output.mkdir(parents=True);(output/'gallery').mkdir();(output/'masks').mkdir()
    model,_=load_backend(paths['r1'],config,'cuda');candidate,_=load_backend(paths['tail'],config,'cuda')
    restorer=get_restorer(config,'cuda')
    if (frozen_digest(model)!=frozen_digest(candidate) or frozen_digest(model)!=status['frozen_state_sha256']
            or tensor_digest(restorer.state_dict().items())!=status['restorer_state_sha256']):
        raise RuntimeError('Encoder/semantic/D5a frozen identity differs')
    for m in (model,candidate):
        if any(getattr(m,k,None) is not None for k in ('semantic_lora','geometry_feature_adapter','geometry_highres_refiner')):
            raise RuntimeError('Unexpected deployment adapters/refiners')
    states=dict(r1=tensor_digest(model.state_dict().items()),tail=tensor_digest(candidate.state_dict().items()),
                restorer=tensor_digest(restorer.state_dict().items()))
    manual,_,_=build_datasets(config,1024);manual.set_epoch(1)
    sources={Path(p).name:Path(p) for p in manual.whole.base_dataset.base.samples}
    receipts=[json.loads(s) for s in (paths['r1'].parent/'steps.jsonl').read_text().splitlines()]
    rows=[];mode,kwargs=fusion_options(config)
    if mode!='gated':raise RuntimeError('Only fixed gated deployment allowed')
    with torch.no_grad():
        for saved,before,sealed in zip(old['rows'],spatial_before['rows'],spatial_after['rows']):
            index=saved['draw']
            if any(saved[k]!=sealed[k] or saved[k]!=before[k] for k in ('draw','source','view','seed')):
                raise RuntimeError('Four exact source/view/seed identities differ')
            reference=next(r for r in receipts[:64] if int(r['manual_draw']['draw_index'][0])==index)
            raw=default_collate([manual[index]])
            if (raw['name']!=saved['source'] or raw['training_view'][0]!=saved['view']
                    or raw['crop_box'][0].tolist()!=saved['native_crop']
                    or raw['crop_box'].tolist()!=reference['manual_crop'] or draw_receipt(raw)!=reference['manual_draw']):
                raise RuntimeError('Original draw/crop/augmentation differs')
            batch=move_batch(raw,'cuda');set_seed(saved['seed'])
            restored=restore_first(restorer,batch['image'],saved['seed']+1000000)
            input_sha=tensor_digest([('input',batch['image'])]);restore_sha=tensor_digest([('restored',restored)])
            if input_sha!=saved['input_sha256'] or restore_sha!=saved['restored_sha256']:
                raise RuntimeError('Original input/D5a restoration differs')
            ids=batch['affinity_instance_map'][0].cpu().numpy();content=batch['affinity_valid_content'][0,0].cpu().numpy().astype(bool)
            baseline=root/'spatial_before'/f'draw_{index:03d}.npz';protected[str(baseline)]=sha(baseline)
            with np.load(baseline,allow_pickle=False) as blob:
                if (not np.array_equal(blob['ids'],ids) or not np.array_equal(blob['content'],content)
                        or blob['input_sha256'].item()!=input_sha or blob['restored_sha256'].item()!=restore_sha):
                    raise RuntimeError('Cached newGT/valid/input/restoration differs')
                control_boundary=blob['gated'].copy()
            features=model.encoder(restored)
            sem=model.semantic_decoder(features,restored)
            if isinstance(sem,dict):sem=sem['semantic_logits']
            raw_sem=model.semantic_decoder(model.encoder(batch['image']),batch['image'])
            if isinstance(raw_sem,dict):raw_sem=raw_sem['semantic_logits']
            boundary=affinity_boundary_probability(candidate.affinity_decoder(features)['affinity_logits'].float(),mode=mode,**kwargs)
            actual_control=affinity_boundary_probability(model.affinity_decoder(features)['affinity_logits'].float(),mode=mode,**kwargs)
            if not np.array_equal(actual_control[0,0].cpu().numpy(),control_boundary):
                raise RuntimeError('Real R1 boundary differs from saved control')
            spatial=compare_spatial_boundaries(dict(control_gated=control_boundary,candidate_gated=boundary[0,0].cpu().numpy()),
                ids,content&(ids>0),valid_content=content,reference='control_gated',threshold=config['inference']['boundary_threshold'])['report']
            if spatial!=sealed['report'] or spatial['arms']['control_gated']!=saved['spatial']['arms']['control_gated']:
                raise RuntimeError('Sealed TAIL/R1 spatial response report differs')
            source=sources[saved['source'][0]];rgb=read_rgb(str(source))
            gt_path=Path(project_path(config,config['backend_adaptation']['completed_gt_dir'],source.stem+'_gt.npz'))
            full_gt,lookup=load_completed_semantic_source(gt_path,rgb.shape[:2])
            protected[str(gt_path)]=sha(gt_path);protected[str(gt_path.with_name(source.stem+'_class.json'))]=sha(gt_path.with_name(source.stem+'_class.json'))
            y,x,y1,x1=saved['native_crop'];gt=full_gt[y:y1,x:x1]
            classes={int(i):int(lookup[i]) for i in np.unique(gt) if i>0}
            hf,vf,k=bool(raw['horizontal_flip'].item()),bool(raw['vertical_flip'].item()),int(raw['rotation_k'].item())
            grid,valid,meta=letterbox_instance_geometry(gt,1024,512)
            if (not np.array_equal(_spatial_transform(torch.from_numpy(grid),hf,vf,k).numpy(),ids)
                    or not np.array_equal(_spatial_transform(torch.from_numpy(valid),hf,vf,k).numpy(),content)):
                raise RuntimeError('Original newGT crop/grid alignment differs')
            def native(value):
                return crop_letterbox_output(undo_spatial(value.float(),hf,vf,k),1024,
                    1024-meta.resized_height,1024-meta.resized_width,gt.shape).cpu()
            native_sem=native(sem);probability=native(raw_sem).sigmoid()[0,0].numpy()
            if (tensor_digest([('semantic',native_sem)])!=saved['shared_native_restored_semantic_sha256']
                    or tensor_digest([('raw',torch.from_numpy(probability))])!=saved['shared_raw_probability_sha256']):
                raise RuntimeError('Original frozen restored/raw semantic output differs')
            input_rgb=np.rint(native(batch['image'])[0].permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
            predictions={};arms={};phase={}
            for name,old_name in [('r1','control_gated'),('fused','candidate_gated')]:
                p=cache/'masks'/f'draw_{index:03d}_{old_name}_inst.png';j=p.with_name(p.name.replace('_inst.png','_class.json'))
                mask=cv2.imread(str(p),cv2.IMREAD_UNCHANGED);labels=read(j);verify_prediction(mask,labels,gt.shape)
                arms[name]=diagnose_partition(gt,mask,classes,labels)
                if arms[name]!=saved['arms'][old_name]:raise RuntimeError('Cached final mask no longer reproduces accepted GT diagnostics')
                predictions[name]=dict(instances=mask,classes=labels)
                shutil.copyfile(p,output/'masks'/f'draw_{index:03d}_{name}_inst.png')
                shutil.copyfile(j,output/'masks'/f'draw_{index:03d}_{name}_class.json')
            fresh_control,_=decode_native(native_sem,native(actual_control).clamp(0,1),config)
            fresh_classes,_=revote_instances(fresh_control,probability)
            if not np.array_equal(fresh_control,predictions['r1']['instances']) or fresh_classes!=predictions['r1']['classes']:
                raise RuntimeError('Final R1 postprocess/raw-vote parity failed')
            final_ids,_=decode_native(native_sem,native(boundary).clamp(0,1),config)
            final_classes,_=revote_instances(final_ids,probability);verify_prediction(final_ids,final_classes,gt.shape)
            predictions['tail']=dict(instances=final_ids,classes=final_classes)
            arms['tail']=diagnose_partition(gt,final_ids,classes,final_classes)
            if not cv2.imwrite(str(output/'masks'/f'draw_{index:03d}_tail_inst.png'),final_ids):raise RuntimeError('Final mask write failed')
            write_json(output/'masks'/f'draw_{index:03d}_tail_class.json',final_classes)
            for name in ARMS:phase[name]=phase_diagnostics(gt,predictions[name]['instances'],classes,predictions[name]['classes'],arms[name])
            pairwise={a+'_to_'+b:pairwise_diagnostic(arms[a],arms[b],classes,predictions[a]['classes'],predictions[b]['classes'])
                      for a,b in combinations(ARMS,2)}
            spatial_three=dict(arms=dict(r1=saved['spatial']['arms']['control_gated'],
                fused=saved['spatial']['arms']['candidate_gated'],tail=spatial['arms']['candidate_gated']))
            row=dict(draw=index,source=saved['source'],view=saved['view'],seed=saved['seed'],native_crop=saved['native_crop'],
                shape=list(gt.shape),gt_classes=classes,arms=arms,phase_diagnostics=phase,pairwise=pairwise,
                spatial=spatial_three,response_partition_links=spatial_partition_links(spatial_three,arms),
                input_sha256=input_sha,restored_sha256=restore_sha,
                shared_native_restored_semantic_sha256=saved['shared_native_restored_semantic_sha256'],
                shared_raw_probability_sha256=saved['shared_raw_probability_sha256'])
            rows.append(row);write_json(output/'progress.json',dict(complete=False,processed=len(rows),rows=rows))
            stem=f'draw_{index:03d}';h,w=gt.shape
            render(output/'gallery',stem,input_rgb,gt,predictions)
            render(output/'gallery',stem,input_rgb,gt,predictions,(h//4,w//4,3*h//4,3*w//4),'detail')
            selected=gt==42 if index==37 else np.isin(gt,[127,168]) if index==28 else None
            if selected is not None and selected.any():
                yy,xx=np.nonzero(selected);box=(max(0,int(yy.min())-35),max(0,int(xx.min())-35),
                    min(h,int(yy.max())+36),min(w,int(xx.max())+36))
                render(output/'gallery',stem,input_rgb,gt,predictions,box,'gt42' if index==37 else 'weakpair127_168')
                row['preset_detail']=dict(native_bbox=list(box),gt_ids=[42] if index==37 else [127,168])
            print(json.dumps(dict(draw=index,matched={n:arms[n]['geometry']['matched'] for n in ARMS},
                r1_to_tail_common_delta=pairwise['r1_to_tail']['known']['common_iou_delta_mean'])),flush=True)
    final_states=dict(r1=tensor_digest(model.state_dict().items()),tail=tensor_digest(candidate.state_dict().items()),restorer=tensor_digest(restorer.state_dict().items()))
    if states!=final_states or any(sha(Path(p))!=digest for p,digest in protected.items()):raise RuntimeError('Frozen/protected artifact changed')
    verify_snapshot(root,pipeline['sources']);runtime_end=runtime_files_contract(pipeline['runtime'],config)
    if runtime_end!=runtime or precision_record(torch)!=precision:raise RuntimeError('Runtime/precision changed')
    end_audit=dict(complete=True,frozen_states_sha256=states,protected_sha256=protected,source148_unchanged=True,
        formal_pipeline_unchanged=True,runtime_files=runtime,cache_report_sha256=CACHE_REPORT_SHA256)
    write_json(output/'execution_audit.json',end_audit)
    imported=existing_project_sources(list(sys.modules.values()),ROOT)
    summary=summarize_rows(rows)
    report=dict(complete=True,epoch=20,updates=1280,checkpoints=CHECKPOINTS,rows=rows,summary=summary,
        scope='four seen augmented training draws; full processed newGT including filled; unknown/padding ignored; no test GT',
        deployment_scope='same original crop/full postprocess with frozen restored semantic and raw vote; native crop semantic context is diagnostic-only, not official full-image semantic inference',
        metrics_scope='known-domain class-independent and class-aware matching proxies; conservative frame/unknown whole-object censor; no official mIoU/area or physical crop mean area',
        contract='fixed four draw receipts/input/D5a/GT/control boundary/native restored+raw semantic SHA; cached R1/FUSED final predictions; TAIL only new head; no oracle thresholds',
        cache_manifest_sha256=sha(cache/'download_manifest.json'),cache_report_sha256=CACHE_REPORT_SHA256,
        old_cache_metadata_recovered=old['metadata_recovery']['metadata_recovered'],gpu_fused_forward=False,
        original_run_audit=run_audit,newgt=gt_audit,inference=config['inference'],fusion=dict(mode=mode,**kwargs),precision=precision,
        execution_audit=end_audit,imported_project_sources=imported,
        caveats=['All four source images were seen in training; no independent generalization claim.',
            'Whole vs native draw identities follow sealed receipts, not grouping by draw index.',
            'Significant split/merge uses the unchanged known-domain 50 pixel or 10% contribution rule.',
            'Response gap/component associations use GT IDs; they are not component-level causal proof.',
            'Class-aware seen mean instance IoU is a diagnostic proxy, not the competition evaluator.',
            'Ranking representativeness means response-rank coverage, not spatial/instance balance.'])
    write_json(output/'report.json',report);write_json(output/'summary.json',summary)
    pages=['<!doctype html><meta charset="utf-8"><title>TAIL four seen draws</title><p>Same degraded seen draws; GT colors shared; unknown gray. Diagnostic crop semantics, no test GT.</p>']
    for p in sorted((output/'gallery').glob('*.png')):pages.append(f'<p>{p.stem}<br><img width="1800" src="gallery/{p.name}"></p>')
    (output/'index.html').write_text('\n'.join(pages),encoding='utf8')
    inventory={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='download_manifest.json'}
    write_json(output/'download_manifest.json',dict(complete=True,files=inventory))
    print('COMPLETE',output,flush=True)


def self_test():
    gt=np.ones((40,40),np.int32);gt[10:30,10:30]=2
    labels={1:0,2:1};mask=gt.astype(np.uint16)
    diagnosis=diagnose_partition(gt,mask,labels,{'1':0,'2':1})
    phase=phase_diagnostics(gt,mask,labels,{'1':0,'2':1},diagnosis)
    assert phase['seen_class_aware_mean_instance_iou']==1.
    wrong=phase_diagnostics(gt,mask,labels,{'1':1,'2':0},diagnosis)
    assert wrong['seen_class_aware_mean_instance_iou']==0.
    row=dict(arms={name:diagnosis for name in ARMS},phase_diagnostics={name:phase for name in ARMS},
        pairwise={a+'_to_'+b:pairwise_diagnostic(diagnosis,diagnosis,labels,{'1':0,'2':1},{'1':0,'2':1}) for a,b in combinations(ARMS,2)})
    summary=summarize_rows([row]);assert summary['triple_common']['known']['common_matches']==2
    assert summary['paired']['fused_to_tail']['known']['common_iou_delta_mean']==0.
    json.dumps(summary,allow_nan=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--cache',default='outputs/affinity_fused/analysis/four_gt_v2')
    parser.add_argument('--output',default='outputs/affinity_tail/analysis/four_gt')
    parser.add_argument('--summarize',help='CPU-only reaggregation of an existing completed report; prints JSON')
    args=parser.parse_args()
    if args.self_test:self_test();print('CPU self-test passed')
    elif args.summarize:print(json.dumps(summarize_rows(read(args.summarize)['rows']),ensure_ascii=False,allow_nan=False))
    else:run(ROOT/args.cache,ROOT/args.output)
