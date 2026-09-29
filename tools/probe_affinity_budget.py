# -*- coding: utf-8 -*-
"""同一e20起点、固定8视图的64步预算短测；不启动正式训练或替换部署。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.affinity_connectivity import build_datasets, resize_trusted, undo_spatial
from data.backend_adaptation import PairedDegradationDataset
from models.backend_adaptation import load_backend, restore_first
from tools.affinity_connectivity_views import decode_native
from tools.probe_affinity_chain import relations_at
from tools.render_backend_ablation import overlay, text_line, write_png
from tools.run_affinity_preserve import trusted_changes
from tools.semantic_crossover import revote_instances
from train_affinity_connectivity import auxiliary_loss, capped_weight, configure_training, frozen_digest, get_restorer
from train_backend_adaptation import fusion_options, sha, tensor_digest, write_json
from train_direct_semantic_affinity import compute_affinity_loss, move_batch, set_seed
from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.gradient_budget import adamw_delta, gradient_pair, norm, parameter_weight
from utils.offset_letterbox import geometry_letterbox_metadata, letterbox_instance_geometry


def native(value,item):
    meta=item['meta']
    return crop_letterbox_output(value,1024,1024-meta.resized_height,1024-meta.resized_width,item['shape']).cpu()


def serialize_rows(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf8')


def render(item, ids, classes, boundary, path, title):
    rgb=item['rgb'];bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    heat=cv2.applyColorMap(np.rint(np.clip(boundary,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
    bodies=[bgr,heat,overlay(bgr,item['base_ids'],{int(k):v for k,v in item['base_classes'].items()}),
            overlay(bgr,ids,{int(k):v for k,v in classes.items()})]
    panels=[]
    for label,body in zip([title,'Boundary fixed 0..1','Initial control e20','Short probe: F gold / P blue'],bodies):
        small=cv2.resize(body,(340,round(340*body.shape[0]/body.shape[1])),interpolation=cv2.INTER_AREA)
        small=cv2.copyMakeBorder(small,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        text_line(small,label,(6,21),340,size=.44);panels.append(small)
    path.parent.mkdir(parents=True,exist_ok=True);write_png(path,np.concatenate(panels,1))


def selected_loss(logits,item):
    mask=item['fixed_selected'];target=item['target'];result={}
    for label,value in [('same',1),('different',0)]:
        selected=mask&(target==value)
        result[label]=dict(edges=int(selected.sum()),bce=float(F.binary_cross_entropy_with_logits(logits[selected],target[selected])) if bool(selected.any()) else None,
            affinity_mean=float(logits[selected].sigmoid().mean()) if bool(selected.any()) else None)
    return result


@torch.no_grad()
def evaluate(model,items,config,out,arm,step,indices,full=False):
    model.affinity_decoder.eval();mode,kwargs=fusion_options(config);rows=[]
    for index in indices:
        item=items[index];logits=model.affinity_decoder(item['features'])['affinity_logits'].float()
        boundary=native(affinity_boundary_probability(logits,mode=mode,**kwargs),item)
        ids,_=decode_native(item['semantic_native'],boundary,config)
        assert ids.dtype==np.uint16 and int(ids.max())<=65535 and ids.shape==item['shape']
        classes,_=revote_instances(ids,item['raw_probability'])
        states=relations_at(ids,item['relations']);fixes={}
        for kind,desired in [('merge','different'),('split','same')]:
            vals=[s for r,s in zip(item['relations'],states) if r['kind']==kind]
            fixes[kind]=dict(cases=len(vals),fixed=sum(s==desired for s in vals),blocked=sum(s=='blocked' for s in vals))
        row=dict(source=item['source'],view=item['view'],step=step,arm=arm,instances=len(classes),fixes=fixes,
            selected=selected_loss(logits,item),boundary_mean_abs_delta=float((boundary-item['base_boundary']).abs().mean()))
        if full:
            changes=trusted_changes(item['gt'],item['trust'],item['base_ids'],ids)
            row['new_errors']=changes['summary']
            path=out/arm/f'step_{step:03}'
            path.mkdir(parents=True,exist_ok=True)
            assert cv2.imwrite(str(path/(item['key']+'_inst.png')),ids)
            write_json(path/(item['key']+'_class.json'),classes)
        render(item,ids,classes,boundary[0,0].numpy(),out/arm/'monitor'/f'step_{step:03}'/(item['key']+'.png'),item['key']+f' {arm} s{step}')
        rows.append(row)
        print(json.dumps(dict(stage='evaluate',arm=arm,step=step,key=item['key'],fixes=fixes)),flush=True)
    serialize_rows(out/arm/f'eval_{step:03}.jsonl',rows)
    return rows


def prepare(model,restorer,config,out):
    options=config['affinity_budget_probe'];cfg=config['backend_adaptation']
    manual,pseudo,metadata=build_datasets(config);base=manual.base_dataset
    deg=load_config(project_path(config,cfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
    paired=PairedDegradationDataset(base,{**deg,'profile_probabilities':[0.,1.,0.,0.]},cfg['noise'],cfg['seed'],repeats=1)
    paired.set_epoch(options['fixed_epoch']);pseudo.set_epoch(options['fixed_epoch'])
    references={}
    for path in Path(project_path(config,'outputs/affinity_trace')).rglob('views.jsonl'):
        for line in path.read_text(encoding='utf8').splitlines():
            r=json.loads(line);references[(r['source'],r['view'])]=r
    assert len(references)==64 and len(base)==32
    items=[];mode,kwargs=fusion_options(config)
    for i in options['manual_indices']:
        clean=base[i];sample=paired[i];stem=base.samples[i].stem
        with np.load(base.completed_gt_dir/(stem+'_gt.npz'),allow_pickle=False) as blob:
            gt=blob['instance_map'].copy();trust=blob['original_covered'].astype(bool)&(gt>0)
        blur=undo_spatial(sample['image'],sample['horizontal_flip'],sample['vertical_flip'],sample['rotation_k'])
        for view,tensor in [('clean',clean['image']),('degraded',blur)]:
            record={**clean,'image':tensor,'trusted_pixels':resize_trusted(trust),
                    'native_shape':torch.tensor(gt.shape,dtype=torch.int32),'horizontal_flip':False,'vertical_flip':False,'rotation_k':0}
            batch=move_batch(default_collate([record]),'cuda')
            # native_training_maps 保留单图 CHW 的 strides。stack 会改变内存布局，
            # 当前 GPU 的 encoder 卷积因此产生足以改变最终连通性的数值差异。
            # 缓存优化必须复现原检查入口，不能顺便改变它的计算布局。
            batch['image']=tensor[None].cuda().float()
            item=dict(source=stem,view=view,key=stem+'_'+view,shape=gt.shape,meta=geometry_letterbox_metadata(gt.shape,1024,512),
                gt=gt,trust=trust,batch=batch,relations=references[(stem,view)]['relations'],seed=cfg['restoration']['inference_seed']+i)
            with torch.no_grad():
                image=batch['image'];raw=model.semantic_decoder(model.encoder(image),image)
                restored=restore_first(restorer,image,item['seed']);features=model.encoder(restored)
                semantic=model.semantic_decoder(features,restored);logits=model.affinity_decoder(features)['affinity_logits']
                item.update(features=features,semantic_native=native(semantic,item),raw_probability=native(raw,item).sigmoid()[0,0].numpy(),
                    rgb=np.rint(native(image,item)[0].permute(1,2,0).numpy()*255).astype(np.uint8))
                boundary=native(affinity_boundary_probability(logits,mode=mode,**kwargs),item)
                ids,_=decode_native(item['semantic_native'],boundary,config)
            reference=references[(stem,view)]
            assert int(ids.max())==reference['variants']['base']['instances'], item['key']
            assert relations_at(ids,item['relations'])==[r['stages']['base']['final']['full'] for r in item['relations']]
            pred,_,_=letterbox_instance_geometry(ids,1024,512)
            item.update(base_ids=ids,base_boundary=boundary,partition=torch.from_numpy(pred.astype(np.int64))[None].cuda())
            item['base_classes'],_=revote_instances(ids,item['raw_probability'])
            target,valid=build_affinity_targets_torch(batch['affinity_instance_map'],batch['affinity_valid_content'])
            item.update(target=target,valid=valid)
            leaf=logits.detach().clone().requires_grad_(True)
            extra,stats=auxiliary_loss(config,leaf,target,valid,item['partition'],batch['affinity_instance_map'],batch['trusted_pixels'])
            item['fixed_selected']=torch.autograd.grad(extra,leaf)[0]!=0
            assert int(item['fixed_selected'].sum())==stats['selected_edges']
            item['initial_selected']=selected_loss(logits,item)
            items.append(item)
            print(json.dumps(dict(stage='cache',key=item['key'],edges=stats['selected_edges'])),flush=True)
    pseudos=[]
    for i in range(len(items)):
        sample=pseudo[i];batch=move_batch(default_collate([sample]),'cuda')
        with torch.no_grad():features=model.encoder(restore_first(restorer,batch['image'],cfg['restoration']['inference_seed']+1000000+i))
        pseudos.append(dict(batch=batch,features=features,source=sample['name']))
    write_json(out/'inputs.json',dict(data=metadata,manual=[dict(source=r['source'],view=r['view'],seed=r['seed'],initial_selected=r['initial_selected']) for r in items],
        pseudo=[r['source'] for r in pseudos],features_cached=True,selection_partition='frozen_initial_full_deployment',
        scope='fixed small-cohort overfit mechanism only; no heldout split or generalization measurement'))
    return items,pseudos


def update_arm(model,items,pseudos,config,out,arm,initial):
    options=config['affinity_budget_probe'];cfg=config['backend_adaptation'];extra=config['affinity_connectivity']
    model.affinity_decoder.load_state_dict(initial,strict=True);configure_training(model)
    params=list(model.affinity_decoder.parameters())
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rates']['affinity'],weight_decay=cfg['weight_decay'],eps=1e-4)
    root=out/arm;root.mkdir(parents=True,exist_ok=True);logs=[]
    order=np.random.default_rng(cfg['seed']).permutation(len(items))
    finals=None
    for step in range(options['updates']):
        cycle,position=divmod(step,len(items));index=int(np.roll(order,cycle)[position]);item=items[index];pseudo=pseudos[index]
        set_seed(cfg['seed']+step);configure_training(model);optimizer.zero_grad(set_to_none=True)
        logits=model.affinity_decoder(item['features'])['affinity_logits']
        base,_=compute_affinity_loss(logits,item['batch'],config['direct_semantic_affinity']['affinity_loss'],pseudo=False)
        active=step%extra['every_updates']==0
        weight=0.;details={};b=None
        if active:
            auxiliary,sampling=auxiliary_loss(config,logits,item['target'],item['valid'],item['partition'],
                item['batch']['affinity_instance_map'],item['batch']['trusted_pixels'])
            warmup=min(1.,(cycle+1)/extra['warmup_epochs']);requested=extra['weight']*warmup
            old_weight,logit_stats=capped_weight(base,auxiliary,logits,requested,extra['logit_gradient_ratio'])
            a=torch.autograd.grad(base,params,retain_graph=True)
            b=torch.autograd.grad(auxiliary,params)
            new_weight,parameter_stats=parameter_weight(a,b,requested=requested,target_ratio=options['parameter_ratio']*warmup)
            weight=old_weight if arm=='logit' else new_weight
            pair=gradient_pair(a,b)
            details=dict(warmup=warmup,weight=weight,old_weight=old_weight,parameter_weight=new_weight,
                parameter_ratio=weight*pair['auxiliary_norm']/max(pair['base_norm'],1e-12),gradient=pair,
                logit_ratio=weight*logit_stats['aux_logit_grad']/max(logit_stats['base_logit_grad'],1e-12),
                auxiliary=float(auxiliary.detach()),sampling=sampling,parameter_target=parameter_stats)
        else:a=torch.autograd.grad(base,params)
        plogits=model.affinity_decoder(pseudo['features'])['affinity_logits']
        ploss,_=compute_affinity_loss(plogits,pseudo['batch'],config['direct_semantic_affinity']['affinity_loss'],pseudo=True)
        pg=torch.autograd.grad(ploss,params)
        common=[x.detach()+cfg['pseudo_weight']*p.detach() for x,p in zip(a,pg)]
        combined=common if not active else [x+weight*y.detach() for x,y in zip(common,b)]
        if not all(bool(torch.isfinite(v).all()) for v in combined):raise FloatingPointError('Nonfinite combined gradient')
        predicted,clip=adamw_delta(optimizer,combined,cfg['grad_clip'])
        if active:
            plain,plain_clip=adamw_delta(optimizer,common,cfg['grad_clip'])
            details.update(extra_to_combined_gradient=weight*float(norm(b))/max(float(norm(common)),1e-12),
                adamw_extra_delta_ratio=float(norm([a-b for a,b in zip(predicted,plain)]))/max(float(norm(plain)),1e-12),
                common_gradient_cosine=gradient_pair(common,b)['cosine'],common_clip=plain_clip)
            del plain
        before=[p.detach().clone() for p in params]
        for p,g in zip(params,combined):p.grad=g
        torch.nn.utils.clip_grad_norm_(params,cfg['grad_clip'],error_if_nonfinite=True);optimizer.step()
        actual=[p.detach()-v for p,v in zip(params,before)]
        error=max(float((a-b).abs().max()) for a,b in zip(actual,predicted))
        assert error<2e-7, error
        logs.append(dict(step=step+1,cycle=cycle+1,index=index,source=item['source'],view=item['view'],pseudo=pseudo['source'],
            manual_bce=float(base.detach()),pseudo_bce=float(ploss.detach()),active=active,details=details,
            clip=clip,actual_update_norm=float(norm(actual)),adamw_prediction_max_error=error))
        serialize_rows(root/'steps.jsonl',logs)
        del logits,plogits,base,ploss,a,b,pg,common,combined,predicted,before,actual
        if active:print(json.dumps(dict(arm=arm,step=step+1,parameter_ratio=details['parameter_ratio'],adamw_extra_ratio=details['adamw_extra_delta_ratio'])),flush=True)
        if step+1 in options['snapshots']:
            state={k:v.detach().cpu().clone() for k,v in model.affinity_decoder.state_dict().items()}
            path=root/f'head_{step+1:03}.pt';torch.save(dict(format='affinity_budget_probe_v1',steps=step+1,geometry_state_dict=state,config=config,arm=arm),path)
            is_final=step+1==options['updates']
            evaluated=evaluate(model,items,config,out,arm,step+1,range(len(items)) if is_final else options['monitor_views'],full=is_final)
            if is_final:finals=evaluated
    state=torch.load(root/f'head_{options["updates"]:03}.pt',map_location='cpu',weights_only=False)['geometry_state_dict']
    expected=tensor_digest(model.affinity_decoder.state_dict().items());model.affinity_decoder.load_state_dict(state,strict=True)
    assert tensor_digest(model.affinity_decoder.state_dict().items())==expected
    write_json(root/'status.json',dict(complete=True,updates=options['updates'],strict_reload_equal=True,head_sha256=expected))
    return logs,finals


def summarize(rows):
    return dict(views=len(rows),fixes={kind:{key:sum(r['fixes'][kind][key] for r in rows) for key in ['cases','fixed','blocked']} for kind in ['merge','split']},
        new_split_pairs=sum(r['new_errors']['new_split_pairs'] for r in rows),
        new_merge_pairs=sum(r['new_errors']['fixed_anchor_merges'].get('new_pairs',0) for r in rows),
        selected_bce={direction:float(np.mean([r['selected'][direction]['bce'] for r in rows if r['selected'][direction]['bce'] is not None])) for direction in ['same','different']})


def run(args):
    config=load_config(args.config);options=config['affinity_budget_probe'];out=Path(project_path(config,options['output_dir']))
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(2)
    if not torch.cuda.is_available():raise RuntimeError('Use sam2_env GPU')
    assert options['updates']==64 and options['selection_partition']=='frozen_initial_full_deployment'
    checkpoint=project_path(config,options['checkpoint']);assert sha(checkpoint)==options['checkpoint_sha256']
    started=time.time();model,bundle=load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False)
    assert model.geometry_feature_adapter is None and model.geometry_highres_refiner is None and model.semantic_lora is None
    restorer=get_restorer(config,'cuda');frozen=frozen_digest(model);restore_state=tensor_digest(restorer.state_dict().items())
    initial={k:v.detach().cpu().clone() for k,v in model.affinity_decoder.state_dict().items()}
    write_json(out/'status.json',dict(status='preparing',config=config,checkpoint_sha256=sha(checkpoint),epoch=bundle['epoch'],automatic_full_training=False))
    source=out/'source';source.mkdir()
    for name in ['tools/probe_affinity_budget.py','utils/gradient_budget.py','config/experiments/affinity_budget.yaml']:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    items,pseudos=prepare(model,restorer,config,out)
    initial_rows=[]
    for i,item in enumerate(items):
        initial_rows.append(dict(source=item['source'],view=item['view'],selected=item['initial_selected']))
        render(item,item['base_ids'],item['base_classes'],item['base_boundary'][0,0].numpy(),out/'initial/monitor/step_000'/(item['key']+'.png'),item['key']+' initial')
    write_json(out/'initial.json',initial_rows)
    results={};logs={}
    for arm in ['logit','parameter']:
        write_json(out/'status.json',dict(status='running',arm=arm,config=config,checkpoint_sha256=sha(checkpoint),automatic_full_training=False))
        logs[arm],evaluated=update_arm(model,items,pseudos,config,out,arm,initial);results[arm]=summarize(evaluated)
        assert frozen_digest(model)==frozen and tensor_digest(restorer.state_dict().items())==restore_state
    for a,b in zip(logs['logit'],logs['parameter']):
        assert all(a[k]==b[k] for k in ['step','cycle','index','source','view','pseudo','active'])
    for arm in results:
        active=[r['details'] for r in logs[arm] if r['active']]
        results[arm]['parameter_ratio_median']=float(np.median([r['parameter_ratio'] for r in active]))
        results[arm]['adamw_extra_delta_ratio_median']=float(np.median([r['adamw_extra_delta_ratio'] for r in active]))
        results[arm]['extra_to_combined_gradient_median']=float(np.median([r['extra_to_combined_gradient'] for r in active]))
        results[arm]['active_views']=sorted({r['index'] for r in logs[arm] if r['active']})
    write_json(out/'report.json',dict(complete=True,scope='fixed 4-source 8-view short probe, not generalization, not official score',
        updates_per_arm=64,checkpoint_sha256=sha(checkpoint),config=config,results=results,frozen_exact=True,paired_inputs=True,
        elapsed_seconds=time.time()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,automatic_full_training=False))
    write_json(out/'status.json',dict(status='complete',results=results,automatic_full_training=False))
    print('BUDGET PROBE COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/experiments/affinity_budget.yaml')
    args=parser.parse_args()
    try:
        run(args)
    except Exception as error:
        config=load_config(args.config)
        out=Path(project_path(config,config['affinity_budget_probe']['output_dir']))
        if out.is_dir() and not isinstance(error,FileExistsError):
            write_json(out/'status.json',dict(status='failed',error=repr(error),automatic_full_training=False))
        raise
