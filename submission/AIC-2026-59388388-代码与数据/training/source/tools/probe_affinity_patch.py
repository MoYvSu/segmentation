# -*- coding: utf-8 -*-
"""固定control的原生patch、D5a放大恢复及逐张/batch诊断；不训练。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.backend_adaptation import CanonicalBackendDataset
from data.dataset import letterbox
from data.mim_dataset import list_images
from data.rgb_spatial_blur import apply_sigma_map_blur
from models.backend_adaptation import load_backend,restore_first
from tools.affinity_connectivity_views import decode_native
from tools.analyze_affinity_connectivity import load_directory,render
from tools.render_backend_ablation import text_line,write_png
from tools.run_affinity_preserve import trusted_changes
from tools.semantic_crossover import revote_instances
from train_affinity_connectivity import get_restorer
from train_backend_adaptation import fusion_options,prediction_maps,sha,tensor_digest,write_json
from utils.affinity_deployment import crop_letterbox_output,prepare_image
from utils.affinity_fusion import affinity_boundary_probability
from utils.config import load_config,project_path
from utils.patch_diagnostic import BlendMap,phase_description,reconstruction_metrics,tile_boxes,trusted_phase_metrics


def tensor_from_rgb(rgb,device='cuda'):
    value,_,ph,pw=letterbox(rgb,1024)
    tensor=torch.from_numpy(value).permute(2,0,1).float().unsqueeze(0).to(device)
    if rgb.dtype==np.uint8:tensor=tensor/255.
    return tensor,ph,pw


@torch.no_grad()
def boundary_grid(model,image,config):
    features=model.encoder(image);output=model.affinity_decoder(features)['affinity_logits'].float()
    mode,kwargs=fusion_options(config)
    return affinity_boundary_probability(output,mode=mode,**kwargs)


def cases_for(config):
    opt=config['affinity_patch_probe'];cfg=config['backend_adaptation']
    manual=CanonicalBackendDataset(project_path(config,config['paths']['raw_data_dir']),project_path(config,cfg['completed_gt_dir']))
    assert len(manual)==32
    base_seed=cfg['restoration']['inference_seed']
    cases=[dict(kind='train',name=manual.samples[i].stem,path=str(manual.samples[i]),index=i,seed=base_seed+i) for i in opt['manual_indices']]
    files=[Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))]
    assert len(files)==100
    fixed=opt['test_fixed'];remaining=[p.stem for p in files if p.stem not in fixed]
    chosen=fixed+sorted(np.random.default_rng(opt['selection_seed']).choice(remaining,opt['test_random'],replace=False).tolist())
    assert set(chosen).issubset({p.stem for p in files})
    cases += [dict(kind='test',name=p.stem,path=str(p),index=i,seed=base_seed+i) for i,p in enumerate(files) if p.stem in chosen]
    return cases


def load_gt(case,config):
    if case['kind']!='train':return None
    root=Path(project_path(config,config['backend_adaptation']['completed_gt_dir']))
    with np.load(root/(case['name']+'_gt.npz'),allow_pickle=False) as blob:
        gt=blob['instance_map'].copy();trust=blob['original_covered'].astype(bool)&(gt>0)
    classes={int(k):int(v) for k,v in json.loads((root/(case['name']+'_class.json')).read_text()).items()}
    return gt,trust,classes


def save_variant(case,name,boundary,semantic,raw_probability,config,out,baseline=None,extra=None):
    root=out/case['name'];folder=root/name;folder.mkdir(parents=True,exist_ok=True)
    boundary=np.asarray(boundary,dtype=np.float32)
    ids,_=decode_native(semantic,torch.from_numpy(boundary)[None,None],config)
    classes,_=revote_instances(ids,raw_probability)
    assert ids.shape==boundary.shape and ids.dtype==np.uint16 and int(ids.max())<=65535
    assert {str(int(x)) for x in np.unique(ids) if x}==set(classes)
    assert cv2.imwrite(str(folder/(case['name']+'_inst.png')),ids)
    write_json(folder/(case['name']+'_class.json'),classes)
    np.savez_compressed(folder/'boundary.npz',boundary=boundary)
    row=dict(source=case['name'],kind=case['kind'],variant=name,shape=list(ids.shape),phases=phase_description(ids,classes),extra=extra or {})
    labelled=load_gt(case,config)
    if labelled is not None:
        gt,trust,gc=labelled;row['known_domain']=trusted_phase_metrics(gt,gc,trust,ids,classes)
        if baseline is not None:
            changes=trusted_changes(gt,trust,baseline,ids)
            row['new_errors']=dict(split_by_class=dict(Counter('ferrite' if gc[p['gt']]==1 else 'pearlite' for p in changes['new_split_pairs'])),
                merge_by_class=dict(Counter('FF' if all(gc[i]==1 for i in p['gt_ids']) else 'PP' if all(gc[i]==0 for i in p['gt_ids']) else 'mixed' for p in changes['new_merge_pairs'])),summary=changes['summary'])
            write_json(folder/'trusted_changes.json',changes)
    write_json(folder/'summary.json',row)
    print(json.dumps(dict(stage='decoded',source=case['name'],variant=name,phases=row['phases'])),flush=True)
    return ids,classes,row


@torch.no_grad()
def full_image(case,model,restorer,config,out):
    rgb,semantic,boundary=prediction_maps(model,case['path'],config,'cuda',restorer,case['seed'])
    _,image,ph,pw=prepare_image(case['path'],1024,'cuda')
    raw=model(image);mode,kwargs=fusion_options(config)
    raw_sem=crop_letterbox_output(raw['semantic_logits'].float(),1024,ph,pw,rgb.shape[:2]).cpu()
    raw_probability=raw_sem.sigmoid()[0,0].numpy()
    raw_boundary=crop_letterbox_output(affinity_boundary_probability(raw['affinity_logits'].float(),mode=mode,**kwargs),1024,ph,pw,rgb.shape[:2]).cpu()[0,0].numpy()
    root=out/case['name'];root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'fixed_maps.npz',semantic=semantic.numpy(),raw_probability=raw_probability)
    base,classes,row=save_variant(case,'global_d5a',boundary[0,0].numpy(),semantic,raw_probability,config,out)
    if case['kind']=='test':
        expected,ec=load_directory(project_path(config,config['affinity_patch_probe']['control_predictions']),case['name'])
        assert np.array_equal(base,expected) and classes==ec, 'Official control output mismatch: '+case['name']
    _,_,rawrow=save_variant(case,'global_raw',raw_boundary,semantic,raw_probability,config,out,base,
        extra=dict(semantic_source='fixed_global_d5a_geometry_and_raw_final_vote'))
    return [row,rawrow]


@torch.no_grad()
def tiled_image(case,size,model,restorer,config,out):
    rgb=cv2.cvtColor(cv2.imread(case['path']),cv2.COLOR_BGR2RGB);shape=rgb.shape[:2]
    with np.load(out/case['name']/'fixed_maps.npz') as maps:
        semantic=torch.from_numpy(maps['semantic'].copy());probability=maps['raw_probability'].copy()
    base,_=load_directory(out/case['name']/'global_d5a',case['name'])
    boxes=tile_boxes(shape,size,config['affinity_patch_probe']['overlap'])
    raw_blend,d5a_blend=BlendMap(shape),BlendMap(shape)
    seeds=[]
    for i,box in enumerate(boxes):
        y,x,y1,x1=box;tensor,ph,pw=tensor_from_rgb(rgb[y:y1,x:x1]);native=(y1-y,x1-x)
        seed=case['seed']+100000+size*1000+i;seeds.append(seed)
        raw=boundary_grid(model,tensor,config)
        restored=restore_first(restorer,tensor,seed);repaired=boundary_grid(model,restored,config)
        raw_blend.add(box,crop_letterbox_output(raw,1024,ph,pw,native).cpu()[0,0].numpy())
        d5a_blend.add(box,crop_letterbox_output(repaired,1024,ph,pw,native).cpu()[0,0].numpy())
    rows=[]
    for suffix,blend in [('d5a',d5a_blend),('raw',raw_blend)]:
        result,audit=blend.finish();name=f'patch{size}_{suffix}'
        _,_,row=save_variant(case,name,result,semantic,probability,config,out,base,
            extra=dict(tile_size=size,tile_count=len(boxes),overlap=config['affinity_patch_probe']['overlap'],boxes=boxes,seeds=seeds,
                       blend=audit,execution='sequential',global_watershed_passes=1))
        rows.append(row)
    names=['global_d5a',f'patch{size}_d5a',f'patch{size}_raw']
    predictions=[load_directory(out/case['name']/n,case['name']) for n in names]
    for label,crop in [('full',None),('upper',(.10,.10,.45,.45)),('lower',(.50,.40,.90,.80))]:
        render(case['path'],predictions,names,out/'gallery'/f'{case["name"]}_{size}_{label}.png',case['name'],crop=crop,width=360)
    panels=[]
    for name in names:
        with np.load(out/case['name']/name/'boundary.npz') as maps:value=maps['boundary']
        heat=cv2.applyColorMap(np.rint(value*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
        heat=cv2.resize(heat,(420,round(420*shape[0]/shape[1])),interpolation=cv2.INTER_AREA)
        heat=cv2.copyMakeBorder(heat,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));text_line(heat,name,(5,20),420,size=.5);panels.append(heat)
    write_png(out/'gallery'/f'{case["name"]}_{size}_boundary.png',np.concatenate(panels,1))
    return rows


@torch.no_grad()
def first_noise(restorer,image,noise):
    t=torch.full((len(image),),restorer.num_steps,dtype=torch.long,device=image.device)
    state=restorer.q_sample(image,image,t,noise=noise)
    clean=restorer.denoise(state,image,t).float()
    return (image.float()+(clean-image.float())).clamp(0,1)


def difference(a,b):
    d=(a-b).abs().float()
    return dict(max=float(d.max()),mean=float(d.mean()),equal=bool(torch.equal(a,b)))


@torch.no_grad()
def batch_check(case,model,restorer,config,out):
    rgb=cv2.cvtColor(cv2.imread(case['path']),cv2.COLOR_BGR2RGB)
    boxes=tile_boxes(rgb.shape[:2],1024,config['affinity_patch_probe']['overlap'])[:config['affinity_patch_probe']['batch_probe']]
    tensors=[];noises=[];restored=[];probabilities=[];raws=[];seeds=[]
    for i,(y,x,y1,x1) in enumerate(boxes):
        image,_,_=tensor_from_rgb(rgb[y:y1,x:x1]);seed=case['seed']+900000+i
        noise=torch.randn(image.shape,device=image.device,dtype=torch.float32,generator=torch.Generator(device='cuda').manual_seed(seed))
        repair=first_noise(restorer,image,noise)
        assert torch.equal(repair,restore_first(restorer,image,seed))
        tensors.append(image);noises.append(noise);restored.append(repair);seeds.append(seed)
        probabilities.append(boundary_grid(model,repair,config));raws.append(boundary_grid(model,image,config))
    images=torch.cat(tensors,0).to(memory_format=torch.channels_last);noise=torch.cat(noises,0)
    repaired=first_noise(restorer,images,noise);batch=boundary_grid(model,repaired,config);batch_raw=boundary_grid(model,images,config)
    rows=[dict(index=i,seed=seeds[i],restored_rgb=difference(restored[i],repaired[i:i+1]),
        boundary_d5a=difference(probabilities[i],batch[i:i+1]),boundary_raw=difference(raws[i],batch_raw[i:i+1]),
        threshold_flip_fraction=float(((probabilities[i]>.65)!=(batch[i:i+1]>.65)).float().mean())) for i in range(len(boxes))]
    # 只用已有训练图：固定首块对应的全图语义，另检查最终局部分区是否因batch数值变化。
    # 此处不是全图精度指标；其余正式尺度比较全部逐张执行。
    y,x,y1,x1=boxes[0]
    with np.load(out/case['name']/'fixed_maps.npz') as maps:sem=torch.from_numpy(maps['semantic'][:,:,y:y1,x:x1].copy())
    native=(y1-y,x1-x);_,ph,pw=tensor_from_rgb(rgb[y:y1,x:x1])
    ba=crop_letterbox_output(probabilities[0],1024,ph,pw,native).cpu()
    bb=crop_letterbox_output(batch[0:1],1024,ph,pw,native).cpu()
    ia,_=decode_native(sem,ba,config);ib,_=decode_native(sem,bb,config)
    result=dict(source=case['name'],batch_size=len(boxes),per_patch=rows,explicit_noise_seed_replay_exact=True,
        single_input_stride=list(tensors[0].stride()),batch_input_stride=list(images.stride()),
        first_patch_partition=dict(single_instances=int(ia.max()),batch_instances=int(ib.max()),pixel_equal=bool(np.array_equal(ia,ib))),
        main_experiment_execution='sequential_even_if_batch_differs',cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    write_json(out/'batch_check.json',result);print('BATCH CHECK',json.dumps(result),flush=True)
    return result


@torch.no_grad()
def restoration_check(cases,restorer,config,out):
    opt=config['affinity_patch_probe'];rows=[];folder=out/'restoration';folder.mkdir(parents=True,exist_ok=True)
    for case in [c for c in cases if c['kind']=='train']:
        rgb=cv2.cvtColor(cv2.imread(case['path']),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        h,w=rgb.shape[:2];size=min(1024,h,w)
        for i,fraction in enumerate(np.linspace(.2,.8,opt['restoration_crops'])):
            y,x=round((h-size)*fraction),round((w-size)*(1-fraction));target=rgb[y:y+size,x:x+size]
            if size!=1024:target=cv2.resize(target,(1024,1024),interpolation=cv2.INTER_LINEAR)
            for profile in opt['restoration_profiles']:
                condition=target.copy();scale=1.
                if profile=='spatial050':
                    phase=np.linspace(0,1,1024,dtype=np.float32);blend=phase**3*(phase*(6*phase-15)+10)
                    sigma=np.broadcast_to(np.sqrt(.6**2+(4.5**2-.6**2)*blend),(1024,1024)).copy()
                    condition=apply_sigma_map_blur(condition,sigma,(.4,5.2),8);scale=.5
                elif profile=='down075':scale=.75
                elif profile=='down050':scale=.5
                elif profile!='identity':raise ValueError(profile)
                if scale<1:
                    low=cv2.resize(condition,(round(1024*scale),)*2,interpolation=cv2.INTER_AREA)
                    condition=cv2.resize(low,(1024,1024),interpolation=cv2.INTER_LINEAR)
                tensor,_,_=tensor_from_rgb(condition);seed=case['seed']+800000+i
                repaired=restore_first(restorer,tensor,seed)[0].permute(1,2,0).cpu().numpy()
                before,after=reconstruction_metrics(target,condition),reconstruction_metrics(target,repaired)
                row=dict(source=case['name'],crop=i,box=[y,x,size,size],profile=profile,seed=seed,input=before,d5a=after)
                rows.append(row)
                panels=[]
                # 固定中心256像素，显示真实细节，不仅是全图缩略。
                for title,image in [('Clear reference',target),('Interpolated input',condition),('D5a first_x0',repaired)]:
                    body=cv2.cvtColor(np.rint(np.clip(image[384:640,384:640],0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
                    body=cv2.copyMakeBorder(body,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));text_line(body,title,(4,20),256,size=.45);panels.append(body)
                write_png(folder/f'{case["name"]}_{i}_{profile}.png',np.concatenate(panels,1))
            print(json.dumps(dict(stage='restoration',source=case['name'],crop=i)),flush=True)
    write_json(folder/'rows.json',rows)
    return rows


def run(args):
    config=load_config(args.config);opt=config['affinity_patch_probe'];out=Path(project_path(config,opt['output_dir']))
    out.mkdir(parents=True,exist_ok=False);(out/'gallery').mkdir();torch.set_num_threads(4);cv2.setNumThreads(2)
    if not torch.cuda.is_available():raise RuntimeError('Use sam2_env GPU')
    checkpoint=project_path(config,opt['checkpoint']);assert sha(checkpoint)==opt['checkpoint_sha256']
    model,bundle=load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False)
    assert model.semantic_lora is None and model.geometry_feature_adapter is None and model.geometry_highres_refiner is None
    restorer=get_restorer(config,'cuda');digest=tensor_digest(model.state_dict().items());rdigest=tensor_digest(restorer.state_dict().items())
    total_parameters=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    assert total_parameters<500000000
    cases=cases_for(config);write_json(out/'cases.json',cases);write_json(out/'config.json',config)
    source=out/'source';source.mkdir()
    sources=['tools/probe_affinity_patch.py','utils/patch_diagnostic.py','config/experiments/affinity_patch.yaml']
    for name in sources:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    started=time.time();rows=[]
    for stage in ['global',*opt['tile_sizes']]:
        write_json(out/'status.json',dict(status='running',stage=stage,training=False))
        for case in cases:
            rows.extend(full_image(case,model,restorer,config,out) if stage=='global' else tiled_image(case,stage,model,restorer,config,out))
            write_json(out/'rows.json',rows)
        if stage=='global':batch_check(cases[0],model,restorer,config,out)
    write_json(out/'status.json',dict(status='running',stage='restoration',training=False))
    reconstruction=restoration_check(cases,restorer,config,out)
    assert tensor_digest(model.state_dict().items())==digest and tensor_digest(restorer.state_dict().items())==rdigest
    report=dict(complete=True,training=False,source_count=len(cases),variants=len(rows),frozen_exact=True,
        checkpoint=dict(path=opt['checkpoint'],epoch=bundle['epoch'],sha256=sha(checkpoint)),
        d5a=config['backend_adaptation']['restoration'],config=opt,
        scope='small-cohort frozen diagnostic; GT metrics restricted to original known domain; no official score',
        total_parameters=total_parameters,
        elapsed_seconds=time.time()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
        source_sha256={name:sha(ROOT/name) for name in sources},restoration_pairs=len(reconstruction))
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='complete',training=False));print('PATCH PROBE COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config/experiments/affinity_patch.yaml');args=p.parse_args()
    try:run(args)
    except Exception as error:
        c=load_config(args.config);out=Path(project_path(c,c['affinity_patch_probe']['output_dir']))
        if out.is_dir() and not isinstance(error,FileExistsError):write_json(out/'status.json',dict(status='failed',error=repr(error),training=False))
        raise
