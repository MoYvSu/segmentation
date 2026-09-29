# -*- coding: utf-8 -*-
"""清晰参照/插值/D5a的affinity结构响应；仅使用已标训练图的可信铁素体内部。"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.rgb_spatial_blur import apply_sigma_map_blur
from models.backend_adaptation import load_backend,restore_first
from tools.probe_affinity_patch import boundary_grid,tensor_from_rgb,load_gt
from tools.render_backend_ablation import text_line,write_png
from train_affinity_connectivity import get_restorer
from train_backend_adaptation import sha,tensor_digest,write_json
from utils.config import load_config,project_path
from utils.patch_diagnostic import reconstruction_metrics


def trusted_ferrite_core(ids,trust,classes,radius):
    grid=cv2.resize(ids.astype(np.float32),(512,512),interpolation=cv2.INTER_NEAREST)
    covered=cv2.resize(trust.astype(np.float32),(512,512),interpolation=cv2.INTER_AREA)>=1-1e-6
    kernel=np.ones((2*radius+1,)*2,np.uint8)
    same=cv2.erode(grid,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)==cv2.dilate(grid,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    inside=cv2.erode(covered.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    lookup=np.zeros(int(ids.max())+1,np.bool_)
    for key,c in classes.items():
        if int(key)<len(lookup):lookup[int(key)]=c==1
    return same&inside&(grid>0)&lookup[grid.astype(np.int64)]


def condition_for(target,profile):
    value=target.copy();scale=1.
    if profile=='spatial050':
        phase=np.linspace(0,1,1024,dtype=np.float32);blend=phase**3*(phase*(6*phase-15)+10)
        sigma=np.broadcast_to(np.sqrt(.6**2+(4.5**2-.6**2)*blend),(1024,1024)).copy()
        value=apply_sigma_map_blur(value,sigma,(.4,5.2),8);scale=.5
    elif profile=='down075':scale=.75
    elif profile=='down050':scale=.5
    elif profile!='identity':raise ValueError(profile)
    if scale<1:value=cv2.resize(cv2.resize(value,(round(1024*scale),)*2,interpolation=cv2.INTER_AREA),(1024,1024),interpolation=cv2.INTER_LINEAR)
    return value


@torch.no_grad()
def run(config_path):
    config=load_config(config_path);opt=config['patch_structure'];source=Path(project_path(config,opt['source_dir']));out=Path(project_path(config,opt['output_dir']))
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(2)
    checkpoint=project_path(config,config['affinity_patch_probe']['checkpoint']);assert sha(checkpoint)==config['affinity_patch_probe']['checkpoint_sha256']
    model,_=load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False);restorer=get_restorer(config,'cuda')
    frozen=tensor_digest(model.state_dict().items());rfrozen=tensor_digest(restorer.state_dict().items())
    cases={c['name']:c for c in json.loads((source/'cases.json').read_text()) if c['kind']=='train'}
    pairs=json.loads((source/'restoration/rows.json').read_text());assert len(pairs)==32
    high=float(config['inference']['boundary_threshold']);low=float(config['inference']['marker_boundary_low_threshold'])
    started=time.time();cache={};rows=[]
    for item in pairs:
        key=(item['source'],item['crop']);case=cases[item['source']]
        if key not in cache:
            rgb=cv2.cvtColor(cv2.imread(case['path']),cv2.COLOR_BGR2RGB).astype(np.float32)/255
            gt,trust,classes=load_gt(case,config);y,x,h,w=item['box'];assert h==w==1024
            target=rgb[y:y+h,x:x+w];core=trusted_ferrite_core(gt[y:y+h,x:x+w],trust[y:y+h,x:x+w],classes,opt['interior_radius_grid'])
            tensor,_,_=tensor_from_rgb(target);reference=boundary_grid(model,tensor,config)[0,0].cpu().numpy()
            cache[key]=(target,core,reference)
        target,core,reference=cache[key];condition=condition_for(target,item['profile'])
        tensor,_,_=tensor_from_rgb(condition);restored=restore_first(restorer,tensor,item['seed']);repaired=restored[0].permute(1,2,0).cpu().numpy()
        for name,value in [('input',condition),('d5a',repaired)]:
            metrics=reconstruction_metrics(target,value)
            assert all(abs(metrics[k]-item[name][k])<1e-9 for k in metrics), 'Restoration pair not exactly replayed'
        maps={'clear':reference,'input':boundary_grid(model,tensor,config)[0,0].cpu().numpy(),'d5a':boundary_grid(model,restored,config)[0,0].cpu().numpy()}
        count=int(core.sum());assert count>0
        row=dict(source=item['source'],crop=item['crop'],profile=item['profile'],trusted_F_core_grid_pixels=count,variants={})
        for name,boundary in maps.items():
            row['variants'][name]=dict(mean=float(boundary[core].mean()),high_cells=int(((boundary>high)&core).sum()),
                low_cells=int(((boundary>low)&core).sum()),extra_high_vs_clear=int(((boundary>high)&(reference<=high)&core).sum()),
                new_high_from_clear_low=int(((boundary>high)&(reference<low)&core).sum()),
                mae_to_clear=float(np.abs(boundary[core]-reference[core]).mean()))
        rows.append(row)
        # 与前次缩略图取同一中心256，RGB和固定0..1概率并列，所有样本保留。
        top=[];bottom=[]
        for label,image in [('Clear',target),('Input',condition),('D5a',repaired)]:
            body=cv2.cvtColor(np.rint(np.clip(image[384:640,384:640],0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
            body=cv2.copyMakeBorder(body,26,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));text_line(body,label,(5,18),256,size=.45);top.append(body)
        for name in ['clear','input','d5a']:
            heat=cv2.applyColorMap(np.rint(np.clip(maps[name][192:320,192:320],0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
            heat=cv2.resize(heat,(256,256),interpolation=cv2.INTER_NEAREST);bottom.append(heat)
        write_png(out/f'{item["source"]}_{item["crop"]}_{item["profile"]}.png',np.concatenate([np.concatenate(top,1),np.concatenate(bottom,1)],0))
        print(json.dumps(row),flush=True)
    assert tensor_digest(model.state_dict().items())==frozen and tensor_digest(restorer.state_dict().items())==rfrozen
    summary={}
    for profile in dict.fromkeys(r['profile'] for r in rows):
        selected=[r for r in rows if r['profile']==profile];n=sum(r['trusted_F_core_grid_pixels'] for r in selected)
        entry=dict(pairs=len(selected),core_grid_pixels=n,variants={})
        for name in ['clear','input','d5a']:
            values={k:sum(r['variants'][name][k] for r in selected) for k in ['high_cells','low_cells','extra_high_vs_clear','new_high_from_clear_low']}
            values.update(high_fraction=values['high_cells']/n,mae_to_clear=float(np.mean([r['variants'][name]['mae_to_clear'] for r in selected])))
            entry['variants'][name]=values
        summary[profile]=entry
    write_json(out/'rows.json',rows);write_json(out/'report.json',dict(complete=True,pairs_replayed_exact=True,frozen_exact=True,summary=summary,
        scope='known F interiors of seen training crops; fixed clear-model reference; not final-instance or test accuracy',
        config=opt,checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),elapsed_seconds=time.time()-started))
    print('STRUCTURE COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config/experiments/patch_structure.yaml');run(p.parse_args().config)
