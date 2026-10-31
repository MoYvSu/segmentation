# -*- coding: utf-8 -*-
"""RGB条件补边固定过程图；只读训练配对与冻结N0控制，不读取测试GT。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.boundary_restore_monitor import (
    _sha, _heat, _overlay, panels, paired_metrics, zoom_box, preserved_state,
)
from tools.boundary_restore_readout import (
    _probability, _weights, _evaluation, tiles, evaluate_frozen_readout,
)
from tools.semantic_dino_monitor import prepare_monitor_folder
from utils.config import load_config, project_path


def predict_context_residual_tiled(boundary, rgb, model, *, device='cpu', tile_size=512,
                                  overlap=128, reflect_pad=32, strength=1.):
    """同旧读出拼接有符号残差；RGB与heat逐像素同步reflect/裁窗，不重采样。"""
    boundary = _probability(boundary, 'boundary')
    if (not isinstance(rgb, np.ndarray) or rgb.dtype!=np.uint8
        or rgb.shape!=(*boundary.shape, 3)):
        raise ValueError('RGB must be native uint8 HWC exactly aligned with boundary')
    if not np.isfinite(strength) or not 0<=float(strength)<=1:
        raise ValueError('strength must lie in [0,1]')
    if type(reflect_pad) is not int or reflect_pad<0:
        raise ValueError('reflect_pad must be a nonnegative integer')
    boxes = tiles((boundary.shape[0]+2*reflect_pad,boundary.shape[1]+2*reflect_pad),tile_size,overlap)
    audit = dict(strength=float(strength),tile_size=tile_size,overlap=overlap,reflect_pad=reflect_pad,
        tiles=0,bypass=False,all_zero_residual=False,context_scaling='uint8_RGB_div255',
        context_rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        boundary_input_sha256=hashlib.sha256(boundary.tobytes()).hexdigest())
    if float(strength)==0:
        audit.update(bypass=True,boundary_exact=True,reason='strength_zero')
        return boundary,audit
    if not callable(getattr(model,'residual',None)):
        raise ValueError('Context model must expose residual(heat, RGB); probabilities are not residuals')
    padded = np.pad(boundary,reflect_pad,mode='reflect') if reflect_pad else boundary
    context = np.pad(rgb,((reflect_pad,reflect_pad),(reflect_pad,reflect_pad),(0,0)),mode='reflect') if reflect_pad else rgb
    total,weight = np.zeros(padded.shape,np.float32),np.zeros(padded.shape,np.float32)
    local_zero = True
    target = torch.device(device)
    with _evaluation(model),torch.autocast(device_type=target.type,enabled=False):
        for y,x,y1,x1 in boxes:
            height,width=y1-y,x1-x
            value=padded[y:y1,x:x1]; color=context[y:y1,x:x1]
            if height!=tile_size or width!=tile_size:
                pads=((0,tile_size-height),(0,tile_size-width))
                value=np.pad(value,pads,mode='reflect')
                color=np.pad(color,(*pads,(0,0)),mode='reflect')
            heat=torch.from_numpy(np.ascontiguousarray(value))[None,None].to(target)
            rgb_tensor=torch.from_numpy(np.ascontiguousarray(color)).permute(2,0,1)[None].to(target).float().div_(255.)
            residual=model.residual(heat,rgb_tensor)
            if (not isinstance(residual,torch.Tensor) or residual.shape!=heat.shape
                or residual.dtype!=torch.float32 or not bool(torch.isfinite(residual).all())):
                raise ValueError('Context residual must be finite FP32 B1HW matching heat')
            delta=residual.detach().cpu()[0,0,:height,:width].numpy()
            local_zero=local_zero and not bool(np.any(delta))
            blend=_weights(height,width)
            total[y:y1,x:x1]+=delta*blend;weight[y:y1,x:x1]+=blend
            audit['tiles']+=1
    if not np.all(weight>0):
        raise RuntimeError('Context residual left uncovered native pixels')
    merged=total/weight
    if reflect_pad:
        merged=merged[reflect_pad:-reflect_pad,reflect_pad:-reflect_pad]
    if not np.any(merged):
        audit.update(all_zero_residual=local_zero,boundary_exact=True,reason='zero_merged_residual',
                     residual_min=0.,residual_max=0.,residual_abs_mean=0.)
        return boundary,audit
    restored=np.clip(boundary+np.float32(strength)*merged,0.,1.).astype(np.float32,copy=False)
    audit.update(boundary_exact=bool(np.array_equal(restored,boundary)),reason='residual_applied',
        residual_min=float(merged.min()),residual_max=float(merged.max()),
        residual_abs_mean=float(np.abs(merged).mean()),
        boundary_output_sha256=hashlib.sha256(restored.tobytes()).hexdigest())
    return restored,audit


def context_pair_metrics(source,target,output,valid,gap,noise):
    result=paired_metrics(source,target,output,valid,gap,noise)
    # 旧untouched保留可比；新pristine严格排除人工损伤的弱尾部，不影响训练loss。
    pristine=valid & (source==target)
    masks={'pristine':pristine,'strong_original':valid & (target>=.65)}
    if noise.any():
        distance=cv2.distanceTransform((~noise).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        masks.update(noise_pristine_ring1_8=pristine & (distance>0)&(distance<=8),
                     noise_pristine_ring8_16=pristine & (distance>8)&(distance<=16))
    for name,mask in masks.items():
        count=int(mask.sum()); result[name+'_pixels']=count
        result[name+'_after_mae']=float(np.abs(output-target)[mask].mean()) if count else None
        result[name+'_signed_change']=float((output-source)[mask].mean()) if count else None
    return result


def _control_groups(config):
    settings=config['boundary_restore'].get('monitor',{})
    if not settings.get('control_manifest'):
        raise ValueError('Fixed original three-image N0 controls are required for context monitor')
    groups=[];seen=set()
    for key in ('control_manifest','additional_control_manifest'):
        if not settings.get(key):
            continue
        path=Path(project_path(config,settings[key]));receipt=json.loads(path.read_text(encoding='utf-8'))
        if (receipt.get('complete') is not True or receipt.get('training') is not False
            or _sha(receipt['config'])!=receipt.get('config_sha256')):
            raise RuntimeError('Fixed N0 controls incomplete or decoder config changed')
        sources=[row['source'] for row in receipt['files']]
        if len(sources)!=len(set(sources)) or seen.intersection(sources):
            raise ValueError('Duplicate context monitor control sources')
        seen.update(sources)
        if key=='control_manifest':
            expected=set(settings.get('required_control_sources',['test_101','test_116','test_166']))
            if set(sources)!=expected:
                raise ValueError('Original N0 monitor must retain the exact three fixed controls')
        groups.append((key,receipt,load_config(receipt['config'])))
    return groups


def structure_pair_panels(sample, output, recipe):
    """几何来源和选路单列；彩色走廊不是精确边界真值。"""
    from models.boundary_structure_loss import select_path
    rgb=np.rint(sample['context_rgb'].permute(1,2,0).numpy()*255).astype(np.uint8)
    target=sample['target'][0].numpy()
    corridor=sample['structure_corridor'][0].numpy().astype(bool)
    free=sample['structure_free_mask'][0].numpy().astype(bool)
    cores=sample.get('structure_cores')
    cores=np.zeros_like(corridor) if cores is None else cores[0].numpy().astype(bool)
    location=rgb.copy()
    location[corridor]=np.rint(location[corridor]*.45+np.array([0,210,255])*.55).astype(np.uint8)
    location[cores]=np.rint(location[cores]*.45+np.array([245,110,40])*.55).astype(np.uint8)
    selected_before,selected_after=np.zeros_like(corridor),np.zeros_like(corridor)
    rows=[]
    for path in sample['structure_paths']:
        paths=[]
        for values,destination in ((target,selected_before),(output,selected_after)):
            chosen=select_path(values,target,path,margin=float(recipe['response_margin']),
                               change_cost=float(recipe['selection_change_cost']))
            if chosen is None:
                raise RuntimeError('Monitor received an infeasible structural path')
            yx=chosen['coords_yx'];destination[yx[:,0],yx[:,1]]=True
            paths.append(dict(points=len(yx),mean_response=float(values[yx[:,0],yx[:,1]].mean()),
                              below_margin_points=int((values[yx[:,0],yx[:,1]]<float(recipe['response_margin'])).sum())))
        rows.append(dict(meta=path.get('meta',{}),closed=path['closed'],before=paths[0],after=paths[1]))
    before,after=rgb.copy(),rgb.copy()
    before[selected_before]=(255,40,210);after[selected_after]=(255,40,210)
    cells=[('RGB / training source',rgb),('Cyan corridor / orange deep core',location),
           ('N0 response [0,1]',_heat(target)),('Candidate [0,1]',_heat(output)),
           ('N0-selected separator',before),('Candidate-selected separator',after)]
    return cells,dict(path_count=len(rows),corridor_pixels=int(corridor.sum()),free_pixels=int(free.sum()),
        deep_core_pixels=int(cores.sum()),paths=rows,
        interpretation='Fixed instance identities; corridor is location prior, selected path is not ground-truth heat')


@torch.no_grad()
def monitor(model,config,output_dir,epoch,device):
    from train_boundary_context import build_dataset,collate_samples,combined_loss,STRUCTURE_MODE
    cfg=config['boundary_restore'];settings=cfg.get('monitor',{})
    data_mode=cfg['data_mode'];structured=data_mode==STRUCTURE_MODE
    width=int(settings.get('thumbnail_width',300))
    dataset=build_dataset(config,seed_offset=917);dataset.set_epoch(1)
    groups=_control_groups(config)
    folder,backup=prepare_monitor_folder(output_dir,int(epoch));rows=[]
    roi_spec={}
    if settings.get('roi_spec'):
        roi_spec=json.loads(Path(project_path(config,settings['roi_spec'])).read_text(encoding='utf-8')).get('images',{})
    with preserved_state(model):
        for index in settings.get('pair_indices',[0,1,2,5,4,6]):
            sample=dataset[int(index)]
            source=sample['input'][0].numpy();target=sample['target'][0].numpy()
            rgb=np.rint(sample['context_rgb'].permute(1,2,0).numpy()*255).astype(np.uint8)
            valid=sample['valid'][0].numpy()>0;gap=sample['gap_mask'][0].numpy()>0;noise=sample['noise_mask'][0].numpy()>0
            with torch.autocast(device_type=torch.device(device).type,enabled=False):
                output=model(sample['input'][None].to(device),sample['context_rgb'][None].to(device))[0,0].float().cpu().numpy()
            cells=[('Aligned original RGB',rgb),('Original N0 teacher / NOT GT',_heat(target)),
                   ('Corrupted heat [0,1]',_heat(source)),('RGB-context output [0,1]',_heat(output))]
            for _,array in cells[1:]:
                array[~valid]=96
            (x,y,x1,y1),label=zoom_box(target,valid,gap,noise)
            cells += [(label+' target zoom',_heat(target[y:y1,x:x1])),
                      (label+' input zoom',_heat(source[y:y1,x:x1])),
                      (label+' output zoom',_heat(output[y:y1,x:x1]))]
            panels(folder/f'pair_{int(index):03d}.jpg',f'{sample["source"]} | aligned RGB pair {index} | e{epoch:03d}',cells,width,
                   'N0 response is NOT GT. Structural free regions may change; pristine means input==N0.' if structured else
                   'Original N0 is teacher, NOT GT. Same training pairs/loss. Pristine means exact input==target.')
            rows.append(dict(kind='paired_training_source',source=sample['source'],draw=int(index),
                identity=bool(sample['identity']),data_mode=data_mode,target_kind='frozen_N0_teacher_response',
                meta=sample['meta'],**context_pair_metrics(source,target,output,valid,gap,noise)))
            if structured:
                structure_cells,structure_stats=structure_pair_panels(sample,output,cfg['structure'])
                panels(folder/f'structure_{int(index):03d}.jpg',
                    f'{sample["source"]} | fixed instance relation {index} | e{epoch:03d}',structure_cells,width,
                    'Cyan is allowed search region, orange protected deep core. Pink selected path is NOT GT. No test supervision.')
                structure_batch=collate_samples([sample])
                _,loss_stats=combined_loss(torch.from_numpy(output.copy())[None,None],structure_batch,cfg)
                rows[-1].update(structure=structure_stats,loss_diagnostics=loss_stats,
                    supervision='N0 reconstruction outside free region; trusted instance separation within legal corridor')
        for group,receipt,frozen_config in groups:
            for row in receipt['files']:
                path=Path(row['path']);class_path=Path(row['control_classes']);png_path=Path(row['control_png'])
                for asset,digest in ((path,row['sha256']),(class_path,row['control_classes_sha256']),
                                     (png_path,row['control_png_sha256'])):
                    if _sha(asset)!=digest:
                        raise RuntimeError('Fixed control asset changed: '+str(asset))
                old_classes=json.loads(class_path.read_text(encoding='utf-8'))
                with np.load(path,allow_pickle=False) as arrays:
                    values={key:arrays[key].copy() for key in arrays.files}
                boundary,rgb,old_ids=values['boundary'],values['rgb'],values['instances']
                old_png=cv2.imdecode(np.frombuffer(png_path.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
                if old_png is None or old_png.dtype!=np.uint16 or not np.array_equal(old_png,old_ids):
                    raise RuntimeError('Fixed PNG and cached instances disagree')
                restored,inference=predict_context_residual_tiled(boundary,rgb,model,device=device,
                    tile_size=int(cfg['crop_size']),overlap=int(settings.get('tile_overlap',128)),strength=1.)
                ids,classes,readout=evaluate_frozen_readout(values['semantic'],values['raw_probability'],restored,frozen_config)
                boundary_exact=np.array_equal(boundary,restored)
                ids_exact=np.array_equal(old_png,ids);classes_exact=old_classes==classes
                if epoch==0 and not (boundary_exact and ids_exact and classes_exact):
                    raise RuntimeError('Zero-initialized RGB completion changed final N0 boundary/PNG/classes')
                stem=row['source']
                panels(folder/(stem+'_whole.jpg'),f'{stem} | e{epoch:03d} | raw RGB context / frozen N0 readout',
                    [('RGB / no test GT',rgb),('N0 boundary [0,1]',_heat(boundary)),('Completed boundary [0,1]',_heat(restored)),
                     ('N0 outlines',_overlay(rgb,old_ids)),('Candidate outlines',_overlay(rgb,ids))],width,
                    'No test GT, no loss or epoch selection. Changed instance count is not accuracy.')
                roi=roi_spec.get(stem,{}).get('roi')
                if roi:
                    x,y,x1,y1=roi;cut=np.s_[y:y1,x:x1]
                    panels(folder/(stem+'_local.jpg'),f'{stem} | fixed ROI | e{epoch:03d}',
                        [('RGB',rgb[cut]),('N0',_heat(boundary[cut])),('RGB-context',_heat(restored[cut])),
                         ('N0 outlines',_overlay(rgb[cut],old_ids[cut])),('Candidate outlines',_overlay(rgb[cut],ids[cut]))],width)
                rows.append(dict(kind='unlabelled_test_monitor',source=stem,control_group=group,no_test_gt=True,
                    output_change_mae=float(np.abs(restored-boundary).mean()),original_instances=int(old_ids.max()),
                    candidate_instances=int(ids.max()),boundary_exact=bool(boundary_exact),
                    png_pixels_exact=bool(ids_exact),classes_exact=bool(classes_exact),
                    identity_epoch_exact=bool(boundary_exact and ids_exact and classes_exact),
                    inference=inference,readout=readout))
    result=dict(format='boundary_context_monitor_v1',data_mode=data_mode,epoch=int(epoch),rows=rows,
        interrupted_backup=str(backup) if backup else None,
        status='mechanism_diagnostics_only; no accuracy/official score; no epoch selection')
    (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result
