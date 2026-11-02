# -*- coding: utf-8 -*-
"""NAF候选的固定缩略图：训练GT配对与冻结后端终态分开，不做测试评分。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

from tools.boundary_restore_monitor import panels, _heat, _overlay, paired_metrics, zoom_box
from tools.boundary_restore_readout import tiles, _weights, _probability, evaluate_frozen_readout
from utils.config import load_config, project_path


def canonical_input(boundary, mode='binary', threshold=.5):
    """候选模块的显式表示适配；不能把它与NAF网络本身的作用混淆。"""
    if mode=='continuous': return boundary
    if mode=='skeleton':
        if torch.is_tensor(boundary) or np.asarray(boundary).ndim!=2 or not 0<float(threshold)<1:
            raise ValueError('Skeleton输入须先对完整原生二维图适配，禁止逐tile张量细化')
        value=_probability(boundary,'skeleton boundary')
        return skeletonize(value>=threshold).astype(np.float32)
    if mode!='binary' or not 0<float(threshold)<1:
        raise ValueError('Unknown boundary input representation')
    if torch.is_tensor(boundary): return (boundary>=threshold).to(boundary.dtype)
    return (np.asarray(boundary)>=threshold).astype(np.float32)


def predict_boundary_tiled(boundary,model,settings,*,device='cpu',enabled=True):
    """一次分块NAF前向，批量拼有符号残差；无迭代、无失败扩展回退。"""
    boundary=_probability(boundary,'boundary')
    if settings.get('iterations',1)!=1 or settings.get('revert_unconnected',False):
        raise ValueError('本候选不允许迭代或未连接扩展回退')
    if not enabled:
        return boundary,dict(bypass=True,boundary_exact=True,reason='module_disabled')
    if settings.get('input_mode')=='skeleton':
        adapted=canonical_input(boundary,'skeleton',settings.get('input_threshold',.5))
        nested=dict(settings,input_mode='continuous')
        result,audit=predict_boundary_tiled(adapted,model,nested,device=device,enabled=True)
        audit.update(input_mode='skeleton',global_skeletonization_before_tiling=True,
            input_skeleton_sha256=hashlib.sha256(adapted.tobytes()).hexdigest(),
            absolute_change_mean=float(np.abs(result-boundary).mean()))
        return result,audit
    size=int(settings.get('tile_size',256)); overlap=int(settings.get('overlap',64))
    pad=int(settings.get('reflect_pad',32)); batch_size=int(settings.get('batch_size',16))
    if pad<0 or batch_size<1: raise ValueError('Invalid tiling configuration')
    padded=np.pad(boundary,pad,mode='reflect') if pad else boundary
    boxes=tiles(padded.shape,size,overlap)
    total=np.zeros(padded.shape,np.float32); weights=np.zeros_like(total)
    before=model.training; model.eval(); target=torch.device(device)
    try:
        with torch.inference_mode(),torch.autocast(device_type=target.type,enabled=False):
            for start in range(0,len(boxes),batch_size):
                chunk=boxes[start:start+batch_size]; inputs=[]
                for y,x,y1,x1 in chunk:
                    a=padded[y:y1,x:x1]; h,w=a.shape
                    if (h,w)!=(size,size):
                        a=np.pad(a,((0,size-h),(0,size-w)),mode='reflect')
                    inputs.append(a)
                tensor=torch.from_numpy(np.stack(inputs)[:,None].copy()).to(target)
                adapted=canonical_input(tensor,settings.get('input_mode','binary'),settings.get('input_threshold',.5))
                prediction=model(adapted)
                if (not torch.is_tensor(prediction) or not prediction.is_floating_point()
                        or prediction.shape!=tensor.shape or not torch.isfinite(prediction).all()
                        or ((prediction<0)|(prediction>1)).any()):
                    raise RuntimeError('NAF输出必须同尺寸、有限且位于[0,1]')
                residual=(prediction-tensor).float().cpu().numpy()[:,0]
                for delta,(y,x,y1,x1) in zip(residual,chunk):
                    h,w=y1-y,x1-x; weight=_weights(h,w)
                    total[y:y1,x:x1]+=delta[:h,:w]*weight; weights[y:y1,x:x1]+=weight
    finally: model.train(before)
    if not np.all(weights>0): raise RuntimeError('NAF拼接遗漏原图像素')
    residual=total/weights
    if pad: residual=residual[pad:-pad,pad:-pad]
    result=np.clip(boundary+residual,0,1).astype(np.float32)
    return result,dict(bypass=False,tiles=len(boxes),tile_size=size,overlap=overlap,
        batch_size=batch_size,reflect_pad=pad,input_mode=settings.get('input_mode','binary'),
        input_threshold=settings.get('input_threshold',.5),iterations=1,revert_unconnected=False,
        output_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        absolute_change_mean=float(np.abs(result-boundary).mean()))


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixed_controls(config,settings):
    selected=set(settings.get('source_names',[])); results=[]
    for name in settings.get('control_manifests',[]):
        path=Path(project_path(config,name)); manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('complete') is not True or _sha(manifest['config'])!=manifest['config_sha256']:
            raise ValueError('原部署控制清单或配置未通过身份检查')
        frozen=load_config(manifest['config'])
        for row in manifest['files']:
            if selected and row['source'] not in selected: continue
            if _sha(row['path'])!=row['sha256'] or _sha(row['control_classes'])!=row['control_classes_sha256']:
                raise RuntimeError('固定缓存或原类别改变')
            results.append((row,frozen))
    if len({r['source'] for r,_ in results})!=len(results):
        raise ValueError('固定来源重名')
    if selected and selected!={r['source'] for r,_ in results}:
        raise ValueError('固定过程图缺少已登记来源')
    order={n:i for i,n in enumerate(settings.get('source_names',[]))}
    return sorted(results,key=lambda item:order.get(item[0]['source'],0))


def _rgb_view(dataset,sample):
    row=next(r for r in dataset.records if r['source']==sample['source'])
    image=cv2.imdecode(np.fromfile(row['image'],np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('过程图原RGB无法读取')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    y,x,h,w=sample['box']; image=image[y:y+h,x:x+w]
    geometry=sample['meta']['geometry']
    image=np.rot90(image,int(geometry['rotation']))
    if geometry['flip']: image=image[:,::-1]
    h,w=image.shape[:2]; side=sample['input'].shape[-1]
    if h!=side or w!=side: image=np.pad(image,((0,side-h),(0,side-w),(0,0)),mode='reflect')
    return np.ascontiguousarray(image)


def real_pair_metrics(source,target,prediction,valid,original,*,support_kind='original_covered'):
    """只对记录目标及明确来源的支持域作诊断，不参与训练loss。"""
    if support_kind not in ('original_covered','teacher_known'):
        raise ValueError('Unknown diagnostic support provenance')
    source=np.asarray(source)>0.5;target=np.asarray(target)>0.5
    valid=np.asarray(valid,bool);original=np.asarray(original,bool)
    if not (source.shape==target.shape==prediction.shape==valid.shape==original.shape):
        raise ValueError('Real pair diagnostic shapes differ')
    def distance(mask):
        return (cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                if mask.any() else np.full(mask.shape,np.inf,np.float32))
    supported=target&valid&(distance(source)<=4)
    missing=target&valid&(distance(source)>4)
    wrong_input=source&valid&original&(distance(target)>8)
    near2=cv2.dilate(prediction,np.ones((5,5),np.uint8))
    near1=cv2.dilate(prediction,np.ones((3,3),np.uint8))
    result={}
    deep_name='teacher_known_deep_input' if support_kind=='teacher_known' else 'original_deep_input'
    for name,mask,values in [('supported_gt',supported,near2),('natural_missing_gt',missing,near2),
                             (deep_name,wrong_input,near1)]:
        result[name+'_pixels']=int(mask.sum())
        result[name+'_probability']=float(values[mask].mean()) if mask.any() else None
        result[name+'_above065']=float((values[mask]>.65).mean()) if mask.any() else None
    return result,wrong_input


def diagnostic_support_kind(dataset,sample):
    """教师支持域不能冒充人工原标；兼容旧monitor的轻量Dataset替身。"""
    row=next((r for r in getattr(dataset,'records',[]) if r['source']==sample['source']),{})
    return 'teacher_known' if row.get('source_kind')=='model_teacher' else 'original_covered'


def original_view(dataset,sample):
    """来源只用于诊断；teacher读已知域，r8仍读原标域，不改GT文件。"""
    from data.boundary_naf import _crop_pair
    row=next(r for r in dataset.records if r['source']==sample['source'])
    if diagnostic_support_kind(dataset,sample)=='teacher_known':
        from data.teacher_supervision import load_teacher_instance_source
        _,_,original,_,_=load_teacher_instance_source(row['gt'],class_path=row.get('classes'))
    else:
        with np.load(row['gt'],allow_pickle=False) as arrays:original=arrays['original_covered'].astype(bool)
    g=sample['meta']['geometry']
    result,_,_=_crop_pair(original,np.ones_like(original),sample['box'],sample['input'].shape[-1],
        int(g['rotation']),bool(g['flip']),0)
    return result


def save_monitor(model,config,output,step,dataset,*,force_full=False,source_limit=None):
    cfg=config['boundary_naf']; settings=cfg['monitor']; inf=cfg['inference']
    folder=Path(output)/'monitor'/f's{int(step):06d}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        raise ValueError('过程图目录已有完整回执，不覆盖既有结果')
    device=next(model.parameters()).device; width=int(settings['thumbnail_width']); rows=[]
    baseline_name=settings.get('baseline_name','N0')
    real_input=cfg.get('input_source')=='real_n1'
    for index in settings['pair_indices']:
        sample=dataset[int(index)]
        support_kind=diagnostic_support_kind(dataset,sample)
        teacher_target=support_kind=='teacher_known'
        source=sample['input'][0].numpy(); target=sample['target'][0].numpy()
        valid=sample['valid'][0].numpy()>0; gap=sample['gap_mask'][0].numpy()>0
        prediction=model(sample['input'][None].to(device))[0,0].detach().float().cpu().numpy()
        diagnostic={};noise=np.zeros_like(gap)
        if real_input:
            diagnostic,noise=real_pair_metrics(source,target,prediction,valid,original_view(dataset,sample),
                                              support_kind=support_kind)
        rgb=_rgb_view(dataset,sample)
        cells=[('RGB / training source',rgb),('Frozen teacher boundary target' if teacher_target else 'Complete processed GT',_heat(target)),
               ('Real N1 skeleton input' if real_input else 'Broken GT input',_heat(source)),
               ('NAF restored',_heat(prediction))]
        for _,array in cells[1:]: array[~valid]=96
        panels(folder/f'pair_{int(index):03d}.jpg',f'{sample["source"]} | step {step} | fixed draw {index}',
               cells,width,'Seen frozen teacher target; teacher-known support; gray ignored; no manual GT or independent accuracy claim.'
               if teacher_target else 'Seen training GT; gray ignored; real N1 errors are paired with unchanged GT, not synthetic gaps.'
               if real_input else 'GT source seen during training; gray is ignored; no independent accuracy claim.')
        (x,y,x1,y1),label=zoom_box(target,valid,gap,noise,side=128)
        panels(folder/f'pair_{int(index):03d}_local.jpg',f'{label} | {sample["source"]} | step {step}',
               [('RGB',rgb[y:y1,x:x1]),('Teacher target' if teacher_target else 'GT',_heat(target[y:y1,x:x1])),
                ('Input',_heat(source[y:y1,x:x1])),('Output',_heat(prediction[y:y1,x:x1]))],width)
        rows.append(dict(kind='fixed_training_pair',source=sample['source'],draw=int(index),
                         gap_interpretation='natural_GT_missing_input_support_not_artificial_gap' if real_input else 'synthetic_erased_GT',
                         noise_interpretation=('teacher_known_input_skeleton_farther_than_8px_from_frozen_teacher_target'
                             if teacher_target else 'original_covered_input_skeleton_farther_than_8px_from_recorded_GT') if real_input else 'none',
                         untouched_interpretation='other_known_region_error_not_identity_preservation' if real_input else 'uncorrupted_GT_region',
                         meta=sample['meta'],**diagnostic,**paired_metrics(source,target,prediction,valid,gap,noise)))
        if teacher_target:
            rows[-1].update(target_source_kind='model_teacher',diagnostic_support_mask='teacher_known',
                            diagnostic_support_is_manual_original=False)
    full=force_full or step in settings['full_readout_steps']
    if full:
        controls=_fixed_controls(config,settings)
        if source_limit: controls=controls[:source_limit]
        rois={}
        if settings.get('roi_spec'):
            rois=json.loads(Path(project_path(config,settings['roi_spec'])).read_text()).get('images',{})
        baseline_receipt_path=Path(output)/'fixed_baseline_check.json'
        previous=json.loads(baseline_receipt_path.read_text()) if baseline_receipt_path.exists() else {}
        baseline_checks=previous.get('sources',{})
        for row,frozen_config in controls:
            with np.load(row['path'],allow_pickle=False) as arrays:
                a={k:arrays[k].copy() for k in arrays.files}
            classes=json.loads(Path(row['control_classes']).read_text())
            name=row['source']; boundary=a['boundary']; rgb=a['rgb']
            baseline=evaluate_frozen_readout(a['semantic'],a['raw_probability'],boundary,frozen_config,
                         reference=dict(instances=a['instances'],classes=classes)) if name not in baseline_checks else None
            if baseline is not None:
                baseline_checks[name]=dict(cache_sha256=row['sha256'],reference_final_exact=True,readout=baseline[2])
            elif baseline_checks[name]['cache_sha256']!=row['sha256']:
                raise RuntimeError('原图控制缓存被替换')
            restored,infer=predict_boundary_tiled(boundary,model,inf,device=device)
            ids,new_classes,readout=evaluate_frozen_readout(a['semantic'],a['raw_probability'],restored,frozen_config)
            adapted=canonical_input(boundary,inf['input_mode'],inf['input_threshold'])
            adapter_ids,adapter_classes,adapter_readout=evaluate_frozen_readout(a['semantic'],a['raw_probability'],adapted,frozen_config)
            panels(folder/f'{name}_whole.jpg',f'{name} | step {step} | frozen geometry/semantic controls',
                [('RGB / no test GT',rgb),(baseline_name+' heatmap',_heat(boundary)),('NAF heatmap',_heat(restored)),
                 (baseline_name+' outlines',_overlay(rgb,a['instances'])),('NAF outlines',_overlay(rgb,ids))],width,
                'Only boundary module changes; no rollback, no test GT, no official accuracy claim.')
            panels(folder/f'{name}_adapter.jpg',f'{name} | representation-only control | step {step}',
                [('RGB',rgb),('Global skeleton only' if inf['input_mode']=='skeleton' else 'Binary adapter only',_heat(adapted)),('Adapter outlines',_overlay(rgb,adapter_ids)),
                 ('NAF outlines',_overlay(rgb,ids))],width,
                'Input adaptation is a separate control; same frozen final readout.')
            roi=rois.get(name,{}).get('roi')
            if roi:
                x,y,x1,y1=map(int,roi); cut=np.s_[y:y1,x:x1]
                panels(folder/f'{name}_local.jpg',f'{name} | fixed ROI | step {step}',
                    [('RGB',rgb[cut]),(baseline_name,_heat(boundary[cut])),('NAF',_heat(restored[cut])),
                     (baseline_name+' outlines',_overlay(rgb[cut],a['instances'][cut])),('NAF outlines',_overlay(rgb[cut],ids[cut]))],width)
            rows.append(dict(kind='unlabelled_fixed_monitor',source=name,no_test_gt=True,
                original_instances=int(a['instances'].max()),candidate_instances=int(ids.max()),
                adapter_only_instances=int(adapter_ids.max()),classes_changed=new_classes!=classes,
                inference=infer,readout=readout,adapter_readout=adapter_readout,
                reference_final_exact=baseline_checks[name]['reference_final_exact']))
        baseline_receipt_path.write_text(json.dumps(dict(complete=True,sources=baseline_checks),ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(format='boundary_naf_monitor_v1',global_step=int(step),rows=rows,
        full_readout=full,no_test_training=True,random_e0_is_not_identity=True,
        input_mode=inf['input_mode'],baseline_name=baseline_name,no_revert_unconnected=True,
        interpretation='training mechanism or unlabelled changes; no local blackbox score or epoch selection')
    (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result
