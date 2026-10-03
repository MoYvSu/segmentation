# -*- coding: utf-8 -*-
"""仅观察当前原图语义的粗分支与真实RGB修正；固定最终实例，默认只列计划。

一次真实前向通过只读hook捕获coarse logits和高分辨率delta；full必须精确回放
旧概率/类别。coarse-only保留原来的256→1024→原图两级插值，仅移除该次已捕获的
delta用于末端诊断，不修改权重、model.forward或上游几何，也不拟合阈值。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.probe_semantic_fixed import (changes,class_independent_pairs,copy_fixed_mask,
    file_sha,fixed_confusion,probability_details,write)
from tools.probe_semantic_native import (TRAIN,TEST,preview,validate_recipe,verify_frozen_model,verify_whole_baseline)
from tools.probe_semantic_raw_stress import build_cohort,probability_rows,summarize_condition
from utils.semantic_vote import adaptive_instance_core

FORMAT='semantic_components_probe_v1'
ROUTES=('full','coarse')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan(config):
    validate_recipe(config['semantic_native'])
    cfg=config['semantic_components']
    if (cfg.get('format')!=FORMAT or tuple(cfg.get('routes',()))!=ROUTES
            or cfg.get('correction_version')!='highres_v1' or cfg.get('correction_limit')!=2.):
        raise ValueError('Predeclared semantic contribution recipe differs')
    return dict(format=FORMAT,status='plan_only',training=False,sources=list(TRAIN+TEST),routes=list(ROUTES),
        semantic_forward_calls=9,source_selection='Fixed four labeled training and four unlabeled test sources',
        only_change='Terminal vote from captured coarse contribution versus actual full logits',
        capture='Read-only residual forward hook, coarse_logits kwargs and actual delta output',
        reconstruction='bilinear align_corners=True coarse-to-delta grid + actual delta equals actual output exactly',
        fixed='Original encoder/LoRA/head weights; reflect PNG; D5a/upstream semantic/geometry/fusion/postprocess',
        no_optimizer_steps=True,full_test_inference=False,official_submission=False,best_deployment_changed=False)


def component_grids(coarse,delta,actual):
    """重建真实SemanticChallenger运算；不允许猜测或静默替代结构。"""
    import torch
    import torch.nn.functional as F
    if (coarse.ndim!=4 or delta.ndim!=4 or actual.ndim!=4
            or coarse.shape[:2]!=(1,1) or delta.shape[:2]!=(1,1)
            or actual.shape!=delta.shape or any(t.dtype!=torch.float32 for t in (coarse,delta,actual))
            or any(not torch.isfinite(t).all() for t in (coarse,delta,actual))):
        raise RuntimeError('Expected finite FP32 batch1 single-channel current semantic components')
    coarse_up=F.interpolate(coarse,size=delta.shape[-2:],mode='bilinear',align_corners=True)
    if not torch.equal(coarse_up+delta,actual):
        raise RuntimeError('Captured coarse + actual RGB delta does not reproduce actual logit grid')
    return coarse_up


def validate_native_components(actual,coarse,delta):
    """检查原尺寸插值的线性贡献，仅对不同加法顺序的FP32舍入使用尺度相关容差。

    双线性插值有乘法与加法；I(coarse+delta)和I(coarse)+I(delta)的数学值相等，
    但float32计算顺序不同。32*eps*尺度给这些有限次运算留出保守舍入余量。
    不用于放宽实际输出网格、历史概率SHA、类别、冻结权重的精确门禁。
    """
    import torch
    tensors=(actual,coarse,delta)
    if (any(t.ndim!=4 or t.shape[:2]!=(1,1) or t.dtype!=torch.float32 for t in tensors)
            or any(t.shape!=actual.shape for t in tensors)
            or any(not torch.isfinite(t).all() for t in tensors)):
        raise RuntimeError('Matching finite native FP32 component grids required')
    maximums=[float(t.abs().max()) for t in tensors]
    scale=max(1.,maximums[1]+maximums[2]); epsilon=float(np.finfo(np.float32).eps)
    tolerance=32*epsilon*scale
    error=float((actual-(coarse+delta)).abs().max())
    if error>tolerance:
        raise RuntimeError('Native interpolation component linearity gate failed: '
            f'error={error:.9g}, tolerance={tolerance:.9g}, scale={scale:.9g}')
    return dict(native_linear_reconstruction_max_error=error,native_linear_relative_error=error/scale,
        native_linear_tolerance=tolerance,fp32_epsilon=epsilon,tolerance_operation_factor=32,
        tolerance_scale=scale,actual_max_abs=maximums[0],coarse_max_abs=maximums[1],delta_max_abs=maximums[2])


def logit_statistics(tensor):
    """记录实际logit幅度与分布，解释饱和概率；不选择或拟合新的分类阈值。"""
    values=tensor.detach().cpu().numpy()
    if values.dtype!=np.float32 or not values.size or not np.isfinite(values).all():
        raise RuntimeError('Finite FP32 logits required for distribution receipt')
    absolute=np.abs(values)
    return dict(shape=list(values.shape),minimum=float(values.min()),maximum=float(values.max()),
        mean=float(values.mean(dtype=np.float32)),absolute_mean=float(absolute.mean(dtype=np.float32)),
        quantiles={str(q):float(np.quantile(values,q)) for q in (.1,.5,.9,.99)},
        absolute_quantiles={str(q):float(np.quantile(absolute,q)) for q in (.1,.5,.9,.99)},
        abs_ge10_fraction=float((absolute>=10).mean()),abs_ge20_fraction=float((absolute>=20).mean()))


def delta_statistics(delta,limit):
    delta=np.asarray(delta)
    if delta.ndim!=2 or delta.dtype!=np.float32 or not np.isfinite(delta).all() or limit<=0:
        raise ValueError('Finite native float32 delta and positive bound required')
    if float(np.max(np.abs(delta)))>limit+1e-5:
        raise RuntimeError('Actual correction exceeds recorded checkpoint bound')
    return dict(minimum=float(delta.min()),maximum=float(delta.max()),mean=float(delta.mean(dtype=np.float32)),
        absolute_mean=float(np.abs(delta).mean(dtype=np.float32)),positive_fraction=float((delta>0).mean()),
        negative_fraction=float((delta<0).mean()),near_bound_fraction=float((np.abs(delta)>=.95*limit).mean()),
        quantiles={str(q):float(np.quantile(delta,q)) for q in (.01,.1,.5,.9,.99)},limit=float(limit))


def instance_deltas(ids,delta,coarse_details,full_details):
    rows={}
    for key,value in full_details.items():
        ys,xs=np.nonzero(ids==int(key)); region=np.s_[ys.min():ys.max()+1,xs.min():xs.max()+1]
        mask=ids[region]==int(key); core,_=adaptive_instance_core(mask,fraction=.4,min_pixels=8)
        values=delta[region][mask]; core_values=delta[region][core]
        rows[key]=dict(area=value['area'],mean_delta=float(values.mean(dtype=np.float32)),
            mean_delta_core=float(core_values.mean(dtype=np.float32)),
            delta_quantiles={str(q):float(np.quantile(values,q)) for q in (.1,.5,.9)},
            positive_fraction=float((values>0).mean()),negative_fraction=float((values<0).mean()),
            coarse_probability=coarse_details[key]['probability_all'],full_probability=value['probability_all'],
            probability_shift=value['probability_all']-coarse_details[key]['probability_all'],
            coarse_class=int(coarse_details[key]['probability_all']>.5),full_class=int(value['probability_all']>.5),
            flipped=(coarse_details[key]['probability_all']>.5)!=(value['probability_all']>.5),
            large=value['area']>=10000)
    return rows


def run(args):
    import torch
    import yaml
    from data.backend_adaptation import CanonicalBackendDataset
    from models.backend_adaptation import load_backend
    from tools.probe_semantic_reflect import ferrite_composition,load_fixed,protected_sources,validate_prediction
    from tools.run_marker_reflect import precision_record
    from tools.semantic_crossover import revote_instances
    from train_backend_adaptation import tensor_digest
    from utils.affinity_deployment import crop_letterbox_output,prepare_image
    from utils.config import load_config,project_path

    config=load_config(args.config); selection=plan(config)
    base=config['semantic_native']; cfg=config['semantic_components']
    if not torch.cuda.is_available(): raise RuntimeError('Use server sam2_env CUDA')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    resolve=lambda path:Path(project_path(config,path)).resolve()
    out=resolve(args.output or cfg['output_dir'])
    if out.exists() or not out.is_relative_to((ROOT/'outputs').resolve()): raise RuntimeError('Fresh ignored outputs directory required')
    out.mkdir(parents=True); (out/'gallery').mkdir()
    protected={}; hooks=[]; write(out/'status.json',dict(status='running',training=False))
    def protect(path,expected=None):
        path=resolve(path); digest=file_sha(path)
        if expected is not None and digest!=expected: raise RuntimeError('Protected SHA differs: '+str(path))
        protected[str(path)]=digest; return path
    try:
        path=Path(args.config).resolve(); seen=set(); config_sources={}
        while True:
            if path in seen or not path.is_relative_to(ROOT): raise RuntimeError('Invalid config inheritance')
            seen.add(path); protect(path); relative=path.relative_to(ROOT)
            archived=out/'config_source'/relative; archived.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,archived)
            if file_sha(archived)!=file_sha(path): raise RuntimeError('Config snapshot differs')
            config_sources[relative.as_posix()]=file_sha(path)
            parent=(yaml.safe_load(path.read_text(encoding='utf-8')) or {}).get('_base')
            if not parent: break
            path=(path.parent/parent).resolve()
        checkpoint=protect(base['checkpoint'],base['checkpoint_sha256'])
        inputs=read(protect(base['reference_inputs'])); reference=read(protect(base['reference_report']))
        refrows={r['source']:r for r in read(protect(base['reference_rows']))}; precision=precision_record()
        if (inputs.get('format')!='semantic_reflect_inputs_v1' or not inputs.get('complete')
                or reference.get('format')!='semantic_reflect_probe_v1' or not reference.get('complete')
                or reference.get('precision')!=precision
                or reference['checkpoints']['baseline']['sha256']!=base['checkpoint_sha256']):
            raise RuntimeError('Current completed raw/reflect reference differs')
        cases={r['name']:r for r in inputs['cases']}; names=TRAIN+TEST
        if len(cases)!=len(inputs['cases']) or not set(names).issubset(cases) or not set(names).issubset(refrows): raise RuntimeError('Fixed source receipt missing/repeated')
        gtroot=resolve(config['backend_adaptation']['completed_gt_dir'])
        dataset=CanonicalBackendDataset(resolve(config['paths']['raw_data_dir']),gtroot)
        if len(dataset.samples)!=32 or not set(TRAIN).issubset(p.stem for p in dataset.samples): raise RuntimeError('Processed32 training source contract differs')
        model,bundle=load_backend(checkpoint,config,'cuda')
        if bundle['epoch']!=20 or model.semantic_lora is not None: raise RuntimeError('Expected original R1 raw semantic route')
        model.eval().requires_grad_(False); state=tensor_digest(model.state_dict().items())
        verify_frozen_model(model,state)
        head=model.semantic_decoder; residual=head.semantic_residual
        if head.__class__.__name__!='SemanticChallenger' or head.semantic_residual_version!='highres_v1' or residual is None or residual.max_logit_delta!=cfg['correction_limit']:
            raise RuntimeError('Expected original SemanticChallenger highres_v1 correction architecture')
        params=sum(p.numel() for p in model.parameters()); rp=reference['parameter_counts']['restorer']
        if params!=reference['checkpoints']['baseline']['parameters'] or params+rp>=500000000: raise RuntimeError('Parameter accounting differs')
        restoration=config['backend_adaptation']['restoration']; protect(restoration['checkpoint'],restoration['checkpoint_sha256'])
        architecture=bundle['architecture']; del bundle
        captured=[]
        def capture(module,positionals,kwargs,output):
            if module is not residual or 'coarse_logits' not in kwargs or len(positionals)!=2:
                raise RuntimeError('Actual residual hook input contract differs')
            captured.append((kwargs['coarse_logits'].detach().clone(),output.detach().clone()))
            return None
        hooks.append(residual.register_forward_hook(capture,with_kwargs=True))
        def forbidden(*_): raise RuntimeError('No affinity forward allowed')
        hooks.append(model.affinity_decoder.register_forward_pre_hook(forbidden))
        code,virtual=protected_sources(out); write(out/'selection.json',selection); write(out/'config.json',config)
        calls=0; records=[]; started=time.time(); first_probability=None; first_path=None
        def forward(image,ph,pw,shape):
            nonlocal calls
            captured.clear(); actual=model.semantic_decoder(model.encoder(image.float()),image); calls+=1
            if isinstance(actual,dict): actual=actual['semantic_logits']
            if len(captured)!=1: raise RuntimeError('Expected exactly one actual residual call')
            coarse,delta=captured[0]; coarse_up=component_grids(coarse,delta,actual)
            full_native=crop_letterbox_output(actual.float(),1024,ph,pw,shape).cpu()
            coarse_native=crop_letterbox_output(coarse_up,1024,ph,pw,shape).cpu()
            delta_native=crop_letterbox_output(delta,1024,ph,pw,shape).cpu()
            linear_receipt=validate_native_components(full_native,coarse_native,delta_native)
            probs={route:np.ascontiguousarray(grid.sigmoid()[0,0].numpy(),dtype=np.float32) for route,grid in (('full',full_native),('coarse',coarse_native))}
            native_delta=np.ascontiguousarray(delta_native[0,0].numpy(),dtype=np.float32)
            return probs,native_delta,dict(coarse_grid=list(coarse.shape),delta_grid=list(delta.shape),actual_grid=list(actual.shape),
                reconstructed_actual_grid_exact=True,**linear_receipt,input_stride=list(image.stride()),
                coarse_grid_statistics=logit_statistics(coarse),coarse_native_statistics=logit_statistics(coarse_native),
                full_native_statistics=logit_statistics(full_native))
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False), \
                patch('cv2.watershed',side_effect=RuntimeError('No watershed allowed')), \
                patch('models.backend_adaptation.restore_first',side_effect=RuntimeError('No D5a allowed')):
            for name in names:
                case=cases[name]
                if case['kind']!=('train' if name in TRAIN else 'test'): raise RuntimeError('Source kind differs')
                source=protect(case['path'],case['source_sha256']); ids,original_classes,png,labels=load_fixed(resolve(case['fixed_dir']),name)
                protect(png,case['fixed_png_sha256']); protect(labels,case['fixed_classes_sha256'])
                rgb,image,ph,pw=prepare_image(source,1024,'cuda'); validate_prediction(ids,original_classes,rgb.shape[:2])
                probabilities,native_delta,receipt=forward(image,ph,pw,rgb.shape[:2])
                verify_whole_baseline(name,probabilities['full'],refrows[name]['routes']['salign_raw']['probability_sha256'],ids,original_classes)
                gt=None; cohort=None; pairs=None; match_stats=None
                if name in TRAIN:
                    gp=protect(gtroot/(name+'_gt.npz'),case['gt_npz_sha256']); cp=protect(gtroot/(name+'_class.json'),case['gt_classes_sha256'])
                    with np.load(gp,allow_pickle=False) as blob:
                        gids=blob['instance_map'].copy(); original=blob['original_covered'].astype(bool)
                        if not np.array_equal(blob['filled'].astype(bool),(gids>0)&~original) or not np.array_equal(blob['residual_unknown'].astype(bool),gids==0): raise RuntimeError('GT partition differs')
                    classes={int(k):v for k,v in read(cp).items()}
                    if gids.shape!=ids.shape: raise RuntimeError('Native GT grid differs')
                    gt=(gids,original,classes); cohort=build_cohort(gids,original,classes,base['minimum_native_area'],base['core_fraction'],base['core_min_pixels'])
                    pairs,match_stats=class_independent_pairs(gids,original,ids,minimum_iou=.5)
                row=dict(source=name,kind=case['kind'],shape=list(ids.shape),fixed_png_sha256=case['fixed_png_sha256'],
                    full_probability_and_classes_exact=True,component_receipt=receipt,
                    delta_statistics=delta_statistics(native_delta,cfg['correction_limit']),routes={})
                details_by_route={}; payloads={}; base_details=None; base_gt=None
                for route in ROUTES:
                    probability=probabilities[route]; classes,scores=revote_instances(ids,probability); validate_prediction(ids,classes,rgb.shape[:2])
                    details=probability_details(ids,probability,scores); details_by_route[route]=details
                    if route=='full': base_details=details
                    folder=out/'predictions'/name/route; folder.mkdir(parents=True)
                    copy_fixed_mask(png,folder/png.name); write(folder/(name+'_class.json'),classes); write(folder/'instances.json',details)
                    np.savez_compressed(folder/'probability.npz',ferrite_probability=probability)
                    result=dict(probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),probability_npz_sha256=file_sha(folder/'probability.npz'),
                        copied_png_sha256=file_sha(folder/png.name),changes=changes(original_classes,classes,details,base_details),
                        ferrite_composition=ferrite_composition(classes,details))
                    large=[key for key in result['changes']['changed_ids'] if details[key]['area']>=10000]
                    result['large_changes']=dict(instances=len(large),pixels=sum(details[key]['area'] for key in large),ids=large,
                        P_to_F=sum(classes[key]==1 for key in large),F_to_P=sum(classes[key]==0 for key in large))
                    if gt is not None:
                        values=probability_rows(cohort,probability)
                        if route=='full': base_gt=values
                        result['training']=dict(objects=values,summary=summarize_condition(base_gt,values),
                            fixed_pair_confusion=fixed_confusion(pairs,gt[2],classes),fixed_class_independent_pairs=pairs,
                            fixed_class_independent_match_stats=match_stats,original_pixels=int(gt[1].sum()),
                            filled_pixels=int(((gt[0]>0)&~gt[1]).sum()),unknown_pixels=int((gt[0]==0).sum()),
                            scope='Seen incomplete/coarse processed GT original_covered only; not independent validation')
                    row['routes'][route]=result; payloads[route]={'classes':classes}
                row['instance_deltas']=instance_deltas(ids,native_delta,details_by_route['coarse'],details_by_route['full'])
                np.savez_compressed(out/'predictions'/name/'delta.npz',native_delta=native_delta)
                preview(out/'gallery'/(name+'.jpg'),rgb,ids,payloads,base['thumbnail_width'])
                if first_probability is None: first_probability=probabilities['full'].copy(); first_path=source
                records.append(row); write(out/'rows.json',records); write(out/'status.json',dict(status='running',completed=len(records),total=8,training=False))
                print('SEMANTIC_COMPONENTS',name,'coarse_changes',row['routes']['coarse']['changes']['changed_instances'],'large',row['routes']['coarse']['large_changes']['instances'],flush=True)
            # 第9次撤去只读捕获hook后直接重放，实际验证hook本身没有改变输出。
            hooks[0].remove()
            rgb,image,ph,pw=prepare_image(first_path,1024,'cuda')
            repeat_logits=model.semantic_decoder(model.encoder(image.float()),image); calls+=1
            if isinstance(repeat_logits,dict): repeat_logits=repeat_logits['semantic_logits']
            repeat=np.ascontiguousarray(crop_letterbox_output(repeat_logits.float(),1024,ph,pw,rgb.shape[:2]).cpu().sigmoid()[0,0].numpy(),dtype=np.float32)
            if not np.array_equal(repeat,first_probability): raise RuntimeError('First full route changed after removing capture hook')
        verify_frozen_model(model,state)
        if calls!=9 or precision_record()!=precision: raise RuntimeError('Forward budget or precision differs')
        if any(file_sha(Path(path))!=digest for path,digest in protected.items()): raise RuntimeError('Protected input changed')
        if any(file_sha(ROOT/path)!=digest for path,digest in code.items()): raise RuntimeError('Protected code changed')
        totals={}
        for route in ROUTES:
            train=[r['routes'][route]['training'] for r in records if r['kind']=='train']; test=[r['routes'][route] for r in records if r['kind']=='test']
            composition={key:sum(r['changes'][key] for r in test) for key in ('instances','ferrite','ferrite_pixels','changed_instances','changed_pixels','P_to_F_instances','P_to_F_pixels','F_to_P_instances','F_to_P_pixels')}
            composition['pooled_ferrite_mean_area']=composition['ferrite_pixels']/composition['ferrite'] if composition['ferrite'] else None
            totals[route]=dict(test_unlabeled_changes=composition,test_large_changes={key:sum(r['large_changes'][key] for r in test) for key in ('instances','pixels','P_to_F','F_to_P')},
                training_cohort={group:{key:sum(r['summary'][group][key] for r in train) for key in train[0]['summary'][group]} for group in ('all','large')},
                training_fixed_pair_confusion={key:sum(r['fixed_pair_confusion'][key] for r in train) for key in train[0]['fixed_pair_confusion']},
                test_ferrite_tails={tail:{key:sum(r['ferrite_composition']['tails'][tail][key] for r in test) for key in ('instances','pixels')} for tail in ('lt200','200to999','1000to9999','ge10000')})
        report=dict(format=FORMAT,complete=True,training=False,no_optimizer_steps=True,full_test_inference=False,official_submission=False,best_deployment_changed=False,
            sources=8,routes=list(ROUTES),semantic_forward_calls=9,affinity_forward_calls=0,D5a_forward_calls=0,watershed_calls=0,
            checkpoint=dict(path=str(checkpoint),sha256=base['checkpoint_sha256'],epoch=20,architecture=architecture),
            parameter_counts=dict(diagnostic_resident=params,original_D5a=rp,conservative_full_deployment=params+rp,new_parameters=0),
            full_eight_probability_and_classes_exact=True,actual_grid_reconstruction_exact=True,frozen_exact=True,first_full_repeat_exact=True,
            first_full_unhooked_repeat_exact=True,
            native_linear_roundoff_gate='32*float32_epsilon*max(1,max_abs(coarse_native)+max_abs(delta_native)); exact actual-grid/full probability SHA/class/frozen gates unchanged',
            fixed_png_bytes_exact=all(r['fixed_png_sha256']==v['copied_png_sha256'] for r in records for v in r['routes'].values()),
            frozen_state_sha256=state,precision=precision,protected_inputs=protected,protected_inputs_exact=True,
            protected_sources=code,virtual_module_paths=virtual,inherited_config_sources=config_sources,totals=totals,
            elapsed_seconds=time.time()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
            caveat='This is a terminal contribution diagnostic; fixed PNG upstream already used original D5a/semantic. Coarse-only is not a ready ablation deployment. Original covered training GT may be coarse physically and has been seen. Unlabeled flips, sharper maps and F mean-area changes do not prove correctness or explain the official score. No test GT inferred or threshold fitted.')
        write(out/'report.json',report)
        page=['<!doctype html><meta charset="utf-8"><title>Semantic contribution</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>Actual full / captured coarse contribution</h1><p>Same raw input, weights and fixed reflect PNG. Green F, red P. Tests have no GT; no class correction claimed.</p>',
            '<p><a href="report.json">Report</a> | <a href="rows.json">GT censoring, component and instance diagnostics</a></p>']
        for path in sorted((out/'gallery').glob('*.jpg')): page.append('<h2>'+html.escape(path.stem)+'</h2><img src="gallery/'+html.escape(path.name)+'">')
        (out/'index.html').write_text('\n'.join(page),encoding='utf-8'); write(out/'status.json',dict(status='complete',sources=8,semantic_forward_calls=9,training=False))
        print('SEMANTIC_COMPONENTS_COMPLETE',8,9,flush=True); return report
    except Exception as error:
        write(out/'status.json',dict(status='failed',error=repr(error),training=False)); raise
    finally:
        for hook in hooks: hook.remove()


def main():
    from utils.config import load_config
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',default='config/inference/semantic_components.yaml')
    parser.add_argument('--output'); parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    if args.run: run(args)
    else: print(json.dumps(plan(load_config(args.config)),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
