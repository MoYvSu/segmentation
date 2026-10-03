# -*- coding: utf-8 -*-
"""固定 reflect 实例，检查当前 raw/S-align 语义的原生尺度与上下文敏感性。

默认只列计划；--run 在 sam2_env CUDA 执行预声明的 4 张训练图和 4 张测试图。
全图1024必须重现既有概率SHA与类别，局部1024/512均只改变原图裁窗视野。
几何、权重及投票阈值不变；该诊断不训练、不全量推理、不生成测试GT。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.probe_semantic_fixed import (changes,class_independent_pairs,copy_fixed_mask,
    file_sha,fixed_confusion,probability_details,write)
from tools.probe_semantic_raw_stress import (build_cohort,probability_rows,summarize_condition)
from utils.semantic_native import NativeLogitBlend,prepare_native_tile,tile_boxes

FORMAT = 'semantic_native_probe_v1'
TRAIN = ('train_001','train_173','train_558','train_889')
TEST = ('test_101','test_084','test_116','test_146')
ROUTES = ('whole','native1024','native512')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_recipe(cfg):
    if (cfg.get('format')!=FORMAT or tuple(cfg.get('train_sources',()))!=TRAIN
            or tuple(cfg.get('test_sources',()))!=TEST or tuple(cfg.get('routes',()))!=ROUTES
            or cfg.get('network_size')!=1024 or cfg.get('native_sizes')!=[1024,512]
            or cfg.get('overlap')!=.25 or cfg.get('minimum_axis_weight')!=.001
            or cfg.get('minimum_native_area')!=10000 or cfg.get('expected_manual_sources')!=32
            or cfg.get('core_fraction')!=.4 or cfg.get('core_min_pixels')!=8
            or cfg.get('checkpoint_epoch')!=20):
        raise ValueError('Predeclared source/route/scale recipe differs')


def plan(config):
    cfg = config['semantic_native']
    validate_recipe(cfg)
    return dict(format=FORMAT,status='plan_only',training=False,no_optimizer_steps=True,
        sources=list(TRAIN+TEST),routes=list(ROUTES),source_selection='Predeclared before outputs',
        only_change='Terminal raw semantic native context and texture scale',
        fixed='Reflect PNG bytes, encoder/LoRA/head weights, D5a/upstream semantic/geometry/fusion/postprocess',
        native_sizes=cfg['native_sizes'],network_size=1024,overlap=.25,
        blend='Original RGB crop; reflect letterbox; native logits positive-tent float32 blend; CPU sigmoid',
        full_baseline_gate='All eight clean probability SHA and fixed classes exact',
        formal_submission=False,full_test_inference=False)


def verify_whole_baseline(name,probability,expected_probability_sha,ids,expected_classes):
    """旧入口必须原尺寸float32概率和原分类都精确重现，才允许解释新视野。"""
    from tools.semantic_crossover import revote_instances
    probability=np.asarray(probability)
    if probability.dtype!=np.float32 or probability.shape!=ids.shape or not np.isfinite(probability).all():
        raise RuntimeError('Whole native float32 probability contract differs: '+name)
    if hashlib.sha256(probability.tobytes()).hexdigest()!=expected_probability_sha:
        raise RuntimeError('Whole raw probability SHA failed exact replay: '+name)
    classes,scores=revote_instances(ids,probability)
    if classes!=expected_classes:
        raise RuntimeError('Whole fixed classes failed exact replay: '+name)
    return classes,scores


def verify_frozen_model(model,expected_state):
    """参数、BN等buffer及eval/无梯度状态均需保持，train开关不在state_dict中。"""
    from train_backend_adaptation import tensor_digest
    if tensor_digest(model.state_dict().items())!=expected_state:
        raise RuntimeError('Frozen model parameters or buffers changed')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()) or any(module.training for module in model.modules()):
        raise RuntimeError('Frozen no-gradient/eval-only state changed')


def preview(path,rgb,ids,route_payloads,width=320):
    """所有结果使用同一实例；测试展示仅为预测，不暗示真值。"""
    panels=[]
    labels=[('Raw RGB',rgb)]
    for route,payload in route_payloads.items():
        table=np.zeros(int(ids.max())+1,dtype=np.uint8)
        for key,value in payload['classes'].items():
            table[int(key)]=value
        colored=np.zeros((*ids.shape,3),dtype=np.uint8)
        colored[ids>0]=np.where(table[ids[ids>0]][:,None]==1,
            np.array([74,190,104],np.uint8),np.array([185,74,80],np.uint8))
        image=np.rint(rgb.astype(np.float32)*.55+colored.astype(np.float32)*.45).astype(np.uint8)
        labels.append((route+' (F green / P red)',image))
    for label,image in labels:
        image=cv2.resize(image,(width,round(image.shape[0]*width/image.shape[1])),interpolation=cv2.INTER_AREA)
        image=cv2.copyMakeBorder(image,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(image,label,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1,cv2.LINE_AA)
        panels.append(image)
    if not cv2.imwrite(str(path),cv2.cvtColor(np.concatenate(panels,axis=1),cv2.COLOR_RGB2BGR)):
        raise IOError('Could not write private preview')


def run(args):
    import torch
    import yaml
    from data.backend_adaptation import CanonicalBackendDataset
    from models.backend_adaptation import load_backend
    from tools.probe_semantic_reflect import (ferrite_composition,load_fixed,protected_sources,validate_prediction)
    from tools.run_marker_reflect import precision_record
    from tools.semantic_crossover import revote_instances
    from train_backend_adaptation import tensor_digest
    from utils.affinity_deployment import crop_letterbox_output,prepare_image
    from utils.config import load_config,project_path

    config=load_config(args.config)
    cfg=config['semantic_native']
    validate_recipe(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the server sam2_env CUDA environment')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    resolve=lambda path:Path(project_path(config,path)).resolve()
    out=resolve(args.output or cfg['output_dir'])
    if out.exists() or not out.is_relative_to((ROOT/'outputs').resolve()):
        raise RuntimeError('Fresh ignored outputs directory required')
    out.mkdir(parents=True); (out/'gallery').mkdir()
    write(out/'status.json',dict(status='running',training=False))
    protected={}; hooks=[]
    def protect(path,expected=None):
        path=resolve(path); digest=file_sha(path)
        if expected is not None and digest!=expected:
            raise RuntimeError('Protected SHA differs: '+str(path))
        protected[str(path)]=digest
        return path
    try:
        path=Path(args.config).resolve(); visited=set()
        while True:
            if path in visited:
                raise RuntimeError('Cyclic config inheritance')
            visited.add(path); protect(path)
            base=(yaml.safe_load(path.read_text(encoding='utf-8')) or {}).get('_base')
            if not base: break
            path=(path.parent/base).resolve()
        checkpoint=protect(cfg['checkpoint'],cfg['checkpoint_sha256'])
        receipt=read(protect(cfg['reference_inputs']))
        reference=read(protect(cfg['reference_report']))
        reference_rows={row['source']:row for row in read(protect(cfg['reference_rows']))}
        precision=precision_record()
        if (receipt.get('format')!='semantic_reflect_inputs_v1' or not receipt.get('complete')
                or not reference.get('complete') or reference.get('format')!='semantic_reflect_probe_v1'
                or reference.get('precision')!=precision
                or reference['checkpoints']['baseline']['sha256']!=cfg['checkpoint_sha256']):
            raise RuntimeError('Completed current raw/reflect reference or precision differs')
        cases={row['name']:row for row in receipt['cases']}
        names=TRAIN+TEST
        if len(cases)!=len(receipt['cases']) or not set(names).issubset(cases) or not set(names).issubset(reference_rows):
            raise RuntimeError('Predeclared fixed source receipts missing or repeated')
        dataset=CanonicalBackendDataset(resolve(config['paths']['raw_data_dir']),resolve(config['backend_adaptation']['completed_gt_dir']))
        manual={p.stem:p for p in dataset.samples}
        if len(manual)!=32 or not set(TRAIN).issubset(manual):
            raise RuntimeError('Expected full processed 32-source GT manifest')
        gtroot=resolve(config['backend_adaptation']['completed_gt_dir'])
        model,bundle=load_backend(checkpoint,config,'cuda')
        if bundle['epoch']!=20 or model.semantic_lora is not None:
            raise RuntimeError('Expected original R1 S-align/raw semantic route')
        model.eval().requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Diagnostic must be fully frozen')
        state=tensor_digest(model.state_dict().items())
        parameters=sum(p.numel() for p in model.parameters())
        if parameters!=reference['checkpoints']['baseline']['parameters']:
            raise RuntimeError('Current original backend parameter count differs')
        restorer_parameters=reference['parameter_counts']['restorer']
        restoration=config['backend_adaptation']['restoration']
        protect(restoration['checkpoint'],restoration['checkpoint_sha256'])
        deployed_parameters=parameters+restorer_parameters
        if deployed_parameters>=500000000:
            raise RuntimeError('Model parameter limit exceeded')
        architecture=bundle['architecture']; del bundle
        def forbidden(*_):
            raise RuntimeError('No affinity forward allowed in semantic native diagnostic')
        hooks.append(model.affinity_decoder.register_forward_pre_hook(forbidden))
        code,virtual=protected_sources(out)
        write(out/'config.json',config); write(out/'selection.json',plan(config))
        records=[]; calls=0; expected_calls=1; started=time.time()
        first_probability=None; first_path=None
        def forward_logits(image,ph,pw,shape):
            nonlocal calls
            logits=model.semantic_decoder(model.encoder(image.float()),image)
            calls+=1
            if isinstance(logits,dict): logits=logits['semantic_logits']
            # 原FPN也输出边界通道；现行末端语义只读取channel0，保持同样的resize顺序。
            if logits.ndim!=4 or logits.shape[0]!=1 or logits.shape[1]<1:
                raise RuntimeError('Expected batch1 and ferrite logits in channel0')
            return crop_letterbox_output(logits.float(),1024,ph,pw,shape).cpu()
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False), \
                patch('cv2.watershed',side_effect=RuntimeError('No watershed allowed')), \
                patch('models.backend_adaptation.restore_first',side_effect=RuntimeError('No D5a allowed')):
            # 所有旧入口先过SHA和类别门禁，再执行任一新视野；不用失败半途的候选作结论。
            whole_cache={}
            for name in names:
                case=cases[name]
                source=protect(case['path'],case['source_sha256'])
                fixed_ids,fixed_classes,png,labels=load_fixed(resolve(case['fixed_dir']),name)
                protect(png,case['fixed_png_sha256']); protect(labels,case['fixed_classes_sha256'])
                rgb,image,ph,pw=prepare_image(source,1024,'cuda')
                validate_prediction(fixed_ids,fixed_classes,rgb.shape[:2])
                probability=np.ascontiguousarray(forward_logits(image,ph,pw,rgb.shape[:2]).sigmoid()[0,0].numpy(),dtype=np.float32)
                expected_calls+=1
                verify_whole_baseline(name,probability,reference_rows[name]['routes']['salign_raw']['probability_sha256'],fixed_ids,fixed_classes)
                whole_cache[name]=probability
                print('SEMANTIC_NATIVE_BASELINE_EXACT',name,flush=True)
            write(out/'baseline_gate.json',dict(complete=True,sources=list(names),probability_and_classes_exact=True,
                note='All eight whole routes passed before any native crop forward'))
            for name in names:
                case=cases[name]
                if case['kind']!=('train' if name in TRAIN else 'test'):
                    raise RuntimeError('Source kind differs')
                source=protect(case['path'],case['source_sha256'])
                ids,original_classes,png,labels=load_fixed(resolve(case['fixed_dir']),name)
                protect(png,case['fixed_png_sha256']); protect(labels,case['fixed_classes_sha256'])
                rgb,image,ph,pw=prepare_image(source,1024,'cuda')
                validate_prediction(ids,original_classes,rgb.shape[:2])
                clean=whole_cache.pop(name)
                clean_sha=hashlib.sha256(clean.tobytes()).hexdigest()
                if clean_sha!=reference_rows[name]['routes']['salign_raw']['probability_sha256']:
                    raise RuntimeError('Whole raw probability SHA failed exact replay: '+name)
                gt=None; cohort=None; pairs=None; match_stats=None
                if name in TRAIN:
                    gtpath=protect(gtroot/(name+'_gt.npz'),case['gt_npz_sha256'])
                    classpath=protect(gtroot/(name+'_class.json'),case['gt_classes_sha256'])
                    with np.load(gtpath,allow_pickle=False) as blob:
                        gtids=blob['instance_map'].copy(); original=blob['original_covered'].astype(bool)
                        if not np.array_equal(blob['filled'].astype(bool),(gtids>0)&~original) or not np.array_equal(blob['residual_unknown'].astype(bool),gtids==0):
                            raise RuntimeError('GT original/fill/unknown partition differs')
                    gtclasses={int(k):v for k,v in read(classpath).items()}
                    if gtids.shape!=ids.shape: raise RuntimeError('Native GT shape differs')
                    gt=(gtids,original,gtclasses)
                    cohort=build_cohort(gtids,original,gtclasses,cfg['minimum_native_area'],cfg['core_fraction'],cfg['core_min_pixels'])
                    pairs,match_stats=class_independent_pairs(gtids,original,ids,minimum_iou=.5)
                row=dict(source=name,kind=case['kind'],shape=list(ids.shape),source_sha256=case['source_sha256'],
                    fixed_png_sha256=case['fixed_png_sha256'],baseline_probability_and_classes_exact=False,routes={})
                base_details=None; base_gt_rows=None; payloads={}
                for route in ROUTES:
                    if route=='whole':
                        probability=clean
                        tile_receipt=dict(tiles=1,native_shape=list(ids.shape),input_stride=list(image.stride()),
                            arithmetic='Actual deployed whole1024 CUDA native logit resize, CPU sigmoid',
                            resize_scale=1024/max(ids.shape),pad_h=ph,pad_w=pw)
                    else:
                        size=int(route.removeprefix('native')); boxes=tile_boxes(ids.shape,size,cfg['overlap'])
                        blend=NativeLogitBlend(ids.shape,cfg['minimum_axis_weight']); views=[]
                        expected_calls+=len(boxes)
                        for box in boxes:
                            tile,tph,tpw,view=prepare_native_tile(rgb,box,1024,'cuda')
                            native=forward_logits(tile,tph,tpw,(box[1]-box[0],box[3]-box[2]))[0,0].numpy()
                            blend.add(box,np.ascontiguousarray(native,dtype=np.float32)); views.append(view)
                        stitched=blend.finish()
                        probability=np.ascontiguousarray(torch.from_numpy(stitched).sigmoid().numpy(),dtype=np.float32)
                        tile_receipt={**blend.receipt(), 'native_size':size,'overlap':cfg['overlap'],'views':views}
                    classes,scores=revote_instances(ids,probability)
                    validate_prediction(ids,classes,rgb.shape[:2])
                    details=probability_details(ids,probability,scores)
                    if route=='whole':
                        if classes!=original_classes: raise RuntimeError('Whole fixed classes failed exact replay: '+name)
                        row['baseline_probability_and_classes_exact']=True; base_details=details
                    folder=out/'predictions'/name/route; folder.mkdir(parents=True)
                    copy_fixed_mask(png,folder/png.name); write(folder/(name+'_class.json'),classes)
                    np.savez_compressed(folder/'probability.npz',ferrite_probability=probability)
                    write(folder/'instances.json',details)
                    result=dict(probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                        probability_npz_sha256=file_sha(folder/'probability.npz'),copied_png_sha256=file_sha(folder/png.name),
                        changes=changes(original_classes,classes,details,base_details),
                        ferrite_composition=ferrite_composition(classes,details),tile_receipt=tile_receipt)
                    large_changed=[key for key in result['changes']['changed_ids'] if details[key]['area']>=cfg['minimum_native_area']]
                    result['large_changes']=dict(instances=len(large_changed),pixels=sum(details[key]['area'] for key in large_changed),
                        ids=large_changed,P_to_F=sum(classes[key]==1 for key in large_changed),F_to_P=sum(classes[key]==0 for key in large_changed))
                    if gt is not None:
                        values=probability_rows(cohort,probability)
                        if route=='whole': base_gt_rows=values
                        result['training']=dict(objects=values,summary=summarize_condition(base_gt_rows,values),
                            fixed_pair_confusion=fixed_confusion(pairs,gt[2],classes),fixed_class_independent_pairs=pairs,
                            fixed_class_independent_match_stats=match_stats,
                            original_pixels=int(gt[1].sum()),unknown_pixels=int((gt[0]==0).sum()),
                            filled_pixels=int(((gt[0]>0)&~gt[1]).sum()),
                            scope='Seen processed training GT; original_covered only; not independent validation or complete physical GT')
                    row['routes'][route]=result; payloads[route]={'classes':classes}
                    write(out/'rows.json',records+[row])
                    print('SEMANTIC_NATIVE_ROUTE',name,route,'tiles',tile_receipt['tiles'],'changes',result['changes']['changed_instances'],flush=True)
                preview(out/'gallery'/(name+'.jpg'),rgb,ids,payloads,cfg['thumbnail_width'])
                if first_probability is None: first_probability=clean.copy(); first_path=source
                records.append(row); write(out/'rows.json',records)
                write(out/'status.json',dict(status='running',completed=len(records),total=len(names),training=False))
            rgb,image,ph,pw=prepare_image(first_path,1024,'cuda')
            repeat=np.ascontiguousarray(forward_logits(image,ph,pw,rgb.shape[:2]).sigmoid()[0,0].numpy(),dtype=np.float32)
            if not np.array_equal(repeat,first_probability): raise RuntimeError('First whole probability changed after local forwards')
        if calls!=expected_calls: raise RuntimeError('Unexpected forward budget')
        verify_frozen_model(model,state)
        if precision_record()!=precision: raise RuntimeError('CUDA precision changed')
        if any(file_sha(Path(path))!=digest for path,digest in protected.items()): raise RuntimeError('Protected input changed')
        if any(file_sha(ROOT/path)!=digest for path,digest in code.items()): raise RuntimeError('Protected source changed')
        totals={}
        for route in ROUTES:
            train=[row['routes'][route]['training'] for row in records if row['kind']=='train']
            test=[row['routes'][route] for row in records if row['kind']=='test']
            test_change={key:sum(row['changes'][key] for row in test) for key in ('instances','ferrite','ferrite_pixels','changed_instances','changed_pixels','P_to_F_instances','P_to_F_pixels','F_to_P_instances','F_to_P_pixels')}
            test_change['pooled_ferrite_mean_area']=test_change['ferrite_pixels']/test_change['ferrite'] if test_change['ferrite'] else None
            tails={key:{field:sum(row['ferrite_composition']['tails'][key][field] for row in test) for field in ('instances','pixels')}
                for key in ('lt200','200to999','1000to9999','ge10000')}
            totals[route]=dict(test_unlabeled_changes=test_change,test_ferrite_tails=tails,
                training_cohort={group:{key:sum(row['summary'][group][key] for row in train) for key in train[0]['summary'][group]} for group in ('all','large')},
                training_fixed_pair_confusion={key:sum(row['fixed_pair_confusion'][key] for row in train) for key in train[0]['fixed_pair_confusion']})
        report=dict(format=FORMAT,complete=True,training=False,no_optimizer_steps=True,full_test_inference=False,
            official_submission=False,best_deployment_changed=False,sources=len(names),routes=list(ROUTES),
            checkpoint=dict(path=str(checkpoint),sha256=cfg['checkpoint_sha256'],epoch=20,architecture=architecture,parameters=parameters),
            parameter_counts=dict(diagnostic_resident=parameters,original_deployment_restorer=restorer_parameters,
                conservative_full_deployment=deployed_parameters,new_parameters=0,
                scope='Original protected reflect report accounts for same D5a, which is not loaded or run in this diagnostic'),
            semantic_forward_calls=calls,expected_forward_calls=expected_calls,affinity_forward_calls=0,watershed_calls=0,D5a_forward_calls=0,
            baseline_eight_probability_and_classes_exact=all(row['baseline_probability_and_classes_exact'] for row in records),
            fixed_png_bytes_exact=all(item['copied_png_sha256']==row['fixed_png_sha256'] for row in records for item in row['routes'].values()),
            frozen_state_sha256=state,frozen_exact=True,first_whole_repeat_exact=True,precision=precision,
            protected_inputs=protected,protected_inputs_exact=True,protected_sources=code,virtual_module_paths=virtual,
            totals=totals,elapsed_seconds=time.time()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
            caveat='Only terminal semantic context/scale varies on eight predeclared sources. Fixed geometry upstream already used original D5a/semantic; this is not geometry independent of semantics. Seen incomplete/coarse GT is diagnostic. Unlabeled class flips and F area composition are not correctness, instance mIoU, official area score or proof of compensating errors. No test GT inferred.')
        write(out/'report.json',report)
        page=['<!doctype html><meta charset="utf-8"><title>Frozen semantic native contexts</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>Frozen raw/S-align semantic: whole / native1024 / native512</h1>',
            '<p>Same reflect instance PNG and model weights. Green F, red P; test panels are predictions, without GT. Continuous native logits stitched before sigmoid and existing strict mean vote.</p>',
            '<p><a href="report.json">Report</a> | <a href="rows.json">Objects, censoring, crop receipts</a></p>']
        for path in sorted((out/'gallery').glob('*.jpg')):
            page.append('<h2>'+html.escape(path.stem)+'</h2><img src="gallery/'+html.escape(path.name)+'">')
        (out/'index.html').write_text('\n'.join(page),encoding='utf-8')
        write(out/'status.json',dict(status='complete',sources=len(names),semantic_forward_calls=calls,training=False))
        print('SEMANTIC_NATIVE_COMPLETE',len(names),calls,flush=True)
        return report
    except Exception as error:
        write(out/'status.json',dict(status='failed',error=repr(error),training=False)); raise
    finally:
        for hook in hooks: hook.remove()


def main():
    from utils.config import load_config
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/inference/semantic_native.yaml')
    parser.add_argument('--output',help='Fresh ignored outputs subdirectory; default outputs/semantic_native')
    parser.add_argument('--run',action='store_true',help='Execute bounded CUDA diagnostic; default prints plan only')
    args=parser.parse_args()
    if args.run: run(args)
    else: print(json.dumps(plan(load_config(args.config)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
