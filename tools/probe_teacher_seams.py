# -*- coding: utf-8 -*-
"""教师窄零缝支持恢复的固定四图诊断；实际三头共享特征，不训练、不重跑WS。

before/after始终指修补前/后教师e5，绝不映射为人工GT训练组。
训练GT仅定位机制；测试GT不读取、不生成，最终预测复用已保存monitor。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_gt_source_boundary import (array_sha, calibrate_same_count,
    native_gt_masks, probability_summary, read, response_drift, sha, write)

SOURCES = ('train_351', 'train_658', 'test_101', 'test_116')
ARMS = ('r1', 'before', 'after')
EXPECTED = dict(
    r1='43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434',
    before='86d27f74545daf40d555ca22016ff52836aa43919dcf73263fa4fc342e0699b0',
    after='8abb6e374e77fd733c81b04da82f3d62824f27590c70f724ba71075043221fc1')
FOLDERS = dict(before='outputs/gt_source/teacher', after='outputs/teacher_seams/teacher')
CONFIGS = dict(before='config/train/affinity_gt_source.yaml', after='config/train/teacher_seams.yaml')


def validate_candidate_status(status, config, arm):
    """诊断名与训练名分开；两个候选确实都必须是teacher，修补后还须版本匹配。"""
    from tools.evaluate_affinity_gt_source import validate_training_status
    if arm not in ('before', 'after'):
        raise ValueError('Only before/after teacher candidates are supported')
    contract = validate_training_status(status, config, 'teacher', EXPECTED[arm])
    if status.get('paired_rows_verified') != 320:
        raise RuntimeError('All 320 paired draws must be verified')
    if arm == 'after' and (status.get('variant') != 'teacher_zero_seam_v1'
                          or contract.get('variant') != 'teacher_zero_seam_v1'
                          or status.get('paired_actual_rows_verified') is not True):
        raise RuntimeError('Actual repaired seam version and 320 actual pair checks required')
    if arm == 'before' and contract.get('variant') is not None:
        raise RuntimeError('Before must be the unchanged original teacher control')
    return contract


def channel_diagnostics(tiles, probabilities, original, common, class_lookup):
    from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
    from utils.affinity_loss import build_affinity_targets_torch, _edge_slices
    from utils.offset_letterbox import letterbox_instance_geometry
    chunks=[[] for _ in range(8)]
    for tile_index,tile in enumerate(tiles):
        gt=tile['grid_gt'];valid=tile['valid']
        targets,legal=build_affinity_targets_torch(torch.from_numpy(gt.astype(np.int64))[None],
                                                torch.from_numpy(valid.copy())[None,None])
        target=targets.numpy()[0].astype(bool);legal=legal.numpy()[0]
        y,x,y1,x1=tile['box']
        og,_,_=letterbox_instance_geometry(original[y:y1,x:x1].astype(np.uint8),1024,512)
        cg,_,_=letterbox_instance_geometry(common[y:y1,x:x1].astype(np.uint8),1024,512)
        for channel,(dy,dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
            a,b=_edge_slices(*gt.shape,dy,dx)
            original_edge=np.zeros(gt.shape,bool);common_edge=np.zeros(gt.shape,bool)
            original_count=np.zeros(gt.shape,np.uint8);ferrite_count=np.zeros(gt.shape,np.uint8)
            original_edge[a]=(og[a]>0)&(og[b]>0)
            common_edge[a]=(cg[a]>0)&(cg[b]>0)
            original_count[a]=(og[a]>0).astype(np.uint8)+(og[b]>0).astype(np.uint8)
            ferrite_count[a]=(class_lookup[gt[a]]==1).astype(np.uint8)+(class_lookup[gt[b]]==1).astype(np.uint8)
            mask=legal[channel]
            chunks[channel].append(dict(same=target[channel][mask],original=original_edge[mask],
                common=common_edge[mask],original_count=original_count[mask],ferrite_count=ferrite_count[mask],
                **{arm:probabilities[arm][tile_index][channel][mask] for arm in ARMS}))
    result=[]
    for channel,offset in enumerate(DEFAULT_AFFINITY_OFFSETS):
        arrays={key:np.concatenate([r[key] for r in chunks[channel]])
                for key in ('same','original','common','original_count','ferrite_count',*ARMS)}
        masks=dict(all_processed_known=np.ones(len(arrays['same']),bool),
            original_original=arrays['original'], involving_filled=~arrays['original'],
            original_filled=arrays['original_count']==1, filled_filled=arrays['original_count']==0,
            common_teacher_support=arrays['common'],
            original_F_F=arrays['original']&(arrays['ferrite_count']==2),
            original_F_P=arrays['original']&(arrays['ferrite_count']==1),
            original_P_P=arrays['original']&(arrays['ferrite_count']==0))
        if not np.all((masks['original_original'].astype(np.uint8)+masks['original_filled']+masks['filled_filled'])==1):
            raise RuntimeError('Original/filled edge strata must be a disjoint exhaustive partition')
        anchor_mask=masks['original_original'];anchor_same=arrays['same'][anchor_mask]
        anchor_probs={arm:arrays[arm][anchor_mask] for arm in ARMS}
        anchor_count=int(np.count_nonzero(anchor_probs['r1'][anchor_same]>=.5))
        anchor_calibration={arm:calibrate_same_count(anchor_probs[arm][anchor_same],anchor_probs[arm][~anchor_same],anchor_count)
                            for arm in ARMS}
        anchor_calibration['r1'].update(threshold=.5,same_count=anchor_count,
            same_recall=anchor_count/int(anchor_same.sum()) if anchor_same.any() else None,
            different_specificity=float(np.mean(anchor_probs['r1'][~anchor_same]<.5)) if (~anchor_same).any() else None,
            tied_same_at_threshold=int(np.count_nonzero(anchor_probs['r1'][anchor_same]==.5)),recall_excess_due_ties=0.)
        strata={}
        for name,mask in masks.items():
            same=arrays['same'][mask];ps={arm:arrays[arm][mask] for arm in ARMS}
            npos=int(same.sum());nneg=int((~same).sum())
            required=int(np.count_nonzero(ps['r1'][same]>=.5))
            levels=dict(r1_threshold05_same_count=required, fixed095=int(math.ceil(.95*npos)))
            strata[name]=dict(legal_same=npos,legal_different=nneg,
                target_sha256=array_sha(same),arms={arm:dict(
                    same_probability=probability_summary(ps[arm][same]),
                    different_probability=probability_summary(ps[arm][~same]),
                    same_recall_at05=float(np.mean(ps[arm][same]>=.5)) if npos else None,
                    different_specificity_at05=float(np.mean(ps[arm][~same]<.5)) if nneg else None,
                    local_at_least_reference_recall={label:calibrate_same_count(ps[arm][same],ps[arm][~same],count)
                                    for label,count in levels.items()},
                    shared_original_original_threshold=anchor_calibration[arm]['threshold'],
                    same_recall_at_shared_threshold=float(np.mean(ps[arm][same]>=anchor_calibration[arm]['threshold']))
                        if npos and anchor_calibration[arm]['threshold'] is not None else None,
                    different_specificity_at_shared_threshold=float(np.mean(ps[arm][~same]<anchor_calibration[arm]['threshold']))
                        if nneg and anchor_calibration[arm]['threshold'] is not None else None) for arm in ARMS})
        result.append(dict(channel=channel,offset=list(offset),strata=strata,
                           shared_original_original_calibration=anchor_calibration))
    return dict(per_channel=result, test_GT_used=False,
        targets='Original processed manual IDs, identical legal endpoints for all three heads; uncovered-as-boundary disabled',
        provenance='original_original requires both nearest-grid endpoints originally covered; involving_filled is the remaining processed-known legal edge',
        sampling='All original native1024 windows; overlapping window relations are intentionally counted repeatedly',
        calibration='same=1; sigmoid >= threshold. Shared thresholds fixed on original_original to at least R1 .5 TP count, then applied to all provenance/phase strata. Local per-stratum recalibration and .95 recall are auxiliary ROC diagnostics, not cross-stratum calibration evidence.',
        native_path_checked=False, independent_validation=False)


def save_gallery(path, rgb, maps, source):
    width=300;height=max(1,round(rgb.shape[0]*width/rgb.shape[1]))
    panels=[]
    for arm in ('rgb',*ARMS):
        if arm=='rgb':
            image=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(width,height),interpolation=cv2.INTER_AREA)
            title=source+' original'
        else:
            thumbnail=cv2.resize(maps[arm],(width,height),interpolation=cv2.INTER_AREA)
            image=cv2.applyColorMap(np.rint(np.clip(thumbnail,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
            title=arm+' mean %.4f'%maps[arm].mean(dtype=np.float64)
        image=cv2.copyMakeBorder(image,30,32,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(image,title,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        panels.append(image)
    board=np.concatenate(panels,axis=1)
    ramp=np.linspace(0,1,width,dtype=np.float32)[None,:].repeat(12,axis=0)
    color=cv2.applyColorMap(np.rint(ramp*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
    for i in range(1,4):
        start=i*width;board[-28:-16,start:start+width]=color
        cv2.putText(board,'P(boundary): 0    .45 .65    1',(start+5,board.shape[0]-3),
                    cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),board):
        raise RuntimeError('Gallery write failed')



def save_final_gallery(path, rgb, before_ids, before_classes, after_ids, after_classes, source):
    """明确修补前后名称，橙色轮廓、蓝色铁素体、棕色珠光体。"""
    from tools.evaluate_affinity_gt_source import _edges, _phase
    panels = [rgb.copy()]
    titles = ['Original', 'Before edges', 'After edges', 'Before phase', 'After phase']
    for ids in (before_ids, after_ids):
        image = rgb.copy(); image[_edges(ids)] = (255, 90, 0); panels.append(image)
    for ids, classes in ((before_ids, before_classes), (after_ids, after_classes)):
        phase = _phase(ids, classes); image = rgb.copy()
        for value, color in ((0,(220,100,50)),(1,(40,190,230))):
            mask = phase == value
            image[mask] = np.rint(.45 * image[mask] + .55 * np.asarray(color)).astype(np.uint8)
        image[_edges(ids)] = 255; panels.append(image)
    h, w = rgb.shape[:2]
    crops = [np.s_[:,:], np.s_[max(0,h//2-256):min(h,h//2+256),max(0,w//2-256):min(w,w//2+256)]]
    boards = []
    for crop in crops:
        tiles = []
        for image, title in zip(panels, titles):
            image = image[crop]; ih, iw = image.shape[:2]
            image = cv2.resize(image, (250,max(1,round(ih*250/iw))), interpolation=cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image,25,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(image,title,(4,17),cv2.FONT_HERSHEY_SIMPLEX,.37,(0,0,0),1,cv2.LINE_AA)
            tiles.append(image)
        boards.append(np.concatenate(tiles,axis=1))
    if not cv2.imwrite(str(path),cv2.cvtColor(np.concatenate(boards,axis=0),cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Final gallery write failed: '+source)


def run(args):
    from data.mim_dataset import list_images
    from data.rgb_restoration_dataset import read_rgb
    from models.backend_adaptation import load_backend
    from tools import affinity_bottleneck_capture
    from tools.affinity_feature_capture import capture_frozen_features, replay_head_boundary
    from tools.evaluate_affinity_gt_source import compare_predictions
    from tools.run_affinity_sweep import runtime_contract
    from tools.run_marker_reflect import load_prediction, precision_record, source_snapshot
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_affinity_coverage import load_original_core
    from train_affinity_gt_source import D5A_SHA, relative_input, validate_pair
    from train_backend_adaptation import tensor_digest
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA sam2_env required; no real-data CPU fallback')
    output = (ROOT / args.output).resolve()
    if output != ROOT / 'outputs/teacher_seams/boundary_probe' or output.exists():
        raise ValueError('Fresh fixed private outputs/teacher_seams/boundary_probe required')
    configs = {arm: load_config(str(ROOT / path)) for arm, path in CONFIGS.items()}
    config = configs['before']
    if tuple(config['gt_source']['train_monitors']) + tuple(config['gt_source']['monitor_images']) != SOURCES:
        raise ValueError('Original fixed process sources differ')
    if any(Path(project_path(value)).resolve() != ROOT for value in configs.values()):
        raise ValueError('Actual project root differs')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    precision = precision_record(); runtime = runtime_contract(config)
    if runtime_contract(configs['after']) != runtime:
        raise RuntimeError('Repair inference runtime contract changed')
    manifest_path = relative_input(config, config['gt_source']['manifest'])
    manifest = read(manifest_path)
    if (manifest.get('status') != 'complete' or manifest.get('teacher_checkpoint_sha256') != EXPECTED['r1']
            or manifest.get('restorer_checkpoint_sha256') != D5A_SHA
            or manifest['precision'] != precision or manifest['runtime'] != runtime):
        raise RuntimeError('Source manifest/R1/D5a/FP32 runtime differs')
    protected = {str(ROOT / value): sha(ROOT / value) for value in CONFIGS.values()}
    protected[str(manifest_path)] = sha(manifest_path)
    statuses = {}; contracts = {}; receipts = {}; checkpoints = {}
    for arm in ('before', 'after'):
        folder = ROOT / FOLDERS[arm]
        checkpoints[arm] = folder / 'final.pt'
        protected[str(checkpoints[arm])] = EXPECTED[arm]
        status_path = folder / 'status.json'; statuses[arm] = read(status_path)
        protected[str(status_path)] = sha(status_path)
        contracts[arm] = validate_candidate_status(statuses[arm], configs[arm], arm)
        for epoch in (0, 5):
            path = folder / 'monitor' / f'e{epoch:02d}' / 'receipt.json'
            receipt = read(path); protected[str(path)] = sha(path)
            if receipt.get('epoch') != epoch or len(receipt.get('rows', [])) != 4 or {r['source'] for r in receipt['rows']} != set(SOURCES):
                raise RuntimeError('Exact four-source monitor receipt required')
            receipts[(arm, epoch)] = {row['source']: row for row in receipt['rows']}
    for key in ('checkpoint_sha256','restorer_checkpoint_sha256','manifest_sha256',
                'frozen_state_sha256','restorer_state_sha256','initial_head_sha256'):
        if contracts['before'][key] != contracts['after'][key]:
            raise RuntimeError('Candidate isolation contract differs: ' + key)
    if sha(manifest_path) != contracts['before']['manifest_sha256']:
        raise RuntimeError('Original common manifest SHA differs')
    repair_manifest = relative_input(configs['after'], configs['after']['gt_source']['seam_repair']['manifest'])
    protected[str(repair_manifest)] = contracts['after']['seam_manifest_sha256']
    for source in SOURCES:
        if receipts[('before',0)][source] != receipts[('after',0)][source]:
            raise RuntimeError('Original and repaired e0 source receipts differ')
    checkpoints['r1'] = relative_input(config, config['gt_source']['checkpoint'])
    protected[str(checkpoints['r1'])] = EXPECTED['r1']
    restorer_path = relative_input(config, config['backend_adaptation']['restoration']['checkpoint'])
    protected[str(restorer_path)] = D5A_SHA
    for path, digest in protected.items():
        if sha(path) != digest:
            raise RuntimeError('Protected identity differs: ' + path)
    core = load_original_core(config); views = core.coverage_original_views
    model, bundle = load_backend(checkpoints['r1'], config, 'cuda')
    model.eval().requires_grad_(False)
    frozen = frozen_digest(model); state = tensor_digest(model.state_dict().items())
    if frozen != contracts['before']['frozen_state_sha256']:
        raise RuntimeError('Actual frozen R1 state differs')
    base_state = bundle.get('model_state_dict')
    if not isinstance(base_state, dict) or not base_state:
        raise RuntimeError('Complete R1 model_state_dict required')
    heads = dict(r1=model.affinity_decoder); head_records = {}
    for arm in ('before','after'):
        candidate = torch.load(checkpoints[arm],map_location='cpu',weights_only=False)
        values = candidate.get('model_state_dict')
        if (not isinstance(values,dict) or set(values)!=set(base_state) or candidate.get('epoch')!=5
                or candidate.get('extra',{}).get('arm')!='teacher' or candidate['extra'].get('updates')!=320
                or candidate['extra'].get('contract')!=contracts[arm]
                or tensor_digest(values.items())!=statuses[arm]['final_state_sha256']):
            raise RuntimeError('Complete candidate state/epoch/identity differs: '+arm)
        if any(not torch.equal(values[key],base_state[key]) for key in base_state if not key.startswith('affinity_decoder.')):
            raise RuntimeError('Candidate non-head values differ from R1')
        head_state={key[len('affinity_decoder.'):]:value for key,value in values.items() if key.startswith('affinity_decoder.')}
        head=deepcopy(model.affinity_decoder).cpu();head.load_state_dict(head_state,strict=True)
        heads[arm]=head.to('cuda').eval().requires_grad_(False)
        head_records[arm]=dict(checkpoint_sha256=EXPECTED[arm],training_arm='teacher',
            diagnostic_arm=arm,head_state_sha256=tensor_digest(head_state.items()),
            full_state_sha256=statuses[arm]['final_state_sha256'],all_nonhead_values_equal_R1=True)
        del candidate,values,head_state;gc.collect()
    for arm,head in heads.items():
        modules=list(head.modules())
        if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in modules):
            raise RuntimeError('Actual affinity head unexpectedly contains BatchNorm/Dropout')
        head_records.setdefault(arm,dict(checkpoint_sha256=EXPECTED[arm],head_state_sha256=tensor_digest(head.state_dict().items())))
        head_records[arm].update(groupnorm_modules=sum(isinstance(m,torch.nn.GroupNorm) for m in modules),
                                 batchnorm_modules=0,dropout_modules=0)
    restorer=get_restorer(config,'cuda').eval().requires_grad_(False)
    restored=tensor_digest(restorer.state_dict().items())
    if restored!=contracts['before']['restorer_state_sha256']:
        raise RuntimeError('Frozen D5a state differs')
    tests={Path(p).stem:Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))}
    train_rows={row['source']:row for row in manifest['rows'] if row['source'] in SOURCES}
    if set(train_rows)!=set(SOURCES[:2]) or not set(SOURCES[2:]).issubset(tests):
        raise RuntimeError('Fixed source cohort missing')
    output.mkdir(parents=True)
    for sub in ('maps','gallery','final_gallery'):
        (output/sub).mkdir()
    code,virtual=source_snapshot(output)
    protected.update({str(ROOT/name):digest for name,digest in code.items()})
    protected[str(Path(__file__).resolve())]=sha(__file__)
    report=dict(format='teacher_seams_boundary_probe_v1',complete=False,training=False,
        no_optimizer_steps=True,test_GT_used=False,source_order=list(SOURCES),
        source_rule='Original four fixed e0/e5 monitor sources and seeds; no effect-based selection',
        checkpoints=head_records,epochs=dict(r1=20,before=5,after=5),updates=dict(before=320,after=320),
        frozen_state_sha256=frozen,restorer_checkpoint_sha256=D5A_SHA,restorer_state_sha256=restored,
        protected_inputs=protected,protected_code=code,virtual_code_paths=virtual,config=config,
        candidate_configs=configs,precision=precision,runtime=runtime,images=[],watershed_forward_calls=0,
        interpretation='Seen training GT mechanism diagnosis only; test labels absent; relative topology is not error/correctness or official score.',
        map_artifacts='Float32 width384 area thumbnails; native probabilities/SHA/statistics computed before resize; absolute 0-1 colormap')
    write(output/'report.json',report);started=time.time()
    try:
        for source in SOURCES:
            receipt=receipts[('before',0)][source];seed=int(receipt['seed'])
            if source.startswith('train_'):
                row=train_rows[source];path=relative_input(config,row['image']);gtpath=relative_input(config,row['manual_gt'])
                pairpath=relative_input(config,row['pair_npz'])
                for p,field in ((path,'image_sha256'),(gtpath,'manual_gt_sha256'),(pairpath,'pair_sha256')):
                    protected[str(p)]=row[field]
                    if sha(p)!=row[field]:
                        raise RuntimeError('Fixed training source SHA differs')
                from data.semantic_targets import load_completed_semantic_source
                gt,class_lookup=load_completed_semantic_source(gtpath,tuple(row['shape']))
                classpath=gtpath.with_name(gtpath.name.removesuffix('_gt.npz')+'_class.json')
                protected[str(classpath)]=sha(classpath)
                with np.load(gtpath,allow_pickle=False) as z:
                    if not np.array_equal(gt,z['instance_map']):
                        raise RuntimeError('Semantic IDs differ')
                    gt=z['instance_map'].copy();original=z['original_covered'].astype(bool)
                    with np.load(pairpath,allow_pickle=False) as pair:
                        checked=validate_pair(pair,gt,gt.shape)
                        if not np.array_equal(original,checked['original_covered']) or not np.array_equal(original|checked['filled'],gt>0):
                            raise RuntimeError('Original/filled provenance differs')
                        common=checked['common_known']
                if row['seed']!=seed:
                    raise RuntimeError('Fixed training seed differs')
            else:
                path=tests[source];protected[str(path)]=sha(path)
                gt=np.zeros(read_rgb(str(path)).shape[:2],np.uint16);original=common=None
            predictions={};final_receipts={}
            for arm in ARMS:
                monitor_arm='before' if arm=='r1' else arm;epoch=0 if arm=='r1' else 5
                folder=ROOT/FOLDERS[monitor_arm]/'monitor'/f'e{epoch:02d}'
                expected=receipts[(monitor_arm,epoch)][source]
                predictions[arm]=load_prediction(folder,source,gt.shape)
                final_receipts[arm]=expected['final_output_hashes']
                for suffix,key in (('_inst.png','png_sha256'),('_class.json','classes_sha256')):
                    p=folder/(source+suffix);protected[str(p)]=expected['final_output_hashes'][key]
                    if sha(p)!=protected[str(p)]:
                        raise RuntimeError('Existing final monitor prediction SHA differs')
            ids,classes=predictions['r1']
            def capture_boundary_only(actual_views,actual_model,actual_restorer,actual_path,actual_config,device,actual_seed):
                observed={};calls=[]
                def record_decode(semantic,boundary,_config):
                    if calls:
                        raise RuntimeError('Exactly one boundary-only native callback required')
                    calls.append(True);observed.update(semantic=semantic.detach().cpu().clone(),boundary=boundary.detach().cpu().clone())
                    return ids.copy(),deepcopy(classes)
                def reuse_vote(actual_ids,probability):
                    observed['raw_probability']=np.asarray(probability).copy();return deepcopy(classes),{}
                with patch.object(actual_views,'decode_native',side_effect=record_decode),patch.object(actual_views,'revote_instances',side_effect=reuse_vote):
                    rgb,results=actual_views.predict_views(actual_model,actual_restorer,actual_path,actual_config,device,actual_seed,[1024])
                if len(calls)!=1 or set(results)!={'patch1024'}:
                    raise RuntimeError('Single-view capture differs')
                observed['baseline']=results['patch1024'];return rgb,observed
            with patch.object(affinity_bottleneck_capture,'capture_original',side_effect=capture_boundary_only):
                rgb,captured,tiles,capture_audit=capture_frozen_features(views,model,restorer,path,config,gt,'cuda',seed)
            maps={};replays={};probabilities={}
            for arm,head in heads.items():
                probs=[]
                def collect(_module,_inputs,result):
                    probs.append(result['affinity_logits'].sigmoid().detach().cpu().numpy()[0].copy())
                hook=head.register_forward_hook(collect)
                try:
                    maps[arm],replays[arm]=replay_head_boundary(views,tiles,head,config,gt.shape,empty_exact=arm=='r1')
                finally:
                    hook.remove()
                if len(probs)!=len(tiles):
                    raise RuntimeError('One logits record per actual tile required')
                probabilities[arm]=probs
                epoch=0 if arm=='r1' else 5;monitor_arm='before' if arm=='r1' else arm
                expected=receipts[(monitor_arm,epoch)][source]
                if expected['seed']!=seed or array_sha(maps[arm])!=expected['boundary_sha256']:
                    raise RuntimeError('Actual float boundary monitor SHA failed: '+source+' '+arm)
            image=dict(source=source,kind=receipt['kind'],seed=seed,shape=list(gt.shape),source_sha256=sha(path),
                boundary_sha256={arm:array_sha(value) for arm,value in maps.items()},
                monitor_float_SHA_exact={arm:True for arm in ARMS},R1_empty_replay_exact=True,
                frozen_features=capture_audit,head_replays=replays,
                probability_summary={arm:probability_summary(value) for arm,value in maps.items()},
                response_strata={reference:{name:response_drift(maps[reference],maps['after'],mask)
                    for name,mask in dict(high_ge065=maps[reference]>=.65,low_lt045=maps[reference]<.45,
                    middle_045_to065=(maps[reference]>=.45)&(maps[reference]<.65)).items()}
                    for reference in ('r1','before')},
                final_output_hashes=final_receipts,
                final_comparisons={a+'_to_'+b:compare_predictions(*predictions[a],*predictions[b])
                    for a,b in (('r1','before'),('r1','after'),('before','after'))},test_GT_used=False)
            if original is not None:
                image['native_GT_response']={name:{arm:probability_summary(value[mask]) for arm,value in maps.items()}
                    for name,mask in native_gt_masks(gt,original).items()}
                image['original_GT_channels']=channel_diagnostics(tiles,probabilities,original,common,class_lookup)
            mw=384;mh=max(1,round(gt.shape[0]*mw/gt.shape[1]))
            np.savez_compressed(output/'maps'/(source+'.npz'),**{arm:cv2.resize(value,(mw,mh),interpolation=cv2.INTER_AREA).astype(np.float32) for arm,value in maps.items()})
            save_gallery(output/'gallery'/(source+'.png'),rgb,maps,source)
            save_final_gallery(output/'final_gallery'/(source+'.png'),rgb,*predictions['before'],*predictions['after'],source+' BEFORE -> AFTER')
            report['images'].append(image);write(output/'report.json',report)
            write(output/'status.json',dict(status='running',completed_images=len(report['images']),maximum_images=4,training=False))
            print('TEACHER_SEAMS_PROBE',len(report['images']),source,{arm:round(image['probability_summary'][arm]['mean'],6) for arm in ARMS},flush=True)
            del rgb,captured,tiles,maps,probabilities,probs;gc.collect();torch.cuda.empty_cache()
        for path,digest in protected.items():
            if sha(path)!=digest:
                raise RuntimeError('Protected input/code changed: '+path)
        if (tensor_digest(model.state_dict().items())!=state or frozen_digest(model)!=frozen
                or tensor_digest(restorer.state_dict().items())!=restored
                or precision_record()!=precision or runtime_contract(config)!=runtime):
            raise RuntimeError('Frozen state/runtime changed')
        for arm,head in heads.items():
            if tensor_digest(head.state_dict().items())!=head_records[arm]['head_state_sha256']:
                raise RuntimeError('Read-only head state changed')
        report.update(complete=True,protected_before_after_exact=True,elapsed_seconds=time.time()-started,
                      native_boundary_monitor_SHA_parity_all12=True)
        write(output/'report.json',report)
        write(output/'status.json',dict(status='complete',completed_images=4,maximum_images=4,training=False))
        write(output/'artifact_manifest.json',dict(files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}))
    except Exception as error:
        write(output/'status.json',dict(status='failed',completed_images=len(report['images']),error=str(error),training=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/teacher_seams/boundary_probe')
    run(parser.parse_args())


if __name__=='__main__':
    main()
