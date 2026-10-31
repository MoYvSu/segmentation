# -*- coding: utf-8 -*-
"""固定四图的GT-source概率诊断；仅重放三个头，不训练、不重新watershed。

原CUDA encoder features只保存在内存；测试图的全零占位map只用于窗口几何。
训练源原processed GT只做机制诊断，不能作为独立验证集或竞赛晋级证据。
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
SOURCES = ('train_351', 'train_658', 'test_101', 'test_116')
ARMS = ('r1', 'manual', 'teacher')
EXPECTED = dict(
    r1='43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434',
    manual='db004925caf7048e98cd0dd889a65e4fdd00adbdf220c3ddde633a416b9b977a',
    teacher='86d27f74545daf40d555ca22016ff52836aa43919dcf73263fa4fc342e0699b0')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def probability_summary(values):
    values = np.asarray(values)
    if values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('Finite floating probabilities required')
    if values.size and (values.min() < 0 or values.max() > 1):
        raise ValueError('Probabilities outside [0,1]')
    if not values.size:
        return dict(count=0, mean=None, quantiles=None, ge_045=None, ge_065=None)
    return dict(count=int(values.size), mean=float(values.mean(dtype=np.float64)),
        quantiles={str(q):float(np.quantile(values,q)) for q in (0,.01,.1,.25,.5,.75,.9,.95,.99,1)},
        ge_045=float(np.count_nonzero(values >= .45)/values.size),
        ge_065=float(np.count_nonzero(values >= .65)/values.size))


def calibrate_same_count(positive, negative, required):
    """k>0取满足召回的最大观测阈值；k=0取概率范围外的2，ties显式报告。"""
    positive, negative = np.asarray(positive), np.asarray(negative)
    probability_summary(positive); probability_summary(negative)
    if positive.ndim != 1 or negative.ndim != 1 or type(required) is not int or not 0 <= required <= len(positive):
        raise ValueError('One-dimensional probabilities and feasible integer TP count required')
    if not len(positive):
        return dict(threshold=None, requested_same_count=0, same_count=0,
            same_recall=None, different_specificity=None, different_count=len(negative),
            tied_same_at_threshold=0, recall_excess_due_ties=None)
    # 2在float16/32/64中均精确且大于所有合法概率；避免nextafter在低精度比较时舍入。
    threshold = (2.0
                 if required == 0 else float(np.partition(positive,len(positive)-required)[len(positive)-required]))
    found = int(np.count_nonzero(positive >= threshold))
    return dict(threshold=threshold, requested_same_count=required, same_count=found,
        same_recall=found/len(positive), different_count=int(len(negative)),
        different_specificity=float(np.count_nonzero(negative < threshold)/len(negative)) if len(negative) else None,
        tied_same_at_threshold=int(np.count_nonzero(positive == threshold)),
        recall_excess_due_ties=(found-required)/len(positive))


def response_drift(reference, candidate, mask):
    r,c = reference[mask],candidate[mask]
    delta = c.astype(np.float64)-r.astype(np.float64)
    return dict(reference=probability_summary(r),candidate=probability_summary(c),
        delta_mean=float(delta.mean()) if delta.size else None,
        delta_quantiles={str(q):float(np.quantile(delta,q)) for q in (.1,.5,.9)} if delta.size else None,
        falling_fraction=float(np.mean(delta < 0)) if delta.size else None,
        rising_fraction=float(np.mean(delta > 0)) if delta.size else None)


def native_gt_masks(gt, original):
    """可信指原覆盖双端known；仅局部ID关系，不推断物理界面或整路径。"""
    gt,original = np.asarray(gt),np.asarray(original)
    if gt.ndim != 2 or original.shape != gt.shape or original.dtype != bool or np.any(original & (gt == 0)):
        raise ValueError('Processed GT and original coverage inconsistent')
    edge = np.zeros(gt.shape,bool)
    for dy,dx in ((0,1),(1,0)):
        a=np.s_[:gt.shape[0]-dy,:gt.shape[1]-dx]
        b=np.s_[dy:,dx:]
        crossing = original[a]&original[b]&(gt[a]>0)&(gt[b]>0)&(gt[a]!=gt[b])
        edge[a] |= crossing; edge[b] |= crossing
    kernel=np.ones((5,5),np.uint8)
    safe=cv2.erode(original.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    labels=gt.astype(np.float32)
    interior=safe&(gt>0)&(cv2.erode(labels,kernel)==cv2.dilate(labels,kernel))
    band=(cv2.dilate(edge.astype(np.uint8),kernel)>0)&safe&(gt>0)
    return dict(original_known=original&(gt>0), original_deep_local_same=interior,
                original_different_ID_band_radius2=band)


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


def run(args):
    from data.mim_dataset import list_images
    from data.rgb_restoration_dataset import read_rgb
    from models.backend_adaptation import load_backend
    from tools import affinity_bottleneck_capture
    from tools.affinity_feature_capture import capture_frozen_features, replay_head_boundary
    from tools.evaluate_affinity_gt_source import validate_training_status
    from tools.run_affinity_sweep import runtime_contract
    from tools.run_marker_reflect import load_prediction, precision_record, source_snapshot
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_affinity_coverage import load_original_core
    from train_affinity_gt_source import D5A_SHA, validate_config, relative_input, validate_pair
    from train_backend_adaptation import tensor_digest
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Actual sam2_env CUDA required; no CPU real-data fallback')
    output=(ROOT/args.output).resolve()
    if output != ROOT/'outputs/gt_source/boundary_probe' or output.exists():
        raise ValueError('Fresh fixed private outputs/gt_source/boundary_probe required')
    config_path=(ROOT/args.config).resolve();config=load_config(str(config_path));validate_config(config)
    if tuple(config['gt_source']['train_monitors'])+tuple(config['gt_source']['monitor_images']) != SOURCES:
        raise ValueError('Fixed original monitor source order differs')
    if Path(project_path(config)).resolve()!=ROOT:
        raise ValueError('Actual project root differs')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    precision=precision_record();runtime=runtime_contract(config)
    manifest_path=relative_input(config,config['gt_source']['manifest']);manifest=read(manifest_path)
    if (manifest.get('status')!='complete' or manifest.get('teacher_checkpoint_sha256')!=EXPECTED['r1']
            or manifest.get('restorer_checkpoint_sha256')!=D5A_SHA
            or manifest['precision']!=precision or manifest['runtime']!=runtime):
        raise RuntimeError('Source manifest/model/FP32 runtime identity differs')
    protected={str(config_path):sha(config_path),str(manifest_path):sha(manifest_path)}
    statuses={};contracts={};receipts={};checkpoints={}
    for arm in ('manual','teacher'):
        folder=ROOT/'outputs/gt_source'/arm
        checkpoints[arm]=folder/'final.pt';protected[str(checkpoints[arm])]=EXPECTED[arm]
        status_path=folder/'status.json';statuses[arm]=read(status_path);protected[str(status_path)]=sha(status_path)
        contracts[arm]=validate_training_status(statuses[arm],config,arm,EXPECTED[arm])
        for epoch in (0,5):
            path=folder/'monitor'/f'e{epoch:02d}'/'receipt.json';receipt=read(path);protected[str(path)]=sha(path)
            if receipt.get('epoch')!=epoch or len(receipt.get('rows',[]))!=4 or {r['source'] for r in receipt['rows']}!=set(SOURCES):
                raise RuntimeError('Exact four-image monitor receipt required')
            receipts[(arm,epoch)]={r['source']:r for r in receipt['rows']}
    if contracts['manual']!=contracts['teacher'] or statuses['teacher'].get('paired_rows_verified')!=320:
        raise RuntimeError('Paired same-contract e5/320 arms required')
    if sha(manifest_path)!=contracts['manual']['manifest_sha256']:
        raise RuntimeError('Paired training manifest SHA differs')
    for source in SOURCES:
        if receipts[('manual',0)][source]!=receipts[('teacher',0)][source]:
            raise RuntimeError('Both e0 monitor rows must be identical')
    checkpoints['r1']=relative_input(config,config['gt_source']['checkpoint']);protected[str(checkpoints['r1'])]=EXPECTED['r1']
    restorer_path=relative_input(config,config['backend_adaptation']['restoration']['checkpoint'])
    protected[str(restorer_path)]=D5A_SHA
    for path,digest in protected.items():
        if sha(path)!=digest:
            raise RuntimeError('Protected identity differs: '+path)
    core=load_original_core(config);views=core.coverage_original_views
    model,bundle=load_backend(checkpoints['r1'],config,'cuda');model.eval().requires_grad_(False)
    frozen=frozen_digest(model);state=tensor_digest(model.state_dict().items())
    if frozen!=contracts['manual']['frozen_state_sha256']:
        raise RuntimeError('R1 frozen encoder/LoRA/semantic state differs')
    base_state=bundle.get('model_state_dict')
    if not isinstance(base_state,dict) or not base_state:
        raise RuntimeError('Complete R1 model_state_dict required')
    heads=dict(r1=model.affinity_decoder);head_records={}
    for arm in ('manual','teacher'):
        candidate=torch.load(checkpoints[arm],map_location='cpu',weights_only=False)
        values=candidate.get('model_state_dict')
        if (not isinstance(values,dict) or set(values)!=set(base_state) or candidate.get('epoch')!=5
                or candidate.get('extra',{}).get('arm')!=arm or candidate['extra'].get('updates')!=320
                or candidate['extra'].get('contract')!=contracts[arm]
                or tensor_digest(values.items())!=statuses[arm]['final_state_sha256']):
            raise RuntimeError('Complete candidate state/epoch/identity differs: '+arm)
        frozen_keys=[key for key in base_state if not key.startswith('affinity_decoder.')]
        if any(not torch.equal(values[key],base_state[key]) for key in frozen_keys):
            raise RuntimeError('Candidate non-affinity state differs from R1')
        head_state={key[len('affinity_decoder.'):]:value for key,value in values.items() if key.startswith('affinity_decoder.')}
        head=deepcopy(model.affinity_decoder).cpu();head.load_state_dict(head_state,strict=True)
        heads[arm]=head.to('cuda').eval().requires_grad_(False)
        head_records[arm]=dict(source='Complete model_state_dict affinity_decoder. prefix; strict standalone head load',
            checkpoint_sha256=EXPECTED[arm],head_state_sha256=tensor_digest(head_state.items()),
            full_state_sha256=statuses[arm]['final_state_sha256'],all_nonhead_values_equal_R1=True)
        del candidate,values,head_state;gc.collect()
    for arm,head in heads.items():
        modules=list(head.modules())
        if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in modules):
            raise RuntimeError('Actual affinity head unexpectedly contains BatchNorm/Dropout')
        head_records.setdefault(arm,dict(checkpoint_sha256=EXPECTED[arm],head_state_sha256=tensor_digest(head.state_dict().items())))
        head_records[arm]['groupnorm_modules']=sum(isinstance(m,torch.nn.GroupNorm) for m in modules)
        head_records[arm]['batchnorm_modules']=0;head_records[arm]['dropout_modules']=0
    restorer=get_restorer(config,'cuda').eval().requires_grad_(False)
    restored=tensor_digest(restorer.state_dict().items())
    if restored!=contracts['manual']['restorer_state_sha256']:
        raise RuntimeError('Actual loaded oldD5a state differs')
    tests={Path(p).stem:Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))}
    train_rows={r['source']:r for r in manifest['rows'] if r['source'] in SOURCES}
    if set(train_rows)!=set(SOURCES[:2]) or not set(SOURCES[2:]).issubset(tests):
        raise RuntimeError('Fixed source cohort missing')
    output.mkdir(parents=True);(output/'maps').mkdir();(output/'gallery').mkdir()
    code,virtual=source_snapshot(output)
    protected.update({str(ROOT/name):digest for name,digest in code.items()})
    report=dict(format='affinity_gt_source_boundary_probe_v1',complete=False,training=False,
        no_optimizer_steps=True,test_GT_used=False,test_GT_placeholder='all-zero HW integer map only for existing geometry API',
        source_order=list(SOURCES),source_rule='Original e0/e5 monitors and receipt seed; fixed before inspection',
        checkpoints=head_records,epochs=dict(r1=20,manual=5,teacher=5),updates=dict(manual=320,teacher=320),
        frozen_state_sha256=frozen,restorer_checkpoint_sha256=D5A_SHA,restorer_state_sha256=restored,
        protected_inputs=protected,protected_code=code,virtual_code_paths=virtual,config=config,
        precision=precision,runtime=runtime,images=[],watershed_forward_calls=0,
        interpretation='Four fixed process-observation sources, two seen training sources. Mechanism diagnosis only; no independent accuracy, generalization or official score conclusion.',
        response_strata='R1 >=.65 high response and R1 <.45 low response are prediction-based, not GT.',
        native_GT_strata='5x5 original-covered local same-ID interior and radius2 band of native distinct-ID adjacency with both endpoints originally covered; no full native path or physical boundary proof.',
        map_artifacts='Four float32 thumbnail npz files, width384 area-average of native probabilities; native SHA and statistics always computed before downsampling; no min/max normalization')
    write(output/'report.json',report);started=time.time()
    try:
        for source in SOURCES:
            receipt=receipts[('manual',0)][source];seed=int(receipt['seed'])
            if source.startswith('train_'):
                row=train_rows[source];path=relative_input(config,row['image']);gtpath=relative_input(config,row['manual_gt'])
                pairpath=relative_input(config,row['pair_npz'])
                for p,field in ((path,'image_sha256'),(gtpath,'manual_gt_sha256'),(pairpath,'pair_sha256')):
                    protected[str(p)]=row[field]
                    if sha(p)!=row[field]:
                        raise RuntimeError('Fixed training source identity differs')
                from data.semantic_targets import load_completed_semantic_source
                gt,class_lookup=load_completed_semantic_source(gtpath,tuple(row['shape']))
                classpath=gtpath.with_name(gtpath.name.removesuffix('_gt.npz')+'_class.json')
                protected[str(classpath)]=sha(classpath)
                with np.load(gtpath,allow_pickle=False) as z:
                    if not np.array_equal(gt,z['instance_map']):
                        raise RuntimeError('Semantic source IDs differ from processed GT')
                    gt=z['instance_map'].copy();original=z['original_covered'].astype(bool)
                    with np.load(pairpath,allow_pickle=False) as pair:
                        checked=validate_pair(pair,gt,gt.shape)
                        if not np.array_equal(original,checked['original_covered']):
                            raise RuntimeError('Original provenance differs')
                        if (not np.array_equal(checked['filled'],z['filled'].astype(bool))
                                or not np.array_equal(original|checked['filled'],gt>0)):
                            raise RuntimeError('Original plus filled must exactly cover processed-known GT')
                        common=checked['common_known']
                if row['seed']!=seed:
                    raise RuntimeError('Fixed training seed differs')
            else:
                path=tests[source];protected[str(path)]=sha(path)
                gt=np.zeros(read_rgb(str(path)).shape[:2],np.uint16);original=common=None
            folder=ROOT/'outputs/gt_source/manual/monitor/e00'
            ids,classes=load_prediction(folder,source,gt.shape)
            for suffix,key in (('_inst.png','png_sha256'),('_class.json','classes_sha256')):
                p=folder/(source+suffix);protected[str(p)]=receipt['final_output_hashes'][key]
                if sha(p)!=protected[str(p)]:
                    raise RuntimeError('Existing monitor output differs')
            def capture_boundary_only(actual_views,actual_model,actual_restorer,actual_path,actual_config,device,actual_seed):
                observed={};calls=[]
                def record_decode(semantic,boundary,_config):
                    if calls:
                        raise RuntimeError('Exactly one boundary-only native callback required')
                    calls.append(True);observed.update(semantic=semantic.detach().cpu().clone(),
                        boundary=boundary.detach().cpu().clone())
                    return ids.copy(),deepcopy(classes)
                def reuse_vote(actual_ids,probability):
                    observed['raw_probability']=np.asarray(probability).copy()
                    return deepcopy(classes),{}
                with patch.object(actual_views,'decode_native',side_effect=record_decode), \
                     patch.object(actual_views,'revote_instances',side_effect=reuse_vote):
                    rgb,results=actual_views.predict_views(actual_model,actual_restorer,actual_path,
                        actual_config,device,actual_seed,[1024])
                if len(calls)!=1 or set(results)!={'patch1024'}:
                    raise RuntimeError('Boundary-only actual original single-view contract differs')
                observed['baseline']=results['patch1024']
                return rgb,observed
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
                    raise RuntimeError('One actual logits record per replayed tile required')
                probabilities[arm]=probs
                epoch=0 if arm=='r1' else 5;monitor_arm='manual' if arm=='r1' else arm
                expected=receipts[(monitor_arm,epoch)][source]
                if expected['seed']!=seed or array_sha(maps[arm])!=expected['boundary_sha256']:
                    raise RuntimeError('Actual float boundary monitor SHA parity failed: '+source+' '+arm)
            image=dict(source=source,kind=receipt['kind'],seed=seed,shape=list(gt.shape),source_sha256=sha(path),
                boundary_sha256={arm:array_sha(value) for arm,value in maps.items()},
                monitor_float_SHA_exact={arm:True for arm in ARMS},R1_empty_replay_exact=True,
                frozen_features=capture_audit,head_replays=replays,
                probability_summary={arm:probability_summary(value) for arm,value in maps.items()},
                R1_response_strata={name:{arm:response_drift(maps['r1'],maps[arm],mask) for arm in ARMS[1:]}
                    for name,mask in dict(high_ge065=maps['r1']>=.65,low_lt045=maps['r1']<.45,
                                         middle_045_to065=(maps['r1']>=.45)&(maps['r1']<.65)).items()},
                test_GT_used=False)
            if original is not None:
                image['native_GT_response']={name:{arm:probability_summary(value[mask]) for arm,value in maps.items()}
                                            for name,mask in native_gt_masks(gt,original).items()}
                image['original_GT_channels']=channel_diagnostics(tiles,probabilities,original,common,class_lookup)
            mw=384;mh=max(1,round(gt.shape[0]*mw/gt.shape[1]))
            np.savez_compressed(output/'maps'/(source+'.npz'),
                **{arm:cv2.resize(value,(mw,mh),interpolation=cv2.INTER_AREA).astype(np.float32) for arm,value in maps.items()})
            save_gallery(output/'gallery'/(source+'.png'),rgb,maps,source)
            report['images'].append(image);write(output/'report.json',report)
            write(output/'status.json',dict(status='running',completed_images=len(report['images']),maximum_images=4,training=False))
            print('BOUNDARY_PROBE',len(report['images']),source,
                  {arm:round(image['probability_summary'][arm]['mean'],6) for arm in ARMS},flush=True)
            del rgb,captured,tiles,maps,probabilities,probs;gc.collect();torch.cuda.empty_cache()
        for path,digest in protected.items():
            if sha(path)!=digest:
                raise RuntimeError('Protected input/code changed during probe: '+path)
        if (tensor_digest(model.state_dict().items())!=state or frozen_digest(model)!=frozen
                or tensor_digest(restorer.state_dict().items())!=restored
                or precision_record()!=precision or runtime_contract(config)!=runtime):
            raise RuntimeError('Actual frozen model/restorer/runtime changed')
        for arm,head in heads.items():
            if tensor_digest(head.state_dict().items())!=head_records[arm]['head_state_sha256']:
                raise RuntimeError('Read-only head replay changed its state')
        report.update(complete=True,protected_before_after_exact=True,elapsed_seconds=time.time()-started,
                      native_boundary_monitor_SHA_parity_all12=True)
        write(output/'report.json',report)
        write(output/'status.json',dict(status='complete',completed_images=4,maximum_images=4,training=False))
        write(output/'artifact_manifest.json',dict(files={p.relative_to(output).as_posix():sha(p)
              for p in output.rglob('*') if p.is_file()}))
    except Exception as error:
        write(output/'status.json',dict(status='failed',completed_images=len(report['images']),error=str(error),training=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/affinity_gt_source.yaml')
    parser.add_argument('--output',default='outputs/gt_source/boundary_probe')
    run(parser.parse_args())


if __name__=='__main__':
    main()
