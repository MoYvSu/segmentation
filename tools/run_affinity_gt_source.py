# -*- coding: utf-8 -*-
"""固定已评分部署，在原32训练源生成独立无类别伪几何及公平支持域。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.backend_adaptation import CanonicalBackendDataset
from models.backend_adaptation import load_backend
from tools.probe_affinity_interior import R1_SHA, RESTORER_SHA
from tools.run_marker_reflect import (capture_original, replay_candidates, sha, write,
                                      save_prediction, precision_record, config_gate, array_sha, source_snapshot)
from train_affinity_connectivity import get_restorer, frozen_digest
from train_affinity_coverage import load_original_core
from train_backend_adaptation import tensor_digest
from utils.config import load_config, project_path
from utils.affinity_teacher_targets import load_teacher_pair, resize_teacher_pair
from tools.affinity_bottleneck_capture import capture_native_tiles
from data.semantic_targets import load_completed_semantic_source
from utils.affinity_interior_loss import build_interior_supervision
from tools.run_affinity_sweep import runtime_contract

def boundary(ids):
    out = np.zeros(ids.shape, bool)
    h = (ids[:, 1:] != ids[:, :-1]) & ((ids[:, 1:] > 0) | (ids[:, :-1] > 0))
    v = (ids[1:] != ids[:-1]) & ((ids[1:] > 0) | (ids[:-1] > 0))
    out[:, 1:] |= h; out[:, :-1] |= h
    out[1:] |= v; out[:-1] |= v
    return out

def gallery(path, rgb, manual, teacher, common, title, crop=None):
    if crop is not None:
        y,x,y1,x1 = crop
        rgb,manual,teacher,common = (v[y:y1,x:x1] for v in (rgb,manual,teacher,common))
    pieces = []
    for content, label in [(rgb.copy(), 'Original'), (rgb.copy(), 'Processed GT edges'),
                           (rgb.copy(), 'R1 reflect32 edges')]:
        if label != 'Original':
            edge = boundary(manual if label.startswith('Processed') else teacher)
            content[edge] = [255, 80, 0]
            content[~common] = (content[~common].astype(float)*.55).astype(np.uint8)
        h,w = content.shape[:2]
        thumb = cv2.resize(content,(340,max(1,round(h*340/w))),interpolation=cv2.INTER_AREA)
        thumb = cv2.copyMakeBorder(thumb,26,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(thumb,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv2.LINE_AA)
        pieces.append(thumb)
    strip = np.concatenate(pieces,axis=1)
    if not cv2.imwrite(str(path),cv2.cvtColor(strip,cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Cannot save process thumbnail: '+str(path))

def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\','/')

def prepare(config, opts):
    if not torch.cuda.is_available():
        raise RuntimeError('Actual sam2_env GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    output = Path(project_path(config, opts['output_dir'])).resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Private ignored output required')
    targets = output/'targets'
    targets.mkdir(parents=True,exist_ok=False)
    (targets/'gallery').mkdir();(targets/'teacher').mkdir();(targets/'pairs').mkdir()
    cfg = config['backend_adaptation']
    manual = CanonicalBackendDataset(project_path(config,config['paths']['raw_data_dir']),
                                    project_path(config,cfg['completed_gt_dir']),1024,512)
    whitelist = [p.stem for p in manual.samples]
    if len(whitelist)!=32 or len(set(whitelist))!=32 or any(not n.startswith('train_') for n in whitelist):
        raise ValueError('Exactly original 32 allowed training sources required')
    r1 = ROOT/'outputs/affinity_balance/mixed/final.pt'
    if sha(r1)!=R1_SHA:
        raise RuntimeError('Deployed R1 changed')
    receipt = json.loads((ROOT/'outputs/affinity_weak/capture.json').read_text(encoding='utf8'))
    config_gate(config,receipt['arms']['r1']['config'])
    model, bundle = load_backend(r1,config,'cuda')
    restorer = get_restorer(config,'cuda')
    model.eval().requires_grad_(False);restorer.eval().requires_grad_(False)
    core = load_original_core(config);views = core.coverage_original_views
    frozen = (tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items()))
    if (bundle['epoch']!=20 or frozen!=(receipt['arms']['r1']['model_state_sha256'],receipt['restorer_state_sha256'])
            or precision_record()!=receipt['precision'] or runtime_contract(config)!=receipt['current_runtime']):
        raise RuntimeError('Frozen deployment state/epoch/runtime differs from the sealed actual R1')
    protected_code,virtual_code_paths=source_snapshot(output)
    protected = {relative(r1):R1_SHA,relative(project_path(config,cfg['restoration']['checkpoint'])):RESTORER_SHA}
    structure = json.loads((ROOT/'outputs/geometry_structure/report.json').read_text(encoding='utf8'))
    known_seeds = {row['source']['source_path'].split('/')[-1].split('.')[0]:row['source']['seed']
                   for row in structure['sources']}
    sam_manifest = ROOT/cfg['sam2_geometry']['dataset_dir']/'manifest.jsonl'
    if sha(sam_manifest)!=cfg['sam2_geometry']['manifest_sha256']:
        raise RuntimeError('Authorized 64 SAM2 source manifest changed')
    protected[relative(sam_manifest)]=sha(sam_manifest)
    sam_rows = [json.loads(s) for s in sam_manifest.read_text().splitlines() if s.strip()]
    source_hashes = [sha(p) for p in manual.samples]
    unique = set(source_hashes)|{r['source_sha256'] for r in sam_rows}
    if len(set(source_hashes))!=32 or len(sam_rows)!=64 or len(unique)>=250:
        raise RuntimeError('Source identity/quota mismatch')
    manifest = dict(format='affinity_gt_source_targets_v1',status='running',
        quality_confirmation='pending_user_review',allowed_training_sources=whitelist,
        teacher_checkpoint_sha256=R1_SHA,restorer_checkpoint_sha256=RESTORER_SHA,
        teacher_readout='native1024 overlap .25, reflect32, original continuous WS elevation/raw semantic vote',
        semantic_labels_used_for_teacher_training=False,automatic_best_promotion=False,
        source_quota=dict(manual=32,old_SAM2=64,unique_pseudo_union=len(unique)),
        original_GT_modified=False,test_training=False,rows=[],protected_inputs=protected,
        protected_code=protected_code,virtual_code_paths=virtual_code_paths,
        precision=precision_record(),runtime=runtime_contract(config),config=config,
        interpretation='Hard geometry self-distillation, not independent evidence of label truth',
        valid_policy='both arms processed-known AND teacher-positive; manual unknown/teacher zero seams ignored')
    write(targets/'manifest.json',manifest)
    started=time.time()
    for index,path in enumerate(manual.samples):
        gtpath = manual.completed_gt_dir/(path.stem+'_gt.npz')
        protected[relative(path)] = source_hashes[index]
        protected[relative(gtpath)] = sha(gtpath)
        protected[relative(gtpath.with_name(path.stem+'_class.json'))] = sha(gtpath.with_name(path.stem+'_class.json'))
        seed = known_seeds.get(path.stem,cfg['restoration']['inference_seed']+index)
        if path.stem == 'train_351':
            gt,_ = load_completed_semantic_source(gtpath,cv2.imread(str(path)).shape[:2])
            with np.load(gtpath,allow_pickle=False) as z:
                original=z['original_covered'].astype(bool)
            rgb,captured,tiles,capture_audit = capture_native_tiles(views,model,restorer,path,config,gt,'cuda',seed)
            cache=output/'retention_cache';cache.mkdir(exist_ok=False)
            cache_rows=[]
            fit=json.loads((ROOT/'outputs/interior_fit/report.json').read_text(encoding='utf8'))
            historical={r['index']:r for r in fit['captures']['train_351']['supervision']}
            if array_sha(captured['baseline']['boundary'])!=fit['captures']['train_351']['original_boundary_sha256']:
                raise RuntimeError('Fresh teacher boundary differs from sealed original fit')
            for tile in tiles:
                if tile['index'] not in historical:continue
                supervision,audit=build_interior_supervision(tile,gt,original,tile['logits'],
                    gid=68,margin=4,low=.45,confidence=.9)
                if json.loads(json.dumps(audit))!=historical[tile['index']]['audit']:
                    raise RuntimeError('Fresh teacher/supervision differs from sealed fit window')
                p=cache/('tile_%02d.npz'%tile['index'])
                arrays={k:v.detach().cpu().numpy() for k,v in supervision.items() if torch.is_tensor(v)}
                np.savez_compressed(p,teacher_logits=tile['logits'].cpu().numpy(),grid_gt=tile['grid_gt'],
                    valid=tile['valid'],grid=tile['grid'],box=np.asarray(tile['box']),**arrays)
                cache_rows.append(dict(index=tile['index'],file=relative(p),file_sha256=sha(p),
                    geometry=tile['geometry'],pad_h=1024-tile['geometry']['resized_height'],
                    pad_w=1024-tile['geometry']['resized_width'],seed=tile['seed'],box=tile['box'],audit=audit))
            if not cache_rows or {r['index'] for r in cache_rows}!=set(historical):
                raise RuntimeError('Frozen protection cache misses registered training windows')
            write(cache/'cache.json',dict(format='retention_support_capture_v1',source='train_351',
                gt_path=relative(gtpath),gt_sha256=sha(gtpath),source_path=relative(path),
                source_sha256=source_hashes[index],seed=seed,tiles=cache_rows,
                checkpoint_sha256=R1_SHA,restorer_sha256=RESTORER_SHA,
                initial_head_sha256=tensor_digest(model.affinity_decoder.state_dict().items()),
                historical_fit_sha256=sha(ROOT/'outputs/interior_fit/report.json'),
                original_boundary_sha256=array_sha(captured['baseline']['boundary']),
                complete_original_stitch_exact=True,historical_fit_exact=True,
                capture_audit=capture_audit,training=False))
            del tiles,supervision
        else:
            rgb,captured = capture_original(views,model,restorer,path,config,'cuda',seed)
        _,variants,replay = replay_candidates(captured,config)
        candidate = variants['32']
        pred_hash = save_prediction(targets/'teacher',path.stem,candidate['instances'],candidate['classes'])
        pair,audit = load_teacher_pair(gtpath,targets/'teacher'/(path.stem+'_inst.png'),
                                      source_name=path.stem,allowed_training_sources=whitelist)
        _,grid_stats = resize_teacher_pair(pair,input_size=1024,output_grid=512)
        pairpath=targets/'pairs'/(path.stem+'.npz')
        np.savez_compressed(pairpath,manual_ids=pair['manual_masked'],teacher_ids=pair['teacher_masked'],
                            common_known=pair['common_known'],original_covered=pair['original_covered'],
                            filled=pair['filled'])
        gallery(targets/'gallery'/(path.stem+'.png'),rgb,pair['manual_original'],
                candidate['instances'],pair['common_known'],path.stem)
        h,w=rgb.shape[:2]
        gallery(targets/'gallery'/(path.stem+'_detail.png'),rgb,pair['manual_original'],
                candidate['instances'],pair['common_known'],path.stem,
                (max(0,h//2-256),max(0,w//2-256),min(h,h//2+256),min(w,w//2+256)))
        row=dict(source=path.stem,image=relative(path),image_sha256=source_hashes[index],
            manual_gt=relative(gtpath),manual_gt_sha256=protected[relative(gtpath)],
            pair_npz=relative(pairpath),pair_sha256=sha(pairpath),seed=seed,shape=list(rgb.shape[:2]),
            class_label=None,teacher_classes_diagnostic_only=True,mutually_exclusive=True,
            predicted_instances=len(candidate['classes']),replay=replay,prediction_hashes=pred_hash,
            audit=audit,grid_statistics=grid_stats)
        manifest['rows'].append(row)
        write(targets/'manifest.json',manifest)
        print(json.dumps({'stage':'prepared','source':path.stem,'count':len(manifest['rows']),
                          'common_fraction':float(pair['common_known'].mean())}),flush=True)
        del captured,pair,variants,candidate
        gc.collect();torch.cuda.empty_cache()
    if frozen!=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items())):
        raise RuntimeError('Frozen model/restorer mutated during generation')
    for path,digest in protected.items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Protected original modified: '+path)
    for path,digest in protected_code.items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Protected loaded source changed: '+path)
    if precision_record()!=manifest['precision'] or runtime_contract(config)!=manifest['runtime']:
        raise RuntimeError('Deployment runtime changed during generation')
    manifest.update(status='complete',frozen_unchanged=True,seconds=time.time()-started)
    write(targets/'manifest.json',manifest)
    print(json.dumps({'status':'complete','sources':len(manifest['rows']),
                      'quality_confirmation':manifest['quality_confirmation']}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/train/affinity_gt_source.yaml')
    args=parser.parse_args()
    merged=load_config(args.config);opts=merged.pop('gt_source')
    prepare(merged,opts)

if __name__=='__main__':
    main()
