# -*- coding: utf-8 -*-
"""固定C原生区域的关系监督盘点/分片；不运行模型、WS或训练。"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.boundary_core_supervision import build_trusted_cores
from utils.boundary_relation_supervision import (
    KEEP,MERGE,IGNORE,adjacent_region_pairs,label_region_pair,native_relation_patch,
    original_components,predicted_interiors,region_shape_descriptors,representative_anchor,
)

LABEL_NAMES={KEEP:'keep',MERGE:'merge',IGNORE:'ignore'}
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
def checked(entry,base,seen):
    path=Path(entry['path']);path=path if path.is_absolute() else base/path;path=path.resolve()
    digest=sha(path)
    if digest!=entry['sha256']:raise RuntimeError('Source SHA mismatch: '+str(path))
    if str(path) in seen and seen[str(path)]!=digest:raise RuntimeError('Source changed')
    seen[str(path)]=digest;return path
def load_array(entry,base,seen):
    path=checked(entry,base,seen)
    with np.load(path,allow_pickle=False) as data:array=np.ascontiguousarray(data[entry['key']])
    if entry.get('array_sha256') and array_sha(array)!=entry['array_sha256']:raise RuntimeError('Array SHA mismatch')
    return array
def size_bin(area):return 'lt200' if area<200 else '200_499' if area<500 else '500_2000' if area<=2000 else 'gt2000'
def increment(group,key,label):group.setdefault(str(key),Counter())[LABEL_NAMES[label]]+=1
def serialize_groups(groups):return {key:dict(value) for key,value in groups.items()}

def render_review(path,rgb,features,core_map,audit,pair,anchor):
    pic=features['rgb'].transpose(1,2,0).copy();mask=features['masks'];overlay=pic.copy()
    for plane,color in [(mask[0]>0,(50,230,80)),(mask[1]>0,(70,100,255)),(mask[2]>0,(255,40,40))]:
        overlay[plane]=np.rint(overlay[plane]*.55+np.asarray(color)*.45).astype(np.uint8)
    x0,y0,x1,y1=features['metadata']['native_crop_xyxy'];gt=np.zeros((256,256),np.int32)
    xa,ya,xb,yb=features['metadata']['native_valid_box_xyxy'];gt[ya-y0:yb-y0,xa-x0:xb-x0]=core_map[ya:yb,xa:xb]
    colors=np.stack([(gt*53)%191+32,(gt*97)%191+32,(gt*157)%191+32],axis=-1).astype(np.uint8);colors[gt==0]=80
    truth=np.rint(pic*.4+colors*.6).astype(np.uint8)
    canvas=np.full((326,768,3),245,np.uint8)
    for i,(name,image) in enumerate([('RGB native256',pic),('Predicted A / B / shared seam',overlay),('Original GT cores: REVIEW ONLY',truth)]):
        canvas[28:284,i*256:(i+1)*256]=image;cv2.putText(canvas,name,(i*256+4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
    text=f'{LABEL_NAMES[audit["label"]]} | pair {pair["pair_ids"]} | {audit["reason"]}'
    cv2.putText(canvas,text,(5,306),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
    ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88]);assert ok
    path.write_bytes(jpeg.tobytes())

def build_unlabeled_monitors(sources,base,seen,output):
    """旧C固定ROI附近选缝；只有无GT输入，labels255不进入训练分片。"""
    shards=[];selected=[]
    for source in sources:
        assert source['kind']=='monitor_unlabeled' and source['source'].startswith('test_')
        assert not any(key in source for key in ('GT','GT_classes','original_covered','filled','residual_unknown'))
        name=source['source'];image_path=checked(source['image'],base,seen)
        rgb=cv2.imdecode(np.frombuffer(image_path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)[...,::-1].copy()
        N1=load_array(source['N1'],base,seen);NAF=load_array(source['NAF'],base,seen);ids=load_array(source['C_final200'],base,seen)
        assert ids.dtype==np.uint16 and ids.shape==N1.shape==NAF.shape==rgb.shape[:2]
        assert N1.dtype==NAF.dtype==np.float32 and np.isfinite(N1).all() and np.isfinite(NAF).all()
        pairs,_=adjacent_region_pairs(ids);h,w=ids.shape;features=[];samples=[]
        for roi in source['monitor_ROIs']:
            x0,y0,x1,y1=roi['box'];assert 0<=x0<x1<=w and 0<=y0<y1<=h
            cx,cy=(x0+x1-1)/2,(y0+y1-1)/2;choices=[]
            for pair in pairs:
                yy,xx=np.divmod(pair['seam_flat'],w);inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
                indices=np.flatnonzero(inside) if inside.any() else np.arange(len(xx))
                distance=(xx[indices]-cx)**2+(yy[indices]-cy)**2
                local=int(np.lexsort((xx[indices],yy[indices],distance))[0]);i=int(indices[local])
                choices.append((not bool(inside.any()),float(distance[local]),*pair['pair_ids'],int(yy[i]),int(xx[i]),pair))
            if not choices:
                selected.append(dict(source=name,ROI=roi,status='no_C_adjacency_anywhere_no_monitor_generated'));continue
            choice=min(choices,key=lambda entry:entry[:-1]);fallback,d,_,_,ay,ax,pair=choice
            feature=native_relation_patch(rgb,N1,NAF,ids,pair,(ax,ay));sample_id=f'{name}:monitor:{roi["kind"]}:{pair["pair_ids"][0]}:{pair["pair_ids"][1]}:{ax}:{ay}'
            meta=dict(sample_id=sample_id,source=name,label=IGNORE,pair_ids=pair['pair_ids'],
                native_selection='nearest shared-seam pixel to frozen ROI center; prefer a pair touching ROI; lexicographic tie break',
                ROI=roi,ROI_fallback_to_nearest_whole_C_pair=bool(fallback),squared_distance_to_ROI_center=d,
                no_GT_or_accuracy_judgment=True,**feature['metadata'])
            features.append(feature);samples.append(meta);selected.append(meta)
        if features:
            folder=output/'monitor_patches';folder.mkdir(exist_ok=True);path=folder/(name+'.npz')
            np.savez_compressed(path,rgb=np.stack([f['rgb'] for f in features]),maps=np.stack([f['maps'] for f in features]),
                masks=np.stack([f['masks'] for f in features]),labels=np.full(len(features),IGNORE,np.uint8))
            shards.append(dict(path=path.relative_to(output).as_posix(),sha256=sha(path),source=name,count=len(samples),samples=samples))
    return shards,selected


def run(args):
    manifest_path=Path(args.sources).resolve();source_manifest=read(manifest_path);base=manifest_path.parent
    assert source_manifest['format']=='boundary_relation_sources_v1'
    output=Path(args.output).resolve();assert not output.exists();output.mkdir(parents=True)
    for name in ('pairs','patches','gallery'): (output/name).mkdir()
    write(output/'sources.json',source_manifest)
    seen={str(manifest_path):sha(manifest_path)};started=time.perf_counter();all_rows=[];shards=[];all_meta=[];funnel=Counter();by_size={};by_source={};by_phase={};source_stats=[];gallery=[]
    for source in source_manifest['sources']:
        assert source['kind']=='train' and source['source'].startswith('train_'),'No test GT / unlabeled inputs in supervised builder'
        name=source['source'];t=time.perf_counter()
        image_path=checked(source['image'],base,seen);bgr=cv2.imdecode(np.frombuffer(image_path.read_bytes(),np.uint8),cv2.IMREAD_COLOR);assert bgr is not None
        rgb=np.ascontiguousarray(bgr[...,::-1]);N1=load_array(source['N1'],base,seen);NAF=load_array(source['NAF'],base,seen)
        ids=load_array(source['C_final200'],base,seen);assert ids.dtype==np.uint16 and ids.shape==N1.shape==NAF.shape==rgb.shape[:2]
        assert N1.dtype==NAF.dtype==np.float32 and np.isfinite(N1).all() and np.isfinite(NAF).all()
        gp=checked(source['GT'],base,seen)
        with np.load(gp,allow_pickle=False) as z:
            gt=np.ascontiguousarray(z['instance_map']);original=z['original_covered']>0;filled=z['filled']>0;unknown=z['residual_unknown']>0
        GT_classes=read(checked(source['GT_classes'],base,seen)) if source.get('GT_classes') else {}
        assert gt.shape==ids.shape and not np.any(original&filled) and not np.any(unknown&(gt>0))
        cores=build_trusted_cores(gt,original,radius=4,min_component_pixels=64);components,component_rows=original_components(gt,original)
        interior=predicted_interiors(ids,2);pairs,pair_audit=adjacent_region_pairs(ids);areas,perimeters=region_shape_descriptors(ids)
        counts=Counter();items=[];feature_arrays=[];review_count=Counter();source_meta=[]
        for pair in pairs:
            audit=label_region_pair(ids,pair,gt,original,filled,unknown,cores['core_map'],components,interiors=interior)
            label=audit['label'];counts[LABEL_NAMES[label]]+=1;funnel['all_native_adjacent_pairs']+=1
            sides=[audit['side_A'],audit['side_B']]
            phase=[GT_classes.get(str(side['dominant_GT_id'])) for side in sides]
            if all(value in (0,1) for value in phase):
                phase_key='|'.join(sorted('P' if value==0 else 'F' for value in phase));increment(by_phase,phase_key,label)
                audit['dominant_GT_phase_diagnostic_only']=phase
            predicates=[all(side['side_band_pixels']>0 for side in sides),all(side['trusted_support_pixels']>=32 for side in sides),
                all(side['coverage']>=.8 for side in sides),all(side['purity']>=.95 for side in sides)]
            cumulative=True
            for key,predicate in zip(('both_inner_bands','both_support32','both_coverage080','both_purity095'),predicates):
                cumulative=cumulative and predicate;funnel[key]+=int(cumulative)
            a,b=pair['pair_ids'];anchor=representative_anchor(pair,ids.shape[1]);sample_id=f'{name}:{a}:{b}:{anchor[0]}:{anchor[1]}'
            bins=sorted([size_bin(int(areas[a])),size_bin(int(areas[b]))]);joint='|'.join(bins)
            strength=dict(N1=float(N1.ravel()[pair['seam_flat']].mean()),NAF=float(NAF.ravel()[pair['seam_flat']].mean()))
            thin=min(2*float(areas[i])/max(1,int(perimeters[i])) for i in (a,b))
            metadata=dict(sample_id=sample_id,source=name,label=int(label),pair_ids=[a,b],anchor_native_xy=list(anchor),
                seam_pixels=pair['seam_pixels'],seam_bbox_xyxy=pair['seam_bbox_xyxy'],region_areas=[int(areas[a]),int(areas[b])],
                region_area_bins=bins,thin_side_proxy=thin,thin_proxy_definition='min(2*predicted area / four-neighbor perimeter); diagnostic only',
                seam_strength=strength,labelaudit=audit)
            items.append(metadata);all_rows.append(metadata);increment(by_size,joint,label);increment(by_source,name,label)
            if (label!=IGNORE and not args.inventory_only) or review_count[label]<2:
                features=native_relation_patch(rgb,N1,NAF,ids,pair,anchor)
                if review_count[label]<2:
                    filename=f'{name}_{a}_{b}.jpg';render_review(output/'gallery'/filename,rgb,features,cores['core_map'],audit,pair,anchor)
                    gallery.append(dict(source=name,label=int(label),file=filename,sha256=sha(output/'gallery'/filename),sample_id=sample_id));review_count[label]+=1
                if label!=IGNORE and not args.inventory_only:
                    metadata.update(features['metadata']);feature_arrays.append(features);source_meta.append(metadata);all_meta.append(metadata)
        pair_path=output/'pairs'/(name+'.json');write(pair_path,dict(source=name,candidate_definition=pair_audit,rows=items,
            labels_relative_to_original_GT_only=True,GT_not_model_input=True))
        if feature_arrays:
            npz=output/'patches'/(name+'.npz')
            np.savez_compressed(npz,rgb=np.stack([f['rgb'] for f in feature_arrays]),maps=np.stack([f['maps'] for f in feature_arrays]),
                masks=np.stack([f['masks'] for f in feature_arrays]),labels=np.asarray([m['label'] for m in source_meta],np.uint8))
            shards.append(dict(path=npz.relative_to(output).as_posix(),sha256=sha(npz),source=name,count=len(source_meta),samples=source_meta))
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        stat=dict(source=name,shape=list(ids.shape),pairs=len(pairs),labels=dict(counts),core_summary=cores['summary'],
            raw_original_components=len(component_rows),C_final_instances=int((areas>0).sum()-int(areas[0]>0)),
            brightness_mean=float(gray.mean()),texture_laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_32F).var()),
            pair_receipt=dict(path=pair_path.relative_to(output).as_posix(),sha256=sha(pair_path)),seconds=time.perf_counter()-t)
        if source.get('C_pre200'):
            before=load_array(source['C_pre200'],base,seen);stat['C_pre200_instances']=int(len(np.unique(before[before>0])))
        source_stats.append(stat)
        print('SOURCE_DONE',name,len(pairs),dict(counts),round(stat['seconds'],3),flush=True)
    labels=Counter(LABEL_NAMES[row['label']] for row in all_rows)
    coverage={label:len({row['source'] for row in all_rows if LABEL_NAMES[row['label']]==label}) for label in ('keep','merge','ignore')}
    grouped={};cutpoints={}
    def quantile_group(name,values,rows,lookup):
        cuts=np.quantile(values,[1/3,2/3]).tolist() if len(values) else [0.,0.];cutpoints[name]=cuts;bucket={}
        for row in rows:increment(bucket,int(np.searchsorted(cuts,lookup(row),side='right')),row['label'])
        grouped[name]=serialize_groups(bucket)
    stat_lookup={s['source']:s for s in source_stats}
    for key in ('brightness_mean','texture_laplacian_variance'):
        quantile_group(key,[s[key] for s in source_stats],all_rows,lambda row,k=key:stat_lookup[row['source']][k])
    for key in ('N1','NAF'):
        quantile_group(key+'_seam_strength',[row['seam_strength'][key] for row in all_rows],all_rows,lambda row,k=key:row['seam_strength'][k])
    quantile_group('thin_side_proxy',[row['thin_side_proxy'] for row in all_rows],all_rows,lambda row:row['thin_side_proxy'])
    readiness=dict(keep_pairs=labels['keep'],merge_pairs=labels['merge'],keep_sources=coverage['keep'],merge_sources=coverage['merge'],
        minimum_each_class_pairs=64,minimum_each_class_sources=4,
        eligible_by_count_only=labels['keep']>=64 and labels['merge']>=64 and coverage['keep']>=4 and coverage['merge']>=4,
        no_generalization_or_GT_perfection_claim=True)
    inventory=dict(format='boundary_relation_inventory_v1',complete=True,inventory_only=args.inventory_only,
        source_manifest_sha256=sha(manifest_path),builder_sha256=sha(Path(__file__)),
        supervision_module_sha256=sha(ROOT/'utils/boundary_relation_supervision.py'),trusted_core_module_sha256=sha(ROOT/'utils/boundary_core_supervision.py'),
        sources=source_stats,unique_pair_labels=dict(labels),unique_pair_source_coverage=coverage,side_funnel=dict(funnel),
        ignore_reasons=dict(Counter(row['labelaudit']['reason'] for row in all_rows if row['label']==IGNORE)),
        by_pair_size=serialize_groups(by_size),by_source=serialize_groups(by_source),by_GT_phase=serialize_groups(by_phase),by_tertile=grouped,tertile_cutpoints=cutpoints,
        quantile_reference='only the sources/pairs in this completed inventory; descriptive bins, not physical sample classes',
        unique_pairs=len(all_rows),patch_exposures=sum(s['count'] for s in shards),one_anchor_per_pair=True,
        readiness=readiness,gallery=gallery,no_forward=True,no_watershed=True,no_training=True,
        filled_unknown_not_used_as_GT_core=True,merge_actual_contacts_must_be_original=True,
        valid_channel_is_native_extent_only=True,no_new_holdout=True,source_count=len(source_stats),
        frozen_sources=seen,seconds=time.perf_counter()-started)
    for path,digest in seen.items():
        if sha(path)!=digest:raise RuntimeError('Frozen input changed')
    write(output/'inventory.json',inventory)
    if not args.inventory_only:
        if source_manifest.get('monitor_ROI_receipt'):checked(source_manifest['monitor_ROI_receipt'],base,seen)
        monitor_shards,monitor_selection=build_unlabeled_monitors(source_manifest.get('monitor_sources',[]),base,seen,output)
        for path,digest in seen.items():
            if sha(path)!=digest:raise RuntimeError('Frozen monitor input changed')
        monitor=[];train_selection=[]
        for label in (KEEP,MERGE):
            by=defaultdict(list)
            for row in all_meta:
                if row['label']==label:by[row['source']].append(row)
            available=sorted(by,key=lambda name:(stat_lookup[name]['brightness_mean'],name))
            indices=np.rint(np.linspace(0,len(available)-1,min(6,len(available)))).astype(int) if available else []
            for index in indices:
                name=available[index];position=len(by[name])//2;row=by[name][position]
                monitor.append(row['sample_id']);train_selection.append(dict(sample_id=row['sample_id'],source=name,label=label,
                    source_pair_index_zero_based=position,available_label_pairs_this_source=len(by[name]),
                    brightness_mean=stat_lookup[name]['brightness_mean'],texture_laplacian_variance=stat_lookup[name]['texture_laplacian_variance'],
                    region_area_bins=row['region_area_bins'],thin_side_proxy=row['thin_side_proxy'],seam_strength=row['seam_strength']))
        data=dict(format='boundary_relation_data_v1',complete=True,patch_size=256,shards=shards,monitor_shards=monitor_shards,
            monitor=dict(train_sample_ids=monitor,train_selection=train_selection,
                train_selection_rule='each label: available sources brightness-sorted, up to6 equally spaced distinct sources, upper median pair index per source; no duplicate source padding',
                physical_six_organization_coverage_not_claimed=True,unlabeled_selection=monitor_selection),
            inventory=dict(path='inventory.json',sha256=sha(output/'inventory.json')),training_eligibility=readiness,
            raw_shard_bytes=sum(s['count']*720897 for s in shards),no_holdout=True)
        write(output/'manifest.json',data)
    print('INVENTORY_COMPLETE',dict(labels),readiness,round(inventory['seconds'],3),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sources',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--inventory-only',action='store_true');args=parser.parse_args();cv2.setNumThreads(2)
    previous=cv2.watershed
    def forbidden(*a,**k):raise RuntimeError('Data builder must not run watershed')
    cv2.watershed=forbidden
    try:run(args)
    finally:cv2.watershed=previous

if __name__=='__main__':main()
