# -*- coding: utf-8 -*-
"""既有N1八源缓存的浅盆地独立候选，禁止模型前向/训练/测试GT。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(ids,classes):
    assert ids.dtype==np.uint16 and ids.ndim==2 and int(ids.max(initial=0))<=65535
    assert set(classes)=={str(int(i)) for i in np.unique(ids) if i}
    assert all(type(v)is int and v in (0,1) for v in classes.values())
    sizes=np.bincount(ids.ravel());result=dict(instances=len(classes))
    for label,c in [('F',1),('P',0)]:
        inds=[int(k) for k,v in classes.items() if v==c];a=sizes[inds]
        result[label]=dict(count=len(inds),pixels=int(a.sum()),under500=int((a<500).sum()))
    result['zero_pixels']=int(sizes[0]);return result


def source_gt(row,shape,cores_fn):
    if not row.get('gt'):return None
    assert row['source'].startswith('train_'), 'Never inspect test GT'
    with np.load(row['gt'],allow_pickle=False) as z:
        ids=z['instance_map'].copy();original=z['original_covered'].astype(bool)
        filled=z['filled'].astype(bool);unknown=z['residual_unknown'].astype(bool)
    assert ids.shape==shape and np.array_equal(unknown,ids==0)
    assert np.array_equal(filled,(ids>0)&~original) and not np.any(original&unknown)
    return dict(ids=ids,original=original,filled=filled,unknown=unknown,
                cores=cores_fn(ids,original,radius=4,min_component_pixels=64))


def annotate_depths(info,markers,gt):
    if gt is None:return None
    core=gt['cores']['core_map'];byid={r['seed_id']:r for r in info['seeds']}
    groups={};rows=[]
    for r in info['seeds']:
        x,y=r['anchor_xy'];gid=int(core[y,x]);elder=r['elder_seed_id']
        if elder is None:other=0
        else:
            ex,ey=byid[elder]['anchor_xy'];other=int(core[ey,ex])
        kind=('two_core_same_GT' if gid and other and gid==other else
              'two_core_different_GT' if gid and other else 'unreliable_or_root')
        groups.setdefault(kind,[]).append(r['depth'])
        rows.append(dict(seed_id=r['seed_id'],anchor_core_GT=gid,elder_core_GT=other,
                         relation=kind,depth=r['depth'],birth=r['birth']))
    summary={}
    for k,vals in groups.items():
        v=np.array([x for x in vals if x is not None],np.float64)
        summary[k]=dict(total=len(vals),finite=len(v),below005=int((v<.05).sum()),
            below010=int((v<.10).sum()),median=float(np.median(v)) if len(v) else None)
    return dict(groups=summary,seeds=rows,
        note='Anchor core relations from seen incomplete GT; not verified physical truth or complete grain accuracy.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--basin-module',type=Path,
        default=Path(__file__).resolve().parents[1]/'utils/marker_basin_depth.py')
    args=p.parse_args();root=args.run_root.resolve();out=args.output.resolve()
    assert not (out/'complete.json').exists() and not (out/'results').exists(),'Preserve prior results'
    out.mkdir(parents=True,exist_ok=True)
    (out/'gallery').mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'code'))
    from utils.config import load_config
    from utils.image_io import read_image,write_image
    from utils.boundary_core_supervision import build_trusted_cores,compare_marker_atoms
    from tools.semantic_marker_diagnostic import marker_stages
    from tools.probe_affinity_marker_replay import build_marker_variant,decode_marker_replay
    from tools.semantic_crossover import revote_instances
    from tools.boundary_restore_monitor import panels,_heat,_overlay
    spec=importlib.util.spec_from_file_location('independent_marker_basin',args.basin_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    seed_depth=module.seed_basin_depths;filter_depth=module.filter_shallow_seeds
    torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest_path=root/'capture/manifest.json';manifest=read(manifest_path)
    assert manifest['complete'] and len(manifest['rows'])==8
    hashes={str(manifest_path):sha(manifest_path),str(args.basin_module):sha(args.basin_module),
            str(Path(__file__).resolve()):sha(__file__)}
    records=[];start=time.monotonic()
    try:
        for row in manifest['rows']:
            name=row['source'];record=dict(source=name,source_group=row.get('source_group'),arms={})
            for k in ['cache','config','N1_png','N1_classes','gt','gt_classes']:
                if row.get(k):
                    actual=sha(row[k]);assert not row.get(k+'_sha256') or row[k+'_sha256']==actual,(name,k)
                    hashes[row[k]]=actual
            config=load_config(row['config'])
            with np.load(row['cache'],allow_pickle=False) as z:
                a={k:z[k].copy() for k in ['rgb','semantic','raw_probability','boundary_before']}
            B=a['boundary_before'];shape=B.shape
            assert B.dtype==np.float32 and a['semantic'].dtype==np.float32
            assert a['semantic'].shape==(1,1,*shape) and a['raw_probability'].shape==shape
            fixed={k:array_sha(v) for k,v in a.items()}
            stages=build_marker_variant(marker_stages(B,config),'reflect',32)
            markers=stages['marker_labels'];ts=time.monotonic();info=seed_depth(markers,B)
            record.update(depth_analysis_seconds=time.monotonic()-ts,depths=info)
            print(json.dumps(dict(source=name,phase='depth',seconds=record['depth_analysis_seconds'],
                seeds=info['input_seed_count'],visited=info['tree_visited_nodes'],
                below005=sum(r['depth']is not None and r['depth']<.05 for r in info['seeds']),
                below010=sum(r['depth']is not None and r['depth']<.10 for r in info['seeds']))),flush=True)
            gt=source_gt(row,shape,build_trusted_cores)
            record['depth_GT_relation']=annotate_depths(info,markers,gt)
            original_png=read_image(row['N1_png'],cv2.IMREAD_UNCHANGED);original_classes=read(row['N1_classes'])
            outputs={};baseline_elevation=None
            for label,h in [('N1',0.),('h005',.05),('h010',.10)]:
                selected,audit=filter_depth(markers,B,h,analysis=info)
                if label!='N1' and audit['markers_unchanged']:
                    ids=outputs['N1']['ids'].copy();classes=dict(outputs['N1']['classes'])
                    replay_elevation=baseline_elevation;capture=dict(no_changed_markers_reuse_exact_baseline=True)
                else:
                    replay=decode_marker_replay(a['semantic'],torch.from_numpy(B)[None,None],config,selected)
                    assert np.array_equal(replay['actual_markers'],selected)
                    assert replay['capture_audit']['real_barrier_changed']is False
                    ids=replay['instances'];classes,_=revote_instances(ids,a['raw_probability'])
                    replay_elevation=replay['elevation_sha256'];capture=replay['capture_audit']
                if label=='N1':
                    assert np.array_equal(ids,original_png) and classes==original_classes,(name,'baseline')
                    baseline_elevation=replay_elevation
                assert replay_elevation==baseline_elevation
                result=stats(ids,classes)
                outputs[label]=dict(ids=ids,classes=classes)
                arm=dict(statistics=result,filter=audit,reference_exact=label=='N1',
                    capture_audit=capture,elevation_sha256=replay_elevation,
                    boundary_unchanged=True,semantic_raw_unchanged=True)
                if gt is not None:
                    tracking=compare_marker_atoms(gt['cores'],outputs['N1']['ids'],ids)
                    arm['GT_core_tracking']=tracking
                    arm['provenance']={k:dict(pixels=int(v.sum()),old_positive_to_zero=int((v&(original_png>0)&(ids==0)).sum()))
                        for k,v in [('original',gt['original']),('filled',gt['filled'])]}
                dest=out/'results'/label;dest.mkdir(parents=True,exist_ok=True)
                write_image(dest/(name+'_inst.png'),ids);write(dest/(name+'_class.json'),classes)
                record['arms'][label]=arm
                print(json.dumps(dict(source=name,phase='readout',arm=label,removed=audit['removed_seed_count'],
                    instances=result['instances'])),flush=True)
            x,y,x1,y1=map(int,row['roi']);sl=np.s_[y:y1,x:x1]
            panels(out/'gallery'/(name+'_outlines.jpg'),name+' | original B; only existing seed depth filter',
                [('RGB',a['rgb'][sl])]+[(k,_overlay(a['rgb'][sl],v['ids'][sl])) for k,v in outputs.items()],280,
                'N1 / depth .05 / depth .10. No test GT, area or class condition; same B and watershed height.')
            mark=a['rgb'].copy();removed=np.isin(markers,record['arms']['h010']['filter']['removed_ids'])
            mark[removed]=(.25*mark[removed]+.75*np.array([255,30,30])).astype(np.uint8)
            panels(out/'gallery'/(name+'_seeds.jpg'),name+' | native RGB and continuous B',
                [('RGB',a['rgb'][sl]),('B',_heat(B[sl])),('Revoked seed pixels at .10',mark[sl])],280)
            assert fixed=={k:array_sha(v) for k,v in a.items()},'Input arrays mutated'
            records.append(record);write(out/'progress.json',dict(complete=False,source_count=len(records),records=records))
        for f,h in hashes.items():assert sha(f)==h,('Input changed',f)
        totals={}
        for label in ['N1','h005','h010']:
            rr=[r['arms'][label] for r in records]
            totals[label]=dict(instances=sum(x['statistics']['instances'] for x in rr),
                F_count=sum(x['statistics']['F']['count'] for x in rr),
                P_count=sum(x['statistics']['P']['count'] for x in rr),
                F_under500=sum(x['statistics']['F']['under500'] for x in rr),
                removed_seeds=sum(x['filter']['removed_seed_count'] for x in rr))
        result=dict(complete=True,source_count=len(records),records=records,totals=totals,
            inputs_sha256=hashes,elapsed_seconds=time.monotonic()-start,
            no_model_forward=True,no_training=True,no_full150=True,no_test_GT=True,
            no_proxy_score=True,no_package=True,existing_deployment_unchanged=True,
            interpretation='Mechanism/seed-persistence candidate only; observed GT can omit boundaries.')
        write(out/'complete.json',result)
        print(json.dumps(dict(complete=True,totals=totals,seconds=result['elapsed_seconds'])),flush=True)
    except Exception as exc:
        write(out/'failed.json',dict(complete=False,error=repr(exc),completed_sources=len(records)))
        raise


if __name__=='__main__':main()
