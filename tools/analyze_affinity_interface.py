# -*- coding: utf-8 -*-
"""连续界面训练的完整输出与固定训练源诊断；不估计测试精度或选择checkpoint。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_affinity_connectivity import description, load_directory, read, relation, render, rows, write
from utils.config import load_config, project_path


def quantiles(values):
    return dict(zip(['min', 'p10', 'median', 'p90', 'max'], map(float, np.quantile(values, [0, .1, .5, .9, 1])))) if len(values) else {}


def train_summary(run, control):
    pipeline = read(run / 'pipeline_status.json')
    assert pipeline['status'] == 'complete' and pipeline['paired_control_updates'] == 1280
    ee, ss, result = {}, {}, {}
    for name, folder in [('control', control), ('candidate', run / 'candidate')]:
        status = read(folder / 'status.json')
        assert status['status'] == 'complete' and status['updates'] == 1280 and status['epoch'] == 20
        assert all(status[k] for k in ['frozen_unchanged', 'restorer_unchanged', 'strict_reload_equal'])
        ee[name], ss[name] = rows(folder / 'epochs.jsonl'), rows(folder / 'steps.jsonl')
        assert len(ee[name]) == 20 and len(ss[name]) == 1280
        active = [r for r in ss[name] if r['auxiliary_weight'] > 0]
        measured = [r for r in ss[name] if 'parameter_aux_ratio' in r]
        assert all(np.isfinite(r[k]) for r in ss[name] for k in ['manual_bce', 'pseudo_bce', 'auxiliary', 'grad_norm'])
        result[name] = dict(status={k: v for k, v in status.items() if k not in ['config', 'initialization', 'data']},
            last5_manual=float(np.mean([r['manual_bce'] for r in ee[name][-5:]])),
            last5_pseudo=float(np.mean([r['pseudo_bce'] for r in ee[name][-5:]])),
            active_steps=len(active), auxiliary_weights=quantiles([r['auxiliary_weight'] for r in active]),
            parameter_aux_ratio=quantiles([r['parameter_aux_ratio'] for r in measured]),
            gradient_cosine=quantiles([r['parameter_gradient_cosine'] for r in measured]))
    for a, b in zip(ss['control'], ss['candidate']):
        for k in ['epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw']:
            assert a[k] == b[k], (k, a['updates'])
    for k in ['initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256']:
        assert result['control']['status'][k] == result['candidate']['status'][k]
    return result, ee


def analyze_test(args, config, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run, control = Path(args.run), Path(args.control)
    train, epochs = train_summary(run, control)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for name, values in epochs.items():
        for ax, key in zip(axes[:2], ['manual_bce', 'pseudo_bce']):
            ax.plot([r['epoch'] for r in values], [r[key] for r in values], label=name)
            ax.set_title(key); ax.set_xlabel('Epoch'); ax.grid(alpha=.25)
    measured = [r for r in rows(run / 'candidate/steps.jsonl') if 'parameter_aux_ratio' in r]
    axes[2].plot([r['epoch'] for r in measured], [100*r['parameter_aux_ratio'] for r in measured])
    axes[2].set_title('Extra / manual parameter gradient (%)'); axes[2].set_xlabel('Epoch'); axes[2].grid(alpha=.25)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=150); plt.close(fig)
    files = sorted(Path(project_path(config, config['inference']['test_dir'])).glob('*.jpg'))
    assert len(files) == 100
    fixed = config['backend_adaptation']['monitor']['images']
    random = np.random.default_rng(20260928).choice([p.stem for p in files if p.stem not in fixed], 4, replace=False).tolist()
    selected = set(fixed + random)
    summaries = []
    for path in files:
        a, ac = load_directory(control / 'deployment', path.stem)
        b, bc = load_directory(run / 'candidate/deployment', path.stem)
        assert a.shape == b.shape and set(ac.values()).issubset({0, 1}) and set(bc.values()).issubset({0, 1})
        raw = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        assert raw.shape[:2] == a.shape
        aa, ba = np.bincount(a.ravel()), np.bincount(b.ravel())
        af = np.array([aa[int(k)] for k, v in ac.items() if v == 1]); bf = np.array([ba[int(k)] for k, v in bc.items() if v == 1])
        median = float(np.median(af)); bins = [0, median*.1, median*.25, median, np.inf]
        d = dict(image=path.stem, control=description(a, ac), candidate=description(b, bc), changes=relation(a, b, ac, bc),
            ferrite_area_bins={'control':np.histogram(af, bins)[0].tolist(), 'candidate':np.histogram(bf, bins)[0].tolist()},
            ferrite_pixels={'control':int(af.sum()), 'candidate':int(bf.sum())},
            ferrite_tiny={'control':int((af<200).sum()), 'candidate':int((bf<200).sum())},
            ferrite_mean_area_change_percent=100*(float(bf.mean())/float(af.mean())-1),
            foreground_added=int(((a==0)&(b>0)).sum()), foreground_lost=int(((a>0)&(b==0)).sum()))
        summaries.append(d)
        if path.stem in selected:
            render(path, [(a, ac), (b, bc)], ['Control e20', 'Interface e20'], out/(path.stem+'.png'), path.stem)
        if path.stem in ['test_101', 'test_116']:
            render(path, [(a, ac), (b, bc)], ['Control e20', 'Interface e20'], out/(path.stem+'_detail.png'), path.stem+' right detail', crop=(.48,.05,.95,.7), width=540)
        if len(summaries)%20 == 0: print('test',len(summaries),flush=True)
    totals = {name:{k:sum(r[name][k] for r in summaries) for k in ['instances','ferrite','pearlite','tiny_50_199','covered_pixels']} for name in ['control','candidate']}
    for name in totals:
        totals[name].update(ferrite_pixels=sum(r['ferrite_pixels'][name] for r in summaries),
            ferrite_tiny=sum(r['ferrite_tiny'][name] for r in summaries),
            ferrite_area_bins=np.sum([r['ferrite_area_bins'][name] for r in summaries],axis=0).tolist())
    comparison = {k:sum(r['changes'][k] for r in summaries) for k in ['split_relations','merge_relations','matched_iou50','matched_iou95','class_changed_matched_iou50','class_changed_pixels','common_covered_pixels']}
    comparison.update(mean_area_change_percent=quantiles([r['ferrite_mean_area_change_percent'] for r in summaries]),
        images_count_changed=sum(r['control']['instances']!=r['candidate']['instances'] for r in summaries),
        images_area_increased=sum(r['ferrite_mean_area_change_percent']>0 for r in summaries),
        images_area_decreased=sum(r['ferrite_mean_area_change_percent']<0 for r in summaries),
        foreground_added=sum(r['foreground_added'] for r in summaries),foreground_lost=sum(r['foreground_lost'] for r in summaries))
    ranked = sorted(summaries,key=lambda r:r['ferrite_mean_area_change_percent'])
    extreme = [r['image'] for r in ranked[:2]+ranked[-2:]]
    for stem in extreme:
        path = next(p for p in files if p.stem==stem)
        predictions = [load_directory(control/'deployment',stem),load_directory(run/'candidate/deployment',stem)]
        render(path,predictions,['Control e20','Interface e20'],out/(stem+'_area_extreme.png'),stem+' | selected by area change')
    write(out/'analysis.json',dict(scope='Unlabelled changes, not accuracy',training=train,totals=totals,comparison=comparison,
        images=summaries,visualized_fixed=fixed,visualized_random=random,visualized_area_extreme=extreme))
    print(json.dumps(dict(totals=totals,comparison=comparison,random=random,extreme=extreme)),flush=True)


def analyze_labelled(args, config, out):
    import torch
    from data.affinity_connectivity import undo_spatial
    from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
    from models.backend_adaptation import load_backend
    from tools.affinity_connectivity_views import decode_native, predict_cross
    from tools.probe_affinity_chain import relations_at, CONTROL_SHA
    from tools.run_affinity_preserve import native_training_maps, trusted_changes
    from tools.semantic_crossover import revote_instances
    from train_affinity_connectivity import get_restorer
    from train_backend_adaptation import sha, tensor_digest
    torch.set_num_threads(4); cv2.setNumThreads(2)
    paths = [Path(args.control)/'final.pt',Path(args.run)/'candidate/final.pt']
    assert sha(paths[0]) == CONTROL_SHA
    models = [load_backend(p,config,'cuda')[0].eval().requires_grad_(False) for p in paths]
    restorer = get_restorer(config,'cuda').eval().requires_grad_(False)
    initial = [tensor_digest(m.state_dict().items()) for m in [*models,restorer]]
    cfg = config['backend_adaptation']
    base = CanonicalBackendDataset(project_path(config,config['paths']['raw_data_dir']),project_path(config,cfg['completed_gt_dir']))
    assert len(base)==32
    deg = load_config(project_path(config,cfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
    paired = PairedDegradationDataset(base,{**deg,'profile_probabilities':[0.,1.,0.,0.]},cfg['noise'],cfg['seed'],repeats=1); paired.set_epoch(20)
    refs = {}
    for p in Path(args.trace).rglob('views.jsonl'):
        for r in rows(p): refs[(r['source'],r['view'])]=r
    assert len(refs)==64
    files = sorted(Path(project_path(config,config['inference']['test_dir'])).glob('*.jpg'))
    for i,p in enumerate(files):
        if p.stem not in ['test_101','test_116']: continue
        for m,cp in zip(models,paths):
            pred = predict_cross(m,restorer,p,config,'cuda',cfg['restoration']['inference_seed']+i)
            ids,cls = load_directory(cp.parent/'deployment',p.stem)
            assert np.array_equal(ids,pred['instances']) and cls==pred['classes']
    records = rows(out/'views.jsonl') if args.resume and (out/'views.jsonl').exists() else []
    completed = {(r['source'],r['view']) for r in records}
    assert len(completed)==len(records)
    checkpoint_hashes = [sha(p) for p in paths]
    if records:
        previous = read(out/'provenance.json')
        assert previous['sha256']==checkpoint_hashes and previous['config']==json.loads(json.dumps(config)), 'Resume source mismatch'
        allowed = {base.samples[i].stem for i in range(args.start,min(32,args.start+args.limit))}
        assert all(stem in allowed and view in ['clean','degraded'] for stem,view in completed)
    write(out/'provenance.json',dict(checkpoints=[str(p) for p in paths],sha256=checkpoint_hashes,
        code_sha256=sha(__file__),config=config,source_count=32,no_holdout=True,no_training=True,test_parity=True))
    for i in range(args.start,min(32,args.start+args.limit)):
        clean=base[i];sample=paired[i];stem=base.samples[i].stem
        with np.load(base.completed_gt_dir/(stem+'_gt.npz'),allow_pickle=False) as blob:
            gt=blob['instance_map'].copy();trust=blob['original_covered'].astype(bool)&(gt>0)
        blur=undo_spatial(sample['image'],sample['horizontal_flip'],sample['vertical_flip'],sample['rotation_k'])
        for view,tensor in [('clean',clean['image']),('degraded',blur)]:
            if (stem,view) in completed: continue
            maps=[native_training_maps(m,restorer,tensor,gt.shape,config,cfg['restoration']['inference_seed']+i) for m in models]
            assert torch.equal(maps[0][1],maps[1][1]) and np.array_equal(maps[0][3],maps[1][3])
            ids=[decode_native(t[1],t[2],config)[0] for t in maps]
            reference=refs[(stem,view)];rr=reference['relations']
            assert relations_at(ids[0],rr)==[r['stages']['base']['final']['full'] for r in rr]
            assert int(ids[0].max())==reference['variants']['base']['instances']
            after=relations_at(ids[1],rr);fixes={}
            for kind,desired in [('merge','different'),('split','same')]:
                states=[s for r,s in zip(rr,after) if r['kind']==kind]
                fixes[kind]=dict(cases=len(states),fixed=sum(s==desired for s in states),blocked=sum(s=='blocked' for s in states))
            changes=trusted_changes(gt,trust,*ids)
            r=dict(source=stem,view=view,fixes=fixes,new_errors=changes['summary'],
                base_instances=int(ids[0].max()),candidate_instances=int(ids[1].max()),
                boundary_mean_abs_delta=float((maps[0][2]-maps[1][2]).abs().mean()),
                boundary_threshold_pixels=[int((t[2]>.65).sum()) for t in maps],
                reproduced_baseline=True,semantic_exact=True)
            records.append(r)
            with (out/'views.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
            print(json.dumps({k:r[k] for k in ['source','view','fixes']}),flush=True)
            del maps,ids,changes;gc.collect()
    assert initial==[tensor_digest(m.state_dict().items()) for m in [*models,restorer]]
    summary={}
    for view in ['clean','degraded']:
        rr=[r for r in records if r['view']==view]
        summary[view]=dict(inputs=len(rr),fixes={kind:{k:sum(r['fixes'][kind][k] for r in rr) for k in ['cases','fixed','blocked']} for kind in ['merge','split']},
            new_split_pairs=sum(r['new_errors']['new_split_pairs'] for r in rr),
            new_merge_pairs=sum(r['new_errors']['fixed_anchor_merges'].get('new_pairs',0) for r in rr),
            images_with_fix=sum(any(r['fixes'][k]['fixed'] for k in ['merge','split']) for r in rr),
            gt_more_fragments=sum(r['new_errors']['gt_regions_more_fragments'] for r in rr),
            gt_fewer_fragments=sum(r['new_errors']['gt_regions_fewer_fragments'] for r in rr),
            base_uncovered_known_pixels=sum(r['new_errors']['base_uncovered_known_pixels'] for r in rr),
            candidate_uncovered_known_pixels=sum(r['new_errors']['candidate_uncovered_known_pixels'] for r in rr))
    write(out/'report.json',dict(complete=True,scope='Fixed trained sources; not held-out accuracy or test accuracy',views=len(records),frozen_exact=True,summary=summary))
    print('LABELLED COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['test','labelled'],required=True)
    p.add_argument('--config',default='config/train/affinity_interface.yaml')
    p.add_argument('--run',default='outputs/affinity_interface')
    p.add_argument('--control',default='outputs/affinity_connectivity/control')
    p.add_argument('--trace',default='outputs/affinity_trace')
    p.add_argument('--output',required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int,default=32)
    p.add_argument('--resume',action='store_true',help='仅继续同一权重和数据的中断诊断；不用于训练')
    args=p.parse_args();config=load_config(args.config)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=args.resume)
    (analyze_test if args.mode=='test' else analyze_labelled)(args,config,out)
