# -*- coding: utf-8 -*-
"""冻结control，对连续界面选位作完整部署干预，同时审计新增错误。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.affinity_connectivity import resize_trusted, undo_spatial
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from models.backend_adaptation import load_backend, restore_first
from tools.probe_affinity_chain import CONTROL_SHA, decode_capture, inject, relations_at, trusted_targets
from tools.run_affinity_preserve import trusted_changes
from tools.render_backend_ablation import overlay, text_line, write_png
from tools.semantic_crossover import revote_instances
from train_affinity_connectivity import auxiliary_loss, get_restorer
from train_backend_adaptation import fusion_options, sha, tensor_digest, write_json
from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_fusion import affinity_boundary_probability
from utils.config import load_config, project_path
from utils.offset_letterbox import geometry_letterbox_metadata, letterbox_instance_geometry


def render(path, rgb, before, after, bc, ac, boundary, changed_boundary, selection, title):
    size = (384, round(384 * rgb.shape[0] / rgb.shape[1]))
    small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size)
    panels = []
    bodies = [small, cv2.applyColorMap(np.rint(cv2.resize(boundary, size)*255).astype(np.uint8), cv2.COLORMAP_INFERNO),
              overlay(small, cv2.resize(before, size, interpolation=cv2.INTER_NEAREST), {int(k):v for k,v in bc.items()}),
              cv2.applyColorMap(np.rint(cv2.resize(selection, size)*255).astype(np.uint8), cv2.COLORMAP_INFERNO),
              cv2.applyColorMap(np.rint(cv2.resize(changed_boundary, size)*255).astype(np.uint8), cv2.COLORMAP_INFERNO),
              overlay(small, cv2.resize(after, size, interpolation=cv2.INTER_NEAREST), {int(k):v for k,v in ac.items()})]
    labels = [title, 'Control boundary 0..1', f'Control N={len(bc)}', 'Supervised positions (not GT image)',
              'Injected boundary 0..1', f'Counterfactual N={len(ac)}']
    for body, label in zip(bodies, labels):
        panel = cv2.copyMakeBorder(body, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        text_line(panel, label, (6,21), size[0], size=.43); panels.append(panel)
    write_png(path, np.concatenate([np.concatenate(panels[:3],1), np.concatenate(panels[3:],1)],0))


def diagnose(model, restorer, tensor, gt, trust, config, seed, reference):
    shape = gt.shape
    meta = geometry_letterbox_metadata(shape, 1024, 512)
    native = lambda value: crop_letterbox_output(value, 1024, 1024-meta.resized_height,
        1024-meta.resized_width, shape).cpu()
    grid, content, _ = letterbox_instance_geometry(gt, 1024, 512)
    trusted = resize_trusted(trust)[0]
    target, valid = trusted_targets(grid, trusted, content)
    with torch.no_grad():
        image = tensor[None].cuda().float()
        raw = native(model.semantic_decoder(model.encoder(image), image))
        prediction = model(restore_first(restorer, image, seed))
        logits = prediction['affinity_logits'].float().cpu()
        semantic = native(prediction['semantic_logits'].float())
    mode, kwargs = fusion_options(config)
    boundary = native(affinity_boundary_probability(logits, mode=mode, **kwargs))
    before, _, _, _ = decode_capture(semantic, boundary, config)
    points = reference['relations']
    expected = [r['stages']['base']['final']['full'] for r in points]
    if relations_at(before, points) != expected or int(before.max()) != reference['variants']['base']['instances']:
        raise AssertionError('Fixed baseline does not reproduce previous probe')
    pred_grid, _, _ = letterbox_instance_geometry(before, 1024, 512)
    leaf = logits.clone().requires_grad_(True)
    loss, sampling = auxiliary_loss(config, leaf, target, valid,
        torch.from_numpy(pred_grid.astype(np.int64))[None], torch.from_numpy(grid.astype(np.int64))[None], trusted[None])
    gradient = torch.autograd.grad(loss, leaf)[0]
    selected = gradient != 0
    if bool((selected & ~valid).any()) or bool(selected[:,4:].any()) or int(selected.sum()) != sampling['selected_edges']:
        raise AssertionError('Selected gradient escapes trusted short relations or differs from audit')
    if not bool(torch.isfinite(gradient).all()):
        raise AssertionError('Nonfinite selection gradient')
    changed, _ = inject(logits, target, selected)
    if not torch.equal(changed[~selected], logits[~selected]):
        raise AssertionError('Changed unselected relation')
    after_boundary = native(affinity_boundary_probability(changed, mode=mode, **kwargs))
    after, _, _, _ = decode_capture(semantic, after_boundary, config)
    probability = raw.sigmoid()[0,0].numpy()
    bc, _ = revote_instances(before, probability); ac, _ = revote_instances(after, probability)
    relation_after = relations_at(after, points)
    fixes = {}
    for kind, desired in [('merge','different'), ('split','same')]:
        rows = [(r, state) for r, state in zip(points, relation_after) if r['kind']==kind]
        fixes[kind] = dict(cases=len(rows), fixed=sum(state==desired for _, state in rows),
            blocked=sum(state=='blocked' for _, state in rows),
            old_selected_fixed=sum(r['stages']['selected_only']['final']['full']==desired for r,_ in rows))
    new = trusted_changes(gt, trust, before, after)
    lut0=np.zeros(int(before.max())+1,np.int8); lut1=np.zeros(int(after.max())+1,np.int8)
    for k,v in bc.items(): lut0[int(k)] = v
    for k,v in ac.items(): lut1[int(k)] = v
    common=(before>0)&(after>0)
    row=dict(sampling=sampling, fixes=fixes, new_errors=new['summary'],
        new_merge_pairs=len(new['new_merge_pairs']), new_split_pairs=new['new_split_pairs'],
        class_changed_pixels=int(((lut0[before]!=lut1[after])&common).sum()),
        common_pixels=int(common.sum()), base_instances=len(bc), candidate_instances=len(ac),
        baseline_reproduced=True, unknown_and_long_unchanged=True)
    rgb=np.rint(native(tensor[None])[0].permute(1,2,0).numpy()*255).astype(np.uint8)
    select=native(selected.float().amax(1,keepdim=True))[0,0].numpy()
    return row, (rgb,before,after,bc,ac,boundary[0,0].numpy(),after_boundary[0,0].numpy(),select)


def summarize(rows):
    result={}
    for view in sorted({r['view'] for r in rows}):
        rr=[r for r in rows if r['view']==view]
        result[view]=dict(inputs=len(rr), fixes={kind:{key:sum(r['fixes'][kind][key] for r in rr)
            for key in ['cases','fixed','blocked','old_selected_fixed']} for kind in ['merge','split']},
            new_split_pairs=sum(r['new_errors']['new_split_pairs'] for r in rr),
            new_merge_pairs=sum(r['new_merge_pairs'] for r in rr),
            images_with_fix=sum(any(r['fixes'][k]['fixed'] for k in ['merge','split']) for r in rr),
            base_uncovered_known_pixels=sum(r['new_errors']['base_uncovered_known_pixels'] for r in rr),
            candidate_uncovered_known_pixels=sum(r['new_errors']['candidate_uncovered_known_pixels'] for r in rr),
            selected_edges=sum(r['sampling']['selected_edges'] for r in rr),
            zero_endpoint_edges=sum(r['sampling'][d]['selected_zero_endpoint_edges'] for r in rr for d in ['negative','positive']),
            class_changed_pixels=sum(r['class_changed_pixels'] for r in rr),
            source_count=len({r['source'] for r in rr}))
    return result


def run(args):
    config=load_config(args.config)
    out=Path(project_path(config,args.output));out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    checkpoint=project_path(config,config['affinity_interface']['reused_control'],'final.pt')
    assert sha(checkpoint)==CONTROL_SHA
    model,_=load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False)
    restorer=get_restorer(config,'cuda')
    states=[tensor_digest(x.state_dict().items()) for x in [model,restorer]]
    cfg=config['backend_adaptation']
    base=CanonicalBackendDataset(project_path(config,config['paths']['raw_data_dir']),project_path(config,cfg['completed_gt_dir']))
    assert len(base)==32
    degeneration=load_config(project_path(config,cfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
    paired=PairedDegradationDataset(base,{**degeneration,'profile_probabilities':[0.,1.,0.,0.]},cfg['noise'],cfg['seed'],repeats=1)
    paired.set_epoch(20)
    references={}
    for path in Path(project_path(config,args.reference)).rglob('views.jsonl'):
        for line in path.read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            references[(row['source'],row['view'])]=row
    write_json(out/'provenance.json',dict(config=config,checkpoint_sha256=CONTROL_SHA,source_sha256=sha(__file__),
        loss_sha256=sha(ROOT/'utils/affinity_interface.py'), no_training=True, no_holdout=True,
        selection='trusted original GT, distance1 continuous interfaces',frozen_states=states))
    rows=[]
    for index in range(args.start,min(32,args.start+(args.limit or 32))):
        clean=base[index];path=base.samples[index];sample=paired[index]
        with np.load(base.completed_gt_dir/(path.stem+'_gt.npz'),allow_pickle=False) as blob:
            gt=blob['instance_map'].copy();trust=blob['original_covered'].astype(bool)&(gt>0)
        blur=undo_spatial(sample['image'],sample['horizontal_flip'],sample['vertical_flip'],sample['rotation_k'])
        for view,tensor in [('clean',clean['image']),('degraded',blur)]:
            row,pictures=diagnose(model,restorer,tensor,gt,trust,config,cfg['restoration']['inference_seed']+index,references[(path.stem,view)])
            row.update(source=path.stem,view=view);rows.append(row)
            with (out/'views.jsonl').open('a',encoding='utf8') as stream: stream.write(json.dumps(row)+'\n')
            render(out/(path.stem+'_'+view+'.png'),*pictures,path.stem+' '+view)
            print(json.dumps({k:row[k] for k in ['source','view','fixes','new_merge_pairs']}),flush=True)
            del pictures;gc.collect()
    assert states==[tensor_digest(x.state_dict().items()) for x in [model,restorer]]
    write_json(out/'report.json',dict(complete=True,scope='trained-source mechanism, not generalization/official accuracy',
        views=len(rows),sources=len({r['source'] for r in rows}),frozen_exact=True,summary=summarize(rows)))
    print('INTERFACE PROBE COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='config/train/affinity_interface.yaml')
    p.add_argument('--output',default='outputs/interface_probe')
    p.add_argument('--reference',default='outputs/affinity_trace')
    p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int)
    run(p.parse_args())
