# -*- coding: utf-8 -*-
"""冻结已验证control，沿完整部署链做训练GT可信域的反事实定位；不训练或选模。"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scipy.ndimage import find_objects

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.affinity_connectivity import resize_trusted, undo_spatial
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from models.backend_adaptation import load_backend, restore_first
from tools.affinity_connectivity_views import decode_native, predict_cross
from tools.render_backend_ablation import overlay, text_line, write_png
from tools.semantic_crossover import revote_instances
from tools.semantic_marker_diagnostic import marker_stages
from train_affinity_connectivity import get_restorer
from train_backend_adaptation import fusion_options, sha, tensor_digest
from utils.affinity_connectivity import _line_points, deployment_connectivity_loss
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.offset_letterbox import geometry_letterbox_metadata, letterbox_instance_geometry
from utils.post_process import _boundary_skeleton_belt

CONTROL_SHA = 'e9fd8eb36dc40c538a5b090e9ab059e8358dc0ef2565a95d1efc924c0d93140b'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def trusted_targets(ids, trust, content):
    """整段端点关系均在原标可信域；补缝、未知和padding不被替换。"""
    ids = torch.as_tensor(ids, dtype=torch.long)[None]
    trust = torch.as_tensor(trust, dtype=torch.bool)
    target, valid = build_affinity_targets_torch(ids, torch.as_tensor(content)[None, None])
    height, width = ids.shape[-2:]
    for c, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        source, _ = _edge_slices(height, width, dy, dx)
        y0, y1 = source[0].start, source[0].stop
        x0, x1 = source[1].start, source[1].stop
        for sy, sx in _line_points(dy, dx):
            valid[0, c, source[0], source[1]] &= trust[y0+sy:y1+sy, x0+sx:x1+sx]
    return target, valid


def inject(logits, target, valid, channels=None):
    """只覆盖显式有效关系，不将unknown当负边；极限logits不是可部署输出。"""
    chosen = valid.clone()
    if channels is not None:
        chosen[:, channels:] = False
    return torch.where(chosen, torch.where(target > .5, 16., -16.), logits), chosen


def decode_capture(semantic, boundary, config):
    """捕获真实watershed输入；验证复刻的marker与实际部署逐值相同。"""
    original = cv2.watershed
    captured = {}
    def capture(image, markers):
        captured['markers'] = markers.copy()
        result = original(image, markers)
        captured['watershed'] = np.maximum(result, 0).copy()
        return result
    with patch('utils.post_process.cv2.watershed', side_effect=capture):
        instances, classes = decode_native(semantic, boundary, config)
    stages = marker_stages(boundary[0, 0].numpy(), config)
    if 'markers' not in captured or not np.array_equal(captured['markers'], stages['marker_labels']):
        raise AssertionError('Diagnostic marker differs from actual deployed watershed input')
    return instances, classes, stages, captured['watershed']


def representative(mask, bounds):
    """在已知区域内部选固定点；仅裁切求距离，连通性始终在整图算。"""
    cropped = mask[bounds]
    if not cropped.any():
        return None
    distance = cv2.distanceTransform(np.pad(cropped.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return [int(y + bounds[0].start), int(x + bounds[1].start)]


def fixed_relations(gt, trusted, base_ids, grid_gt, target, valid):
    """GT邻接合并与显著GT内部拆分，固定两点追踪；不计算官方精度。"""
    anchors, fragments, areas = {}, {}, {}
    bounds = find_objects(gt.astype(np.int32))
    for iid, box in enumerate(bounds, 1):
        if box is None:
            continue
        mask = (gt[box] == iid) & trusted[box]
        area = int(mask.sum())
        if area < 64:
            continue
        ids, counts = np.unique(base_ids[box][mask], return_counts=True)
        pairs = sorted([(int(n), int(p)) for p, n in zip(ids, counts) if p > 0], reverse=True)
        if not pairs:
            continue
        chosen = []
        for n, pid in pairs[:2]:
            if n < max(50, .10 * area):
                continue
            local = mask & (base_ids[box] == pid)
            distance = cv2.distanceTransform(np.pad(local.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            y, x = np.unravel_index(int(distance.argmax()), distance.shape)
            chosen.append({'prediction': pid, 'point': [int(y+box[0].start), int(x+box[1].start)], 'pixels': n})
        if chosen:
            anchors[iid], fragments[iid], areas[iid] = chosen[0], chosen, area
    contacts = Counter()
    g = np.asarray(grid_gt)
    for c, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        source, destination = _edge_slices(*g.shape, dy, dx)
        mask = valid[0, c, source[0], source[1]].numpy() & (target[0, c, source[0], source[1]].numpy() < .5)
        left, right = g[source][mask], g[destination][mask]
        if left.size:
            pairs, counts = np.unique(np.sort(np.stack([left, right], 1), axis=1), axis=0, return_counts=True)
            contacts.update({tuple(map(int, pair)): int(n) for pair, n in zip(pairs, counts)})
    relations = []
    # 从不同已知GT内部点共享预测实例枚举，不能先按可信边筛掉监督不到的粘连。
    by_prediction = {}
    for iid, anchor in anchors.items():
        by_prediction.setdefault(anchor['prediction'], []).append(iid)
    for pid, ids in by_prediction.items():
        for a, b in combinations(sorted(ids), 2):
            relations.append({'kind': 'merge', 'gt_ids': [a, b], 'base_prediction_ids': [pid, pid],
                              'points': [anchors[a]['point'], anchors[b]['point']],
                              'trusted_contact_edges': contacts.get((a, b), 0), 'known_area': areas[a]+areas[b]})
    for iid, pieces in fragments.items():
        if len(pieces) > 1:
            relations.append({'kind': 'split', 'gt_ids': [iid, iid], 'base_prediction_ids': [p['prediction'] for p in pieces],
                              'points': [p['point'] for p in pieces],
                              'trusted_contact_edges': None, 'known_area': areas[iid]})
    return relations, {'trusted_gt_instances_with_anchor': len(anchors), 'trusted_gt_contact_pairs': len(contacts)}


def relations_at(labels, relations):
    result = []
    for relation in relations:
        a, b = [int(labels[y, x]) for y, x in relation['points']]
        result.append('blocked' if min(a, b) <= 0 else 'same' if a == b else 'different')
    return result


def trace(stages, watershed, final, trusted, relations):
    """全原尺寸连通域逐层追踪，同时分离经过unknown才连通的路径。"""
    bridge = stages['resolved']['bridge_width']
    dilate = stages['resolved']['watershed_dilate_width']
    marker = stages['marker_boundary_mask']
    bridged = marker.copy()
    if bridge:
        bridged = cv2.dilate(bridged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bridge+1, 2*bridge+1)))
    skeleton, _ = _boundary_skeleton_belt(marker, bridge, 0)
    belt = skeleton
    if dilate:
        belt = cv2.dilate(skeleton, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1)))
    barriers = {'threshold': stages['high_mask'], 'reconstruction': marker, 'bridge': bridged,
                'skeleton': skeleton, 'dilation': belt, 'sealed_core': stages['marker_belt']}
    result = {}
    for name, barrier in barriers.items():
        free = (barrier == 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(free, connectivity=8)
        full = relations_at(labels, relations)
        del labels
        _, labels = cv2.connectedComponents((free & trusted).astype(np.uint8), connectivity=8)
        result[name] = {'full': full, 'trusted_only': relations_at(labels, relations)}
        del labels
    for name, labels in [('markers', stages['marker_labels']), ('watershed', watershed), ('final', final)]:
        result[name] = {'full': relations_at(labels, relations)}
    return result


def diagnose_view(model, restorer, tensor, native_gt, native_trust, config, seed):
    device = next(model.parameters()).device
    shape = native_gt.shape
    meta = geometry_letterbox_metadata(shape, 1024, 512)
    ph, pw = 1024-meta.resized_height, 1024-meta.resized_width
    native = lambda value: crop_letterbox_output(value, 1024, ph, pw, shape).cpu()
    grid_gt, content, _ = letterbox_instance_geometry(native_gt, 1024, 512)
    trusted = resize_trusted(native_trust, input_size=1024, output_grid=512)[0]
    target, valid = trusted_targets(grid_gt, trusted, content)
    with torch.no_grad():
        image = tensor[None].to(device).float()
        raw_sem = native(model.semantic_decoder(model.encoder(image), image))
        restored = restore_first(restorer, image, seed)
        output = model(restored)
        logits = output['affinity_logits'].float().cpu()
        semantic = native(output['semantic_logits'].float())
    mode, kwargs = fusion_options(config)
    base_fused = affinity_boundary_probability(logits, mode=mode, **kwargs)
    base_boundary = native(base_fused)
    base_ids, _, base_stages, base_ws = decode_capture(semantic, base_boundary, config)
    base_grid, _, _ = letterbox_instance_geometry(base_ids, 1024, 512)
    with torch.enable_grad():
        leaf = logits.detach().clone().requires_grad_(True)
        original_target, original_valid = build_affinity_targets_torch(torch.from_numpy(grid_gt.astype(np.int64))[None], torch.from_numpy(content)[None, None])
        options = {k: config['affinity_connectivity'][k] for k in ('max_groups','edges_per_group','minimum_group_edges','negative_margin','positive_margin')}
        loss, sampling = deployment_connectivity_loss(leaf, original_target, original_valid,
            torch.from_numpy(base_grid.astype(np.int64))[None], torch.from_numpy(grid_gt.astype(np.int64))[None],
            trusted_pixels=trusted[None], **options)
        selected = torch.autograd.grad(loss, leaf)[0] != 0
    if bool((selected & ~valid).any()) or int(selected.sum()) != sampling['selected_edges']:
        raise AssertionError('Selected supervision differs from actual loss or trusted domain')
    relations, anchor_summary = fixed_relations(native_gt, native_trust, base_ids, grid_gt, target, valid)
    selected_relations = set()
    for _, c, y, x in selected.nonzero().tolist():
        dy, dx = DEFAULT_AFFINITY_OFFSETS[c]
        gs,gd=int(grid_gt[y,x]),int(grid_gt[y+dy,x+dx])
        ps,pd=int(base_grid[y,x]),int(base_grid[y+dy,x+dx])
        key=('merge',min(gs,gd),max(gs,gd),ps) if gs!=gd else ('split',gs,min(ps,pd),max(ps,pd))
        selected_relations.add(key)
    for relation in relations:
        gs,gd=relation['gt_ids'];ps,pd=relation['base_prediction_ids']
        key=('merge',gs,gd,ps) if relation['kind']=='merge' else ('split',gs,min(ps,pd),max(ps,pd))
        relation['gt_relation_selected'] = key in selected_relations
    variants = {}
    small = {}
    probability = torch.sigmoid(raw_sem)[0,0].numpy()
    for name in ('base','ideal_short','ideal_all','selected_only'):
        mask = valid if name != 'selected_only' else selected
        if name == 'base':
            fused, boundary = base_fused, base_boundary
            ids, stages, watershed = base_ids, base_stages, base_ws
            overwritten = 0
        else:
            changed, active = inject(logits, target, mask, 4 if name == 'ideal_short' else None)
            if not torch.equal(changed[~active], logits[~active]):
                raise AssertionError('Counterfactual modified an ignored position')
            fused = affinity_boundary_probability(changed, mode=mode, **kwargs)
            boundary = native(fused)
            ids, _, stages, watershed = decode_capture(semantic, boundary, config)
            overwritten = int(active.sum())
        classes, _ = revote_instances(ids, probability)
        chain = trace(stages, watershed, ids, native_trust, relations)
        delta = (fused-base_fused).abs()[0,0]
        result = {'instances': len(classes), 'ferrite': sum(v==1 for v in classes.values()),
                  'overwritten_edges': overwritten, 'fused_max_abs_delta': float(delta.max()),
                  'fused_pixels_changed_gt_001': int((delta>.01).sum()),
                  'marker_count': stages['marker_count'], 'marker_capture_exact': True,
                  'chain': chain, 'anchor_classes': [[classes.get(str(int(ids[y,x]))) for y,x in r['points']] for r in relations]}
        variants[name] = result
        width, height = 384, round(384*shape[0]/shape[1])
        small[name] = {'ids': cv2.resize(ids,(width,height),interpolation=cv2.INTER_NEAREST),
                       'classes': {int(k):v for k,v in classes.items()},
                       'boundary': cv2.resize(boundary[0,0].numpy(),(width,height),interpolation=cv2.INTER_AREA)}
        if name != 'base':
            del stages, watershed
    for index, relation in enumerate(relations):
        relation['stages'] = {name:{stage:{domain:values[index] for domain,values in item.items()} for stage,item in result['chain'].items()} for name,result in variants.items()}
        relation['classes'] = {name:result['anchor_classes'][index] for name,result in variants.items()}
    for result in variants.values():
        result.pop('chain');result.pop('anchor_classes')
    input_native = native(tensor[None].float())[0].permute(1,2,0).numpy()
    input_small = cv2.resize(np.rint(input_native*255).astype(np.uint8),(384,round(384*shape[0]/shape[1])))
    return {'sampling':sampling,'anchor_summary':anchor_summary,'relations':relations,'variants':variants,
            'trusted_pixel_fraction':float(native_trust.mean()), 'trusted_edges':int(valid.sum())}, small, input_small


def render(path, image, trusted, results, pictures, title):
    bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    panels = []
    def add(body,label):
        panel=cv2.copyMakeBorder(body,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        text_line(panel,label,(6,21),body.shape[1],size=.45)
        panels.append(panel)
    add(bgr,title)
    known=cv2.resize(trusted.astype(np.uint8), (bgr.shape[1],bgr.shape[0]),interpolation=cv2.INTER_NEAREST)
    shaded=bgr.copy();shaded[known==0]=(shaded[known==0]*.22).astype(np.uint8)
    add(shaded,'Original labelled support; unknown dark')
    for name,picture in pictures.items():
        add(overlay(bgr,picture['ids'],picture['classes']),f'{name}: N={results["variants"][name]["instances"]}')
    write_png(path,np.concatenate([np.concatenate(panels[:3],1),np.concatenate(panels[3:],1)],0))


def summarize(rows):
    result = {}
    for view in sorted({row['view'] for row in rows}):
        subset = [row for row in rows if row['view']==view]
        relations = [r for row in subset for r in row['relations']]
        summary={'images':len(subset),'directions':{}}
        for kind,desired in [('merge','different'),('split','same')]:
            cases=[r for r in relations if r['kind']==kind]
            metrics={'cases':len(cases),'selected_gt_relation':sum(r['gt_relation_selected'] for r in cases),'variants':{}}
            for variant in ('base','ideal_short','ideal_all','selected_only'):
                metrics['variants'][variant]={'final_fixed':sum(r['stages'][variant]['final']['full']==desired for r in cases),
                    'final_blocked':sum(r['stages'][variant]['final']['full']=='blocked' for r in cases),
                    'marker_relation':dict(Counter(r['stages'][variant]['markers']['full'] for r in cases))}
            if kind=='merge':
                metrics['no_trusted_contact_edges']=sum(r['trusted_contact_edges']==0 for r in cases)
                chain=['threshold','reconstruction','bridge','skeleton','dilation','sealed_core','markers','watershed','final']
                metrics['baseline_stages']={stage:dict(Counter(r['stages']['base'][stage]['full'] for r in cases)) for stage in chain}
                metrics['baseline_transitions']={a+'_to_'+b:{'reopened':sum(r['stages']['base'][a]['full']=='different' and r['stages']['base'][b]['full']=='same' for r in cases),
                    'closed':sum(r['stages']['base'][a]['full']=='same' and r['stages']['base'][b]['full']=='different' for r in cases)} for a,b in zip(chain,chain[1:])}
                metrics['same_marker_with_trusted_route']=sum(r['stages']['base']['sealed_core']['full']=='same' and r['stages']['base']['sealed_core']['trusted_only']=='same' for r in cases)
                metrics['same_marker_without_trusted_route']=sum(r['stages']['base']['sealed_core']['full']=='same' and r['stages']['base']['sealed_core']['trusted_only']!='same' for r in cases)
            summary['directions'][kind]=metrics
        result[view]=summary
    return result


def run(args):
    config=load_config(args.config)
    output=Path(project_path(config,args.output));output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    checkpoint=project_path(config,args.checkpoint)
    if sha(checkpoint)!=CONTROL_SHA:
        raise ValueError('Expected winning control e20 checkpoint')
    device=torch.device('cuda')
    model,bundle=load_backend(checkpoint,config,device)
    model.eval().requires_grad_(False)
    restorer=get_restorer(config,device)
    initial=tensor_digest(model.state_dict().items())
    restorer_initial=tensor_digest(restorer.state_dict().items())
    cfg=config['backend_adaptation']
    base=CanonicalBackendDataset(project_path(config,config['paths']['raw_data_dir']),project_path(config,cfg['completed_gt_dir']))
    if len(base)!=32:
        raise ValueError('Expected all 32 manual sources')
    degradation=load_config(project_path(config,cfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
    degradation={**degradation,'profile_probabilities':[0.,1.,0.,0.]}
    paired=PairedDegradationDataset(base,degradation,cfg['noise'],cfg['seed'],repeats=1)
    paired.set_epoch(20)
    # 先在既有test输出上做一次无标签复现；不将测试图用于下面GT分析。
    files=sorted(Path(project_path(config,config['inference']['test_dir'])).glob('*.jpg'))
    parity=[]
    for index,path in enumerate(files):
        if path.stem not in ['test_101','test_116']:
            continue
        prediction=predict_cross(model,restorer,path,config,device,cfg['restoration']['inference_seed']+index)
        directory=Path(checkpoint).parent/'deployment'
        expected=cv2.imread(str(directory/(path.stem+'_inst.png')),-1)
        classes=json.loads((directory/(path.stem+'_class.json')).read_text())
        assert np.array_equal(expected,prediction['instances']) and classes==prediction['classes'],path.stem
        parity.append(path.stem)
    if len(parity)!=2:
        raise AssertionError('Missing parity images')
    dump(output/'provenance.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':CONTROL_SHA,'epoch':bundle['epoch'],
        'config':config,'no_training':True,'no_holdout':True,'test_output_parity':parity,
        'diagnostic_blur':'all manual sources, forced blur profile, epoch20 source stream, undo flips/rotation, same restoration seed as clean',
        'counterfactual':'trusted original-labelled full-path relations only; other logits and semantic probabilities fixed',
        'source_sha256':sha(__file__)})
    rows=[]
    if not 0 <= args.start < len(base):
        raise ValueError('start outside manual cohort')
    for index in range(args.start,min(args.start+(args.limit or len(base)),len(base))):
        clean=base[index];path=base.samples[index]
        with np.load(base.completed_gt_dir/(path.stem+'_gt.npz'),allow_pickle=False) as archive:
            gt=archive['instance_map'].copy();trusted=archive['original_covered'].astype(bool)&(gt>0)
        sample=paired[index]
        degraded=undo_spatial(sample['image'],sample['horizontal_flip'],sample['vertical_flip'],sample['rotation_k'])
        for view,tensor in [('clean',clean['image']),('degraded',degraded)]:
            row,pictures,thumb=diagnose_view(model,restorer,tensor,gt,trusted,config,cfg['restoration']['inference_seed']+index)
            row.update(source=path.stem,view=view,seed=cfg['restoration']['inference_seed']+index)
            row['degradation']={k:sample[k] for k in ('profile','is_spatial','endpoint_applied','noise_sigma','base_sigma')} if view=='degraded' else None
            rows.append(row)
            with (output/'views.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            render(output/(path.stem+'_'+view+'.png'),thumb,trusted,row,pictures,path.stem+' '+view)
            print(json.dumps({'source':path.stem,'view':view,'relations':dict(Counter(r['kind'] for r in row['relations'])),
                'instances':{k:v['instances'] for k,v in row['variants'].items()}},ensure_ascii=False),flush=True)
            del row,pictures,thumb;gc.collect()
    if tensor_digest(model.state_dict().items())!=initial:
        raise AssertionError('Frozen model state changed')
    if tensor_digest(restorer.state_dict().items())!=restorer_initial:
        raise AssertionError('Frozen restoration state changed')
    dump(output/'report.json',{'scope':'training-source mechanism probe, not held-out accuracy, no test GT or checkpoint selection',
        'views':len(rows),'sources':len({r['source'] for r in rows}),'frozen_model_exact':True,'frozen_restorer_exact':True,'summary':summarize(rows)})
    print('PROBE COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/affinity_connectivity.yaml')
    parser.add_argument('--checkpoint',default='outputs/affinity_connectivity/control/final.pt')
    parser.add_argument('--output',default='outputs/affinity_trace')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--start',type=int,default=0)
    run(parser.parse_args())
