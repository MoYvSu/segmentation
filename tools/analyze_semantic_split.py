# -*- coding: utf-8 -*-
"""固定最终实例轮廓，拆解逐像素语义、边缘影响及空间混合；不训练或推断测试标签。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from models.backend_adaptation import load_backend, restore_first
from tools.semantic_instance_metrics import analyze_instance, instance_bands
from tools.semantic_marker_diagnostic import marker_stages, component_seed_relation
from train_backend_adaptation import fusion_options, save_prediction, write_json
from train_semantic_d5a import get_restorer, sha
from utils.affinity_deployment import prepare_image, crop_letterbox_output, crop_affinity_boundary_output
from utils.config import load_config, project_path
from utils.semantic_vote import instance_semantic_vote


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_prediction(directory, stem):
    labels = cv2.imread(str(directory / f'{stem}_inst.png'), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.ndim != 2 or labels.dtype != np.uint16:
        raise ValueError(f'Invalid uint16 instance map: {directory}/{stem}')
    classes = {int(k): int(v) for k, v in read_json(directory / f'{stem}_class.json').items()}
    audit = read_json(directory / f'{stem}_class_confidence.json')
    if set(np.unique(labels)) - {0} != set(classes):
        raise ValueError('Instance IDs and class map differ')
    if audit['mode'] != 'probability_mean' or audit['threshold'] != .5 or audit['erode_width'] != 0:
        raise ValueError('This diagnostic is defined for the current probability-mean contract')
    return labels, classes, audit


def verify_votes(labels, probability, classes, audit):
    """沿用实际投票实现；每个版本先在自身已保存最终轮廓上复现。"""
    max_delta = 0.0
    for iid, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None:
            continue
        mask, p = labels[sl] == iid, probability[sl]
        cls, score, details = instance_semantic_vote(mask, (p > .5).astype(np.uint8),
            semantic_probability=p, mode='probability_mean', erode_width=0, threshold=.5,
            return_details=True)
        expected = audit['instances'][str(iid)]
        delta = abs(score - expected['ferrite_score'])
        max_delta = max(max_delta, delta)
        if cls != classes[iid] or cls != expected['class'] or delta > 2e-6:
            raise RuntimeError(f'Vote reproduction failed: id={iid}, score={score}, saved={expected}')
        if abs(details['hard_ratio'] - expected['hard_ratio']) > 1e-10:
            raise RuntimeError('Hard pixel ratio does not reproduce saved vote')
        if int(mask.sum()) != expected['area'] or details['core_pixels'] != expected['core_pixels']:
            raise RuntimeError('Saved vote region differs from final mask')
    return max_delta


def contours(labels):
    return (cv2.dilate(labels.astype(np.float32), np.ones((3, 3), np.uint8)) !=
            cv2.erode(labels.astype(np.float32), np.ones((3, 3), np.uint8)))


def prob_rgb(probability):
    # 固定同一色标：珠光体蓝、0.5白、铁素体橙；不逐图归一化。
    p = np.clip(probability, 0, 1)[..., None]
    low, mid, high = np.array([35, 83, 176]), np.array([248, 248, 248]), np.array([233, 150, 24])
    return np.where(p < .5, low + p * 2 * (mid-low), mid + (p-.5)*2*(high-mid)).astype(np.uint8)


def class_rgb(image, labels, classes):
    colours = np.zeros((max(int(labels.max()), max(classes, default=0)) + 1, 3), np.uint8)
    for iid, cls in classes.items():
        colours[iid] = (239, 180, 35) if cls else (65, 125, 220)
    out = (.5 * image + .5 * colours[labels]).astype(np.uint8)
    out[contours(labels)] = [230, 65, 180]
    return out


def tile(image, title, width=420, height=354):
    scale = min(width / image.shape[1], (height-32) / image.shape[0])
    thumb = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    canvas = np.full((height, width, 3), 250, np.uint8)
    y, x = 32+(height-32-thumb.shape[0])//2, (width-thumb.shape[1])//2
    canvas[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
    cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (25,25,25), 1, cv2.LINE_AA)
    return canvas


def write_png(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(path)


def render_board(path, image, restored, boundary, anchor, probabilities, predictions, crop=None, title='', instance_id=None):
    sl = crop or (slice(None), slice(None))
    anchor_crop = anchor[sl]
    edge = contours(anchor_crop)
    cell_height = min(452,max(160,32+round(420*anchor_crop.shape[0]/anchor_crop.shape[1])))
    make_tile = lambda rgb, caption: tile(rgb,caption,height=cell_height)
    selected_edge = contours((anchor_crop == instance_id).astype(np.uint8)) if instance_id is not None else None
    def mark(rgb):
        if selected_edge is not None:
            rgb = rgb.copy()
            rgb[selected_edge] = [255,35,35]
        return rgb
    images = [make_tile(mark(image[sl]), title+' | Raw'), make_tile(mark(restored[sl]), 'D5a input (resized)'),
              make_tile(mark(np.repeat((boundary[sl]*255).astype(np.uint8)[...,None], 3, 2)), 'Affinity: white = strong boundary')]
    for arm, p in probabilities.items():
        rgb = prob_rgb(p[sl])
        rgb[edge] = [30,190,90]
        caption = arm+' P(F): blue=0 white=.5 orange=1'
        if instance_id is not None:
            score = float(p[anchor == instance_id].mean())
            caption = f'{arm}: whole P(F)={score:.3f} | red=target'
        images.append(make_tile(mark(rgb),caption))
    for arm, (labels, classes, _) in predictions.items():
        images.append(make_tile(mark(class_rgb(image[sl], labels[sl], classes)), arm+' actual final: F=gold P=blue'))
    board = np.concatenate([np.concatenate(images[i:i+3], axis=1) for i in range(0,9,3)], axis=0)
    write_png(path, board)


def describe(values):
    if not values:
        return None
    a = np.asarray(values, np.float64)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a,.1)), p90=float(np.quantile(a,.9)), maximum=float(a.max()))


def render_channels(path,image,probability,anchor,stages,crop,instance_id):
    """所有连通域在完整原图计算，此处只裁切显示，不在裁块内重新连通。"""
    target_edge=contours((anchor[crop]==instance_id).astype(np.uint8))
    def marked(rgb):
        rgb=rgb.copy()
        rgb[target_edge]=[255,35,35]
        return rgb
    def binary(array):
        return np.repeat(((array[crop]>0)*255).astype(np.uint8)[...,None],3,2)
    labels=stages['marker_labels'][crop]
    rng=np.random.default_rng(20260928)
    palette=rng.integers(45,230,size=(int(stages['marker_labels'].max())+1,3),dtype=np.uint8)
    palette[0]=[20,20,20]
    panels=[(image[crop],'Raw | red = selected final instance'),
        (prob_rgb(probability[crop]),'LoRA P(F): blue=0 orange=1'),
        (np.repeat((stages['boundary_probability'][crop]*255).astype(np.uint8)[...,None],3,2),'Affinity probability'),
        (binary(stages['high_mask']),'Current high-threshold boundary'),
        (binary(stages['marker_belt']),'Actual marker barrier (after repair)'),
        (palette[labels],'Actual seed IDs: same colour = same seed')]
    height=min(412,max(160,32+round(380*labels.shape[0]/labels.shape[1])))
    tiles=[tile(marked(rgb),caption,width=380,height=height) for rgb,caption in panels]
    write_png(path,np.concatenate([np.concatenate(tiles[:3],1),np.concatenate(tiles[3:],1)],0))


def summarize(rows, arms):
    total_area = sum(r['area'] for r in rows)
    result = {'instances': len(rows), 'foreground_area': total_area, 'models': {}, 'pairs': {}}
    for arm in arms:
        entries = [(r, r['models'][arm]) for r in rows]
        selectors = {
            'heterogeneous': lambda m: m['heterogeneous'],
            'interior_heterogeneous': lambda m: m.get('interior_heterogeneous', False),
            'core_flip': lambda m: m['whole_vs_core_flip'] is True,
            'core_p_to_f': lambda m: m['whole_vs_core_flip'] is True and m['mean_class'] == 0,
            'core_f_to_p': lambda m: m['whole_vs_core_flip'] is True and m['mean_class'] == 1,
            'uniform_strong_p': lambda m: m['strong_p']['fraction'] >= .9,
            'uniform_strong_f': lambda m: m['strong_f']['fraction'] >= .9,
            'mostly_uncertain': lambda m: m['uncertain_fraction'] >= .5,
            'near_threshold': lambda m: abs(m['mean_probability']-.5) <= .05,
        }
        metrics = {}
        for name, select in selectors.items():
            selected = [(r,m) for r,m in entries if select(m)]
            metrics[name] = dict(instances=len(selected), area=sum(r['area'] for r,m in selected),
                area_fraction=sum(r['area'] for r,m in selected)/max(1,total_area))
        metrics['ferrite_instances'] = sum(m['mean_class'] for r,m in entries)
        metrics['meaningful_core_instances'] = sum(m['meaningful_core'] for r,m in entries)
        metrics['core_minus_outer'] = describe([m['core_minus_outer'] for r,m in entries if m['meaningful_core']])
        metrics['distance_layers'] = []
        for index in range(5):
            layers = [m['layers'][index] for r,m in entries if m['layers'][index]['area']]
            n = sum(x['area'] for x in layers)
            metrics['distance_layers'].append(dict(area=n, mean_probability=sum(x['area']*x['mean_probability'] for x in layers)/max(n,1)))
        result['models'][arm] = metrics
    for a,b in [('simple','gray4'),('gray4','lora'),('simple','lora')]:
        changed = [r for r in rows if r['models'][a]['mean_class'] != r['models'][b]['mean_class']]
        result['pairs'][a+'_to_'+b] = dict(changed_instances=len(changed),
            p_to_f=sum(r['models'][a]['mean_class']==0 for r in changed),
            f_to_p=sum(r['models'][a]['mean_class']==1 for r in changed),
            changed_area=sum(r['area'] for r in changed),
            changed_area_fraction=sum(r['area'] for r in changed)/max(1,total_area),
            near_threshold_before=sum(abs(r['models'][a]['mean_probability']-.5)<=.05 for r in changed),
            mean_absolute_score_delta=float(np.mean([abs(r['models'][b]['mean_probability']-r['models'][a]['mean_probability']) for r in rows])))
    return result


@torch.inference_mode()
def run(args):
    config = load_config(args.config)
    cfg = config['semantic_split']
    out = Path(project_path(config, args.output_dir or cfg['output_dir']))
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    if len(files) != cfg['expected_images']:
        raise ValueError('Test image inventory differs')
    selected = [(i,p) for i,p in enumerate(files) if not args.images or p.stem in args.images]
    if args.images and len(selected) != len(set(args.images)):
        raise ValueError('Requested image missing')
    excluded = set(config['semantic_adaptation']['monitor']['images']) | {'test_101','test_116'}
    random_stems = np.random.default_rng(cfg['random_seed']).choice(
        [p.stem for p in files if p.stem not in excluded], cfg['random_images'], replace=False).tolist()
    models, dirs, manifests, lineage = {}, {}, {}, {}
    for arm, entry in cfg['models'].items():
        dirs[arm] = Path(project_path(config, entry['deployment']))
        manifests[arm] = read_json(dirs[arm]/'manifest.json')
        checkpoint = Path(project_path(config, entry['checkpoint']))
        digest = sha(checkpoint)
        if digest != entry['sha256'] or digest != manifests[arm]['checkpoint_sha256']:
            raise ValueError('Checkpoint lineage differs: '+arm)
        historical = manifests[arm]['config']
        for key in ['affinity_deployment']:
            if historical[key] != config[key]:
                raise ValueError('Deployment config differs: '+key)
        for key in ['threshold','semantic_vote_mode','semantic_vote_erode_width','semantic_vote_threshold','boundary_threshold']:
            if historical['inference'].get(key) != config['inference'].get(key):
                raise ValueError('Inference setting differs: '+key)
        if historical['semantic_adaptation']['restoration'] != config['semantic_adaptation']['restoration']:
            raise ValueError('D5a contract differs')
        if [x['image'] for x in manifests[arm]['images']] != [p.name for p in files]:
            raise ValueError('Image ordering differs from original seeded inference')
        model, saved = load_backend(checkpoint, config, device)
        model.eval().requires_grad_(False)
        if (model.semantic_lora is not None) != (arm == 'lora'):
            raise ValueError('Unexpected semantic LoRA route')
        models[arm] = model
        lineage[arm] = dict(checkpoint=entry['checkpoint'], sha256=digest, epoch=int(saved['epoch']))
        del saved
    restorer = get_restorer(config, device)
    write_json(out/'manifest.json', dict(config=config, checkpoints=lineage, random_images=random_stems,
        anchor=cfg['anchor'], output_dir=str(out), diagnostic_only=True, no_training=True, official_score=None,
        probability_scale=[0,1], internal_strong_thresholds=[.2,.8]))
    (out/'source').mkdir()
    for path in [Path(__file__), ROOT/'tools/semantic_instance_metrics.py', ROOT/'tools/semantic_marker_diagnostic.py', Path(args.config)]:
        shutil.copy2(path, out/'source'/path.name)
    all_rows, image_rows, candidates = [], [], []
    maximum_vote_delta, maximum_affinity_delta = 0., 0.
    started = time.time()
    mode, kwargs = fusion_options(config)
    for count,(index,path) in enumerate(selected, 1):
        image,tensor,ph,pw = prepare_image(path, 1024, device)
        restored = restore_first(restorer, tensor, config['semantic_adaptation']['restoration']['inference_seed']+index)
        restored_rgb = (crop_letterbox_output(restored,1024,ph,pw,image.shape[:2])[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
        probs, predictions, boundaries, vote_deltas, native_logits = {}, {}, {}, {}, {}
        for arm,model in models.items():
            output = model(restored)
            semantic = crop_letterbox_output(output['semantic_logits'],1024,ph,pw,image.shape[:2]).cpu()
            # 与部署一致：先在logit域恢复尺寸，再在CPU sigmoid。
            probs[arm] = torch.sigmoid(semantic)[0,0].numpy()
            native_logits[arm] = semantic
            boundaries[arm] = crop_affinity_boundary_output(output,1024,ph,pw,image.shape[:2],mode,kwargs).cpu()
            predictions[arm] = load_prediction(dirs[arm],path.stem)
            own,classes,audit = predictions[arm]
            if own.shape != image.shape[:2]:
                raise ValueError('Saved prediction shape differs')
            vote_deltas[arm] = verify_votes(own,probs[arm],classes,audit)
            maximum_vote_delta = max(maximum_vote_delta,vote_deltas[arm])
            # 首张完整走后处理，核对uint16轮廓及类别，防止手工前向漏掉部署步骤。
            if count == 1:
                check_dir = out/'reproduction'/arm
                check_dir.mkdir(parents=True)
                check_labels,check_classes = save_prediction(image,semantic,boundaries[arm],config,check_dir,path.stem)
                if not np.array_equal(check_labels,own) or check_classes != classes:
                    raise RuntimeError('Full deployed output does not reproduce: '+arm)
            del output
        for b in boundaries.values():
            delta = float((b-boundaries['lora']).abs().max())
            maximum_affinity_delta = max(maximum_affinity_delta,delta)
            if delta > 1e-6:
                raise RuntimeError('Frozen affinity differs across arms')
        anchor = predictions[cfg['anchor']][0]
        boundary = boundaries['lora'][0,0].numpy()
        stages = marker_stages(boundary,config)
        rows = []
        gray_raw = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)/255.
        gray_restored = cv2.cvtColor(restored_rgb,cv2.COLOR_RGB2GRAY)/255.
        for iid,sl in enumerate(ndimage.find_objects(anchor),1):
            if sl is None:
                continue
            mask = anchor[sl] == iid
            bands = instance_bands(mask)
            row = dict(stem=path.stem,id=iid,area=int(mask.sum()),bbox=[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop],
                raw_gray_mean=float(gray_raw[sl][mask].mean()),d5a_gray_mean=float(gray_restored[sl][mask].mean()),models={})
            for arm,p in probs.items():
                metrics = analyze_instance(mask,p[sl],bands=bands)
                cls, score = instance_semantic_vote(mask,(p[sl]>.5).astype(np.uint8),
                    semantic_probability=p[sl],mode='probability_mean',erode_width=0,threshold=.5)
                metrics['descriptive_float64_mean'] = metrics['mean_probability']
                metrics['mean_probability'],metrics['mean_class'] = score,cls
                metrics['mean_margin'] = abs(score-.5)
                metrics['whole_vs_core_flip'] = bool(cls != int(metrics['core']['mean_probability']>.5)) if metrics['meaningful_core'] else None
                if arm == cfg['anchor'] and cls != predictions[arm][1][iid]:
                    raise RuntimeError('Fixed anchor class does not reproduce actual class')
                row['models'][arm] = metrics
            # 内部预测转换附近的affinity响应，仅描述结构线索，不能作真值。
            distance = bands['distance']
            sem_hard = (probs['lora'][sl]>.5).astype(np.uint8)
            interface = (cv2.dilate(sem_hard,np.ones((3,3),np.uint8)) != cv2.erode(sem_hard,np.ones((3,3),np.uint8))) & (distance>3)
            row['internal_transition'] = dict(pixels=int(interface.sum()),
                boundary_mean=float(boundary[sl][interface].mean()) if interface.any() else None,
                boundary_above_065_fraction=float((boundary[sl][interface]>.65).mean()) if interface.any() else None)
            if row['models']['lora']['interior_heterogeneous']:
                row['seed_relation'] = component_seed_relation(mask,probs['lora'][sl],stages['marker_labels'][sl],bands=bands)
            rows.append(row)
        render_board(out/'overview'/f'{path.stem}.png',image,restored_rgb,boundary,anchor,probs,predictions,title=path.stem)
        if path.stem == 'test_101':
            render_board(out/'examples'/'test_101_upper.png',image,restored_rgb,boundary,anchor,probs,predictions,
                crop=(slice(0,image.shape[0]//2),slice(None)),title='test_101 upper half')
        # 每张只保留三类最突出的区域；正式选图按全局固定量排序。
        selectors = {
            'mixed': lambda r: min(r['models']['lora']['strong_f']['largest_fraction'],r['models']['lora']['strong_p']['largest_fraction']),
            'interior': lambda r: min(r['models']['lora']['interior_strong_f']['largest_fraction'],r['models']['lora']['interior_strong_p']['largest_fraction']),
            'rim': lambda r: abs(r['models']['lora']['core_minus_outer'] or 0) if r['models']['lora']['whole_vs_core_flip'] is True else -1,
            'changed': lambda r: abs(r['models']['lora']['mean_probability']-r['models']['simple']['mean_probability']),
        }
        for kind,score in selectors.items():
            eligible = [r for r in rows if r['area']>=128]
            if kind in {'mixed','interior'}:
                key = 'heterogeneous' if kind == 'mixed' else 'interior_heterogeneous'
                eligible = [r for r in eligible if r['models']['lora'][key]]
            if not eligible:
                continue
            row = max(eligible,key=score)
            if score(row)<0:
                continue
            x0,y0,x1,y1 = row['bbox']
            margin = max(16,round(max(x1-x0,y1-y0)*.12))
            crop=(slice(max(0,y0-margin),min(image.shape[0],y1+margin)),slice(max(0,x0-margin),min(image.shape[1],x1+margin)))
            filename=f'{kind}_{path.stem}_{row["id"]}.png'
            render_board(out/'crops'/filename,image,restored_rgb,boundary,anchor,probs,predictions,crop=crop,
                title=f'{path.stem} id={row["id"]} {kind}',instance_id=row['id'])
            render_channels(out/'channels'/filename,image,probs['lora'],anchor,stages,crop,row['id'])
            candidates.append(dict(kind=kind,score=float(score(row)),file='crops/'+filename,stem=path.stem,id=row['id'],area=row['area']))
        image_rows.append(dict(stem=path.stem,shape=list(image.shape[:2]),vote_max_deltas=vote_deltas,
            marker_count=stages['marker_count'],summary=summarize(rows,models)))
        with (out/'instances.jsonl').open('a',encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        all_rows.extend(rows)
        print(json.dumps(dict(images_done=count,total=len(selected),stem=path.stem,instances=len(rows),
            elapsed_seconds=round(time.time()-started,2),vote_max_delta=maximum_vote_delta)),flush=True)
    report = summarize(all_rows,models)
    report.update(complete=not args.images and len(selected)==cfg['expected_images'],images=len(selected),
        checkpoint_lineage=lineage,anchor=cfg['anchor'],vote_max_absolute_delta=maximum_vote_delta,
        affinity_max_absolute_delta=maximum_affinity_delta,first_image_full_deployment_equal=True,
        elapsed_seconds=time.time()-started,official_score=None,no_test_labels=True,no_training=True,
        diagnostic_thresholds=dict(strong_f=.8,strong_p=.2,uncertain=[.4,.6],mixed_min_fraction=.1,mixed_min_pixels=32))
    assert all(len({tuple(m['layers'][i]['area'] for i in range(5)) for m in r['models'].values()})==1
               and len({(m['core']['area'],m['outer']['area'],m['interior']['area']) for m in r['models'].values()})==1
               for r in all_rows), 'Spatial supports differ between arms'
    report['shared_instance_bands_verified'] = True
    relations=[r['seed_relation'] for r in all_rows if 'seed_relation' in r]
    report['seed_relations']=dict(interior_mixed_instances=len(relations),
        same_seed=sum(r['same_seed'] is True for r in relations),
        different_seeds=sum(r['same_seed'] is False for r in relations),
        unresolved=sum(r['same_seed'] is None for r in relations))
    (out/'examples').mkdir(exist_ok=True)
    selection={'random_seed':cfg['random_seed'],'random':random_stems,'automatic':{},'focus':['test_101','test_116']}
    for kind in ['mixed','interior','rim','changed']:
        best = sorted([r for r in candidates if r['kind']==kind],key=lambda r:r['score'],reverse=True)[:3]
        selection['automatic'][kind]=best
        for item in best:
            shutil.copy2(out/item['file'],out/'examples'/Path(item['file']).name)
            shutil.copy2(out/'channels'/Path(item['file']).name,out/'examples'/('channels_'+Path(item['file']).name))
    for stem in set(random_stems+selection['focus']):
        source=out/'overview'/f'{stem}.png'
        if source.exists():
            shutil.copy2(source,out/'examples'/source.name)
    write_json(out/'summary.json',report)
    write_json(out/'images.json',image_rows)
    write_json(out/'selection.json',selection)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/experiments/semantic_split.yaml')
    parser.add_argument('--output-dir')
    parser.add_argument('--images',nargs='+')
    run(parser.parse_args())
