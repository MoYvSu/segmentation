# -*- coding: utf-8 -*-
"""四个封存训练draw的新GT最终划分诊断；只读权重，不改部署或训练入口。"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_semantic_fixed import class_independent_pairs, fixed_confusion
from tools.analyze_affinity_sweep_gt import geometry, paired_changes


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def unsafe_ids(labels, known):
    """含未知、邻接未知或图框的整个实例均censor；不把截断面积当完整晶粒。"""
    if labels.shape != known.shape or labels.ndim != 2:
        raise ValueError('Same two-dimensional grids required')
    safe_pixels = cv2.erode(np.asarray(known, np.uint8), np.ones((3, 3), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return {int(i) for i in np.unique(labels[~safe_pixels]) if i > 0}


def diagnose_partition(gt, pred, gt_classes, pred_classes):
    """相同新GT已知域匹配；另给unknown/frame保守删失，不重新选择推理阈值。"""
    known = gt > 0
    pairs, summary = class_independent_pairs(gt, known, pred)
    stats = geometry(gt, known, pred, pairs, summary) if known.any() else dict(
        **summary, iou_sum=0., matched_iou=0., gt_penalized_iou=0., split_gt=0,
        merged_pred=0, split_gt_ids=[], merged_predictions=[], merged_gt_pairs=[])
    bad_gt, bad_pred = unsafe_ids(gt, known), unsafe_ids(pred, known)
    safe_pairs = [p for p in pairs if p['gt'] not in bad_gt and p['pred'] not in bad_pred]
    # 与既有geometry相同的显著贡献阈值；在共同可信域求交，不将unknown当背景。
    accepted_split, censored_split = [], []
    for value in stats['split_gt_ids']:
        values, counts = np.unique(pred[known & (gt == value)], return_counts=True)
        contributors = {int(i) for i, n in zip(values, counts)
                        if i > 0 and n >= max(50, .1 * int((known & (gt == value)).sum()))}
        (censored_split if value in bad_gt or contributors & bad_pred else accepted_split).append(value)
    accepted_merge, censored_merge = [], []
    for row in stats['merged_predictions']:
        (censored_merge if row['pred'] in bad_pred or set(row['gt_ids']) & bad_gt
         else accepted_merge).append(row)
    safe_merge_pairs = sorted({tuple(pair) for row in accepted_merge
                              for pair in combinations(row['gt_ids'], 2)})
    return dict(pairs=pairs, geometry=stats, confusion=fixed_confusion(pairs, gt_classes, pred_classes),
        conservative=dict(censored_gt_ids=sorted(bad_gt), censored_pred_ids=sorted(bad_pred),
            safe_pairs=safe_pairs, matched=len(safe_pairs),
            safe_matched_iou=float(np.mean([p['iou'] for p in safe_pairs])) if safe_pairs else None,
            split_gt_ids=accepted_split, censored_split_gt_ids=censored_split,
            merged_predictions=accepted_merge, censored_merged_predictions=censored_merge,
            merged_gt_pairs=[list(p) for p in safe_merge_pairs]),
        counts=dict(all_pred_instances=int(len(np.unique(pred[pred > 0]))),
            all_gt_instances=int(len(np.unique(gt[known])))),
        censor_rule='entire GT/pred region touching unknown (8-neighbor), frame, or containing unknown censored; no physical area metric')


def common_safe_change(a, b):
    left = {p['gt']: p for p in a['conservative']['safe_pairs']}
    right = {p['gt']: p for p in b['conservative']['safe_pairs']}
    common = sorted(set(left) & set(right))
    values = [right[g]['iou']-left[g]['iou'] for g in common]
    return dict(common_matches=len(common), common_iou_delta_sum=float(sum(values)),
        common_iou_delta_mean=float(np.mean(values)) if values else None,
        candidate_only_safe_matched_gt=sorted(set(right)-set(left)),
        control_only_safe_matched_gt=sorted(set(left)-set(right)),
        rows=[dict(gt=g, control_iou=left[g]['iou'], candidate_iou=right[g]['iou'],
                   delta=right[g]['iou']-left[g]['iou']) for g in common])


def spatial_partition_links(spatial, arms):
    """按同GT ID关联响应与最终划分；不把相关性写成每组件造成一次误切。"""
    result = {}
    for name, arm in arms.items():
        s = spatial['arms'][name]
        merged = {tuple(p) for p in arm['geometry']['merged_gt_pairs']}
        safe_merged = {tuple(p) for p in arm['conservative']['merged_gt_pairs']}
        split = set(arm['geometry']['split_gt_ids'])
        safe_split = set(arm['conservative']['split_gt_ids'])
        contacts=[]
        for row in s['interface_rows']:
            pair=tuple(row['gt_pair'])
            gaps=[r for r in s['gap_rows'] if tuple(r['gt_pair'])==pair and not r['censored_unknown_or_frame']]
            contacts.append(dict(gt_pair=list(pair), accepted_gap_components=len(gaps),
                gap_component_pixels=sum(r['component_pixels'] for r in gaps),
                longest_component_pixels=max((r['component_pixels'] for r in gaps), default=0),
                final_significant_merge=pair in merged, final_uncensored_merge=pair in safe_merged))
        internals=[]
        for row in s['instance_rows']:
            value=row['gt_instance']; components=[r for r in s['internal_rows'] if r['gt_instance']==value and not r['censored']]
            internals.append(dict(gt=value, accepted_components=len(components),
                accepted_deep_response_pixels=sum(r['deep_pixels'] for r in components),
                closed_loop_candidates=sum(r['category']=='closed_loop_candidate' for r in components),
                multi_contact_candidates=sum(r['multi_contact_candidate'] for r in components),
                final_significant_split=value in split, final_uncensored_split=value in safe_split))
        result[name]=dict(contact_pairs=contacts, internal_gt=internals)
    return result


def rank_phase_diagnostic(boundary, ids, content, lookup):
    """训练附加项的同域同topk选位；类别由同canonical新GT的ID查表生成。"""
    from utils.affinity_loss import build_affinity_targets_torch
    from utils.affinity_fused_loss import fused_ranking_domains
    target, valid = build_affinity_targets_torch(ids, content)
    domains = fused_ranking_domains(target, valid)
    phase = torch.from_numpy(lookup).to(ids.device)[ids[0]]
    result = {}
    for name, mask, largest in zip(('weak_true_boundary', 'strong_deep_interior'), domains, (False, True)):
        scores, classes = boundary[mask], phase[mask[0]]
        if classes.numel() and not bool(((classes == 0) | (classes == 1)).all()):
            raise RuntimeError('Ranking domain does not map to canonical classes')
        n = min(256, max(1, math.ceil(.1 * scores.numel()))) if scores.numel() else 0
        take = scores.detach().topk(n, largest=largest).indices if n else torch.empty(0, dtype=torch.long, device=ids.device)
        picked = classes[take]
        result[name] = dict(eligible=int(scores.numel()), selected=n,
            F=int((picked == 1).sum()), P=int((picked == 0).sum()),
            F_fraction=float((picked == 1).float().mean()) if n else None,
            score_mean=float(scores[take].mean()) if n else None,
            selected_linear_indices=mask[0].flatten().nonzero()[:, 0][take].cpu().tolist())
    return result


def gallery_panel(rgb, labels, known, gt=None):
    # 预测按最大已知GT重叠着色；黑线显示实际实例边界，避免同色碎块不可见。
    palette = np.random.default_rng(20261002).integers(50, 230, (int((labels if gt is None else gt).max())+1, 3), dtype=np.uint8)
    palette[0] = 80
    if gt is None:
        color = palette[labels]
    else:
        color = np.full((*labels.shape, 3), 80, np.uint8)
        for value in np.unique(labels):
            if not value:
                continue
            overlap = gt[(labels == value) & known]
            parent = int(np.bincount(overlap).argmax()) if overlap.size else 0
            color[labels == value] = palette[parent]
    image = np.rint(.6*rgb+.4*color).astype(np.uint8)
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:] |= labels[1:] != labels[:-1]
    image[edge] = 0
    image[~known] = 90
    return image


def render_gallery(folder, stem, rgb, gt, arms):
    known = gt > 0
    panels = [('Degraded input / seen draw', rgb), ('Processed newGT / unknown gray', gallery_panel(rgb, gt, known))]
    panels.extend((name, gallery_panel(rgb, data['instances'], known, gt)) for name, data in arms.items())
    for suffix, crop in [('full', None), ('detail', (.25, .25, .75, .75))]:
        strips = []
        for title, image in panels:
            if crop:
                h, w = image.shape[:2]
                image = image[int(h*crop[1]):int(h*crop[3]), int(w*crop[0]):int(w*crop[2])]
            image = cv2.resize(image, (360, round(360*image.shape[0]/image.shape[1])), interpolation=cv2.INTER_AREA)
            image = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(image, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
            strips.append(image)
        cv2.imwrite(str(folder/(stem+'_'+suffix+'.png')), cv2.cvtColor(np.concatenate(strips, 1), cv2.COLOR_RGB2BGR))


def run(output):
    from data.affinity_connectivity import undo_spatial
    from data.affinity_mixed import build_datasets
    from data.direct_dual_head_dataset import _spatial_transform
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend, restore_first
    from tools.affinity_connectivity_views import decode_native
    from tools.affinity_native_views import verify_prediction
    from tools.probe_affinity_spatial import compare_spatial_boundaries
    from tools.run_affinity_fused import verify_snapshot, verify_newgt
    from tools.run_affinity_short import precision_record
    from tools.run_affinity_sweep import runtime_contract
    from tools.semantic_crossover import revote_instances
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import draw_receipt, frozen_digest, get_restorer
    from train_affinity_fused import CONFIG_PATH, REFERENCE, REFERENCE_SHA256, verify_run
    from train_backend_adaptation import fusion_options, sha, tensor_digest, write_json
    from train_direct_semantic_affinity import move_batch, set_seed
    from torch.utils.data import default_collate
    from utils.affinity_deployment import crop_letterbox_output
    from utils.affinity_fusion import affinity_boundary_probability
    from utils.config import load_config, project_path
    from utils.offset_letterbox import geometry_letterbox_metadata, letterbox_instance_geometry

    root = ROOT/'outputs/affinity_fused'
    output = Path(output).resolve()
    if output.parent != root/'analysis' or output.exists():
        raise RuntimeError('A fresh child of outputs/affinity_fused/analysis is required')
    if not torch.cuda.is_available():
        raise RuntimeError('Only sam2_env GPU server may run actual four draws')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    config = load_config(CONFIG_PATH)
    pipeline_path = root/'pipeline_status.json'
    pipeline = read(pipeline_path); pipeline_sha = sha(pipeline_path)
    if (pipeline.get('status') != 'complete' or pipeline.get('stage') != 'done'
            or pipeline.get('completed') != ['training','inference','spatial_after','render']
            or len(pipeline.get('sources', {})) != 144
            or json.loads(json.dumps(config)) != pipeline['config']):
        raise RuntimeError('Completed sealed fused e20 pipeline/config required')
    verify_snapshot(root, pipeline['sources'])
    if runtime_contract(config) != pipeline['runtime']:
        raise RuntimeError('Third-party runtime differs')
    precision=precision_record(torch)
    if precision != pipeline['precision']:
        raise RuntimeError('Actual FP32/TF32 precision differs from sealed pipeline')
    gt_audit = audit_new_gt(config); verify_newgt(config, gt_audit)
    run_audit = verify_run(config, root/'candidate', 1280)
    status = read(root/'candidate/status.json')
    if status['epoch'] != 20 or status['smoke'] is not False:
        raise RuntimeError('Only completed formal e20 candidate required')
    before = read(root/'spatial_before/report.json'); after = read(root/'spatial_after/report.json')
    if (not before['complete'] or not after['complete']
            or len(before['rows']) != 4 or len(after['rows']) != 4):
        raise RuntimeError('Four completed sealed spatial draws required')
    control_path = Path(project_path(config, REFERENCE, 'final.pt'))
    candidate_path = root/'candidate/final.pt'
    if sha(control_path) != REFERENCE_SHA256 or sha(candidate_path) != after['checkpoint_sha256']:
        raise RuntimeError('Spatial checkpoint identity differs')
    protected = {str(p): sha(p) for p in [pipeline_path, root/'spatial_before/report.json',
        root/'spatial_after/report.json', control_path, candidate_path, Path(__file__)]}
    for row in after['rows']:
        p = root/'spatial_before'/f"draw_{row['draw']:03d}.npz"; protected[str(p)] = sha(p)
    output.mkdir(parents=True); (output/'gallery').mkdir(); (output/'masks').mkdir()
    model, _ = load_backend(control_path, config, 'cuda')
    candidate, _ = load_backend(candidate_path, config, 'cuda')
    restorer = get_restorer(config, 'cuda')
    if frozen_digest(model) != frozen_digest(candidate) or frozen_digest(model) != status['frozen_state_sha256']:
        raise RuntimeError('Frozen shared encoder/semantic components differ')
    for m in (model, candidate):
        if any(getattr(m, k, None) is not None for k in ('semantic_lora','geometry_feature_adapter','geometry_highres_refiner')):
            raise RuntimeError('This exact deployed diagnostic supports no extra adapters/refiners')
    states = dict(control=tensor_digest(model.state_dict().items()), candidate=tensor_digest(candidate.state_dict().items()),
                  restorer=tensor_digest(restorer.state_dict().items()))
    manual, _, _ = build_datasets(config, 1024); manual.set_epoch(1)
    paths = {Path(p).name: Path(p) for p in manual.whole.base_dataset.base.samples}
    references = [json.loads(s) for s in (control_path.parent/'steps.jsonl').read_text().splitlines()]
    rows = []; mode, kwargs = fusion_options(config)
    if mode != 'gated':
        raise RuntimeError('Fixed gated fusion required')
    with torch.no_grad():
        for old, sealed in zip(before['rows'], after['rows']):
            index = sealed['draw']
            if any(old[k] != sealed[k] for k in ('draw','source','view','seed')):
                raise RuntimeError('Spatial before/after draw identity differs')
            reference = next(r for r in references[:64] if int(r['manual_draw']['draw_index'][0]) == index)
            raw = default_collate([manual[index]])
            if (raw['name'] != sealed['source'] or raw['training_view'][0] != sealed['view']
                    or raw['crop_box'].tolist() != reference['manual_crop'] or draw_receipt(raw) != reference['manual_draw']):
                raise RuntimeError('Exact source/crop/augmentation receipt differs')
            batch = move_batch(raw, 'cuda'); set_seed(sealed['seed'])
            restored = restore_first(restorer, batch['image'], sealed['seed']+1000000)
            ids = batch['affinity_instance_map'][0].cpu().numpy()
            content = batch['affinity_valid_content'][0, 0].cpu().numpy().astype(bool)
            with np.load(root/'spatial_before'/f'draw_{index:03d}.npz', allow_pickle=False) as cache:
                if (not np.array_equal(ids, cache['ids']) or not np.array_equal(content, cache['content'])
                        or cache['input_sha256'].item() != tensor_digest([('input', batch['image'])])
                        or cache['restored_sha256'].item() != tensor_digest([('restored', restored)])):
                    raise RuntimeError('Canonical target/input/D5a draw differs')
                old_boundary = cache['gated'].copy()
            features = model.encoder(restored)
            sem = model.semantic_decoder(features, restored)
            if isinstance(sem, dict): sem = sem['semantic_logits']
            raw_sem = model.semantic_decoder(model.encoder(batch['image']), batch['image'])
            if isinstance(raw_sem, dict): raw_sem = raw_sem['semantic_logits']
            control_boundary = affinity_boundary_probability(model.affinity_decoder(features)['affinity_logits'].float(), mode=mode, **kwargs)
            candidate_boundary = affinity_boundary_probability(candidate.affinity_decoder(features)['affinity_logits'].float(), mode=mode, **kwargs)
            if not np.array_equal(control_boundary[0,0].cpu().numpy(), old_boundary):
                raise RuntimeError('Recomputed exact control boundary differs from sealed cache')
            spatial = compare_spatial_boundaries(dict(control_gated=old_boundary,
                candidate_gated=candidate_boundary[0,0].cpu().numpy()), ids, content&(ids>0),
                reference='control_gated', valid_content=content,
                threshold=config['inference']['boundary_threshold'])['report']
            if spatial != sealed['report']:
                raise RuntimeError('Candidate spatial structural report differs from completed report')
            source = paths[sealed['source'][0]]; rgb = read_rgb(str(source))
            gt_path = Path(project_path(config, config['backend_adaptation']['completed_gt_dir'], source.stem+'_gt.npz'))
            full_gt, lookup = load_completed_semantic_source(gt_path, rgb.shape[:2])
            protected[str(gt_path)] = sha(gt_path)
            protected[str(gt_path.with_name(source.stem+'_class.json'))] = sha(gt_path.with_name(source.stem+'_class.json'))
            y, x, y1, x1 = map(int, raw['crop_box'][0].tolist()); gt = full_gt[y:y1,x:x1]
            hflip = bool(raw['horizontal_flip'].item()); vflip = bool(raw['vertical_flip'].item()); turn = int(raw['rotation_k'].item())
            grid, valid, meta = letterbox_instance_geometry(gt, 1024, 512)
            if (not np.array_equal(_spatial_transform(torch.from_numpy(grid), hflip, vflip, turn).numpy(), ids)
                    or not np.array_equal(_spatial_transform(torch.from_numpy(valid), hflip, vflip, turn).numpy(), content)):
                raise RuntimeError('Native newGT crop -> exact augmented affinity grid differs')
            ph, pw = 1024-meta.resized_height, 1024-meta.resized_width
            def native(value):
                return crop_letterbox_output(undo_spatial(value.float(), hflip, vflip, turn), 1024, ph, pw, gt.shape).cpu()
            native_sem = native(sem); raw_probability = native(raw_sem).sigmoid()[0,0].numpy()
            input_rgb = np.rint(native(batch['image'])[0].permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
            classes = {int(i): int(lookup[i]) for i in np.unique(gt) if i > 0}
            arms, rank, predictions = {}, {}, {}
            for name, boundary in [('control_gated',control_boundary), ('candidate_gated',candidate_boundary)]:
                final_ids, _ = decode_native(native_sem, native(boundary).clamp(0,1), config)
                final_classes, _ = revote_instances(final_ids, raw_probability)
                verify_prediction(final_ids, final_classes, gt.shape)
                arms[name] = diagnose_partition(gt, final_ids, classes, final_classes)
                predictions[name] = dict(instances=final_ids, classes=final_classes)
                rank[name] = rank_phase_diagnostic(boundary[:,0], batch['affinity_instance_map'], batch['affinity_valid_content'], lookup)
                cv2.imwrite(str(output/'masks'/f'draw_{index:03d}_{name}_inst.png'), final_ids)
                write_json(output/'masks'/f'draw_{index:03d}_{name}_class.json', final_classes)
            changes = paired_changes(arms['control_gated'], arms['candidate_gated'], classes,
                predictions['control_gated']['classes'], predictions['candidate_gated']['classes'])
            rows.append(dict(draw=index, source=sealed['source'], view=sealed['view'], seed=sealed['seed'],
                native_crop=list(raw['crop_box'][0].tolist()), shape=list(gt.shape), arms=arms, changes=changes,
                common_safe=common_safe_change(arms['control_gated'],arms['candidate_gated']), ranking_phase=rank,
                spatial=spatial, response_partition_links=spatial_partition_links(spatial,arms),
                input_sha256=tensor_digest([('input',batch['image'])]),
                restored_sha256=tensor_digest([('restored',restored)]),
                shared_native_restored_semantic_sha256=tensor_digest([('semantic',native_sem)]),
                shared_raw_probability_sha256=tensor_digest([('raw',torch.from_numpy(raw_probability))])))
            render_gallery(output/'gallery', f'draw_{index:03d}', input_rgb, gt, predictions)
            write_json(output/'progress.json', dict(complete=False, processed=len(rows), rows=rows))
            print(json.dumps(dict(draw=index, geometry={k:v['geometry'] for k,v in arms.items()}, changes=changes, safe=rows[-1]['common_safe'])), flush=True)
    final_states = dict(control=tensor_digest(model.state_dict().items()), candidate=tensor_digest(candidate.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if states != final_states or any(sha(Path(p)) != digest for p,digest in protected.items()):
        raise RuntimeError('Read-only frozen state/protected artifact changed')
    verify_snapshot(root, pipeline['sources'])
    if sha(pipeline_path) != pipeline_sha:
        raise RuntimeError('Formal pipeline JSON mutated')
    imported = {str(Path(m.__file__).resolve().relative_to(ROOT)):sha(m.__file__) for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).suffix == '.py'
        and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(ROOT)}
    report = dict(complete=True, scope='four seen augmented training draws of processed newGT; diagnostic only, not independent accuracy or official full-image semantic deployment',
        contract='same native crop, undo augmentation, inverse letterbox, shared restored semantic, same gated fusion + final postprocess + shared raw semantic revote; no oracle thresholds/seeds',
        censor='unknown/frame touching objects are separately excluded from conservative topology/matches; crop area is never interpreted as physical mean grain area',
        support='all processed newGT known IDs, including filled; remaining unknown/padding ignored; no original_covered wrapper',
        checkpoints=dict(control=REFERENCE_SHA256,candidate=sha(candidate_path)), epoch=20, updates=1280,
        newgt=gt_audit, run_audit=run_audit, protected_sha256=protected, imported_sources_sha256=imported,
        frozen_states_sha256=states, frozen_unchanged=True, source144_unchanged=True, formal_pipeline_unchanged=True,
        inference=config['inference'], precision=precision, fusion=dict(mode=mode,**kwargs), rows=rows,
        caveats=['Shared encoder features verified identical; affinity only changes.',
            'Native crop semantic context and augmented degraded input differ from official whole-image semantics; this is diagnostic-only.',
            'Response gaps/loops/multi-contact are candidates, not proven final split/merge errors.',
            'IoU matching and raw topology use known-pixel areas; conservative rows censor unknown/frame-touching regions.',
            'Training samples were seen: changes locate mechanisms and do not establish generalization.'])
    write_json(output/'report.json', report)
    pages=['<!doctype html><meta charset="utf-8"><title>Four seen newGT draws</title><p>Diagnostic only: same augmented draw; unknown gray; no test GT.</p>']
    for row in rows:
        for suffix in ('full','detail'):
            pages.append(f'<p>draw {row["draw"]} {row["source"][0]} {row["view"]}<br><img width="1440" src="gallery/draw_{row["draw"]:03d}_{suffix}.png"></p>')
    (output/'index.html').write_text('\n'.join(pages), encoding='utf8')
    print('COMPLETE', output, flush=True)


def self_test():
    # 未知缝隙不会变成背景监督；触frame大块不能被称作安全完整晶粒。
    gt = np.zeros((40,40), np.int32); gt[5:35,5:20]=1; gt[5:35,20:35]=2
    pred = gt.astype(np.uint16); classes = {1:1,2:0}
    perfect = diagnose_partition(gt,pred,classes,{'1':1,'2':0})
    assert len(perfect['pairs']) == 2 and perfect['geometry']['matched_iou'] == 1
    assert perfect['conservative']['matched'] == 0  # 两块都邻接unknown，保守口径删除。
    gt = np.ones((60,60), np.int32); gt[10:50,10:50] = 2
    pred = gt.astype(np.uint16); pred[10:50,30:50] = 3
    split = diagnose_partition(gt,pred,{1:0,2:1},{'1':0,'2':1,'3':1})
    assert split['conservative']['split_gt_ids'] == [2]
    assert split['geometry']['split_gt_ids'] == [2]
    assert 1 in split['conservative']['censored_gt_ids']
    gt[29:31,29:31] = 0
    censored = diagnose_partition(gt,pred,{1:0,2:1},{'1':0,'2':1,'3':1})
    assert not censored['conservative']['split_gt_ids']
    empty = diagnose_partition(np.zeros((4,4),np.int32), np.zeros((4,4),np.uint16),{}, {})
    assert empty['geometry']['matched'] == 0
    ids = torch.ones((1,31,31),dtype=torch.long); ids[:,:,16:]=2
    rank=rank_phase_diagnostic(torch.full((1,31,31),.5),ids,torch.ones((1,1,31,31),dtype=torch.bool),np.array([-1,0,1],np.float32))
    for row in rank.values():
        assert row['F']+row['P'] == row['selected'] and row['selected'] <= 256
    json.dumps(split,allow_nan=False); json.dumps(rank,allow_nan=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',default='outputs/affinity_fused/analysis/four_gt')
    args=parser.parse_args()
    if args.self_test:
        self_test(); print('CPU self-test passed')
    else:
        run(ROOT/args.output)
