# -*- coding: utf-8 -*-
"""冻结R1原图末端语义，用新GT原覆盖域检查强光度是否造成大对象错相。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from scipy.ndimage import find_objects

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_semantic_fixed import file_sha, write
from utils.semantic_vote import adaptive_instance_core

FORMAT = 'semantic_raw_stress_v1'
INITIAL = ('train_001', 'train_173', 'train_558', 'train_889')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_order(names, initial=INITIAL, limit=None):
    """先前固定四源先执行；其余源排序，不能根据此次错相选图。"""
    names = list(names)
    if len(names) != len(set(names)) or len(set(initial)) != len(initial):
        raise ValueError('Source names must be unique')
    if not set(initial).issubset(names):
        raise ValueError('Previously fixed sources missing from manual cohort')
    ordered = list(initial) + sorted(set(names) - set(initial))
    if limit is not None and (type(limit) is not int or limit <= 0 or limit > len(ordered)):
        raise ValueError('Invalid source limit')
    return ordered if limit is None else ordered[:limit]


def build_cohort(ids, original, classes, minimum_area=10000, core_fraction=.4, core_min_pixels=8):
    """名单只由GT产生；核心先由已处理实例形状求出，再限制到原覆盖域。"""
    ids, original = np.asarray(ids), np.asarray(original)
    if ids.ndim != 2 or original.shape != ids.shape or original.dtype != bool:
        raise ValueError('Expected matching native GT IDs and boolean original-covered mask')
    if ids.dtype != np.uint16 or np.any(original & (ids == 0)):
        raise ValueError('Processed GT original-covered ownership contract differs')
    present = {int(x) for x in np.unique(ids) if x}
    if not present.issubset(classes) or any(type(classes[x]) is not int or classes[x] not in (0, 1) for x in present):
        raise ValueError('Class 0/1 required for every processed instance')
    counts = np.bincount(ids.ravel())
    boxes = find_objects(ids.astype(np.int32))
    result = []
    for instance in sorted(present):
        box = boxes[instance - 1]
        mask = ids[box] == instance
        trusted = mask & original[box]
        core, _ = adaptive_instance_core(mask, fraction=core_fraction, min_pixels=core_min_pixels)
        core &= trusted
        native_area = int(counts[instance])
        original_pixels = int(trusted.sum())
        touches = bool((box[0].start == 0 and mask[0].any()) or
                       (box[0].stop == ids.shape[0] and mask[-1].any()) or
                       (box[1].start == 0 and mask[:, 0].any()) or
                       (box[1].stop == ids.shape[1] and mask[:, -1].any()))
        result.append(dict(id=instance, cls=classes[instance], native_area=native_area,
            original_pixels=original_pixels, filled_pixels=native_area-original_pixels,
            core_original_pixels=int(core.sum()), frame_censored=touches,
            original_fraction=original_pixels/native_area, large=native_area >= minimum_area,
            usable=original_pixels > 0, entirely_original=original_pixels == native_area,
            bbox=[box[0].start, box[0].stop, box[1].start, box[1].stop],
            box=box, trusted=trusted, core=core))
    return result


def probability_rows(cohort, probability):
    probability = np.asarray(probability)
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Finite native probability in [0,1] required')
    rows = []
    for item in cohort:
        values = probability[item['box']]
        if values.shape != item['trusted'].shape:
            raise ValueError('GT and probability native coordinates differ')
        row = {key:value for key, value in item.items() if key not in ('box', 'trusted', 'core')}
        known = values[item['trusted']]
        core = values[item['core']]
        p = float(known.mean(dtype=np.float32)) if len(known) else None
        pc = float(core.mean(dtype=np.float32)) if len(core) else None
        row.update(probability_original=p, probability_core=pc,
            predicted_class_original=int(p > .5) if p is not None else None,
            predicted_class_core=int(pc > .5) if pc is not None else None,
            wrong_original_pixels=int(((known > .5) != bool(item['cls'])).sum()),
            original_object_wrong=(int(p > .5) != item['cls']) if p is not None else None,
            core_object_wrong=(int(pc > .5) != item['cls']) if pc is not None else None)
        rows.append(row)
    return rows


def summarize_condition(clean, current):
    """保持同一GT对象分母；不把原覆盖区域的相别汇聚当完整实例准确率。"""
    if [x['id'] for x in clean] != [x['id'] for x in current]:
        raise ValueError('Conditions must preserve the fixed GT cohort')
    summary = {}
    for group in ('all', 'large'):
        pairs = [(a,b) for a,b in zip(clean,current) if b['usable'] and (group == 'all' or b['large'])]
        summary[group] = dict(objects=len(pairs), ferrite=sum(b['cls']==1 for _,b in pairs),
            pearlite=sum(b['cls']==0 for _,b in pairs), original_pixels=sum(b['original_pixels'] for _,b in pairs),
            wrong_original_pixels=sum(b['wrong_original_pixels'] for _,b in pairs),
            object_wrong=sum(b['original_object_wrong'] for _,b in pairs),
            F_to_P_wrong=sum(b['original_object_wrong'] and b['cls']==1 for _,b in pairs),
            P_to_F_wrong=sum(b['original_object_wrong'] and b['cls']==0 for _,b in pairs),
            clean_correct_now_wrong=sum(not a['original_object_wrong'] and b['original_object_wrong'] for a,b in pairs),
            clean_wrong_now_correct=sum(a['original_object_wrong'] and not b['original_object_wrong'] for a,b in pairs),
            clean_correct_F_now_P=sum(not a['original_object_wrong'] and b['original_object_wrong'] and b['cls']==1 for a,b in pairs),
            clean_correct_P_now_F=sum(not a['original_object_wrong'] and b['original_object_wrong'] and b['cls']==0 for a,b in pairs),
            core_usable=sum(b['probability_core'] is not None for _,b in pairs),
            core_object_wrong=sum(b['core_object_wrong'] is True for _,b in pairs),
            entirely_original=sum(b['entirely_original'] for _,b in pairs),
            with_fill=sum(b['filled_pixels']>0 for _,b in pairs),
            frame_censored=sum(b['frame_censored'] for _,b in pairs))
    return summary


def save_preview(path, clean_rgb, altered_rgb, probability, title, width=384):
    panels = []
    heat = cv2.cvtColor(cv2.applyColorMap(np.rint(probability * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    for image, label in ((clean_rgb, 'Clean source'), (altered_rgb, title), (heat, 'P(F): fixed 0..1')):
        thumb = cv2.resize(image, (width, round(image.shape[0]*width/image.shape[1])), interpolation=cv2.INTER_AREA)
        thumb = cv2.copyMakeBorder(thumb, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(thumb, label, (5,17), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,0,0), 1, cv2.LINE_AA)
        panels.append(thumb)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels,axis=1), cv2.COLOR_RGB2BGR)):
        raise IOError('Could not write private preview')


def run(args):
    import torch
    from data.backend_adaptation import CanonicalBackendDataset
    from data.semantic_hard import hard_appearance
    from models.backend_adaptation import load_backend
    from tools.probe_semantic_reflect import load_fixed, protected_sources, validate_prediction
    from tools.run_marker_reflect import precision_record
    from tools.semantic_crossover import revote_instances
    from train_backend_adaptation import tensor_digest
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Use the server sam2_env CUDA environment')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = load_config(args.config)
    cfg = config['semantic_raw_stress']
    if (cfg['format'] != FORMAT or tuple(cfg['initial_sources']) != INITIAL or cfg['views_per_family'] != 2
            or cfg['minimum_native_area'] != 10000 or cfg['contrast'] != [.25,.60]
            or cfg['offset'] != [.12,.30] or cfg['max_clipped_fraction'] != .01):
        raise RuntimeError('Predeclared source/photometric stress contract differs')
    resolve = lambda value: Path(project_path(config, value)).resolve()
    out = resolve(args.output or cfg['output_dir'])
    if out.exists() or not out.is_relative_to((ROOT/'outputs').resolve()):
        raise RuntimeError('Fresh ignored outputs directory required')
    out.mkdir(parents=True)
    (out/'gallery').mkdir()
    write(out/'status.json', dict(status='running', training=False, no_optimizer_steps=True))
    protected = {}
    def protect(path, expected=None):
        path = resolve(path)
        digest = file_sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError('Protected SHA differs: '+str(path))
        protected[str(path)] = digest
        return path
    hooks = []
    try:
        protect(args.config)
        checkpoint = protect(cfg['checkpoint'], cfg['checkpoint_sha256'])
        ref_inputs = read(protect(cfg['reference_inputs']))
        ref_rows = {r['source']:r for r in read(protect(cfg['reference_rows']))}
        reference = read(protect(cfg['reference_report']))
        precision = precision_record()
        if (not reference.get('complete') or reference.get('format') != 'semantic_reflect_probe_v1'
                or reference.get('precision') != precision
                or reference['checkpoints']['baseline']['sha256'] != cfg['checkpoint_sha256']):
            raise RuntimeError('Reference semantic probe/precision identity differs')
        fixed = {r['name']:r for r in ref_inputs['cases'] if r['kind']=='train'}
        if set(fixed) != set(INITIAL) or not set(INITIAL).issubset(ref_rows):
            raise RuntimeError('Prior fixed four training source receipts missing')
        dataset = CanonicalBackendDataset(resolve(config['paths']['raw_data_dir']), resolve(config['backend_adaptation']['completed_gt_dir']))
        sources = {p.stem:p for p in dataset.samples}
        if len(sources) != cfg['expected_manual_sources']:
            raise RuntimeError('Expected complete 32-source processed-GT cohort')
        names = source_order(sources, cfg['initial_sources'], args.limit)
        global_indices = {name:i for i,name in enumerate(sorted(sources))}
        model, bundle = load_backend(checkpoint, config, 'cuda')
        if bundle['epoch'] != cfg['checkpoint_epoch'] or model.semantic_lora is not None:
            raise RuntimeError('Expected original R1 full S-align/raw semantic route')
        model.eval().requires_grad_(False)
        state = tensor_digest(model.state_dict().items())
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Pressure probe must be completely frozen')
        def forbidden(*_):
            raise RuntimeError('No affinity forward allowed in raw semantic probe')
        hooks.append(model.affinity_decoder.register_forward_pre_hook(forbidden))
        code, virtual = protected_sources(out)
        write(out/'config.json', config)
        write(out/'selection.json', dict(names=names, all_manual_sources=sorted(sources), limit=args.limit,
            reason='Prior four training sources first, remaining sorted, chosen before current outputs'))
        records, calls, started = [], 0, time.time()
        first_probability, first_path = None, None
        def forward(image, ph, pw, shape):
            nonlocal calls
            logits = model.semantic_decoder(model.encoder(image.float()), image)
            calls += 1
            if isinstance(logits, dict):
                logits = logits['semantic_logits']
            return np.ascontiguousarray(crop_letterbox_output(logits.float(),1024,ph,pw,shape).cpu().sigmoid()[0,0].numpy(),dtype=np.float32)
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), \
                patch('cv2.watershed', side_effect=RuntimeError('No watershed allowed')), \
                patch('models.backend_adaptation.restore_first', side_effect=RuntimeError('No D5a allowed')):
            for name in names:
                source = protect(sources[name], fixed.get(name,{}).get('source_sha256'))
                gtroot = resolve(config['backend_adaptation']['completed_gt_dir'])
                gtpath = protect(gtroot/(name+'_gt.npz'), fixed.get(name,{}).get('gt_npz_sha256'))
                classpath = protect(gtroot/(name+'_class.json'), fixed.get(name,{}).get('gt_classes_sha256'))
                with np.load(gtpath,allow_pickle=False) as blob:
                    ids, original = blob['instance_map'].copy(), blob['original_covered'].astype(bool)
                    if not np.array_equal(blob['filled'].astype(bool),(ids>0)&~original) or not np.array_equal(blob['residual_unknown'].astype(bool),ids==0):
                        raise RuntimeError('Processed GT original/filled/unknown partition differs')
                classes = {int(k):v for k,v in read(classpath).items()}
                rgb, image, ph, pw = prepare_image(source,1024,'cuda')
                if ids.shape != rgb.shape[:2]:
                    raise RuntimeError('Native GT and source dimensions differ')
                cohort = build_cohort(ids,original,classes,cfg['minimum_native_area'],cfg['core_fraction'],cfg['core_min_pixels'])
                clean = forward(image,ph,pw,rgb.shape[:2])
                clean_sha = hashlib.sha256(clean.tobytes()).hexdigest()
                baseline_exact = None
                if name in fixed:
                    case = fixed[name]
                    if clean_sha != ref_rows[name]['routes']['salign_raw']['probability_sha256']:
                        raise RuntimeError('Actual clean raw probability does not reproduce prior SHA: '+name)
                    fixed_ids, fixed_classes, png, labels = load_fixed(resolve(case['fixed_dir']),name)
                    protect(png,case['fixed_png_sha256']); protect(labels,case['fixed_classes_sha256'])
                    voted,_ = revote_instances(fixed_ids,clean)
                    validate_prediction(fixed_ids,voted,rgb.shape[:2])
                    if voted != fixed_classes:
                        raise RuntimeError('Actual clean raw classes differ from fixed reflect baseline')
                    baseline_exact = True
                clean_rows = probability_rows(cohort,clean)
                record = dict(source=name, source_sha256=file_sha(source), shape=list(ids.shape),
                    clear_probability_sha256=clean_sha, reference_baseline_exact=baseline_exact,
                    unknown_pixels=int((ids==0).sum()), filled_pixels=int(((ids>0)&~original).sum()),
                    original_pixels=int(original.sum()), gt_objects=len(cohort), input_stride=list(image.stride()),
                    conditions={'clean':dict(summary=summarize_condition(clean_rows,clean_rows),objects=clean_rows)})
                save_preview(out/'gallery'/(name+'_clean.png'),rgb,rgb,clean,'Clean raw',cfg['thumbnail_width'])
                valid = torch.zeros((1,1,1024,1024),dtype=torch.bool,device='cuda')
                valid[:,:,:1024-ph,:1024-pw] = True
                for family in ('global','local'):
                    for view in range(cfg['views_per_family']):
                        seed = int(cfg['source_seed']) + global_indices[name]*100003 + view*1009
                        altered,appearance = hard_appearance(image,valid,cfg,seed,family)
                        if appearance['minimum_slope'] <= 0 or appearance['clipped_fraction'] > cfg['max_clipped_fraction']+1e-7:
                            raise RuntimeError('Positive slope/clipping photometric gate failed')
                        probability = forward(altered,ph,pw,rgb.shape[:2])
                        values = probability_rows(cohort,probability)
                        label = family+'_'+str(view)
                        record['conditions'][label] = dict(appearance=appearance,
                            probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                            summary=summarize_condition(clean_rows,values),objects=values)
                        altered_rgb = np.rint(crop_letterbox_output(altered,1024,ph,pw,rgb.shape[:2]).cpu()[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
                        save_preview(out/'gallery'/(name+'_'+label+'.png'),rgb,altered_rgb,probability,label,cfg['thumbnail_width'])
                if first_probability is None:
                    first_probability,first_path = clean.copy(),source
                records.append(record)
                write(out/'rows.json',records)
                write(out/'status.json',dict(status='running',completed=len(records),total=len(names),training=False))
                print('RAW_STRESS',len(records),name,'clear_exact',baseline_exact,flush=True)
            rgb,image,ph,pw = prepare_image(first_path,1024,'cuda')
            if not np.array_equal(forward(image,ph,pw,rgb.shape[:2]),first_probability):
                raise RuntimeError('Clean first source changed after stress forwards')
        if tensor_digest(model.state_dict().items()) != state or precision_record() != precision:
            raise RuntimeError('Frozen model/precision changed')
        if calls != len(names)*5+1:
            raise RuntimeError('Unexpected semantic forward budget')
        if any(file_sha(Path(path)) != digest for path,digest in protected.items()):
            raise RuntimeError('Protected input changed')
        if any(file_sha(ROOT/name) != digest for name,digest in code.items()):
            raise RuntimeError('Protected code changed')
        totals = {}
        for condition in records[0]['conditions']:
            totals[condition] = {group:{key:sum(row['conditions'][condition]['summary'][group][key] for row in records)
                for key in records[0]['conditions'][condition]['summary'][group]} for group in ('all','large')}
        report = dict(format=FORMAT,complete=True,training=False,no_optimizer_steps=True,full_test_inference=False,
            official_submission=False,best_deployment_changed=False,sources=len(names),semantic_forward_calls=calls,
            expected_forward_calls=len(names)*5+1,affinity_forward_calls=0,watershed_calls=0,D5a_forward_calls=0,
            checkpoint=dict(path=str(checkpoint),sha256=cfg['checkpoint_sha256'],epoch=cfg['checkpoint_epoch']),
            frozen_state_sha256=state,frozen_exact=True,first_clear_repeat_exact=True,
            reference_four_clear_probability_and_classes_exact=all(r['reference_baseline_exact'] is True for r in records if r['source'] in INITIAL),
            minimum_native_area=cfg['minimum_native_area'],precision=precision,totals=totals,
            protected_inputs=protected,protected_inputs_exact=True,protected_sources=code,virtual_module_paths=virtual,
            elapsed_seconds=time.time()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
            caveat='Seen processed-GT original covered regions only. Unknown/fill/frame censoring retained; not complete physical GT, independent validation, or official instance mIoU. Pressure-induced wrong classes do not prove the cause of real test errors.')
        write(out/'report.json',report)
        page=['<!doctype html><meta charset="utf-8"><title>Raw semantic stress</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>Frozen raw semantic: labeled training sources only</h1>',
            '<p>Left: original. Middle: stress input. Right: fixed 0..1 P(F). No training, D5a, affinity, watershed or test labels.</p>']
        page += ['<p><a href="report.json">Summary</a> | <a href="rows.json">Objects and censoring</a></p>']
        for path in sorted((out/'gallery').glob('*.png')):
            page.append('<h2>'+html.escape(path.stem)+'</h2><img src="gallery/'+html.escape(path.name)+'">')
        (out/'index.html').write_text('\n'.join(page),encoding='utf-8')
        write(out/'status.json',dict(status='complete',sources=len(names),semantic_forward_calls=calls,training=False))
        print('RAW_STRESS_COMPLETE',len(names),calls,flush=True)
        return report
    except Exception as error:
        write(out/'status.json',dict(status='failed',error=repr(error),training=False))
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/inference/semantic_raw_stress.yaml')
    parser.add_argument('--output')
    parser.add_argument('--limit',type=int,help='4 means the prior four training sources; omitted means all32')
    run(parser.parse_args())
