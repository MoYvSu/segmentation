# -*- coding: utf-8 -*-
"""固定全图 control 与原生 p1024 连续边界各半融合；仅小样本诊断，不训练。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import html
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fuse_probability(whole, local, local_weight=.5):
    """同坐标、同概率定义的连续图融合；端点保持原值，不能先阈值化。"""
    whole, local = np.asarray(whole), np.asarray(local)
    weight = float(local_weight)
    if not 0 <= weight <= 1 or not np.isfinite(weight):
        raise ValueError('Invalid fusion weight')
    if whole.shape != local.shape or whole.ndim != 2:
        raise ValueError('Fusion needs equal two-dimensional maps')
    if any(not np.isfinite(value).all() or (value < 0).any() or (value > 1).any()
           for value in (whole, local)):
        raise ValueError('Invalid boundary probability')
    if weight == 0:
        return whole.copy()
    if weight == 1:
        return local.copy()
    return ((1 - weight) * whole + weight * local).astype(np.float32)


@contextmanager
def affinity_head(model, head):
    original = model.affinity_decoder
    try:
        model.affinity_decoder = head
        yield
    finally:
        model.affinity_decoder = original


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sources(config):
    from data.mim_dataset import list_images
    from utils.config import project_path
    prior = Path(project_path(config, config['affinity_native']['prior_probe']))
    train = [case for case in json.loads((prior/'cases.json').read_text()) if case['kind'] == 'train']
    if len(train) != 4:
        raise RuntimeError('Expected original four seen GT sources')
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    if len(files) != 100:
        raise RuntimeError('Expected exactly 100 test sources')
    fixed = ['test_101', 'test_116']
    excluded = config['backend_adaptation']['monitor']['images']
    random = sorted(np.random.default_rng(20261001).choice(
        [path.stem for path in files if path.stem not in excluded], 6, replace=False).tolist())
    selected = set(fixed + random)
    seed = config['backend_adaptation']['restoration']['inference_seed']
    test = [dict(kind='test', name=path.stem, path=str(path), index=i, seed=seed+i)
            for i, path in enumerate(files) if path.stem in selected]
    return train + test, dict(fixed=fixed, random=random, seed=20261001)


def run(args):
    import cv2
    import torch
    from models.backend_adaptation import load_backend, restore_first
    from tools.affinity_connectivity_views import decode_native
    from tools.affinity_native_views import verify_prediction
    from tools.analyze_affinity_connectivity import load_directory, relation, render
    from tools.probe_affinity_patch import boundary_grid, load_gt, tensor_from_rgb
    from tools.render_backend_ablation import text_line, write_png
    from tools.run_affinity_preserve import trusted_changes
    from tools.semantic_crossover import revote_instances
    from train_affinity_connectivity import get_restorer
    from train_backend_adaptation import prediction_maps, sha, tensor_digest
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    from utils.config import load_config, project_path
    from utils.patch_diagnostic import BlendMap, phase_description, tile_boxes, trusted_phase_metrics

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env GPU for actual deployment diagnostic')
    config = load_config(args.config)
    out = Path(project_path(config, args.output))
    out.mkdir(parents=True, exist_ok=False)
    cases, selection = sources(config)
    write(out/'cases.json', cases)
    write(out/'config.json', config)
    write(out/'status.json', dict(status='running', training=False))
    checkpoint = Path(project_path(config, config['affinity_native']['reused_control'], 'final.pt'))
    native_checkpoint = Path(project_path(config, 'outputs/affinity_native/p1024/final.pt'))
    expected = config['affinity_native']['control_sha256']
    native_expected = 'b750a3da2e93195bfa1f220aa1873ac7e85fe6ab34302afc614959fb0e583fca'
    if sha(checkpoint) != expected or sha(native_checkpoint) != native_expected:
        raise RuntimeError('Wrong registered comparison checkpoint')
    model, whole_bundle = load_backend(checkpoint, config, 'cuda')
    candidate, native_bundle = load_backend(native_checkpoint, config, 'cpu')
    if whole_bundle['architecture'] != native_bundle['architecture'] or whole_bundle['deployment'] != native_bundle['deployment']:
        raise RuntimeError('Architecture or deployed inference contract changed')
    frozen = lambda value: tensor_digest((n, t) for n, t in value.state_dict().items()
                                          if not n.startswith('affinity_decoder.'))
    shared = frozen(model)
    if shared != frozen(candidate):
        raise RuntimeError('Encoder/LoRA/semantic or other frozen module differs')
    local_head = candidate.affinity_decoder.to('cuda').eval().requires_grad_(False)
    # 两个 affinity 头共享完整冻结编码器、LoRA和语义；不重复计数两个 SAM2。
    del candidate, whole_bundle, native_bundle
    restorer = get_restorer(config, 'cuda')
    model.eval().requires_grad_(False)
    params = dict(shared_model=sum(p.numel() for p in model.parameters()),
                  extra_local_affinity=sum(p.numel() for p in local_head.parameters()),
                  d5a=sum(p.numel() for p in restorer.parameters()))
    params['total'] = sum(params.values())
    if params['total'] >= 500000000:
        raise RuntimeError('Combined deployment exceeds competition parameter limit')
    original_digest = tensor_digest(model.state_dict().items())
    local_digest = tensor_digest(local_head.state_dict().items())
    restore_digest = tensor_digest(restorer.state_dict().items())
    source_dir = out/'source'
    source_dir.mkdir()
    source_names = ['tools/probe_affinity_scale_fusion.py', 'tools/affinity_native_views.py',
                    'tools/affinity_connectivity_views.py', 'utils/patch_diagnostic.py']
    for name in source_names:
        target = source_dir/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    gallery = out/'gallery'
    gallery.mkdir()
    started = time.time()
    records = []
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        for case in cases:
            rgb, semantic, whole_map = prediction_maps(model, case['path'], config, 'cuda', restorer, case['seed'])
            _, raw, ph, pw = prepare_image(case['path'], 1024, 'cuda')
            raw_logits = model.semantic_decoder(model.encoder(raw.float()), raw.float())
            if isinstance(raw_logits, dict):
                raw_logits = raw_logits['semantic_logits']
            probability = crop_letterbox_output(raw_logits.float(), 1024, ph, pw, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()
            semantic_hash = tensor_digest([('geometry_semantic', semantic)])
            probability_hash = tensor_digest([('raw_probability', torch.from_numpy(probability))])
            boxes = tile_boxes(rgb.shape[:2], 1024, config['affinity_native']['overlap'])
            blend = BlendMap(rgb.shape[:2])
            with affinity_head(model, local_head):
                for i, box in enumerate(boxes):
                    y, x, y1, x1 = box
                    image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], 'cuda')
                    restored = restore_first(restorer, image, case['seed'] + 100000 + 1024 * 1000 + i)
                    grid = boundary_grid(model, restored, config)
                    value = crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
                    blend.add(box, value)
            local_map, blend_audit = blend.finish()
            whole_map = whole_map[0, 0].numpy()
            maps = dict(whole=fuse_probability(whole_map, local_map, 0),
                        native=fuse_probability(whole_map, local_map, 1),
                        half=fuse_probability(whole_map, local_map, .5))
            predictions = {}
            record = dict(source=case['name'], kind=case['kind'], seed=case['seed'],
                          tiles=len(boxes), blend=blend_audit, semantic_sha256=semantic_hash,
                          raw_probability_sha256=probability_hash, variants={}, endpoint_parity={})
            gt = load_gt(case, config)
            for name, boundary in maps.items():
                ids, _ = decode_native(semantic, torch.from_numpy(boundary)[None, None], config)
                classes, _ = revote_instances(ids, probability)
                verify_prediction(ids, classes, rgb.shape[:2])
                folder = out/'predictions'/case['name']/name
                folder.mkdir(parents=True)
                if not cv2.imwrite(str(folder/(case['name']+'_inst.png')), ids):
                    raise RuntimeError('Could not save final PNG')
                write(folder/(case['name']+'_class.json'), classes)
                predictions[name] = ids, classes
                row = dict(phases=phase_description(ids, classes))
                if gt is not None:
                    g, trust, gc = gt
                    row['known_domain'] = trusted_phase_metrics(g, gc, trust, ids, classes)
                if name == 'half':
                    row['relations_vs_whole'] = relation(*[predictions['whole'][0], ids, predictions['whole'][1], classes])
                    if gt is not None:
                        changes = trusted_changes(g, trust, predictions['whole'][0], ids)
                        write(folder/'trusted_changes.json', changes)
                        row['trusted_changes_vs_whole'] = changes['summary']
                record['variants'][name] = row
            if case['kind'] == 'test':
                references = dict(whole=Path(project_path(config, config['affinity_native']['reused_control'], 'deployment')),
                                  native=Path(project_path(config, 'outputs/affinity_native/p1024/deployment/patch1024')))
            else:
                references = dict(whole=Path(project_path(config, config['affinity_native']['prior_probe'], case['name'], 'global_d5a')),
                                  native=Path(project_path(config, 'outputs/native_analysis/train', case['name'], 'p1024_native')))
            for name, directory in references.items():
                ids, classes = load_directory(directory, case['name'])
                current, cc = predictions[name]
                exact = bool(np.array_equal(ids, current) and cc == classes)
                record['endpoint_parity'][name] = exact
                if not exact:
                    raise RuntimeError('Complete final endpoint output differs: '+case['name']+'/'+name)
            for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
                render(case['path'], [predictions[n] for n in ('whole', 'native', 'half')],
                       ['control whole', 'p1024 native', 'fixed 50/50'], gallery/(case['name']+'_'+suffix+'.png'),
                       case['name']+' prediction only; no inferred test GT', crop=crop, width=420)
            panels = []
            for name, value in maps.items():
                value = cv2.resize(value, (420, round(420*rgb.shape[0]/rgb.shape[1])), interpolation=cv2.INTER_AREA)
                heat = cv2.applyColorMap(np.rint(np.clip(value, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                heat = cv2.copyMakeBorder(heat, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                text_line(heat, name+' boundary: fixed 0..1', (5, 21), 420, size=.45)
                panels.append(heat)
            write_png(gallery/(case['name']+'_boundary.png'), np.concatenate(panels, axis=1))
            records.append(record)
            write(out/'rows.json', records)
            write(out/'status.json', dict(status='running', completed=len(records), total=len(cases), source=case['name'], training=False))
            print(json.dumps(dict(source=case['name'], endpoint_parity=record['endpoint_parity'],
                                  phases={n:r['phases'] for n,r in record['variants'].items()})), flush=True)
    if (original_digest != tensor_digest(model.state_dict().items()) or
        local_digest != tensor_digest(local_head.state_dict().items()) or
        restore_digest != tensor_digest(restorer.state_dict().items())):
        raise RuntimeError('Frozen deployment tensor changed')
    totals = {}
    for name in ('whole', 'native', 'half'):
        totals[name] = {}
        for phase in ('ferrite', 'pearlite'):
            values = [r['variants'][name]['known_domain'][phase] for r in records if r['kind'] == 'train']
            sums = {key:sum(r[key] for r in values) for key in
                    ('gt_instances', 'pred_instances_in_known_domain', 'matches', 'iou_sum', 'split_gt', 'merged_pred')}
            sums['matched_iou'] = sums['iou_sum']/max(1, sums['matches'])
            sums['gt_penalized_iou'] = sums['iou_sum']/max(1, sums['gt_instances'])
            totals[name][phase] = sums
    report = dict(complete=True, training=False, official_submission=False, full_test_inference=False,
                  scope='4 seen training sources original covered GT only; 8 unlabeled test images. Mechanism diagnosis, not official accuracy.',
                  selection=selection, parameters=params, frozen_exact=True, shared_frozen_sha256=shared,
                  full_output_endpoint_parity=True, sources={n:sha(ROOT/n) for n in source_names},
                  checkpoints=dict(control=dict(path=str(checkpoint), epoch=20, sha256=expected),
                                   native=dict(path=str(native_checkpoint), epoch=20, sha256=native_expected),
                                   d5a=config['backend_adaptation']['restoration']),
                  local_weight=.5, samples=len(records), known_domain_totals=totals,
                  elapsed_seconds=time.time()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2)
    write(out/'report.json', report)
    page = ['<!doctype html><meta charset="utf-8"><title>固定尺度融合诊断</title>',
            '<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:100%;background:white}section{margin:28px 0}</style>',
            '<h1>全图 control / p1024 原生 / 固定50%融合</h1>',
            '<p>仅12图诊断，不训练、不提交。金=铁素体，蓝=珠光体。训练GT图存在遗漏；测试图没有GT。</p>',
            '<p><a href="report.json">审计及训练已知域汇总</a> · <a href="rows.json">逐图诊断</a></p>']
    for path in sorted(gallery.glob('*_detail.png')):
        name = html.escape(path.relative_to(out).as_posix())
        page.append(f'<section><h2>{html.escape(path.stem)}</h2><img src="{name}"></section>')
    (out/'index.html').write_text('\n'.join(page), encoding='utf-8')
    with zipfile.ZipFile(out/'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in [*out.glob('*.json'), out/'index.html', *gallery.glob('*.png')]:
            archive.write(path, path.relative_to(out).as_posix())
    write(out/'status.json', dict(status='complete', training=False, completed=len(records)))
    print('FUSION COMPLETE '+json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_native.yaml')
    parser.add_argument('--output', default='outputs/scale_fusion')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        from utils.config import load_config, project_path
        out = Path(project_path(load_config(args.config), args.output))
        if out.is_dir() and not isinstance(error, FileExistsError):
            write(out/'status.json', dict(status='failed', training=False, error=repr(error)))
        raise
