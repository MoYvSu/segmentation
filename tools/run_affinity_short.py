# -*- coding: utf-8 -*-
"""固定mixed balance e20，仅改short融合；100图只读比较，不生成提交包。"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import gc
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_affinity_connectivity import relation, render
from tools.analyze_affinity_sweep_test import (
    area_bins, areas_by_phase, changes, load_prediction, pixel_class_changes,
    quantiles, read, summarize_comparison, summarize_totals)
from tools.run_affinity_distance_diagnostic import digest as array_digest
from tools.semantic_crossover import _archive_stems, _read_prediction
from train_backend_adaptation import sha, write_json
from utils.patch_diagnostic import phase_description, tile_boxes

CONTROL = 'outputs/affinity_balance/mixed'
CONTROL_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
CONTROL_PACKAGE = 'outputs/affinity_balance/mixedbalance1024.zip'
CONTROL_PACKAGE_SHA = 'ac7533b529a35ce98a94ac8b2822d9f7543ea015762c15343b62dd3207fa3c77'
SEED = 20261002
FIXED = ('test_101', 'test_116')
PHASES = ('ferrite', 'pearlite')


def require_short_config(control, candidate):
    """除模式外逐键一致；窗口/短程聚合及所有部署阈值不能顺带修改。"""
    if (control['affinity_deployment']['fusion_mode'] != 'gated'
            or control['affinity_deployment']['short_reduction'] != 'top2'
            or control['affinity_native']['overlap'] != .25):
        raise RuntimeError('Registered gated/top2/native1024 overlap=.25 contract required')
    expected = deepcopy(control)
    expected['affinity_deployment']['fusion_mode'] = 'short'
    if candidate != expected:
        raise RuntimeError('Only affinity_deployment.fusion_mode may change')


def short_config(control):
    candidate = deepcopy(control)
    candidate['affinity_deployment']['fusion_mode'] = 'short'
    require_short_config(control, candidate)
    return candidate


def area_description(ids, classes, control_median):
    areas = areas_by_phase(ids, classes)
    result = phase_description(ids, classes)
    for phase in PHASES:
        a = areas[phase]
        result[phase].update(mean_area=float(a.mean()) if len(a) else None,
                             area_quantiles=quantiles(a))
    a = areas['ferrite']
    result['ferrite'].update(tiny_under200_pixels=int(a[a < 200].sum()),
        fixed_reference_bins=area_bins(a, control_median),
        relative_tails=None if control_median is None else {
            name: dict(instances=int((a < fraction * control_median).sum()),
                       pixels=int(a[a < fraction * control_median].sum()))
            for name, fraction in [('lt_one_percent', .01), ('lt_five_percent', .05)]})
    return result


def overlap_mapping(old, new, old_classes, new_classes, control_median):
    """全图重叠对应；无GT，不把新小块或未匹配区域称为错误。"""
    if old.shape != new.shape:
        raise ValueError('Prediction shapes differ')
    left, li = np.unique(old, return_inverse=True)
    right, ri = np.unique(new, return_inverse=True)
    counts = np.bincount(li.ravel() * len(right) + ri.ravel(),
                        minlength=len(left) * len(right)).reshape(len(left), len(right))
    la, ra = counts.sum(1), counts.sum(0)
    positive_left, positive_right = np.flatnonzero(left > 0), np.flatnonzero(right > 0)
    common = counts[np.ix_(positive_left, positive_right)]
    iou = common / np.maximum(1, la[positive_left, None] + ra[None, positive_right] - common)
    a, b = linear_sum_assignment(-iou)
    pairs = {int(right[positive_right[j]]): (int(left[positive_left[i]]), float(iou[i, j]))
             for i, j in zip(a, b) if iou[i, j] >= .5}
    matched_old = {v[0] for v in pairs.values()}
    children = {int(k): [] for k, c in old_classes.items() if c == 1}
    origins = dict(old_ferrite=0, old_pearlite=0, background=0, ambiguous=0)
    rows = []
    bg = int(np.flatnonzero(left == 0)[0]) if (left == 0).any() else None
    for j in positive_right:
        new_id, area = int(right[j]), int(ra[j])
        if new_classes[str(new_id)] != 1:
            continue
        column = counts[:, j]
        maximum = int(column.max())
        winners = np.flatnonzero(column == maximum)
        fraction = maximum / area
        old_id = int(left[winners[0]]) if len(winners) == 1 and fraction > .5 else None
        origin = ('ambiguous' if old_id is None else 'background' if old_id == 0
                  else 'old_ferrite' if old_classes[str(old_id)] == 1 else 'old_pearlite')
        origins[origin] += 1
        if origin == 'old_ferrite':
            children[old_id].append((new_id, area, maximum))
        matched = pairs.get(new_id)
        rows.append(dict(new_id=new_id, area=area, dominant_old_id=old_id,
            origin=origin, maximum_old_overlap_fraction=fraction,
            old_background_fraction=float(column[bg] / area) if bg is not None else 0.,
            matched_old_iou50_id=matched[0] if matched else None,
            matched_iou50=matched[1] if matched else None, under200=area < 200,
            below_control_median_one_percent=area < .01 * control_median if control_median is not None else None,
            below_control_median_five_percent=area < .05 * control_median if control_median is not None else None))
    contribution = sum(len(value) - 1 for value in children.values())
    delta = len(rows) - len(children)
    reconstructed = contribution + sum(origins[k] for k in ('old_pearlite', 'background', 'ambiguous'))
    if delta != reconstructed:
        raise RuntimeError('Ferrite count decomposition does not close')
    parents = []
    for old_id, value in children.items():
        value = sorted(value, key=lambda v: (-v[1], v[0]))
        rest = value[1:]
        parents.append(dict(old_id=old_id, mapped_ferrite_children=len(value),
            largest_child=value[0][0] if value else None,
            additional_children=len(rest), additional_child_pixels=sum(v[1] for v in rest),
            additional_under200=sum(v[1] < 200 for v in rest),
            additional_under200_pixels=sum(v[1] for v in rest if v[1] < 200)))
    unmatched = {}
    for name, ids, classes, areas, matched in (
            ('old', left, old_classes, la, matched_old),
            ('new', right, new_classes, ra, set(pairs))):
        unmatched[name] = {}
        for phase, target in [('ferrite', 1), ('pearlite', 0)]:
            selected = np.array([int(areas[i]) for i, k in enumerate(ids)
                                 if k > 0 and int(k) not in matched and classes[str(int(k))] == target], np.int64)
            unmatched[name][phase] = dict(instances=len(selected), pixels=int(selected.sum()),
                                         area_quantiles=quantiles(selected))
    return dict(new_ferrite_regions=rows, old_ferrite_parents=parents,
        ferrite_count_decomposition=dict(old_ferrite=len(children), new_ferrite=len(rows),
            delta=delta, old_ferrite_children_minus_parents=contribution,
            new_origins=origins, reconstructed_delta=reconstructed, exact=True),
        unmatched_iou50=unmatched,
        coverage=dict(old_background_to_new_covered=int(((old == 0) & (new > 0)).sum()),
                      old_covered_to_new_background=int(((old > 0) & (new == 0)).sum())))


def verify_package(cache, archive_path, receipt_path, files, manifest, *,
                   expected_sha=CONTROL_PACKAGE_SHA, expected_images=100):
    """控制缓存绑定既有包，并核验源图原尺寸及每一个PNG/JSON。"""
    cache, archive_path, receipt_path = map(Path, (cache, archive_path, receipt_path))
    files = [Path(p) for p in files]
    stems = {p.stem for p in files}
    receipt = read(receipt_path)
    actual = sha(archive_path)
    if (len(files) != expected_images or len(stems) != expected_images or actual != expected_sha
            or receipt.get('sha256') != actual or receipt.get('images') != expected_images
            or receipt.get('files') != expected_images * 2 or receipt.get('epoch') != 20
            or receipt.get('checkpoint_sha256') != CONTROL_SHA or receipt.get('inference') != 'native1024'
            or receipt.get('flat') is not True or receipt.get('crc_ok') is not True):
        raise RuntimeError('Registered complete mixed balance result package identity differs')
    selected = [r for r in manifest.get('images', []) if r.get('view') == 'patch1024']
    records = {r['source']: r for r in selected}
    if (manifest.get('complete') is not True or manifest.get('epoch') != 20
            or manifest.get('precision') != 'FP32' or manifest.get('checkpoint_sha256') != CONTROL_SHA
            or 1024 not in manifest.get('sizes', []) or len(selected) != expected_images
            or set(records) != stems):
        raise RuntimeError('Control final manifest differs')
    expected = {s + suffix for s in stems for suffix in ('_inst.png', '_class.json')}
    if {p.name for p in cache.iterdir() if p.is_file()} != expected:
        raise RuntimeError('Control cache must contain exactly the registered 200 files')
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        if set(_archive_stems(archive)) != stems or archive.testzip() is not None:
            raise RuntimeError('Control archive cohort or CRC differs')
        for path in files:
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError('Cannot read original source: ' + str(path))
            pred = load_prediction(cache, path.stem, image.shape[:2])
            archived = _read_prediction(archive, path.stem)
            png, labels = path.stem + '_inst.png', path.stem + '_class.json'
            png_sha = sha(cache / png)
            json_sha = sha(cache / labels)
            if (png_sha != hashlib.sha256(archive.read(png)).hexdigest()
                    or not np.array_equal(pred[0], archived[0]) or pred[1] != archived[1]
                    or phase_description(*pred) != records[path.stem]['phases']):
                raise RuntimeError('Control cache differs from existing package/manifest: ' + path.stem)
            rows.append(dict(source=path.stem, shape=list(image.shape[:2]), source_sha256=sha(path),
                png_sha256=png_sha, classes_sha256=json_sha,
                class_json_bytes_exact=json_sha == hashlib.sha256(archive.read(labels)).hexdigest()))
    return dict(passed=True, archive_sha256=actual, receipt_sha256=sha(receipt_path), images=rows,
                png_bytes_exact=True, class_dicts_exact=True, crc_ok=True)


def source_names(parent_sources):
    """新调用闭包单独封存；父134份仅引用和复核，绝不改写。"""
    found = set(parent_sources)
    pending = ['tools/run_affinity_short.py']
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        if not path.is_file() or path.suffix != '.py':
            raise RuntimeError('Missing source: ' + name)
        found.add(name)
        package, modules = Path(name).parent.parts, []
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'), filename=name)):
            if isinstance(node, ast.Import):
                modules.extend(a.name.split('.') for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                module = prefix + (node.module.split('.') if node.module else [])
                modules.append(module)
                modules.extend(module + a.name.split('.') for a in node.names if a.name != '*')
        for parts in modules:
            if parts:
                for candidate in (Path(*parts).with_suffix('.py'), Path(*parts) / '__init__.py'):
                    if (ROOT / candidate).is_file():
                        pending.append(candidate.as_posix())
                for length in range(1, len(parts)):
                    init = Path(*parts[:length]) / '__init__.py'
                    if (ROOT / init).is_file():
                        pending.append(init.as_posix())
    return tuple(sorted(found))


def verify_new_sources(output, sources):
    for name, expected in sources.items():
        if (Path(name).is_absolute() or '..' in Path(name).parts
                or sha(ROOT / name) != expected or sha(Path(output) / 'source' / name) != expected):
            raise RuntimeError('New current/snapshot source changed: ' + name)


def precision_record(torch):
    record = dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(), autocast=False,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if record['cudnn_tf32'] is not True or record['matmul_tf32'] is not False or record['matmul_precision'] != 'highest':
        raise RuntimeError('Original FP32 convolution-TF32/matmul precision contract differs')
    return record


def save_candidate(directory, stem, result, shape):
    from tools.affinity_native_views import verify_prediction
    verify_prediction(result['instances'], result['classes'], shape)
    path = directory / (stem + '_inst.png')
    if not cv2.imwrite(str(path), result['instances']):
        raise RuntimeError('Failed to save uint16 PNG')
    write_json(directory / (stem + '_class.json'), result['classes'])
    actual = load_prediction(directory, stem, shape)
    if not np.array_equal(actual[0], result['instances']) or actual[1] != result['classes']:
        raise RuntimeError('Saved candidate did not reproduce memory')
    return dict(png_sha256=sha(path), classes_sha256=sha(directory / (stem + '_class.json')))


def summarize(images):
    totals = {arm: summarize_totals(images, arm) for arm in ('gated', 'short')}
    for arm in totals:
        selected = [r['views'][arm]['ferrite']['relative_tails'] for r in images
                    if r['views'][arm]['ferrite']['relative_tails'] is not None]
        totals[arm]['ferrite']['relative_tails'] = {name: {
            key: sum(v[name][key] for v in selected) for key in ('instances', 'pixels')}
            for name in ('lt_one_percent', 'lt_five_percent')}
    pooled = changes(*[{p: dict(totals[a][p], mean_area=totals[a][p]['pooled_mean_area']) for p in PHASES}
                       for a in ('gated', 'short')])
    delta = [r['changes']['short']['ferrite']['mean_area_percent'] for r in images
             if r['changes']['short']['ferrite']['mean_area_percent'] is not None]
    decompositions = [r['mapping']['ferrite_count_decomposition'] for r in images]
    regions = [v for r in images for v in r['mapping']['new_ferrite_regions']]
    parents = [v for r in images for v in r['mapping']['old_ferrite_parents']]
    return dict(totals=totals, pooled_changes=pooled,
        comparisons=summarize_comparison(images, 'short'),
        ferrite_per_image_mean_area={arm: quantiles([r['views'][arm]['ferrite']['mean_area'] for r in images
            if r['views'][arm]['ferrite']['mean_area'] is not None]) for arm in ('gated', 'short')},
        ferrite_per_image_mean_area_percent_change=quantiles(delta),
        ferrite_mean_area_change_gt_one_percent=sum(abs(v) > 1 for v in delta),
        ferrite_mean_area_change_gt_five_percent=sum(abs(v) > 5 for v in delta),
        new_ferrite_origin_tails={origin: dict(
            instances=sum(v['origin'] == origin for v in regions),
            pixels=sum(v['area'] for v in regions if v['origin'] == origin),
            under200=sum(v['origin'] == origin and v['under200'] for v in regions),
            under200_pixels=sum(v['area'] for v in regions if v['origin'] == origin and v['under200']))
            for origin in ('old_ferrite', 'old_pearlite', 'background', 'ambiguous')},
        old_ferrite_parent_children=dict(no_mapped_child=sum(v['mapped_ferrite_children'] == 0 for v in parents),
            multiple_mapped_children=sum(v['mapped_ferrite_children'] > 1 for v in parents),
            **{k: sum(v[k] for v in parents) for k in ('additional_children', 'additional_child_pixels',
                                                       'additional_under200', 'additional_under200_pixels')}),
        background_coverage_changes={k: sum(r['mapping']['coverage'][k] for r in images)
            for k in ('old_background_to_new_covered', 'old_covered_to_new_background')},
        unmatched_iou50={arm: {phase: dict(
            instances=sum(r['mapping']['unmatched_iou50'][arm][phase]['instances'] for r in images),
            pixels=sum(r['mapping']['unmatched_iou50'][arm][phase]['pixels'] for r in images))
            for phase in PHASES} for arm in ('old', 'new')},
        ferrite_count_decomposition=dict(delta=sum(v['delta'] for v in decompositions),
            old_ferrite_children_minus_parents=sum(v['old_ferrite_children_minus_parents'] for v in decompositions),
            new_origins={k: sum(v['new_origins'][k] for v in decompositions)
                         for k in ('old_ferrite', 'old_pearlite', 'background', 'ambiguous')},
            all_exact=all(v['exact'] for v in decompositions)))


def make_gallery(output, files, cache, candidate, selection):
    gallery = output / 'gallery'
    gallery.mkdir()
    paths = {p.stem: p for p in files}
    pages = ['<!doctype html><meta charset="utf-8"><title>Fixed e20 short fusion</title>',
        '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
        '<h1>固定mixed balance e20：gated / short</h1>',
        '<p>金色=铁素体；蓝色=珠光体。无测试GT，数量/面积/小块是预测变化，不是正确性。</p>',
        '<p>窗口位置仅供定位，不能据此认定接缝缺陷。固定两图+预选随机八图；另附面积变化最大两图。</p>',
        '<a href="summary.json">汇总</a> | <a href="report.json">逐图统计与身份回执</a>']
    for group, names in [('fixed', selection['fixed']), ('seeded_random_nonmonitor', selection['random']),
                         ('largest_observed_area_changes_not_random', selection['largest_area_changes'])]:
        pages.append('<h2>' + html.escape(group) + '</h2>')
        for stem in names:
            path = paths[stem]
            original = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            shape = original.shape[:2]
            predictions = [load_prediction(folder, stem, shape) for folder in (cache, candidate)]
            for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
                name = stem + '_' + suffix + '.png'
                if not (gallery / name).exists():
                    render(path, predictions, ['gated control', 'short only'], gallery / name,
                           stem + ' | e20/native1024; no GT', crop=crop, width=360)
                pages.append('<p>' + html.escape(stem + ' ' + group + ' ' + suffix)
                             + '<br><img loading="lazy" src="gallery/' + name + '"></p>')
            if group == 'fixed':
                boxes = tile_boxes(shape, 1024, .25)
                outline = original.copy()
                for y, x, y1, x1 in boxes:
                    cv2.rectangle(outline, (x, y), (x1-1, y1-1), (0, 160, 255), max(2, round(max(shape)/1000)))
                width = 650
                cv2.imwrite(str(gallery / (stem + '_windows.png')),
                            cv2.resize(outline, (width, round(shape[0]*width/shape[1])), interpolation=cv2.INTER_AREA))
                # ROI围绕实际窗口起点，说明位置，不把变化解释成窗口错误。
                y, x, _, _ = next((b for b in boxes if b[0] > 0 or b[1] > 0), boxes[0])
                h, w = shape
                cy, cx = min(h-256, max(256, y)), min(w-256, max(256, x))
                crop = [max(0, cx-256)/w, max(0, cy-256)/h, min(w, cx+256)/w, min(h, cy+256)/h]
                render(path, predictions, ['gated control', 'short only'], gallery / (stem + '_window_detail.png'),
                       stem + ' window-position ROI; no seam accuracy claim', crop=crop, width=360)
                for suffix in ('windows', 'window_detail'):
                    pages.append('<p><img src="gallery/' + stem + '_' + suffix + '.png"></p>')
    (output / 'index.html').write_text('\n'.join(pages), encoding='utf8')


def run(args):
    import torch
    from data.mim_dataset import list_images
    from models.backend_adaptation import build_backend, load_backend
    from tools.affinity_native_views import predict_views
    from tools.analyze_affinity_p100 import preflight
    from train_affinity_balance import comparable
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_backend_adaptation import tensor_digest
    from train_direct_semantic_affinity import set_seed
    from utils.config import load_config, project_path
    if not torch.cuda.is_available():
        raise RuntimeError('Use GPU server sam2_env; no CPU substitute for formal inference')
    output = (ROOT / args.output).resolve()
    if output.exists() or not output.is_relative_to((ROOT / 'outputs').resolve()):
        raise RuntimeError('A fresh ignored outputs directory is required')
    output.mkdir(parents=True)
    started = time.time()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    write_json(output / 'status.json', dict(status='running', stage='preflight', training=False))
    try:
        pconfig = load_config('config/train/affinity_source_p100.yaml')
        checks = preflight(pconfig, ROOT / 'outputs/affinity_p100')
        if len(checks['sources']) != 134 or checks['identities']['r1']['checkpoint_sha256'] != CONTROL_SHA:
            raise RuntimeError('Registered 134-source/current best control identity differs')
        config = deepcopy(checks['states']['r1']['config'])
        if comparable(config) != comparable(checks['configs']['r1']):
            raise RuntimeError('Actual control config differs from sealed lineage')
        candidate_config = short_config(config)
        precision = precision_record(torch)
        files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
        if len(files) != 100 or len({p.stem for p in files}) != 100:
            raise RuntimeError('Exactly 100 unique test sources required')
        if not set(FIXED).issubset(p.stem for p in files):
            raise RuntimeError('Fixed technical parity sources missing')
        monitors = set(config['backend_adaptation']['monitor']['images']) | set(FIXED)
        eligible = sorted(p.stem for p in files if p.stem not in monitors)
        random = sorted(np.random.default_rng(SEED).choice(eligible, 8, replace=False).tolist())
        selection = dict(seed=SEED, fixed=list(FIXED), random=random,
            excluded_original_monitor=sorted(monitors), largest_area_changes=[],
            largest_selection_rule='Two largest absolute per-image ferrite mean-area percent changes; distribution slice, not accuracy')
        write_json(output / 'selection.json', selection)
        folder = checks['folders']['r1']
        cache = folder / 'deployment/patch1024'
        manifest_path = folder / 'deployment/manifest.json'
        manifest = read(manifest_path)
        records = {r['source']: r for r in manifest['images'] if r['view'] == 'patch1024'}
        package_receipt = verify_package(cache, ROOT / CONTROL_PACKAGE,
            ROOT / 'outputs/affinity_balance/mixed_package.json', files, manifest)
        source_inventory = source_names(checks['sources'])
        sources = {name: sha(ROOT / name) for name in source_inventory if name not in checks['sources']}
        for name in sources:
            target = output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        verify_new_sources(output, sources)
        distance_report_path = ROOT / 'outputs/affinity_distance/report.json'
        distance_report = read(distance_report_path)
        if (distance_report.get('complete') is not True or distance_report.get('no_optimizer_steps') is not True
                or distance_report['control']['checkpoint_sha256'] != CONTROL_SHA
                or distance_report['runtime'] != checks['runtime']
                or distance_report['parent_pipeline_sha256'] != checks['pipeline_sha256']
                or distance_report['parent_sources'] != checks['sources']):
            raise RuntimeError('Prior authorized distance diagnostic identity differs')
        for name, expected in distance_report['sources'].items():
            if (name not in sources or sources[name] != expected
                    or sha(ROOT / 'outputs/affinity_distance/source' / name) != expected):
                raise RuntimeError('Prior diagnostic source seal differs: ' + name)
        distance_sha = sha(distance_report_path)
        state = checks['states']['r1']
        set_seed(config['backend_adaptation']['seed'])
        initial, metadata = build_backend(config, 'cuda')
        if (tensor_digest(initial.state_dict().items()) != state['initial_state_sha256']
                or frozen_digest(initial) != state['frozen_state_sha256'] or metadata != state['initialization']):
            raise RuntimeError('Common original Stage1/joint-v3 initialization/frozen state differs')
        del initial
        gc.collect()
        torch.cuda.empty_cache()
        model, bundle = load_backend(folder / 'final.pt', config, 'cuda')
        restorer = get_restorer(config, 'cuda')
        before = tensor_digest(model.state_dict().items())
        restore_before = tensor_digest(restorer.state_dict().items())
        if (bundle['epoch'] != 20 or before != state['final_state_sha256']
                or frozen_digest(model) != state['frozen_state_sha256']
                or restore_before != state['restorer_state_sha256']
                or any(p.requires_grad for p in list(model.parameters()) + list(restorer.parameters()))
                or any(m.training for m in list(model.modules()) + list(restorer.modules()))):
            raise RuntimeError('Actual control/D5a model or frozen inference state differs')
        parameters = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in restorer.parameters())
        if parameters >= 500_000_000:
            raise RuntimeError('Competition parameter budget exceeded')
        target = output / 'deployment/patch1024'
        target.mkdir(parents=True)
        maps = output / 'maps'
        maps.mkdir()
        report = dict(training=False, official_accuracy=False, no_optimizer_steps=True,
            best_deployment_changed=False, submission_package_created=False,
            only_variable='affinity_deployment.fusion_mode: gated -> short',
            config=config, candidate_config=candidate_config, control=checks['identities']['r1'],
            parameter_count=parameters, parent_pipeline_sha256=checks['pipeline_sha256'],
            parent_sources=checks['sources'], new_sources=sources, runtime=checks['runtime'], precision=precision,
            parent_smoke_binding=checks['smoke_binding'], prior_distance_report_sha256=distance_sha,
            initial_metadata=metadata, initial_state_sha256=state['initial_state_sha256'],
            model_state_sha256=before, frozen_state_sha256=state['frozen_state_sha256'],
            restorer_state_sha256=restore_before, control_cache_package=package_receipt,
            semantic_isolation=dict(full_100='Code and config path fixed; old semantic tensors are not cached for all sources',
                actual_tensor_byte_parity_sources=list(FIXED)), selection=selection, parity=[], images=[])
        write_json(output / 'preflight.json', report)
        prepared = {}
        # 固定两源先放行，仍沿完整100图索引取得seed；其余98图不重复控制推理。
        write_json(output / 'status.json', dict(status='running', stage='two_source_gate', training=False))
        for index, path in enumerate(files):
            if path.stem not in FIXED:
                continue
            seed = config['backend_adaptation']['restoration']['inference_seed'] + index
            fixed, short_fixed = {}, {}
            rgb, gated = predict_views(model, restorer, path, config, 'cuda', seed, [1024], fixed_audit=fixed)
            _, short = predict_views(model, restorer, path, candidate_config, 'cuda', seed, [1024], fixed_audit=short_fixed)
            old, classes = load_prediction(cache, path.stem, rgb.shape[:2])
            g, s = gated['patch1024'], short['patch1024']
            if (not np.array_equal(old, g['instances']) or classes != g['classes']
                    or g['phases'] != records[path.stem]['phases'] or g['blend'] != records[path.stem]['blend']
                    or fixed != short_fixed or len(fixed) != 2):
                raise RuntimeError('Exact control/fixed semantic technical parity failed: ' + path.stem)
            receipt = save_candidate(target, path.stem, s, rgb.shape[:2])
            np.savez_compressed(maps / (path.stem + '_gated.npz'), boundary=g['boundary'])
            np.savez_compressed(maps / (path.stem + '_short.npz'), boundary=s['boundary'])
            prepared[path.stem] = dict(phases=s['phases'], blend=s['blend'], fixed_semantic_inputs=short_fixed,
                output=receipt, boundary_sha256=array_digest(s['boundary']))
            report['parity'].append(dict(source=path.stem, full_source_index=index, seed=seed,
                final_png_array_exact=True, class_dict_exact=True, manifest_phases_blend_exact=True,
                fixed_semantic_inputs=fixed, short_fixed_semantic_inputs=short_fixed))
            del rgb, gated, short, g, s
        write_json(output / 'parity.json', dict(passed=len(report['parity']) == 2, images=report['parity']))
        rows = []
        for index, path in enumerate(files):
            seed = config['backend_adaptation']['restoration']['inference_seed'] + index
            if path.stem in prepared:
                current = prepared[path.stem]
                shape = next(r['shape'] for r in package_receipt['images'] if r['source'] == path.stem)
            else:
                fixed = {}
                rgb, views = predict_views(model, restorer, path, candidate_config, 'cuda', seed, [1024], fixed_audit=fixed)
                value = views['patch1024']
                shape = list(rgb.shape[:2])
                current = dict(phases=value['phases'], blend=value['blend'], fixed_semantic_inputs=fixed,
                    output=save_candidate(target, path.stem, value, shape), boundary_sha256=array_digest(value['boundary']))
                if path.stem in random:
                    np.savez_compressed(maps / (path.stem + '_short.npz'), boundary=value['boundary'])
                del rgb, views, value
            old = load_prediction(cache, path.stem, shape)
            new = load_prediction(target, path.stem, shape)
            if phase_description(*old) != records[path.stem]['phases'] or phase_description(*new) != current['phases']:
                raise RuntimeError('Saved final phases differ: ' + path.stem)
            area = areas_by_phase(*old)['ferrite']
            median = float(np.median(area)) if len(area) else None
            descriptions = {arm: area_description(*prediction, median) for arm, prediction in [('gated', old), ('short', new)]}
            rel = relation(old[0], new[0], old[1], new[1])
            rel.update(pixel_class_changes(old, new), final_ids_exact=bool(np.array_equal(old[0], new[0])), class_json_exact=old[1] == new[1])
            row = dict(source=path.stem, full_source_index=index, seed=seed, shape=list(shape),
                control_ferrite_median=median, views=descriptions, changes={'short': changes(descriptions['gated'], descriptions['short'])},
                relations={'short': rel}, mapping=overlap_mapping(old[0], new[0], old[1], new[1], median),
                fixed_semantic_inputs=current['fixed_semantic_inputs'], boundary_sha256=current['boundary_sha256'], output=current['output'])
            report['images'].append(row)
            rows.append(dict(source=path.stem, view='patch1024', seed=seed, phases=current['phases'], blend=current['blend'],
                             output=current['output'], fixed_semantic_inputs=current['fixed_semantic_inputs']))
            write_json(output / 'status.json', dict(status='running', stage='short_inference', training=False, completed_images=index+1))
            print('SHORT', index+1, path.stem, flush=True)
        expected = {p.stem + suffix for p in files for suffix in ('_inst.png', '_class.json')}
        if {p.name for p in target.iterdir()} != expected or len(rows) != 100:
            raise RuntimeError('Incomplete candidate 100-image output')
        largest = sorted(report['images'], key=lambda r: (-abs(r['changes']['short']['ferrite']['mean_area_percent'] or 0), r['source']))[:2]
        selection['largest_area_changes'] = [r['source'] for r in largest]
        write_json(output / 'selection.json', selection)
        summary = summarize(report['images'])
        make_gallery(output, files, cache, target, selection)
        # 完成时重新核验父源、runtime、原缓存、模型及新封存脚本；不改最佳部署。
        after_checks = preflight(pconfig, ROOT / 'outputs/affinity_p100')
        if (after_checks != checks or sha(distance_report_path) != distance_sha
                or precision_record(torch) != precision or tensor_digest(model.state_dict().items()) != before
                or tensor_digest(restorer.state_dict().items()) != restore_before
                or verify_package(cache, ROOT / CONTROL_PACKAGE, ROOT / 'outputs/affinity_balance/mixed_package.json', files, read(manifest_path)) != package_receipt):
            raise RuntimeError('Read-only source/runtime/model/control receipts changed')
        verify_new_sources(output, sources)
        report.update(complete=True, summary=summary, source_and_parent_receipts_unchanged=True,
            model_and_restorer_states_unchanged=True, elapsed_seconds=time.time()-started,
            caveats=['无测试GT：新小块、未匹配区域、数量与平均面积变化不是错误或成绩。',
                     '只复用既有控制100图；两源实际语义字节一致，其余源通过代码与配置隔离。',
                     '类别翻转可由实例投票区域变化引起，未训练或更换语义头。'])
        write_json(output / 'deployment/manifest.json', dict(complete=True, checkpoint_sha256=CONTROL_SHA,
            epoch=20, updates=1280, sizes=[1024], precision='FP32', fusion_mode='short',
            semantic='fixed full-image D5a geometry plus raw final vote', images=rows))
        write_json(output / 'report.json', report)
        write_json(output / 'summary.json', dict(complete=True, only_variable=report['only_variable'], control=report['control'],
            selection=selection, parity=report['parity'], semantic_isolation=report['semantic_isolation'],
            parameter_count=parameters, caveats=report['caveats'], **summary))
        write_json(output / 'status.json', dict(status='complete', stage='done', training=False, completed_images=100,
            elapsed_seconds=report['elapsed_seconds'], submission_package_created=False))
        # 仅诊断图册，明确排除预测提交目录与源码；不建立submission.zip。
        with zipfile.ZipFile(output / 'report.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                relative = path.relative_to(output)
                if path.is_file() and relative.parts[0] not in {'source', 'deployment', 'maps'} and path.suffix in {'.json', '.png', '.html'}:
                    archive.write(path, relative.as_posix())
        with zipfile.ZipFile(output / 'report.zip') as archive:
            if archive.testzip() is not None:
                raise RuntimeError('Diagnostic report archive CRC failed')
        print('COMPLETE SHORT100', json.dumps(summary['pooled_changes']), flush=True)
    except Exception as exc:
        write_json(output / 'status.json', dict(status='failed', stage='error', training=False, error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/affinity_short')
    run(parser.parse_args())
