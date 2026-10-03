# -*- coding: utf-8 -*-
"""CPU读取固定r1/N075/N125最终输出；组成和拓扑变化不是测试准确率。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from tools.analyze_affinity_connectivity import relation, render, write
from train_affinity_balance import comparable
from train_affinity_sweep import reference_config
from utils.config import load_config, project_path
from utils.patch_diagnostic import phase_description

CONTROL = 'outputs/affinity_balance/mixed'
CONTROL_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
ARMS = {'mixed': 1., 'n075': .75, 'n125': 1.25}
PHASES = ('ferrite', 'pearlite')
BINS = (('lt_one_percent', 0., .01), ('one_percent_to_quarter', .01, .25),
        ('ge_quarter', .25, float('inf')))
RELATION_SUMS = ('old_instances', 'new_instances', 'split_relations', 'merge_relations',
    'matched_iou50', 'matched_iou95', 'class_changed_matched_iou50',
    'class_changed_pixels', 'common_covered_pixels')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique)


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite distribution')
    labels = ('minimum', 'p01', 'p05', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'maximum')
    return dict(zip(labels, map(float, np.quantile(values, [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]))))


def load_prediction(folder, source, shape):
    path = folder / (source + '_inst.png')
    ids = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    classes = read(folder / (source + '_class.json'))
    if ids is None or ids.dtype != np.uint16 or ids.ndim != 2 or tuple(ids.shape) != tuple(shape):
        raise ValueError('Expected original-shape single-channel uint16 PNG: ' + str(path))
    if int(ids.max()) > 65535 or not isinstance(classes, dict):
        raise ValueError('Invalid final instance/class contract: ' + source)
    expected = {str(int(value)) for value in np.unique(ids) if value > 0}
    if set(classes) != expected or any(type(value) is not int or value not in (0, 1) for value in classes.values()):
        raise ValueError('Exact positive ID to integer class 0/1 mapping required: ' + source)
    return ids, classes


def areas_by_phase(ids, classes):
    areas = np.bincount(ids.ravel())
    return {name: np.asarray([areas[int(key)] for key, value in classes.items() if value == target], np.int64)
            for name, target in (('ferrite', 1), ('pearlite', 0))}


def area_bins(areas, median):
    if median is None:
        return None
    result = {}
    for name, low, high in BINS:
        chosen = (areas >= low * median) & (areas < high * median)
        result[name] = dict(instances=int(chosen.sum()), pixels=int(areas[chosen].sum()))
    if sum(value['instances'] for value in result.values()) != len(areas):
        raise RuntimeError('Fixed reference bins do not partition ferrite areas')
    return result


def changes(before, after):
    result = {}
    for phase in PHASES:
        a, b = before[phase], after[phase]
        result[phase] = {}
        for key in ('instances', 'pixels', 'mean_area'):
            old, new = a[key], b[key]
            result[phase][key + '_delta'] = new - old if old is not None and new is not None else None
            result[phase][key + '_percent'] = 100 * (new / old - 1) if old and new is not None else None
        if a['mean_area'] and b['mean_area']:
            value = result[phase]
            value.update(log_mean_area_change=float(np.log(b['mean_area'] / a['mean_area'])),
                log_pixel_change=float(np.log(b['pixels'] / a['pixels'])),
                log_count_change=float(np.log(b['instances'] / a['instances'])))
            if not np.isclose(value['log_mean_area_change'], value['log_pixel_change'] - value['log_count_change'], atol=1e-12):
                raise RuntimeError('Mean-area count/pixel decomposition differs')
    return result


def pixel_class_changes(before, after):
    maps = []
    for ids, classes in (before, after):
        lut = np.full(int(ids.max()) + 1, -1, np.int8)
        for key, value in classes.items():
            lut[int(key)] = value
        maps.append(lut[ids])
    a, b = maps
    return dict(ferrite_to_pearlite_pixels=int(((a == 1) & (b == 0)).sum()),
                pearlite_to_ferrite_pixels=int(((a == 0) & (b == 1)).sum()))


def verify_inputs(config, run, files):
    pipeline = read(run / 'pipeline_status.json')
    if pipeline.get('status') != 'complete' or pipeline.get('smoke') is not False or pipeline.get('control_retrained') is not False:
        raise RuntimeError('The complete two-candidate pipeline is required')
    if set(pipeline['configs']) != {'n075', 'n125'} or not pipeline.get('sources'):
        raise RuntimeError('Candidate cohort or source inventory differs')
    for name, expected in pipeline['sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('Invalid training source path: ' + name)
        if digest(ROOT / name) != expected or digest(run / 'source' / name) != expected:
            raise RuntimeError('Current runtime or training snapshot differs: ' + name)
    expected_sources = {path.stem for path in files}
    directories, identities, manifests, states = {}, {}, {}, {}
    for arm, weight in ARMS.items():
        folder = Path(project_path(config, CONTROL)) if arm == 'mixed' else run / arm
        state = read(folder / 'status.json')
        actual_sha = digest(folder / 'final.pt')
        manifest = read(folder / 'deployment/manifest.json')
        selected = [row for row in manifest['images'] if row.get('view') == 'patch1024']
        if (state['status'] != 'complete' or state['epoch'] != 20 or state['updates'] != 1280
                or state.get('smoke') is not False or state.get('precision') != 'FP32'
                or actual_sha != state['final_checkpoint_sha256']
                or state['config']['direct_semantic_affinity']['affinity_loss']['negative_weight'] != weight
                or not all(state.get(key) is True for key in ('frozen_unchanged', 'restorer_unchanged', 'strict_reload_equal'))):
            raise RuntimeError('Training identity/frozen audit differs: ' + arm)
        if arm == 'mixed' and actual_sha != CONTROL_SHA:
            raise RuntimeError('The registered mixed balance control changed')
        if arm != 'mixed':
            if state['config'] != pipeline['configs'][arm] or comparable(reference_config(state['config'])) != comparable(states['mixed']['config']):
                raise RuntimeError('Candidate config changed beyond the authorized weight: ' + arm)
            if any(state[key] != states['mixed'][key] for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data')):
                raise RuntimeError('Candidate initialization/frozen/data dependency differs: ' + arm)
            if not pipeline['audits'][arm]['passed'] or pipeline['audits'][arm]['final_checkpoint_sha256'] != actual_sha:
                raise RuntimeError('Candidate final receipt differs: ' + arm)
        if (manifest.get('complete') is not True or manifest['epoch'] != 20 or manifest['precision'] != 'FP32'
                or manifest['checkpoint_sha256'] != actual_sha or 1024 not in manifest['sizes']
                or len(selected) != 100 or {row['source'] for row in selected} != expected_sources
                or (arm != 'mixed' and manifest['sizes'] != [1024])):
            raise RuntimeError('Final deployment manifest identity differs: ' + arm)
        directory = folder / 'deployment/patch1024'
        for suffix in ('_inst.png', '_class.json'):
            if {path.name for path in directory.glob('*' + suffix)} != {source + suffix for source in expected_sources}:
                raise RuntimeError('Expected exactly 100 output files of each kind: ' + arm + '/' + suffix)
        directories[arm], states[arm] = directory, state
        manifests[arm] = {row['source']: row for row in selected}
        identities[arm] = dict(checkpoint=str(folder / 'final.pt'), checkpoint_sha256=actual_sha, epoch=20,
            negative_weight=weight, manifest_sha256=digest(folder / 'deployment/manifest.json'),
            status_sha256=digest(folder / 'status.json'))
    return directories, identities, manifests, dict(passed=True, files=len(pipeline['sources']),
        pipeline_sha256=digest(run / 'pipeline_status.json'), all_current_and_snapshot_sha256_equal=True,
        sources=pipeline['sources'], runtime=pipeline.get('runtime'))


def summarize_totals(images, arm):
    result = {}
    for phase in PHASES:
        values = [row['views'][arm][phase] for row in images]
        total = {key: sum(value[key] for value in values) for key in ('instances', 'pixels')}
        total['pooled_mean_area'] = total['pixels'] / total['instances'] if total['instances'] else None
        result[phase] = total
    ferrite = result['ferrite']
    ferrite['tiny_under200'] = sum(row['views'][arm]['ferrite']['tiny_under200'] for row in images)
    ferrite['tiny_under200_pixels'] = sum(row['views'][arm]['ferrite']['tiny_under200_pixels'] for row in images)
    valid = [row['views'][arm]['ferrite']['fixed_reference_bins'] for row in images
             if row['views'][arm]['ferrite']['fixed_reference_bins'] is not None]
    ferrite['fixed_reference_bins'] = {name: {metric: sum(value[name][metric] for value in valid)
        for metric in ('instances', 'pixels')} for name, _, _ in BINS}
    ferrite['fixed_reference_bins_images'] = len(valid)
    return result


def summarize_comparison(images, arm):
    groups = [('all', images), ('long_side_le2048', [r for r in images if max(r['shape']) <= 2048]),
              ('long_side_gt2048', [r for r in images if max(r['shape']) > 2048])]
    for shape in sorted({tuple(row['shape']) for row in images}):
        groups.append(('shape_' + 'x'.join(map(str, shape)), [r for r in images if tuple(r['shape']) == shape]))
    result = {}
    for name, selected in groups:
        result[name] = dict(images=len(selected), phases={}, relations={})
        for phase in PHASES:
            result[name]['phases'][phase] = {}
            for key in ('instances', 'pixels', 'mean_area'):
                values = [row['changes'][arm][phase][key + '_percent'] for row in selected
                          if row['changes'][arm][phase][key + '_percent'] is not None]
                result[name]['phases'][phase][key + '_percent'] = dict(valid_images=len(values),
                    quantiles=quantiles(values), decreased=sum(value < 0 for value in values),
                    increased=sum(value > 0 for value in values), equal=sum(value == 0 for value in values))
        relation_counts = {key: sum(row['relations'][arm][key] for row in selected) for key in RELATION_SUMS}
        relation_counts['changed_class_fraction'] = relation_counts['class_changed_pixels'] / max(1, relation_counts['common_covered_pixels'])
        for key in ('ferrite_to_pearlite_pixels', 'pearlite_to_ferrite_pixels'):
            relation_counts[key] = sum(row['relations'][arm][key] for row in selected)
        result[name]['relations'] = relation_counts
    return result


def run(args):
    cv2.setNumThreads(2)
    config = load_config(args.config)
    run_root = Path(project_path(config, args.run))
    output = Path(project_path(config, args.output))
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    if len(files) != 100 or len({path.stem for path in files}) != 100:
        raise RuntimeError('Expected exactly 100 distinct semifinal sources')
    directories, identities, manifests, source_audit = verify_inputs(config, run_root, files)
    fixed = ['test_101', 'test_116']
    excluded = set(config['backend_adaptation']['monitor']['images'])
    random = sorted(np.random.default_rng(20261001).choice([p.stem for p in files if p.stem not in excluded], 6, replace=False).tolist())
    if not set(fixed).issubset({p.stem for p in files}):
        raise RuntimeError('Fixed visual sources are missing')
    output.mkdir(parents=True, exist_ok=False)
    gallery = output / 'gallery'
    gallery.mkdir()
    report = dict(scope='Unlabeled final prediction composition/topology, not test GT accuracy or official area score',
        training=False, inference_rerun=False, epoch_selection=False, postprocessing_tuned=False,
        identities=identities, source_audit=source_audit,
        selection=dict(fixed=fixed, random=random, seed=20261001, excluded_original_monitor=sorted(excluded)),
        area_bins=dict(reference='Each image mixed-r1 ferrite median, shared by all three arms',
            intervals={name: [low, None if np.isinf(high) else high] for name, low, high in BINS},
            interpretation='Disjoint size bins; small predictions are not labelled GT errors or fragments'),
        relation_definition='Class-independent one-to-one Hungarian overlap matching. IoU50/95 compare two predictions; split/merge require overlap >=50 pixels and >=10% of the corresponding instance. No correctness claim.',
        images=[], output_receipts=[])
    labels = ['mixed balance e20 r1', 'N075 e20 native1024', 'N125 e20 native1024']
    for index, path in enumerate(files):
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read original source shape: ' + path.stem)
        predictions = {arm: load_prediction(folder, path.stem, image.shape[:2]) for arm, folder in directories.items()}
        area = {arm: areas_by_phase(*pred) for arm, pred in predictions.items()}
        median = float(np.median(area['mixed']['ferrite'])) if len(area['mixed']['ferrite']) else None
        row = dict(source=path.stem, shape=list(image.shape[:2]), original_sha256=digest(path),
                   mixed_ferrite_median=median, views={}, changes={}, relations={})
        for arm, pred in predictions.items():
            description = phase_description(*pred)
            if description != manifests[arm][path.stem]['phases']:
                raise RuntimeError('Output phase receipt differs from manifest: ' + arm + '/' + path.stem)
            for phase in PHASES:
                description[phase]['mean_area'] = float(area[arm][phase].mean()) if len(area[arm][phase]) else None
                description[phase]['area_quantiles'] = quantiles(area[arm][phase])
            description['ferrite'].update(fixed_reference_bins=area_bins(area[arm]['ferrite'], median),
                tiny_under200_pixels=int(area[arm]['ferrite'][area[arm]['ferrite'] < 200].sum()))
            row['views'][arm] = description
            report['output_receipts'].append(dict(source=path.stem, arm=arm,
                png_sha256=digest(directories[arm] / (path.stem + '_inst.png')),
                classes_sha256=digest(directories[arm] / (path.stem + '_class.json'))))
        for arm in ('n075', 'n125'):
            row['changes'][arm] = changes(row['views']['mixed'], row['views'][arm])
            row['relations'][arm] = relation(predictions['mixed'][0], predictions[arm][0], predictions['mixed'][1], predictions[arm][1])
            row['relations'][arm].update(pixel_class_changes(predictions['mixed'], predictions[arm]),
                final_ids_exact=bool(np.array_equal(predictions['mixed'][0], predictions[arm][0])),
                class_json_exact=predictions['mixed'][1] == predictions[arm][1])
        report['images'].append(row)
        if path.stem in fixed + random:
            for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
                render(path, list(predictions.values()), labels, gallery / (path.stem + '_' + suffix + '.png'),
                       path.stem + ' | fixed e20/native1024, no test GT', crop=crop, width=360)
        if index % 20 == 0:
            print('CPU_TEST', index + 1, path.stem, flush=True)
    report['totals'] = {arm: summarize_totals(report['images'], arm) for arm in ARMS}
    report['comparisons'] = {arm: summarize_comparison(report['images'], arm) for arm in ('n075', 'n125')}
    report['pooled_changes'] = {arm: changes(
        {phase: dict(report['totals']['mixed'][phase], mean_area=report['totals']['mixed'][phase]['pooled_mean_area']) for phase in PHASES},
        {phase: dict(report['totals'][arm][phase], mean_area=report['totals'][arm][phase]['pooled_mean_area']) for phase in PHASES})
        for arm in ('n075', 'n125')}
    report['complete'] = len(report['images']) == 100 and len(report['output_receipts']) == 300
    report['analysis_sources_sha256'] = {name: digest(ROOT / name) for name in (
        'tools/analyze_affinity_sweep_test.py', 'tools/analyze_affinity_connectivity.py',
        'tools/render_backend_ablation.py', 'utils/patch_diagnostic.py', 'train_affinity_sweep.py',
        'train_affinity_balance.py', 'utils/config.py', 'data/mim_dataset.py')}
    write(output / 'test.json', report)
    write(output / 'summary.json', {key: report[key] for key in (
        'scope', 'complete', 'identities', 'source_audit', 'selection', 'area_bins', 'relation_definition',
        'totals', 'comparisons', 'pooled_changes', 'analysis_sources_sha256')})
    pages = ['<!doctype html><meta charset="utf-8"><title>Affinity weight sweep e20</title>',
        '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
        '<h1>mixed r1 / N075 / N125: fixed e20/native1024</h1>',
        '<p>Gold=ferrite, blue=pearlite. No test GT. Area/size/overlap changes are predictions, not accuracy.</p>',
        '<p>Fixed test101/116 + six seeded sources outside the original monitor. No best-image or epoch selection.</p>',
        '<p><a href="summary.json">Summary</a> | <a href="test.json">100-image statistics and SHA receipts</a></p>']
    for source in fixed + random:
        for suffix in ('full', 'detail'):
            name = source + '_' + suffix + '.png'
            pages.append(f'<p>{name}<br><a href="gallery/{name}"><img loading="lazy" src="gallery/{name}"></a></p>')
    (output / 'index.html').write_text('\n'.join(pages), encoding='utf8')
    with zipfile.ZipFile(output / 'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.html', '.png'):
                archive.write(path, path.relative_to(output).as_posix())
    print('TOTALS', json.dumps(report['totals']), flush=True)
    print('POOLED_CHANGES', json.dumps(report['pooled_changes']), flush=True)
    print('COMPLETE CPU analysis', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_sweep_n075.yaml')
    parser.add_argument('--run', default='outputs/affinity_sweep')
    parser.add_argument('--output', default='outputs/affinity_sweep/analysis/test')
    run(parser.parse_args())
