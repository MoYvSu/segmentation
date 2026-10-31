# -*- coding: utf-8 -*-
"""CPU读取三组既有最终实例；分布与拓扑变化不是测试GT准确率。"""
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
from tools.affinity_native_views import verify_prediction
from tools.analyze_affinity_connectivity import load_directory, read, relation, render, write
from utils.config import load_config, project_path
from utils.patch_diagnostic import phase_description

IDENTITIES = {
    'native': ('outputs/affinity_balance/native', '2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b'),
    'mixed': ('outputs/affinity_balance/mixed', '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'),
    'pair': ('outputs/affinity_pair/candidate', '8b834fb8ad04077a350d9b6fedd2bfeb93aab4a6741381140add81927c1b973a'),
}
NAMES = ['native balance e20', 'mixed balance e20', 'paired views e20']
BINS = [('lt_quarter', 0., .25), ('quarter_half', .25, .5), ('half_one', .5, 1.),
        ('one_two', 1., 2.), ('ge_two', 2., float('inf'))]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite distribution values')
    probabilities = [0., .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.]
    labels = ['minimum', 'p01', 'p05', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'maximum']
    return dict(zip(labels, map(float, np.quantile(values, probabilities))))


def ferrite_areas(ids, classes):
    areas = np.bincount(ids.ravel())
    return np.array([areas[int(k)] for k, value in classes.items() if value == 1], dtype=np.int64)


def area_bins(areas, median):
    if median is None:
        return None
    result = {}
    for name, low, high in BINS:
        selected = (areas >= low * median) & (areas < high * median)
        result[name] = dict(instances=int(selected.sum()), pixels=int(areas[selected].sum()))
    if sum(v['instances'] for v in result.values()) != len(areas):
        raise RuntimeError('Fixed reference bins do not partition positive ferrite areas')
    return result


def summarize_changes(images, reference):
    selected = [(row, row['changes'][reference]) for row in images]
    selected = [(r, value) for r, value in selected if value['mean_area_percent'] is not None]
    groups = {}
    for name, items in [('all', selected), ('large', [(r, v) for r, v in selected if max(r['shape']) > 2048]),
                        ('other', [(r, v) for r, v in selected if max(r['shape']) <= 2048])]:
        area = [v['mean_area_percent'] for _, v in items]
        groups[name] = dict(images=len(items), area_percent=quantiles(area),
            ferrite_count_percent=quantiles([v['ferrite_count_percent'] for _, v in items]),
            ferrite_pixels_percent=quantiles([v['ferrite_pixels_percent'] for _, v in items]),
            area_decreased=sum(x < 0 for x in area), area_increased=sum(x > 0 for x in area),
            area_equal=sum(x == 0 for x in area))
    ordered = sorted(selected, key=lambda pair: (pair[1]['mean_area_percent'], pair[0]['source']))
    keys = ['source', 'shape', 'changes']
    tails = dict(lower=[{k: r[k] for k in keys} for r, _ in ordered[:6]],
                 upper=[{k: r[k] for k in keys} for r, _ in ordered[-6:][::-1]])
    totals = {key: sum(row['relations'][reference + '__pair'][key] for row in images) for key in
        ('old_instances', 'new_instances', 'split_relations', 'merge_relations', 'matched_iou50', 'matched_iou95',
         'class_changed_matched_iou50', 'class_changed_pixels', 'common_covered_pixels')}
    totals['changed_class_fraction'] = totals['class_changed_pixels'] / max(1, totals['common_covered_pixels'])
    return dict(groups=groups, tails=tails, relation_totals=totals)


def run(args):
    cv2.setNumThreads(2)
    config = load_config(args.config)
    output = Path(project_path(config, args.output))
    output.mkdir(parents=True, exist_ok=False)
    gallery = output / 'gallery'
    gallery.mkdir()
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    if len(files) != 100 or len({p.stem for p in files}) != 100:
        raise RuntimeError('Expected exactly 100 distinct semifinal sources')
    directories, identities = {}, {}
    expected_sources = {p.stem for p in files}
    for name, (relative, expected_sha) in IDENTITIES.items():
        folder = Path(project_path(config, relative))
        manifest = read(folder / 'deployment/manifest.json')
        actual_sha = digest(folder / 'final.pt')
        selected = [r for r in manifest['images'] if r.get('view') == 'patch1024']
        if (not manifest['complete'] or manifest['epoch'] != 20 or manifest['precision'] != 'FP32'
                or 1024 not in manifest['sizes'] or actual_sha != expected_sha
                or manifest['checkpoint_sha256'] != actual_sha or len(selected) != 100
                or {r['source'] for r in selected} != expected_sources):
            raise RuntimeError('Final output identity differs: ' + name)
        directories[name] = folder / 'deployment/patch1024'
        if {p.name for p in directories[name].glob('*_inst.png')} != {p.stem + '_inst.png' for p in files}:
            raise RuntimeError('Final output PNG cohort is incomplete or has extras: ' + name)
        identities[name] = dict(checkpoint=relative + '/final.pt', checkpoint_sha256=actual_sha,
                                manifest_sha256=digest(folder / 'deployment/manifest.json'), epoch=20)
    fixed = ['test_101', 'test_116']
    excluded = set(config['backend_adaptation']['monitor']['images'])
    random = sorted(np.random.default_rng(20261001).choice(
        [p.stem for p in files if p.stem not in excluded], 6, replace=False).tolist())
    if not set(fixed).issubset(expected_sources):
        raise RuntimeError('Fixed visual sources are missing')
    report = dict(scope='Unlabeled final predicted distributions and topology changes; not GT accuracy or official score',
        training=False, inference_rerun=False, epoch_selection=False, postprocessing_tuned=False,
        identities=identities, selection=dict(fixed=fixed, random=random, seed=20261001,
            random_excluded_monitor=sorted(excluded), extreme_rule='3 lowest + 3 highest pair-vs-mixed predicted mean-area changes'),
        area_bins=dict(reference='Per-image mixed-balance ferrite area median; identical cut points for all three routes',
            intervals={name: [low, None if np.isinf(high) else high] for name, low, high in BINS}),
        images=[], output_receipts=[])
    for index, path in enumerate(files):
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Could not read original source shape: ' + path.stem)
        predictions = {key: load_directory(folder, path.stem) for key, folder in directories.items()}
        for key, (ids, classes) in predictions.items():
            verify_prediction(ids, classes, image.shape[:2])
            if any(not isinstance(c, int) or isinstance(c, bool) for c in classes.values()):
                raise RuntimeError('Final class values must be integer 0/1: ' + key + '/' + path.stem)
            report['output_receipts'].append(dict(source=path.stem, route=key,
                png_sha256=digest(directories[key]/(path.stem + '_inst.png')),
                classes_sha256=digest(directories[key]/(path.stem + '_class.json'))))
        areas = {key: ferrite_areas(*prediction) for key, prediction in predictions.items()}
        mixed_median = float(np.median(areas['mixed'])) if len(areas['mixed']) else None
        row = dict(source=path.stem, shape=list(image.shape[:2]), mixed_ferrite_median=mixed_median,
                   views={}, changes={}, relations={})
        for key, prediction in predictions.items():
            stat = phase_description(*prediction)
            stat['ferrite'].update(area_quantiles=quantiles(areas[key]),
                fixed_reference_bins=area_bins(areas[key], mixed_median),
                tiny_under200_pixels=int(areas[key][areas[key] < 200].sum()))
            row['views'][key] = stat
        for reference in ('native', 'mixed'):
            before, after = row['views'][reference]['ferrite'], row['views']['pair']['ferrite']
            valid = before['instances'] > 0 and after['instances'] > 0
            row['changes'][reference] = dict(
                ferrite_count_delta=after['instances']-before['instances'],
                ferrite_pixels_delta=after['pixels']-before['pixels'],
                ferrite_count_percent=100*(after['instances']/before['instances']-1) if before['instances'] else None,
                ferrite_pixels_percent=100*(after['pixels']/before['pixels']-1) if before['pixels'] else None,
                mean_area_percent=100*(after['mean_area']/before['mean_area']-1) if valid else None,
                log_mean_area_change=float(np.log(after['mean_area']/before['mean_area'])) if valid else None,
                log_pixel_change=float(np.log(after['pixels']/before['pixels'])) if valid else None,
                log_count_change=float(np.log(after['instances']/before['instances'])) if valid else None)
            if valid and not np.isclose(row['changes'][reference]['log_mean_area_change'],
                    row['changes'][reference]['log_pixel_change']-row['changes'][reference]['log_count_change'], atol=1e-12):
                raise RuntimeError('Ferrite mean-area decomposition differs')
            row['relations'][reference + '__pair'] = relation(
                predictions[reference][0], predictions['pair'][0], predictions[reference][1], predictions['pair'][1])
        report['images'].append(row)
        if path.stem in fixed + random:
            for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
                render(path, list(predictions.values()), NAMES, gallery/(path.stem+'_'+suffix+'.png'),
                       path.stem+' | fixed e20 native1024, no test GT', crop=crop, width=360)
        if index % 20 == 0:
            print('CPU_TEST', index + 1, path.stem, flush=True)
    report['totals'] = {}
    for route in IDENTITIES:
        totals = {phase: {key: sum(row['views'][route][phase][key] for row in report['images'])
                         for key in ('instances', 'pixels')} for phase in ('ferrite', 'pearlite')}
        ferrite = totals['ferrite']
        ferrite['pooled_mean_area'] = ferrite['pixels']/ferrite['instances'] if ferrite['instances'] else None
        ferrite['tiny_under200'] = sum(r['views'][route]['ferrite']['tiny_under200'] for r in report['images'])
        ferrite['tiny_under200_pixels'] = sum(r['views'][route]['ferrite']['tiny_under200_pixels'] for r in report['images'])
        ferrite['fixed_reference_bins'] = {name: {metric: sum(r['views'][route]['ferrite']['fixed_reference_bins'][name][metric]
            for r in report['images'] if r['views'][route]['ferrite']['fixed_reference_bins'] is not None)
            for metric in ('instances', 'pixels')} for name, _, _ in BINS}
        report['totals'][route] = totals
    report['comparisons'] = {reference: summarize_changes(report['images'], reference) for reference in ('native', 'mixed')}
    ordered = sorted((r for r in report['images'] if r['changes']['mixed']['mean_area_percent'] is not None),
                     key=lambda r: (r['changes']['mixed']['mean_area_percent'], r['source']))
    extremes = [r['source'] for r in ordered[:3] + ordered[-3:][::-1]]
    report['selection']['area_extremes'] = extremes
    by_source = {path.stem: path for path in files}
    for source in extremes:
        predictions = [load_directory(directory, source) for directory in directories.values()]
        render(by_source[source], predictions, NAMES, gallery/(source+'_area_extreme.png'),
               source+' | predicted area tail; not selected by accuracy', crop=[.25, .25, .75, .75], width=400)
    report['complete'] = len(report['images']) == 100 and len(report['output_receipts']) == 300
    report['analysis_source_sha256'] = digest(__file__)
    write(output/'test.json', report)
    write(output/'summary.json', {key: report[key] for key in
        ('scope', 'complete', 'identities', 'selection', 'totals', 'comparisons', 'analysis_source_sha256')})
    pages = ['<!doctype html><meta charset="utf-8"><title>Affinity paired views e20</title>',
        '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
        '<h1>Fixed e20/native1024: native, mixed, paired</h1>',
        '<p>Gold=ferrite; blue=pearlite. Test images have no GT. Distribution changes are not accuracy.</p>',
        '<p><a href="summary.json">Summary</a> | <a href="test.json">100-image report and SHA receipts</a></p>',
        '<h2>Fixed typical + seeded random sources</h2>']
    for source in fixed + random:
        for suffix in ('full', 'detail'):
            name = source+'_'+suffix+'.png'
            pages.append(f'<p>{name}<br><a href="gallery/{name}"><img loading="lazy" src="gallery/{name}"></a></p>')
    pages.append('<h2>Predicted mean-area change tails (separate from fixed cohort)</h2>')
    for source in extremes:
        name = source+'_area_extreme.png'
        pages.append(f'<p>{name}<br><a href="gallery/{name}"><img loading="lazy" src="gallery/{name}"></a></p>')
    (output/'index.html').write_text('\n'.join(pages), encoding='utf-8')
    with zipfile.ZipFile(output/'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.html', '.png'):
                archive.write(path, path.relative_to(output).as_posix())
    print('TOTALS', json.dumps(report['totals']), flush=True)
    print('COMPARISONS', json.dumps(report['comparisons']), flush=True)
    print('COMPLETE CPU analysis', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_pair.yaml')
    parser.add_argument('--output', default='outputs/affinity_pair/analysis/cpu')
    args = parser.parse_args()
    run(args)
