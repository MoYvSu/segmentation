# -*- coding: utf-8 -*-
"""原 C 在 SHA 白名单32训练源上生成独立教师监督；旧GT始终只读。

generate：原 compact C 完整推理后转换；convert：仅转换已完成的本工具推理；
check：CPU核验封存manifest及所有数组/类别/来源，不推理、不训练。
训练源契约只在本进程的受限上下文中替换，原提交入口不接受 train_*。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

FORMAT = 'teacher_supervision_v1'
CONTEXT_FORMAT = 'teacher_generation_context_v1'
WHITELIST_FORMAT = 'teacher_manual_whitelist_v1'
DATASET = 'teacher_train32_v1'
COUNT = 32
MAX_PIXELS = 6_000_000
BACKEND_SHA = '13b3ab1bc509fa45848e2eb8a40ee458b178fe747fc83eb65333c527a8161f04'
RESTORER_SHA = '9315da872ead66735f799e27f85eb97f8c2c265d44d64c6f8a271be50417007b'
NAF_STATE_SHA = '20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8'
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def write(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def checked_context(context):
    value = read(context) if isinstance(context, (str, Path)) else deepcopy(context)
    require(value.get('format') == CONTEXT_FORMAT, 'Expected ' + CONTEXT_FORMAT)
    for key in ('project_root', 'output_dir', 'raw_dir', 'original_gt_dir', 'manual_whitelist',
                'config', 'model_dir', 'naf_checkpoint'):
        require(Path(value[key]).is_absolute(), key + ' must be absolute')
        value[key] = str(Path(value[key]).resolve())
    require(Path(value['project_root']) == ROOT.resolve(), 'Run the generator from its declared source checkout')
    output = Path(value['output_dir'])
    for key in ('raw_dir', 'original_gt_dir', 'model_dir'):
        path = Path(value[key])
        require(not output.is_relative_to(path) and not path.is_relative_to(output),
                'Teacher output must not overlap original data/model directories')
    for key, field in (('manual_whitelist', 'manual_whitelist_sha256'),
                       ('naf_checkpoint', 'naf_checkpoint_sha256')):
        require(re.fullmatch('[0-9a-f]{64}', str(value.get(field))) is not None
                and sha(value[key]) == value[field], 'Sealed input SHA differs: ' + key)
    return value


def source_rows(context):
    """白名单为根编排从既有人工资产封存的输入，不按模型结果选图。"""
    document = read(context['manual_whitelist'])
    rows = document.get('rows')
    require(document.get('format') == WHITELIST_FORMAT and isinstance(rows, list) and len(rows) == COUNT,
            'Require a sealed 32-source manual whitelist')
    names = [row['source'] for row in rows]
    require(len(set(names)) == COUNT and all(re.fullmatch(r'train_[0-9]{3,4}', n) for n in names),
            'Only 32 distinct train_* stems are allowed; test sources are forbidden')
    raw, old_gt = Path(context['raw_dir']), Path(context['original_gt_dir'])
    candidates = [p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    require(not any(p.stem.startswith('test_') for p in candidates), 'Test images are forbidden in the input directory')
    labeled = [p for p in candidates if p.with_suffix('.json').is_file()]
    require(len(labeled) == COUNT and {p.stem for p in labeled} == set(names),
            'Manual image/LabelMe cohort differs from the whitelist')
    files = {p.stem: p for p in labeled}
    result = []
    for row in sorted(rows, key=lambda item: item['source']):
        name = row['source']
        paths = dict(image=files[name], labelme=files[name].with_suffix('.json'),
                     gt=old_gt / (name + '_gt.npz'), classes=old_gt / (name + '_class.json'))
        for key, path in paths.items():
            expected = row.get(key + '_sha256')
            require(re.fullmatch('[0-9a-f]{64}', str(expected)) is not None and sha(path) == expected,
                    'Whitelisted source identity differs: ' + name + '/' + key)
        with Image.open(paths['image']) as image:
            w, h = image.size
        require(0 < h * w <= MAX_PIXELS, 'Native source exceeds the bounded pixel contract')
        labelme = read(paths['labelme'])
        require(isinstance(labelme, dict) and labelme.get('imageHeight', h) == h
                and labelme.get('imageWidth', w) == w, 'LabelMe/source dimensions differ')
        result.append(dict(source=name, shape=[h, w], paths={k: str(p.resolve()) for k, p in paths.items()},
                           hashes={key + '_sha256': row[key + '_sha256'] for key in paths}))
    return result


@dataclass(frozen=True)
class TrainingContract:
    names: tuple[str, ...]
    name: str = DATASET

    @property
    def count(self):
        return len(self.names)

    def source_index(self, stem):
        require(isinstance(stem, str) and stem in self.names, 'Unwhitelisted teacher training source')
        return self.names.index(stem)


@contextmanager
def training_contract(entry, context, rows):
    """只替换命名/白名单分派；网络、浮点运算、S0、WS和最终读出不改。"""
    names = tuple(row['source'] for row in rows)
    contract = TrainingContract(names)
    original_contract = entry.dataset_contract

    def dataset(name=None):
        return contract if name == DATASET else original_contract(name)

    def inputs(directory, only=None, dataset=None):
        require(Path(directory).resolve() == Path(context['raw_dir']) and dataset == DATASET,
                'Teacher input dispatch must use the declared training cohort')
        require(only is None or tuple(only) == names, 'Teacher generation must cover the complete sorted32')
        fresh = source_rows(context)
        require(fresh == rows, 'Source whitelist changed before inference')
        return [Path(row['paths']['image']) for row in rows]

    with patch.object(entry, 'dataset_contract', new=dataset), patch.object(entry, 'input_files', new=inputs):
        yield contract


def validate_teacher_models(context):
    from tools import infer_compact as compact
    cfg = compact.entry.load_config(context['config'])
    compact.flat.execution_config(cfg)  # 原C/旧NAF/50种子/200末态的既有固定参数门禁。
    paths = compact.entry.model_paths(cfg, Path(context['model_dir']))
    require(set(paths) == {'backend', 'restorer'}, 'Teacher must use exactly the original two upstream models')
    require(sha(paths['backend']) == BACKEND_SHA and sha(paths['restorer']) == RESTORER_SHA,
            'Teacher backend/restorer are not the original C models')
    return cfg, {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}


def source_code():
    names = ('repro/generate_teacher.py', 'tools/infer_compact.py', 'utils/compact_instance_readout.py',
             'tools/infer_zero.py', 'tools/infer_boundary_rgb.py', 'tools/infer_semifinal.py',
             'tools/affinity_native_views.py', 'utils/post_process.py', 'utils/restoration_probe.py',
             'utils/instance_area_policy.py', 'utils/affinity_loss.py', 'utils/offset_letterbox.py')
    return {name: sha(ROOT / name) for name in names}


def prediction(ids, classes, shape):
    require(ids.dtype == np.uint16 and ids.ndim == 2 and ids.shape == tuple(shape) and ids.size,
            'Teacher prediction must be native uint16 HW')
    require(isinstance(classes, dict) and set(classes) == {str(int(i)) for i in np.unique(ids) if i},
            'Teacher positive IDs and class keys must match exactly')
    require(all(re.fullmatch(r'[1-9][0-9]*', k) and type(v) is int and v in (0, 1)
                for k, v in classes.items()), 'Teacher classes must be integer P0/F1')
    require(np.any(ids > 0), 'Teacher prediction has no positive support')


def probability_classes(ids, classes):
    lookup = np.full(int(ids.max()) + 1, -1, np.int8)
    for key, value in classes.items():
        if int(key) < len(lookup):
            lookup[int(key)] = int(value)
    require(np.all(lookup[ids[ids > 0]] >= 0), 'Original class lookup is incomplete')
    return lookup[ids]


def native_topology(ids):
    from data.boundary_naf import known_instance_boundary, reliable_region
    known = ids > 0
    pairs = same = different = 0
    for a, b in ((ids[1:], ids[:-1]), (ids[:, 1:], ids[:, :-1])):
        valid = (a > 0) & (b > 0)
        pairs += int(valid.sum())
        same += int((valid & (a == b)).sum())
        different += int((valid & (a != b)).sum())
    # 仅诊断0像素四邻域中至少出现两个不同正ID；绝不据此填0或制造新监督。
    lower = np.full(ids.shape, 65536, np.int32)
    upper = np.zeros(ids.shape, np.int32)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        from utils.affinity_graph import _edge_slices
        source, dest = _edge_slices(*ids.shape, dy, dx)
        values = ids[dest]
        lower[source] = np.minimum(lower[source], np.where(values > 0, values, 65536))
        upper[source] = np.maximum(upper[source], values)
    seam = (~known) & (lower < upper)
    boundary, valid = known_instance_boundary(ids), reliable_region(ids, 4)
    return dict(native_known_pixels=int(known.sum()), native_unknown_pixels=int((~known).sum()),
        teacher_zero_fraction=float((~known).mean()), native_four_neighbor_known_pairs=pairs,
        native_four_neighbor_same_pairs=same, native_four_neighbor_different_pairs=different,
        zero_four_neighbors_two_distinct_IDs=int(seam.sum()), zero_seam_count_is_diagnostic_only=True,
        naf_raw_boundary_pixels=int(boundary.sum()), naf_valid_margin=4, naf_valid_pixels=int(valid.sum()),
        naf_valid_boundary_pixels=int((boundary & valid).sum()),
        naf_valid_negative_pixels=int((~boundary & valid).sum()))


def diagnostic(old_path, old_classes, teacher, teacher_classes):
    """CPU逐源诊断；教师有效域保持自身正ID，比较域另行交集。"""
    import torch
    from utils.affinity_loss import build_affinity_targets_torch
    from utils.offset_letterbox import letterbox_instance_geometry
    with np.load(old_path, allow_pickle=False) as saved:
        old = saved['instance_map'].copy()
        original = saved['original_covered'].astype(bool) if 'original_covered' in saved else None
        filled = saved['filled'].astype(bool) if 'filled' in saved else None
    require(old.shape == teacher.shape and old.dtype.kind in 'iu' and old.min() >= 0 and old.max() <= 65535,
            'Original target geometry differs from the source shape/range')
    require(original is None or original.shape == old.shape, 'Original provenance shape differs')
    a, b = probability_classes(old, old_classes), probability_classes(teacher, teacher_classes)
    common = (old > 0) & (teacher > 0)
    phase = dict(common_known_pixels=int(common.sum()), P_to_F_pixels=int((common & (a == 0) & (b == 1)).sum()),
                 F_to_P_pixels=int((common & (a == 1) & (b == 0)).sum()),
                 old_known_teacher_zero_pixels=int(((old > 0) & (teacher == 0)).sum()),
                 old_unknown_teacher_known_pixels=int(((old == 0) & (teacher > 0)).sum()))
    for label, mask in (('original_covered', original), ('filled', filled)):
        if mask is not None:
            selected = common & mask
            phase[label] = dict(compared_pixels=int(selected.sum()),
                phase_disagreement_pixels=int((selected & (a != b)).sum()))
    grids = [letterbox_instance_geometry(value, 1024, 512) for value in (old, teacher)]
    targets = [build_affinity_targets_torch(torch.from_numpy(grid.astype(np.int64))[None],
               torch.from_numpy(valid)[None, None]) for grid, valid, _ in grids]
    (mt, mv), (tt, tv) = targets
    shared = mv & tv
    relations = dict(grid=512, input_size=1024, rule='nearest letterbox; both positive known endpoints; outside ignored',
        old_valid_edges=int(mv.sum()), teacher_valid_edges=int(tv.sum()), common_valid_edges=int(shared.sum()),
        teacher_same_edges=int((tv & (tt == 1)).sum()), teacher_different_edges=int((tv & (tt == 0)).sum()),
        old_valid_lost_to_teacher_zero=int((mv & ~tv).sum()), teacher_valid_added_on_old_unknown=int((tv & ~mv).sum()),
        manual_same_teacher_different=int((shared & (mt == 1) & (tt == 0)).sum()),
        manual_different_teacher_same=int((shared & (mt == 0) & (tt == 1)).sum()),
        per_channel=[dict(channel=i, teacher_valid_edges=int(tv[0, i].sum()),
            teacher_same_edges=int((tv[0, i] & (tt[0, i] == 1)).sum()),
            teacher_different_edges=int((tv[0, i] & (tt[0, i] == 0)).sum())) for i in range(8)])
    return dict(native=native_topology(teacher), phase=phase, affinity512=relations,
                accuracy_claimed=False, original_GT_unchanged=True), old, a, b


def thumbnail(path, source, image_path, old, ids, a, b):
    from utils.image_io import read_image, write_image
    image = read_image(image_path)
    def edge(labels):
        result = np.zeros(labels.shape, bool)
        result[1:] |= labels[1:] != labels[:-1]
        result[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        return result
    old_view, new_view, change = image.copy(), image.copy(), image.copy()
    old_view[edge(old)] = (0, 0, 255)
    new_view[edge(ids)] = (0, 0, 255)
    common = (old > 0) & (ids > 0)
    change[common & (a == 0) & (b == 1)] = (0, 255, 255)  # P→F 黄色。
    change[common & (a == 1) & (b == 0)] = (200, 0, 200)  # F→P 紫红。
    panels = []
    for title, value in [('Training image ' + source, image), ('Reference: old processed GT', old_view),
                         ('Teacher: current C final200', new_view), ('Old GT -> C: yellow P-F / magenta F-P', change)]:
        value = cv2.resize(value, (320, max(1, round(value.shape[0] * 320 / value.shape[1]))), interpolation=cv2.INTER_AREA)
        value = cv2.copyMakeBorder(value, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(value, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, np.concatenate(panels, axis=1))


def validate_inference(context, rows):
    from utils.instance_area_policy import resolve_instance_area_policy
    output = Path(context['output_dir']) / 'inference'
    report = read(output / 'audit/report.json')
    compact = read(output / 'audit/compact_report.json')
    require(report.get('complete') is True and compact.get('complete') is True
            and report.get('frozen_models_unchanged') is True and report.get('naf_complete') is True
            and report.get('naf_frozen_unchanged') is True and report.get('no_optimizer_steps') is True,
            'Complete frozen original C inference is required')
    require(report.get('config_sha256') == sha(context['config']) and report.get('parameter_count') == 102583318,
            'Teacher config identity or original C parameter count differs')
    require(report.get('geometry_mode') == compact.get('geometry_mode') == 'C'
            and report.get('dataset') == DATASET and report.get('dataset_count') == COUNT,
            'Teacher must be current C on the training32 contract')
    require(report['models']['backend']['sha256'] == BACKEND_SHA
            and report['models']['restorer']['sha256'] == RESTORER_SHA
            and report['models']['boundary_naf']['model_sha256'] == NAF_STATE_SHA
            and report['models']['boundary_naf']['checkpoint_sha256'] == context['naf_checkpoint_sha256'],
            'Teacher model identity differs from original C')
    require(report['config']['submission']['restoration_noise'] == 'explicit_zero_v1'
            and report['local_restoration'] == dict(mode='zero', model='restorer', candidate='N1')
            and report['seed_support'] == 'original-zero', 'Teacher must retain zero/N1/S0')
    require(resolve_instance_area_policy(report['config']['inference']) == dict(seed_min_area=50,
        final_min_instance_area=200, final_small_instance_action='neighbor_merge_v1', pre_vote_final_min_area=0),
        'Teacher must retain the original final200 policy')
    counts = dict(marker_stage_calls=COUNT, reflect_marker_calls=COUNT, S0_filter_calls=COUNT,
                  actual_WS_calls=COUNT, temporary_semantic_vote_calls=0, raw_revote_calls=COUNT, final200_calls=COUNT)
    require(compact.get('actual_compact_counts') == counts and report.get('global_zero_calls') == COUNT,
            'Teacher must execute one original C geometry/raw vote/final200 per source')
    require([r['source'] for r in report['images']] == [r['source'] for r in rows], 'Incomplete or reordered teacher cohort')
    for index, (actual, source) in enumerate(zip(report['images'], rows)):
        require(actual['input_sha256'] == source['hashes']['image_sha256'] and actual['shape'] == source['shape']
                and actual['full_source_index'] == index and actual['seed'] == 314159 + index,
                'Actual inference source/seed identity differs')
    require(all(not row.get('random_noise_generated', True) and row['global_zero_calls'] == 1
                and all(call['q_sample_calls'] == 1 and call['noise_nonzero'] == 0
                        and call['initial_equals_condition'] is True for call in row['calls'])
                for row in report.get('zero_noise_images', []))
            and len(report.get('zero_noise_images', [])) == COUNT, 'Actual zero-noise receipts are incomplete')
    return output, report, compact


def convert(context):
    context = checked_context(context)
    rows = source_rows(context)
    validate_teacher_models(context)
    output, report, compact = validate_inference(context, rows)
    root = Path(context['output_dir'])
    target = root / 'derived_maps'
    require(not target.exists(), 'Teacher targets already exist; use check, never overwrite')
    target.mkdir()
    actual_rows, diagnostics = [], []
    for index, (source, actual) in enumerate(zip(rows, report['images'])):
        name = source['source']
        png, classes_path = output / 'results' / (name + '_inst.png'), output / 'results' / (name + '_class.json')
        require(sha(png) == actual['png_sha256'] and sha(classes_path) == actual['classes_sha256'], 'Actual prediction bytes differ')
        from utils.image_io import read_image
        ids, classes = read_image(png, cv2.IMREAD_UNCHANGED), read(classes_path)
        prediction(ids, classes, source['shape'])
        gt, labels = target / (name + '_gt.npz'), target / (name + '_class.json')
        np.savez_compressed(gt, instance_map=ids, teacher_known=ids > 0, teacher_unknown=ids == 0)
        shutil.copyfile(classes_path, labels)
        stats, old, a, b = diagnostic(source['paths']['gt'], read(source['paths']['classes']), ids, classes)
        if index < 4:
            thumbnail(root / 'gallery' / (name + '.png'), name, source['paths']['image'], old, ids, a, b)
        diagnostics.append(dict(source=name, **stats))
        actual_rows.append(dict(source=name, shape=source['shape'], image=source['paths']['image'],
            labelme=source['paths']['labelme'], gt=str(gt.resolve()), classes=str(labels.resolve()),
            prediction=str(png.resolve()), original_gt_diagnostic_only=source['paths']['gt'],
            original_classes_diagnostic_only=source['paths']['classes'],
            hashes=dict(image_sha256=source['hashes']['image_sha256'], labelme_sha256=source['hashes']['labelme_sha256'],
                gt_sha256=sha(gt), classes_sha256=sha(labels), prediction_sha256=sha(png),
                original_gt_sha256=source['hashes']['gt_sha256'], original_classes_sha256=source['hashes']['classes_sha256']),
            teacher_known_pixels=int((ids > 0).sum()), teacher_unknown_pixels=int((ids == 0).sum())))
        write(root / 'conversion_progress.json', dict(complete=False, converted_sources=index + 1), exclusive=False)
    require(source_rows(context) == rows, 'Original training assets changed during conversion')
    totals = {key: sum(row['native'][key] for row in diagnostics) for key in
              ('native_known_pixels', 'native_unknown_pixels', 'native_four_neighbor_different_pairs',
               'zero_four_neighbors_two_distinct_IDs', 'naf_raw_boundary_pixels', 'naf_valid_boundary_pixels',
               'naf_valid_negative_pixels')}
    totals['teacher_zero_fraction'] = totals['native_unknown_pixels'] / (
        totals['native_known_pixels'] + totals['native_unknown_pixels'])
    summary = dict(format='current_C_teacher_diagnostic_v1', complete=True, source_count=COUNT,
        source_rule='first4 sorted whitelist sources for thumbnails; all32 for statistics', native_totals=totals,
        naf_positive_supervision_available=totals['naf_valid_boundary_pixels'] > 0,
        naf_zero_positive_requires_explicit_resolution=totals['naf_valid_boundary_pixels'] == 0,
        zero_pixels_remain_ignored=True, no_zero_filling=True, accuracy_claimed=False, images=diagnostics)
    write(root / 'diagnostic_summary.json', summary)
    identity = dict(geometry='C', models=report['models'], model_states=report['model_states'],
        config=report['config'], config_sha256=report['config_sha256'], source_code=source_code(),
        report_sha256=sha(output / 'audit/report.json'), compact_report_sha256=sha(output / 'audit/compact_report.json'),
        whitelist_sha256=context['manual_whitelist_sha256'],
        contract='original C zero/native1024/13b3+9315+20a59/S0/raw vote/final200; only source-name dispatch differs')
    manifest = dict(format=FORMAT, source_kind='model_teacher', complete=True, sample_count=COUNT,
        samples=[row['source'] for row in rows], rows=actual_rows, teacher=identity,
        no_test_training=True, original_GT_modified=False, teacher_zero_policy='ignore',
        class_convention={'0': 'pearlite', '1': 'ferrite'},
        provenance_arrays=['instance_map', 'teacher_known', 'teacher_unknown'],
        diagnostic_summary=str(root / 'diagnostic_summary.json'), diagnostic_summary_sha256=sha(root / 'diagnostic_summary.json'))
    manifest_path = target / 'manifest.json'
    write(manifest_path, manifest)
    write(root / 'manifest.seal.json', dict(format='teacher_manifest_seal_v1', complete=True,
        manifest=str(manifest_path), manifest_sha256=sha(manifest_path)))
    return check(context)


def check(context):
    context = checked_context(context)
    sources = source_rows(context)
    root = Path(context['output_dir'])
    path = root / 'derived_maps/manifest.json'
    manifest, seal = read(path), read(root / 'manifest.seal.json')
    require(sha(path) == seal['manifest_sha256'], 'Frozen teacher manifest changed')
    require(manifest.get('format') == FORMAT and manifest.get('source_kind') == 'model_teacher'
            and manifest.get('complete') is True and manifest.get('sample_count') == COUNT
            and manifest.get('samples') == [row['source'] for row in sources]
            and len(manifest.get('rows', [])) == COUNT, 'Teacher manifest schema/cohort differs')
    require(manifest['teacher']['whitelist_sha256'] == context['manual_whitelist_sha256'], 'Teacher whitelist binding differs')
    output, report, _ = validate_inference(context, sources)
    validate_teacher_models(context)
    require(manifest['teacher']['report_sha256'] == sha(output / 'audit/report.json')
            and manifest['teacher']['compact_report_sha256'] == sha(output / 'audit/compact_report.json'),
            'Frozen teacher inference audit changed')
    for row, source, actual in zip(manifest['rows'], sources, report['images']):
        require(row['source'] == source['source'] and row['shape'] == source['shape'], 'Teacher row identity differs')
        for field in ('gt', 'classes', 'prediction'):
            file = Path(row[field]).resolve()
            require(file.is_relative_to(root) and sha(file) == row['hashes'][field + '_sha256'],
                    'Teacher asset changed or escaped output: ' + field)
        require(row['hashes']['image_sha256'] == source['hashes']['image_sha256'], 'Teacher original image identity differs')
        require(Path(row['prediction']).resolve() == output / 'results' / (source['source'] + '_inst.png')
                and row['hashes']['prediction_sha256'] == actual['png_sha256']
                and row['hashes']['classes_sha256'] == actual['classes_sha256'],
                'Teacher assets differ from the actual C inference outputs')
        with np.load(row['gt'], allow_pickle=False) as saved:
            require(set(saved.files) == {'instance_map', 'teacher_known', 'teacher_unknown'}, 'Teacher NPZ must not forge r8 provenance')
            ids, known, unknown = [saved[key] for key in ('instance_map', 'teacher_known', 'teacher_unknown')]
        prediction(ids, read(row['classes']), row['shape'])
        require(known.dtype == unknown.dtype == bool and known.shape == unknown.shape == ids.shape
                and np.array_equal(known, ids > 0) and np.array_equal(unknown, ~known), 'Teacher known/unknown masks differ')
        from utils.image_io import read_image
        require(np.array_equal(ids, read_image(row['prediction'], cv2.IMREAD_UNCHANGED)), 'Teacher NPZ differs from actual C PNG')
    require(sha(manifest['diagnostic_summary']) == manifest['diagnostic_summary_sha256'], 'Teacher diagnostic summary changed')
    return dict(format='teacher_generation_verification_v1', complete=True, no_training=True,
        samples=COUNT, manifest=str(path), manifest_sha256=sha(path), original_GT_unchanged=True,
        npz_prediction_arrays_exact=True, teacher_zero_policy='ignore',
        naf_positive_supervision_available=read(manifest['diagnostic_summary'])['naf_positive_supervision_available'])


def generate(context):
    context = checked_context(context)
    rows = source_rows(context)
    validate_teacher_models(context)
    root = Path(context['output_dir'])
    require(not root.exists(), 'Teacher generation requires a fresh output directory')
    root.mkdir(parents=True)
    write(root / 'generation_context.json', context)
    before = source_code()
    from tools import infer_compact as compact
    args = SimpleNamespace(input_dir=context['raw_dir'], output_dir=str(root / 'inference'),
        config=context['config'], model_dir=context['model_dir'], dataset=DATASET, only=None,
        naf_checkpoint=context['naf_checkpoint'], capture_names=[], global_noise='zero', geometry_mode='C')
    with training_contract(compact.entry, context, rows):
        compact.run(args)
    require(source_code() == before and source_rows(context) == rows, 'Source/data changed during generation')
    result = convert(context)
    write(root / 'generation_complete.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('generate', 'convert', 'check'), required=True)
    parser.add_argument('--context', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(globals()[args.mode](args.context), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
