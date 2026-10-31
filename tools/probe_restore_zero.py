# -*- coding: utf-8 -*-
"""固定局部分割后端，交叉 A/C 训练噪声与带噪/零噪声首步推理。

复用已核验的 A 八种子缓存；只新增 A-zero、C-zero、C 八种子。
零噪声没有随机种子方差是构造事实，本工具不将其解释为准确率收益。
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import infer_semifinal as deployment
from tools.analyze_restoration_noise import (
    compare_maps, ferrite_stats, instability_roi, load_case, make_panel,
    original_picture, phase_picture,
)
from tools.probe_affinity_patch import boundary_grid, tensor_from_rgb
from tools.probe_restoration_noise import OFFSETS, SOURCES, digest, save_case, write
from tools.probe_restore_consistency import (
    boundary_picture, checked_cached_case, contract, load_candidate,
    series_statistics, validate_cache,
)
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import tensor_digest
from utils.affinity_deployment import crop_letterbox_output
from utils.config import load_config
from utils.image_io import read_image
from utils.patch_diagnostic import BlendMap, tile_boxes
from utils.restoration_probe import restore_probe

FORMAT = 'restore_zero_backend_probe_v1'
NEW_CASES = ('a_zero', 'c_zero', *(f'c_noisy_s{i}' for i in range(len(OFFSETS))))
RECHECK_SOURCES = ('test_088', 'test_167')


def require_globals(maps, fixed):
    for key, recorded in (('geometry_semantic', 'geometry_semantic_sha256'),
                          ('raw_probability', 'raw_probability_sha256')):
        if digest(maps[key]) != fixed[recorded]:
            raise ValueError('Fixed global map changed: ' + key)


def checked_case(root, record, source, case, fixed):
    maps, classes = load_case(Path(root), source, case)
    entry = record['images'][source]['cases'][case]
    if digest(maps['boundary']) != entry['boundary_sha256']:
        raise ValueError('Cached boundary differs from receipt: ' + source + '/' + case)
    require_globals(maps, fixed)
    return maps, classes


def validate_adapted_cache(record, config, paths, control, sources):
    if record.get('format') != 'restore_consistency_backend_probe_v1' or record.get('complete') is not True:
        raise ValueError('Matched A/B backend cache is incomplete')
    if list(record.get('offsets', [])) != list(OFFSETS):
        raise ValueError('Matched A noisy offsets changed')
    if contract(record['config']) != contract(config):
        raise ValueError('Matched A backend contract changed')
    if not set(sources) <= set(record.get('sources', [])):
        raise ValueError('Matched A sources are missing')
    for name, path in (('old', paths['restorer']), ('backend', paths['backend']), ('control', control)):
        if deployment.sha(path) != record['checkpoints'][name]['sha256']:
            raise ValueError('Matched A checkpoint changed: ' + name)
    for source in sources:
        gate = record['images'][source]['old_s0_gate']
        if not all(gate.get(key) for key in ('continuous_boundary_exact', 'instances_exact', 'classes_exact')):
            raise ValueError('Original six-image exact gate missing: ' + source)


@torch.no_grad()
def local_boundaries(model, restorer, rgb, config, seed, *, mode, offsets=(0,), device='cuda'):
    """固定原生1024窗口；零模式只跑一次，首窗口额外检查seed确实不起作用。"""
    if mode not in ('single', 'zero'):
        raise ValueError('Only single/zero inference belongs to this protocol')
    if mode == 'zero' and tuple(offsets) != (0,):
        raise ValueError('Zero inference must not fabricate repeated seed statistics')
    shape = rgb.shape[:2]
    boxes = tile_boxes(shape, 1024, config['affinity_native']['overlap'])
    blends = [BlendMap(shape) for _ in offsets]
    for index, box in enumerate(boxes):
        y, x, y1, x1 = box
        image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], device)
        tile_seed = seed + 100000 + 1024 * 1000 + index
        for choice, offset in enumerate(offsets):
            restored, _ = restore_probe(restorer, image, tile_seed + offset, mode)
            if mode == 'zero' and index == 0:
                repeated, _ = restore_probe(restorer, image, tile_seed + 130363, mode)
                if not torch.equal(restored, repeated):
                    raise RuntimeError('Zero-noise restoration changed with inactive seed')
                del repeated
            grid = boundary_grid(model, restored, config)
            native = crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
            blends[choice].add(box, native)
            del restored, grid, native
        del image
        print(json.dumps(dict(stage='local_tile', mode=mode, completed=index+1, tiles=len(boxes))), flush=True)
    return [blend.finish()[0] for blend in blends], boxes


def comparison(left, right):
    return compare_maps(left[0], right[0], left[1], right[1])


def boundary_comparison(left, right):
    return dict(mae=float(np.abs(right-left).mean()), signed_mean_difference=float((right-left).mean()),
                threshold_crossing={str(t): float(np.mean((left > t) != (right > t))) for t in (.45, .65)})


def summarize_cases(old_cases, a_cases, c_cases, a_zero, c_zero):
    """A/C训练方式与推理方式交叉；比较的是两次预测，不是GT正确性。"""
    if not (len(old_cases) == len(a_cases) == len(c_cases) == len(OFFSETS)):
        raise ValueError('All noisy series require eight fixed offsets')
    series, std, mean = {}, {}, {}
    for label, values in (('old_noisy', old_cases), ('a_noisy', a_cases), ('c_noisy', c_cases)):
        names = [f'{label}_s{i}' for i in range(len(OFFSETS))]
        cases = dict(zip(names, values))
        series[label], std[label] = series_statistics(cases, names)
        mean[label] = np.mean(np.stack([v[0]['boundary'] for v in values]), axis=0, dtype=np.float64).astype(np.float32)
    cross = dict(c_zero_vs_a_zero=comparison(a_zero, c_zero),
                 c_zero_vs_old_s0=comparison(old_cases[0], c_zero),
                 a_zero_vs_old_s0=comparison(old_cases[0], a_zero),
                 c_noisy_vs_a_noisy=[comparison(a, c) for a, c in zip(a_cases, c_cases)],
                 a_zero_vs_a_noisy=[comparison(a, a_zero) for a in a_cases],
                 c_zero_vs_c_noisy=[comparison(c, c_zero) for c in c_cases])
    means = dict(c_noisy_vs_a_noisy=boundary_comparison(mean['a_noisy'], mean['c_noisy']),
                 a_zero_vs_a_noisy=boundary_comparison(mean['a_noisy'], a_zero[0]['boundary']),
                 c_zero_vs_c_noisy=boundary_comparison(mean['c_noisy'], c_zero[0]['boundary']),
                 c_zero_vs_a_zero=boundary_comparison(a_zero[0]['boundary'], c_zero[0]['boundary']))
    zero_stats = {label: dict(ferrite=ferrite_stats(case[0]['instances'], case[1]),
                             stochastic_variance_not_evaluated=True, inference_noise='exact_zero')
                  for label, case in (('a_zero', a_zero), ('c_zero', c_zero))}
    return dict(series=series, zero=zero_stats, comparisons=cross,
                seed_mean_boundary_comparisons=means), std


def gallery(output, input_dir, source, values, std):
    destination = output/'gallery'
    destination.mkdir(exist_ok=True)
    original = original_picture(input_dir, source, values['old_noisy'][0]['instances'].shape)
    box = instability_roi(std['old_noisy'], size=256)
    marked = original.copy()
    ImageDraw.Draw(marked).rectangle(box, outline='red', width=3)
    names = []
    order = ('old_noisy', 'a_noisy', 'a_zero', 'c_noisy', 'c_zero')
    rows = [('Input; fixed old-noise ROI', marked)] + [
        (label+' F-blue/P-gold', phase_picture(*values[label])) for label in order]
    name = source+'_full.jpg'; make_panel(rows, destination/name, width=288); names.append(name)
    rows = [('Input: same native-pixel ROI', original.crop(box))] + [
        (label+' boundary [0,1]', boundary_picture(values[label][0]['boundary']).crop(box)) for label in order]
    name = source+'_boundary.jpg'; make_panel(rows, destination/name, width=288); names.append(name)
    rows = [('Input: same native-pixel ROI', original.crop(box))] + [
        (label+' instances', phase_picture(*values[label]).crop(box)) for label in order]
    name = source+'_instances.jpg'; make_panel(rows, destination/name, width=288); names.append(name)
    return dict(box=list(box), rule='Old noisy std maximal 256px ROI, all groups same ROI and [0,1] boundary scale', files=names)


def macro(rows):
    result = {}
    for group in ('old_noisy', 'a_noisy', 'c_noisy'):
        items = [row['series'][group] for row in rows.values()]
        result[group] = dict(boundary_std_mean=float(np.mean([x['boundary_std']['mean'] for x in items])),
            ferrite_count_cv=float(np.mean([x['ferrite_cv']['count'] for x in items])),
            ferrite_mean_area_cv=float(np.mean([x['ferrite_cv']['mean_area'] for x in items])),
            phase_any_seed_change_fraction=float(np.mean([x['phase_any_seed_change_fraction'] for x in items])))
    result['c_zero_vs_a_zero'] = dict(
        boundary_mae=float(np.mean([r['comparisons']['c_zero_vs_a_zero']['boundary']['mae'] for r in rows.values()])),
        phase_changed_fraction=float(np.mean([r['comparisons']['c_zero_vs_a_zero']['phase']['changed_fraction'] for r in rows.values()])),
        mean_matched_iou=float(np.mean([r['comparisons']['c_zero_vs_a_zero']['instances']['mean_matched_iou'] for r in rows.values()])),
        ferrite_count_relative_delta=float(np.mean([r['comparisons']['c_zero_vs_a_zero']['ferrite_relative_delta']['count'] for r in rows.values()])),
        ferrite_total_pixels_relative_delta=float(np.mean([r['comparisons']['c_zero_vs_a_zero']['ferrite_relative_delta']['total_pixels'] for r in rows.values()])),
        ferrite_mean_area_relative_delta=float(np.mean([r['comparisons']['c_zero_vs_a_zero']['ferrite_relative_delta']['mean_area'] for r in rows.values()])))
    return result


def analyze(output, baseline, adapted, input_dir):
    output, baseline, adapted = map(Path, (output, baseline, adapted))
    record = json.loads((output/'record.json').read_text(encoding='utf-8'))
    old_record = json.loads((baseline/'record.json').read_text(encoding='utf-8'))
    a_record = json.loads((adapted/'record.json').read_text(encoding='utf-8'))
    if record.get('format') != FORMAT or record.get('complete') is not True:
        raise ValueError('Zero backend inference is incomplete')
    summary = dict(format=FORMAT, complete=False, diagnostic_only=True, sources={},
                   design='A/C training noise crossed with noisy/zero local inference; fixed global maps and backend',
                   limitation='No test GT or official score; zero seed variance is guaranteed by construction',
                   aggregation='Six predeclared sources equal weight; noisy eight offsets, ddof=0')
    files = []
    for source in record['sources']:
        fixed = old_record['images'][source]['cases']['control']
        old = [checked_cached_case(baseline, old_record, source, i) for i in range(len(OFFSETS))]
        a = [checked_case(adapted, a_record, source, f'control_s{i}', fixed) for i in range(len(OFFSETS))]
        c = [checked_case(output, record, source, f'c_noisy_s{i}', fixed) for i in range(len(OFFSETS))]
        a_zero = checked_case(output, record, source, 'a_zero', fixed)
        c_zero = checked_case(output, record, source, 'c_zero', fixed)
        row, std = summarize_cases(old, a, c, a_zero, c_zero)
        row['gallery'] = gallery(output, Path(input_dir), source,
                                dict(old_noisy=old[0], a_noisy=a[0], c_noisy=c[0], a_zero=a_zero, c_zero=c_zero), std)
        files.extend(row['gallery']['files'])
        summary['sources'][source] = row
        print(json.dumps(dict(stage='analyzed', source=source)), flush=True)
    summary.update(complete=True, macro=macro(summary['sources']))
    write(output/'summary.json', summary)
    lines = ['# D5a零噪声适配：固定后端2×2诊断', '',
             '只改变局部恢复器和其输入噪声；全图语义、原图类别概率和其余分割链固定。',
             '以下是预测组成及一致性，不是测试准确率。zero只有一次预测，不汇报虚构八种子CV。', '',
             '| 图像 | A-noisy s0 F数量/均面积 | A-zero | C-noisy s0 | C-zero | C-zero/A-zero类别像素变化 |',
             '|---|---:|---:|---:|---:|---:|']
    for source, row in summary['sources'].items():
        descriptions = []
        for group in ('a_noisy', 'a_zero', 'c_noisy', 'c_zero'):
            stats = (row['zero'][group]['ferrite'] if group.endswith('zero') else
                     row['series'][group]['each_seed_vs_own_s0'][f'{group}_s0']['ferrite'])
            descriptions.append(f"{stats['count']} / {stats['mean_area']:.2f}")
        changed = row['comparisons']['c_zero_vs_a_zero']['phase']['changed_fraction']
        lines.append(f"| {source} | {' | '.join(descriptions)} | {changed:.4%} |")
    lines.extend(['', '## 带噪八种子：每图先统计，再对六图等权', '',
                  '| 组 | 边界std均值 | F数量CV | F均面积CV | 类别变化面积比例 |', '|---|---:|---:|---:|---:|'])
    for group in ('old_noisy', 'a_noisy', 'c_noisy'):
        r = summary['macro'][group]
        lines.append(f"| {group} | {r['boundary_std_mean']:.6f} | {r['ferrite_count_cv']:.4%} | {r['ferrite_mean_area_cv']:.4%} | {r['phase_any_seed_change_fraction']:.4%} |")
    lines.extend(['', '完整JSON含同噪声配对、同权重推理模式切换、Hungarian实例匹配、类别翻转和面积分子/分母变化。',
                  'C由带噪D5a e60短适配而来，不能据此判断从零训练时噪声是否必要。'])
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (output/'gallery/index.html').write_text('<!doctype html><meta charset="utf-8"><title>D5a zero-noise 2x2</title>'
        '<style>body{background:#181c22;color:#eee;font:16px sans-serif}img{max-width:100%;display:block;margin:12px 0}</style>'
        '<h1>Fixed backend: A-noisy / A-zero / C-noisy / C-zero</h1><p>No test GT. Same native ROI and boundary scale.</p>'+''.join(
            f'<h3>{html.escape(name)}</h3><img src="{html.escape(name, quote=True)}">' for name in files), encoding='utf-8')
    return summary


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Formal local-backend inference requires CUDA')
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh diagnostic output directory')
    output.mkdir(parents=True)
    deployment.configure_runtime()
    config = load_config(str(args.config.resolve())); config['paths']['project_root'] = str(ROOT)
    deployment.validate_config(config)
    paths = deployment.model_paths(config, args.model_dir.resolve())
    baseline, adapted = args.baseline_root.resolve(), args.adapted_root.resolve()
    cached = json.loads((baseline/'record.json').read_text(encoding='utf-8'))
    a_cached = json.loads((adapted/'record.json').read_text(encoding='utf-8'))
    sources = list(SOURCES if args.only is None else args.only)
    validate_cache(cached, config, paths, sources)
    validate_adapted_cache(a_cached, config, paths, args.control, sources)
    metadata = dict(format=FORMAT, complete=False, training=False, optimizer_steps=0,
        sources=sources, offsets=list(OFFSETS), expected_new_cases=list(NEW_CASES),
        baseline_root=str(baseline), adapted_root=str(adapted),
        reused_record_sha256=dict(old=deployment.sha(baseline/'record.json'), a=deployment.sha(adapted/'record.json')),
        config=config, runtime=deployment.runtime_record(), images={}, stage_epoch=5,
        checkpoints={name: dict(path=str(path), sha256=deployment.sha(path)) for name, path in
                     (('backend', paths['backend']), ('old', paths['restorer']),
                      ('control', args.control.resolve()), ('zero', args.zero.resolve()))},
        fixed='Only local D5a weights and injected noise change. Old global D5a, raw semantic, SAM2/LoRA/affinity, reflect32, overlap.25, thresholds.45/.65 fixed')
    write(output/'record.json', metadata)
    started = time.monotonic()
    try:
        model, old, terminal = deployment.load_models(config, paths, 'cuda')
        if terminal is not None:
            raise ValueError('Only fixed original raw semantic is allowed')
        a = load_candidate(args.control, 'control', metadata['checkpoints']['old']['sha256'], old.model_config, 5, 'cuda')
        c = load_candidate(args.zero, 'zero', metadata['checkpoints']['old']['sha256'], old.model_config, 5, 'cuda')
        c_checkpoint = torch.load(args.zero, map_location='cpu', weights_only=True)
        if c_checkpoint.get('training_noise_mode') != 'zero':
            raise ValueError('C is not the declared zero-noise training checkpoint')
        del c_checkpoint
        modules = dict(backend=model, old=old, control=a, zero=c)
        if any(m.training or any(p.requires_grad for p in m.parameters()) for root in modules.values() for m in root.modules()):
            raise ValueError('All inference modules must remain frozen/eval')
        states = {name: tensor_digest(module.state_dict().items()) for name, module in modules.items()}
        for path in deployment.input_files(args.input_dir, sources):
            source = path.stem
            if deployment.sha(path) != cached['images'][source]['input_sha256']:
                raise ValueError('Original input bytes changed: ' + source)
            rgb = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB)
            original, classes = checked_cached_case(baseline, cached, source, 0)
            fixed = cached['images'][source]['cases']['control']
            if original['instances'].shape != rgb.shape[:2]:
                raise ValueError('Input/cache geometry differs')
            boxes = tile_boxes(rgb.shape[:2], 1024, config['affinity_native']['overlap'])
            gate = dict(reused_original_six_exact_gate=a_cached['images'][source]['old_s0_gate'], rechecked=False)
            if source in RECHECK_SOURCES:
                boundaries, gate_boxes = local_boundaries(model, old, rgb, config, deployment.source_seed(source), mode='single')
                ids, _ = deployment.decode_reflect(torch.from_numpy(original['geometry_semantic'])[None, None], torch.from_numpy(boundaries[0])[None, None], config)
                reclasses, _ = revote_instances(ids, original['raw_probability'])
                gate.update(rechecked=True, continuous_boundary_exact=bool(np.array_equal(boundaries[0], original['boundary'])),
                            instances_exact=bool(np.array_equal(ids, original['instances'])), classes_exact=reclasses == classes)
                if gate_boxes != boxes or not all(gate[k] for k in ('continuous_boundary_exact', 'instances_exact', 'classes_exact')):
                    raise RuntimeError('Old s0 exact反验 failed: ' + source)
                del boundaries
            metadata['images'][source] = dict(old_s0_gate=gate, boxes=boxes, cases={}, input_sha256=deployment.sha(path))
            for label, restorer, mode, offsets in (('a_zero', a, 'zero', (0,)), ('c_zero', c, 'zero', (0,)), ('c_noisy', c, 'single', OFFSETS)):
                boundaries, actual_boxes = local_boundaries(model, restorer, rgb, config, deployment.source_seed(source), mode=mode, offsets=offsets)
                if actual_boxes != boxes:
                    raise RuntimeError('Window plan changed')
                for index, boundary in enumerate(boundaries):
                    case = label if mode == 'zero' else f'{label}_s{index}'
                    metadata['images'][source]['cases'][case] = save_case(output, source, case, boundary,
                        original['geometry_semantic'], original['raw_probability'], config)[0]
                    print(json.dumps(dict(stage='decoded', source=source, case=case)), flush=True)
                del boundaries
            if set(metadata['images'][source]['cases']) != set(NEW_CASES):
                raise RuntimeError('Incomplete crossed inference plan')
            write(output/'record.json', metadata)
        if states != {name: tensor_digest(module.state_dict().items()) for name, module in modules.items()}:
            raise RuntimeError('Frozen model state changed')
        for row in metadata['checkpoints'].values():
            if deployment.sha(row['path']) != row['sha256']:
                raise RuntimeError('Checkpoint file changed')
        if metadata['reused_record_sha256'] != dict(old=deployment.sha(baseline/'record.json'), a=deployment.sha(adapted/'record.json')):
            raise RuntimeError('Reused cache receipt changed')
        metadata.update(complete=True, frozen_model_states_unchanged=True, model_states=states,
                        elapsed_seconds=time.monotonic()-started, imported_sources=deployment.local_module_sources())
        write(output/'record.json', metadata)
    except BaseException as error:
        metadata.update(error=repr(error), elapsed_seconds=time.monotonic()-started)
        write(output/'record.json', metadata)
        raise
    analyze(output, baseline, adapted, args.input_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'model-dir', 'input-dir', 'baseline-root', 'adapted-root', 'control', 'zero', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=name in ('input-dir', 'baseline-root', 'adapted-root', 'output-dir'))
    parser.add_argument('--only', nargs='+', choices=SOURCES)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args.output_dir, args.baseline_root, args.adapted_root, args.input_dir)
    else:
        if any(getattr(args, name) is None for name in ('config', 'model_dir', 'control', 'zero')):
            parser.error('GPU inference needs config, model-dir, control and zero')
        run(args)


if __name__ == '__main__':
    main()
