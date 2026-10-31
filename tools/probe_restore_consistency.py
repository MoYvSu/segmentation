# -*- coding: utf-8 -*-
"""固定后端，只比较旧D5a与双噪声短适配恢复器的局部1024路径。

复用已核验的全图几何语义、原图语义和旧八种子输出，不推理100图、不选测试种子。
本工具与云盘infer、训练入口分离；候选恢复器仅以独立对象传入局部函数。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.rgb_restoration import load_rgb_restorer
from tools import infer_semifinal as deployment
from tools.analyze_restoration_noise import (
    compare_maps, ferrite_stats, instability_roi, load_case, make_panel,
    original_picture, phase_picture, series_summary,
)
from tools.probe_affinity_patch import boundary_grid, tensor_from_rgb
from tools.probe_restoration_noise import OFFSETS, SOURCES, digest, save_case, write
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import tensor_digest
from utils.affinity_deployment import crop_letterbox_output
from utils.config import load_config
from utils.image_io import read_image
from utils.patch_diagnostic import BlendMap, tile_boxes
from utils.restoration_probe import restore_probe

FORMAT = 'restore_consistency_backend_probe_v1'
GROUPS = ('old', 'control', 'candidate')


def contract(config):
    """工作目录可不同，算法配置必须逐项相同。"""
    value = copy.deepcopy(config)
    value.pop('paths', None)
    return value


def validate_cache(metadata, config, paths, sources):
    if metadata.get('format') != 'restoration_noise_probe_v1' or metadata.get('complete') is not True:
        raise ValueError('Baseline noise experiment is incomplete or unsupported')
    if list(metadata.get('offsets', [])) != list(OFFSETS):
        raise ValueError('Baseline noise offsets differ')
    if contract(metadata.get('config', {})) != contract(config):
        raise ValueError('Backend/preprocessing/fusion/postprocessing configuration changed')
    if not set(sources) <= set(metadata.get('sources', [])):
        raise ValueError('Required sources missing from baseline experiment')
    for key, path in paths.items():
        if deployment.sha(path) != metadata['models'][key]['sha256']:
            raise ValueError('Deployed model identity changed: ' + key)
    for source in sources:
        row = metadata['images'][source]
        if not row.get('isolation_passed') or not row['cases']['control'].get('formal_control_exact'):
            raise ValueError('Baseline full-output/isolation gate was not passed: ' + source)


def load_candidate(path, arm, initial_sha, model_config, epoch, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(training_format='restore_consistency_v1', arm=arm,
                    initial_sha256=initial_sha, epoch=epoch, updates=epoch * 250, smoke=False)
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError('Not the declared all1000 fixed-stage checkpoint: ' + str(path))
    if checkpoint.get('model_config') != model_config:
        raise ValueError('Candidate restoration architecture differs from old D5a')
    return load_rgb_restorer(str(path), device).eval().requires_grad_(False)


@torch.no_grad()
def local_boundaries(model, local_restorer, rgb, config, seed, *, offsets=OFFSETS, device='cuda'):
    """只有local_restorer参与局部恢复；全图语义从未进入此函数。"""
    shape = rgb.shape[:2]
    boxes = tile_boxes(shape, 1024, config['affinity_native']['overlap'])
    blends = [BlendMap(shape) for _ in offsets]
    for index, box in enumerate(boxes):
        y, x, y1, x1 = box
        image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], device)
        tile_seed = seed + 100000 + 1024 * 1000 + index
        for choice, offset in enumerate(offsets):
            restored, _ = restore_probe(local_restorer, image, tile_seed + offset, 'single')
            grid = boundary_grid(model, restored, config)
            native = crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
            blends[choice].add(box, native)
            del restored, grid, native
        del image
        print(json.dumps(dict(stage='local_tile', completed=index+1, tiles=len(boxes))), flush=True)
    return [blend.finish()[0] for blend in blends], boxes


def checked_cached_case(baseline, metadata, source, index):
    case = f'local_s{index}'
    maps, classes = load_case(baseline, source, case)
    recorded = metadata['images'][source]['cases'][case]
    for key, sha_key in (('boundary', 'boundary_sha256'), ('geometry_semantic', 'geometry_semantic_sha256'),
                         ('raw_probability', 'raw_probability_sha256')):
        if digest(maps[key]) != recorded[sha_key]:
            raise ValueError('Cached continuous map differs from recorded digest: ' + source + '/' + case)
    fixed = metadata['images'][source]['cases']['control']
    for key in ('geometry_semantic_sha256', 'raw_probability_sha256'):
        if recorded[key] != fixed[key]:
            raise ValueError('Cached local variation changed fixed global semantics')
    return maps, classes


def series_statistics(cases, names):
    """每组相对于自己的s0；ddof=0，不把缺失种子当完整实验。"""
    if len(names) != len(OFFSETS) or any(name not in cases for name in names):
        raise ValueError('Exactly the predeclared eight seed cases are required')
    reference = cases[names[0]]
    result, std = series_summary(cases, names, reference)
    comparisons = {name: compare_maps(reference[0], cases[name][0], reference[1], cases[name][1])
                   for name in names}
    boundaries = np.stack([cases[name][0]['boundary'] for name in names])
    result['threshold_any_seed_crossing'] = {
        str(t): float(np.mean((boundaries.min(axis=0) <= t) & (boundaries.max(axis=0) > t)))
        for t in (.45, .65)}
    result['each_seed_vs_own_s0'] = comparisons
    result['seed_boundary_mae_from_s0_excluding_s0'] = float(np.mean([
        comparisons[name]['boundary']['mae'] for name in names[1:]]))
    result['seed_threshold_crossing_from_s0_excluding_s0'] = {
        str(t): float(np.mean([comparisons[name]['boundary']['threshold_crossing'][str(t)]
                              for name in names[1:]])) for t in (.45, .65)}
    stats = [ferrite_stats(cases[name][0]['instances'], cases[name][1])
             for name in names]
    result['ferrite_count_range'] = [min(x['count'] for x in stats), max(x['count'] for x in stats)]
    areas = [x['mean_area'] for x in stats if x['mean_area'] is not None]
    result['ferrite_mean_area_range'] = [min(areas), max(areas)] if areas else None
    result['reference_case'] = names[0]
    mean_boundary = boundaries.mean(axis=0, dtype=np.float64).astype(np.float32)
    result['boundary_mean_distribution'] = dict(mean=float(mean_boundary.mean()),
        quantiles={str(q):float(np.quantile(mean_boundary,q)) for q in (.5,.9,.95,.99)},
        fraction_above={str(t):float(np.mean(mean_boundary>t)) for t in (.45,.65)})
    result['threshold_margin'] = {}
    for threshold in (.45,.65):
        distance=np.abs(mean_boundary-threshold)
        positive_std=std>1e-8
        result['threshold_margin'][str(threshold)]=dict(
            mean_absolute_distance=float(distance.mean()),
            fraction_within_fixed_margin={str(margin):float(np.mean(distance<=margin)) for margin in (.01,.02,.05)},
            fraction_within_one_std=float(np.mean(positive_std & (distance<=std))),
            fraction_within_two_std=float(np.mean(positive_std & (distance<=2*std))),
            positive_std_fraction=float(np.mean(positive_std)))
    return result, std


def macro_statistics(sources):
    """先每源图八种子统计，再对源图等权；禁止以大图像素数增加权重。"""
    result = {}
    for group in GROUPS:
        rows = [source['series'][group] for source in sources.values()]
        if not rows or not all(row.get('complete') for row in rows):
            raise ValueError('Cannot summarize missing or incomplete source series')
        def mean(values):
            valid = [value for value in values if value is not None]
            return dict(mean=float(np.mean(valid)) if valid else None, valid_sources=len(valid))
        result[group] = dict(source_count=len(rows),
            boundary_std_mean=mean([row['boundary_std']['mean'] for row in rows]),
            boundary_std_p99=mean([row['boundary_std']['p99'] for row in rows]),
            ferrite_count_cv=mean([row['ferrite_cv']['count'] for row in rows]),
            ferrite_mean_area_cv=mean([row['ferrite_cv']['mean_area'] for row in rows]),
            phase_any_seed_change_fraction=mean([row['phase_any_seed_change_fraction'] for row in rows]),
            threshold_any_seed_crossing={str(t): mean([row['threshold_any_seed_crossing'][str(t)] for row in rows])
                                         for t in (.45, .65)})
    return result


def boundary_picture(boundary):
    # 三组共用0..1色阶，禁止各图自动拉伸制造锐化印象。
    return Image.fromarray(np.rint((1-np.clip(boundary, 0, 1))*255).astype(np.uint8)).convert('RGB')


def source_gallery(output, input_dir, source, cases0, std_maps):
    gallery = output/'gallery'
    gallery.mkdir(exist_ok=True)
    original = original_picture(input_dir, source, cases0['old'][0]['instances'].shape)
    box = instability_roi(std_maps['old'], size=256)
    marked = original.copy()
    ImageDraw.Draw(marked).rectangle(box, outline='red', width=3)
    names = []
    rows = [('Input; fixed old-noise ROI', marked)] + [(group+' s0: F blue/P gold', phase_picture(*cases0[group])) for group in GROUPS]
    name = source+'_full.jpg'; make_panel(rows, gallery/name, width=320); names.append(name)
    rows = [('Input: same native-pixel ROI', original.crop(box))] + [
        (group+' s0 boundary; fixed [0,1]', boundary_picture(cases0[group][0]['boundary']).crop(box)) for group in GROUPS]
    name = source+'_boundary.jpg'; make_panel(rows, gallery/name, width=320); names.append(name)
    rows = [('Input: same native-pixel ROI', original.crop(box))] + [
        (group+' s0 instances', phase_picture(*cases0[group]).crop(box)) for group in GROUPS]
    name = source+'_instances.jpg'; make_panel(rows, gallery/name, width=320); names.append(name)
    scale = max(float(np.quantile(std_maps['old'], .995)), 1e-8)
    rows = [('Input: same native-pixel ROI', original.crop(box))]
    for group in GROUPS:
        heat = np.rint(np.clip(std_maps[group]/scale, 0, 1)*255).astype(np.uint8)
        heat = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB))
        rows.append((group+f' std; old p99.5={scale:.5f}', heat.crop(box)))
    name = source+'_noise.jpg'; make_panel(rows, gallery/name, width=320); names.append(name)
    return dict(box=list(box), rule='256 native pixels; maximal old boundary-noise variation, same ROI and color scale for all groups', files=names)


def analyze(output, baseline, input_dir):
    output, baseline = Path(output), Path(baseline)
    metadata = json.loads((output/'record.json').read_text(encoding='utf-8'))
    if metadata.get('format') != FORMAT or not metadata.get('complete'):
        raise ValueError('Backend experiment has not completed its predeclared cases')
    old_record = json.loads((baseline/'record.json').read_text(encoding='utf-8'))
    summary = dict(format=FORMAT, complete=False, diagnostic_only=True,
                   reference='each group own s0; cross-group matched noise offset', sources={},
                   aggregation='Eight fixed seeds per image, ddof=0; macro equal weight per source; no GT/official accuracy')
    gallery_files=[]
    for source in metadata['sources']:
        series, cases0, std_maps, mean_maps = {}, {}, {}, {}
        for group in GROUPS:
            names=[f'{group}_s{i}' for i in range(len(OFFSETS))]
            if group == 'old':
                cases={name:checked_cached_case(baseline, old_record, source, i) for i,name in enumerate(names)}
            else:
                cases={name:load_case(output, source, name) for name in names}
            fixed=old_record['images'][source]['cases']['control']
            for maps,_ in cases.values():
                if (digest(maps['geometry_semantic'])!=fixed['geometry_semantic_sha256']
                        or digest(maps['raw_probability'])!=fixed['raw_probability_sha256']):
                    raise ValueError('Fixed global semantic path changed during candidate evaluation')
            series[group], std_maps[group]=series_statistics(cases, names)
            mean_maps[group]=np.mean(np.stack([cases[name][0]['boundary'] for name in names]),axis=0,dtype=np.float64).astype(np.float32)
            cases0[group]=cases[names[0]]
            del cases
        cross={}
        for index in range(len(OFFSETS)):
            old=checked_cached_case(baseline, old_record, source, index)
            control=load_case(output, source, f'control_s{index}')
            candidate=load_case(output, source, f'candidate_s{index}')
            cross[f's{index}']=dict(
                control_vs_old=compare_maps(old[0],control[0],old[1],control[1]),
                candidate_vs_old=compare_maps(old[0],candidate[0],old[1],candidate[1]),
                candidate_vs_control=compare_maps(control[0],candidate[0],control[1],candidate[1]))
        gallery=source_gallery(output, Path(input_dir), source, cases0, std_maps)
        gallery_files.extend(gallery['files'])
        mean_comparisons={}
        for left,right in (('old','control'),('old','candidate'),('control','candidate')):
            a,b=mean_maps[left],mean_maps[right]
            mean_comparisons[right+'_vs_'+left]=dict(mae=float(np.abs(b-a).mean()),
                signed_mean_difference=float((b-a).mean()),
                threshold_crossing={str(t):float(np.mean((a>t)!=(b>t))) for t in (.45,.65)})
        summary['sources'][source]=dict(series=series, matched_offset_comparisons=cross,
                                        seed_mean_boundary_comparisons=mean_comparisons,
                                        old_s0_gate=metadata['images'][source]['old_s0_gate'],gallery=gallery)
        print(json.dumps(dict(stage='analyzed',source=source)),flush=True)
    summary.update(complete=True, macro=macro_statistics(summary['sources']))
    write(output/'summary.json',summary)
    lines=['# 双噪声短适配：固定后端局部路径检查','',
           '全部数值是预测稳定性，不是GT准确率或官方成绩。全图恢复、几何语义、原图类别概率与分割后端固定。',
           '标准差和CV使用ddof=0；CV是标准差除以均值。每组均相对自己的s0，源图等权汇总。','',
           '| 图像 | 恢复器 | 边界std均值 | .65跨种子穿越 | F数量范围 | F数量CV | F均面积CV |',
           '|---|---|---:|---:|---|---:|---:|']
    def number(value):return 'N/A' if value is None else f'{value:.6f}'
    for source,row in summary['sources'].items():
        for group,series in row['series'].items():
            lines.append(f"| {source} | {group} | {number(series['boundary_std']['mean'])} | {number(series['threshold_any_seed_crossing']['0.65'])} | {series['ferrite_count_range']} | {number(series['ferrite_cv']['count'])} | {number(series['ferrite_cv']['mean_area'])} |")
    lines.extend(['','## 每源图等权汇总','','| 恢复器 | 边界std均值 | F数量CV | F均面积CV |','|---|---:|---:|---:|'])
    for group,row in summary['macro'].items():
        lines.append(f"| {group} | {number(row['boundary_std_mean']['mean'])} | {number(row['ferrite_count_cv']['mean'])} | {number(row['ferrite_mean_area_cv']['mean'])} |")
    lines.extend(['','稳定性改善不能证明边界更准确；须同清晰配对恢复质量和最终实例图联合判断。',
                  '图册中的窗口由旧模型最大噪声波动确定，三组共用窗口与色阶，不作GT解释。'])
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'gallery/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed-backend noise consistency</title>'
        '<style>body{background:#181c22;color:#eee;font:16px sans-serif}img{max-width:100%;display:block;margin:12px 0}</style>'
        '<h1>Old D5a / matched control / consistency candidate</h1><p>No test GT; fixed native ROI and color scales.</p>'+''.join(
            f'<h3>{html.escape(name)}</h3><img src="{html.escape(name,quote=True)}">' for name in gallery_files),encoding='utf-8')
    return summary


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Formal backend probe requires CUDA')
    output=args.output_dir.resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    deployment.configure_runtime()
    config=load_config(str(args.config.resolve()));config['paths']['project_root']=str(ROOT)
    deployment.validate_config(config)
    paths=deployment.model_paths(config,args.model_dir.resolve())
    baseline=args.baseline_root.resolve()
    cached=json.loads((baseline/'record.json').read_text(encoding='utf-8'))
    sources=list(SOURCES if args.only is None else args.only)
    validate_cache(cached,config,paths,sources)
    metadata=dict(format=FORMAT,complete=False,training=False,optimizer_steps=0,sources=sources,
                  offsets=list(OFFSETS),groups=list(GROUPS),expected_cases=[f'{g}_s{i}' for g in GROUPS for i in range(len(OFFSETS))],
                  baseline_root=str(baseline),baseline_record_sha256=deployment.sha(baseline/'record.json'),
                  config=config,config_sha256=deployment.sha(args.config),runtime=deployment.runtime_record(),
                  stage_epoch=args.epoch,images={},checkpoints={
                      'backend':dict(path=str(paths['backend']),sha256=deployment.sha(paths['backend'])),
                      'old':dict(path=str(paths['restorer']),sha256=deployment.sha(paths['restorer'])),
                      'control':dict(path=str(args.control.resolve()),sha256=deployment.sha(args.control)),
                      'candidate':dict(path=str(args.candidate.resolve()),sha256=deployment.sha(args.candidate))},
                  fixed='Global old D5a geometry semantic and original raw semantic reused byte-exact; encoder/LoRA/fusion/reflect/threshold/post fixed')
    write(output/'record.json',metadata)
    started=time.monotonic()
    try:
        model,old,terminal=deployment.load_models(config,paths,'cuda')
        if terminal is not None:raise ValueError('Only original fixed raw semantic is allowed')
        restorers={arm:load_candidate(getattr(args,arm),arm,metadata['checkpoints']['old']['sha256'],
                                     old.model_config,args.epoch,'cuda') for arm in ('control','candidate')}
        modules={'backend':model,'old':old,**restorers}
        if any(m.training or any(p.requires_grad for p in m.parameters()) for root in modules.values() for m in root.modules()):
            raise ValueError('All inference modules must be frozen/eval')
        states={name:tensor_digest(module.state_dict().items()) for name,module in modules.items()}
        for path in deployment.input_files(args.input_dir,sources):
            source=path.stem;cached_image=cached['images'][source]
            if deployment.sha(path)!=cached_image['input_sha256']:
                raise ValueError('Original input bytes changed: '+source)
            rgb=cv2.cvtColor(read_image(path),cv2.COLOR_BGR2RGB)
            original,classes=checked_cached_case(baseline,cached,source,0)
            if original['instances'].shape!=rgb.shape[:2]:raise ValueError('Input/cache geometry differs')
            base_control,base_control_classes=load_case(baseline,source,'control')
            if any(not np.array_equal(original[k],base_control[k]) for k in original) or classes!=base_control_classes:
                raise ValueError('Cached local_s0 differs from formal complete control')
            boundaries,boxes=local_boundaries(model,old,rgb,config,deployment.source_seed(source),offsets=(0,))
            ids,_=deployment.decode_reflect(torch.from_numpy(original['geometry_semantic'])[None,None],torch.from_numpy(boundaries[0])[None,None],config)
            reclasses,_=revote_instances(ids,original['raw_probability'])
            deployment.verify_prediction(ids,reclasses,rgb.shape[:2])
            gate=dict(continuous_boundary_exact=bool(np.array_equal(boundaries[0],original['boundary'])),
                      instances_exact=bool(np.array_equal(ids,original['instances'])),classes_exact=reclasses==classes,
                      original_sha256=deployment.sha(path),config_contract_exact=True)
            if not all(gate[key] for key in ('continuous_boundary_exact','instances_exact','classes_exact')):
                raise RuntimeError('Old initial local s0 no longer reproduces complete baseline: '+source)
            metadata['images'][source]=dict(old_s0_gate=gate,boxes=boxes,cases={},source_seed=deployment.source_seed(source))
            for arm,local_restorer in restorers.items():
                boundaries,new_boxes=local_boundaries(model,local_restorer,rgb,config,deployment.source_seed(source))
                if new_boxes!=boxes:raise RuntimeError('Native window plan changed')
                for index,boundary in enumerate(boundaries):
                    case=f'{arm}_s{index}'
                    metadata['images'][source]['cases'][case]=save_case(output,source,case,boundary,
                        original['geometry_semantic'],original['raw_probability'],config)[0]
                    print(json.dumps(dict(stage='decoded',source=source,case=case)),flush=True)
                del boundaries
            write(output/'record.json',metadata)
        after={name:tensor_digest(module.state_dict().items()) for name,module in modules.items()}
        if states!=after:raise RuntimeError('Frozen model state changed')
        for row in metadata['checkpoints'].values():
            if deployment.sha(row['path'])!=row['sha256']:raise RuntimeError('Checkpoint file changed')
        if deployment.sha(baseline/'record.json')!=metadata['baseline_record_sha256']:
            raise RuntimeError('Baseline record changed')
        metadata.update(complete=True,frozen_model_states_unchanged=True,model_states=states,
                        elapsed_seconds=time.monotonic()-started,imported_sources=deployment.local_module_sources())
        write(output/'record.json',metadata)
    except BaseException as error:
        metadata.update(error=repr(error),elapsed_seconds=time.monotonic()-started)
        write(output/'record.json',metadata)
        raise
    analyze(output,baseline,args.input_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--control',type=Path)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--epoch',type=int,default=5)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--only',nargs='+',choices=SOURCES,help='Smoke only; full default is all predeclared six sources')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    if args.analyze_only:
        analyze(args.output_dir,args.baseline_root,args.input_dir)
    else:
        if any(getattr(args,key) is None for key in ('config','model_dir','control','candidate')):
            parser.error('GPU probe requires config, model-dir, control and candidate')
        if args.epoch<=0:parser.error('Stage epoch must be positive')
        run(args)


if __name__=='__main__':main()
