# -*- coding: utf-8 -*-
"""当前全链的NAF适配：动态权重身份→真实N1缓存→原width训练→仅G导出。

所有新文件仅写入context.run_dir；GT只读。旧实验入口和缺省合同不变。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMAT = 'current_training_naf_v1'
STAGES = ('n1_cache', 'naf_train', 'naf_export')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, frozen=False):
    path = Path(path)
    if frozen and path.exists():
        if read(path) != value:
            raise ValueError('已有阶段配置/来源不同，使用新run目录: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def checked_context(context):
    value = read(context) if isinstance(context, (str, Path)) else deepcopy(context)
    if value.get('format') != 'current_training_context_v1':
        raise ValueError('需要current_training_context_v1')
    for name in ('project_root', 'run_dir'):
        if not Path(value[name]).is_absolute():
            raise ValueError(name + '必须是绝对路径')
    return value


def artifact(context, name):
    path = Path(context['artifacts'][name])
    if not path.is_absolute() or not path.resolve().is_relative_to(Path(context['run_dir']).resolve()):
        raise ValueError('阶段产物必须位于新run目录: ' + name)
    return path.resolve()


def asset(context, name):
    path = Path(context['assets'][name])
    if not path.is_absolute():
        raise ValueError('资产路径必须是绝对路径: ' + name)
    return path.resolve()


def model_files(context):
    return {name: dict(path=str(artifact(context, name)), sha256=file_sha(artifact(context, name)))
            for name in ('backend', 'restorer')}


def cache_sources():
    return {}


def prepare_targets(context):
    """仅将32源既有监督数组封装为Dataset清单，不生成或改写GT数组。"""
    from PIL import Image
    from data.boundary_naf import load_processed_binary

    raw = asset(context, 'raw_dir')
    teacher = context['assets'].get('teacher_gt_dir')
    teacher_manifest = context['assets'].get('teacher_gt_manifest')
    if bool(teacher) != bool(teacher_manifest):
        raise ValueError('teacher_gt_dir与teacher_gt_manifest须同时指定')
    completed = asset(context, 'teacher_gt_dir' if teacher else 'completed_gt_dir')
    parent = asset(context, 'teacher_gt_manifest' if teacher else 'completed_gt_manifest')
    origin = read(parent)
    if teacher and origin.get('format') != 'teacher_supervision_v1':
        raise ValueError('显式teacher资产必须使用teacher_supervision_v1')
    policy = context.get('settings', {}).get('teacher_boundary_policy', 'known_neighbors_v1')
    if policy not in ('known_neighbors_v1', 'explicit_separators_v1'):
        raise ValueError('未知teacher_boundary_policy')
    images = sorted(p for p in raw.iterdir() if p.suffix.lower() in
                    {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'} and p.with_suffix('.json').is_file())
    names = [p.stem for p in images]
    if len(names) != 32 or len(set(names)) != 32 or any(not re.fullmatch(r'train_[0-9]{3,4}', n) for n in names):
        raise ValueError('NAF仅使用32个唯一人工训练源')
    declared = origin.get('samples')
    if declared is None and isinstance(origin.get('rows'), list):
        declared = [row['source'] for row in origin['rows']]
    if not isinstance(declared, list) or sorted(declared) != sorted(names):
        raise ValueError('GT版本清单与人工32源不一致')
    provided = context['assets'].get('naf_target_manifest')
    provided_document = read(provided) if provided else None
    prior = {r['source']: r for r in provided_document.get('rows', [])} if provided_document else None
    if prior is not None and set(prior) != set(names):
        raise ValueError('现有NAF训练封装源集合不同')
    rows = []
    for image in images:
        with Image.open(image) as handle:
            width, height = handle.size
        if min(height, width) < 1:
            raise ValueError('训练源尺寸必须为正')
        labelme = read(image.with_suffix('.json'))
        if (not isinstance(labelme, dict)
                or labelme.get('imageHeight', height) != height or labelme.get('imageWidth', width) != width
                or ('imagePath' in labelme and str(labelme['imagePath']).replace('\\', '/').split('/')[-1] != image.name)):
            raise ValueError('人工原图与LabelMe来源/尺寸不同')
        files = dict(image=image, gt=completed / (image.stem + '_gt.npz'),
                     classes=completed / (image.stem + '_class.json'))
        hashes = {key + '_sha256': file_sha(path) for key, path in files.items()}
        boundary, valid, metadata = load_processed_binary(files['gt'], files['classes'],
            boundary_dilation=0, valid_margin=4, teacher_boundary_policy=policy)
        if ((origin.get('format') == 'teacher_supervision_v1') !=
                (metadata.get('source_kind') == 'model_teacher')):
            raise ValueError('GT manifest与数组的teacher/manual来源不同')
        if boundary.shape != (height, width) or valid.shape != boundary.shape:
            raise ValueError('原图与只读GT尺寸不一致: ' + image.stem)
        if prior is not None:
            previous = prior[image.stem]
            if any(previous.get('hashes', previous).get(key) != digest for key, digest in hashes.items()):
                raise ValueError('现有NAF封装与当前GT内容不同: ' + image.stem)
        rows.append(dict(source=image.stem, shape=[height, width],
                         **{k: str(v.resolve()) for k, v in files.items()}, hashes=hashes,
                         labelme=str(image.with_suffix('.json').resolve()),
                         labelme_sha256=file_sha(image.with_suffix('.json')),
                         provenance=metadata))
    document = dict(format='current_training_processed_gt32_v1', complete=True, rows=rows,
        source_count=32, parent_manifest=str(parent), parent_manifest_sha256=file_sha(parent),
        parent_format=origin.get('format'), GT_arrays_modified=False, no_test_training=True,
        supplied_naf_manifest_sha256=file_sha(provided) if provided else None)
    document['target_statistics'] = dict(
        source_kind='model_teacher' if origin.get('format') == 'teacher_supervision_v1' else 'manual_completed',
        valid_positive_pixels=sum(row['provenance']['boundary_pixels'] for row in rows),
        valid_negative_pixels=sum(row['provenance']['valid_pixels'] - row['provenance']['boundary_pixels'] for row in rows),
        known_pixels=sum(row['provenance']['known_pixels'] for row in rows),
        unknown_pixels=sum(row['provenance']['unknown_pixels'] for row in rows),
        geometry_known_pixels=sum(row['provenance'].get('geometry_known_pixels', row['provenance']['known_pixels']) for row in rows),
        boundary_policy=policy,
        separator_pixels=sum(row['provenance'].get('separator_pixels', 0) for row in rows),
        semantic_ID0='ignore', affinity_ID0_endpoint='ignore',
        zero_ID_policy=('instance/semantic/affinity endpoints ignore; certified A-0-B center separator geometry boundary; other zero ignore'
                        if policy == 'explicit_separators_v1' else 'ignore; no boundary labels across unknown contacts'))
    if origin.get('format') == 'teacher_supervision_v1':
        document['teacher_boundary_policy'] = policy
        document['teacher_loader_sha256'] = file_sha(ROOT / 'data/teacher_supervision.py')
    path = Path(context['run_dir']) / 'configs' / 'naf_processed_gt32.json'
    write(path, document, frozen=True)
    return path, document


def cache_config(context):
    from utils.config import load_config

    original = load_config(str(ROOT.parent / 'recipes/inference.yaml'))
    keys = ('fusion_mode', 'distance2_weight', 'distance4_weight', 'support_threshold',
            'support_temperature', 'short_reduction', 'short_softmax_temperature')
    cfg = dict(paths=dict(project_root=context['project_root']),
        sam2=dict(input_normalization='legacy_none', sam2_repo_path=str(asset(context, 'sam2_repo'))),
        affinity_native=dict(overlap=.25),
        affinity_deployment={key: original['affinity_deployment'][key] for key in keys},
        submission=dict(native_sizes=[1024], seed=314159,
                        local_restoration=dict(mode='zero', model='restorer'),
                        restoration_noise='explicit_zero_v1', models=model_files(context)))
    return cfg


def production_ops(config, device):
    """延迟导入；只缓存局部affinity，不运行全图语义或WS。"""
    import cv2
    import numpy as np
    import torch
    from models.backend_adaptation import load_backend
    from models.rgb_restoration import load_rgb_restorer
    from tools.affinity_connectivity_views import _inference_state
    from tools.probe_affinity_patch import tensor_from_rgb, boundary_grid
    from utils.affinity_deployment import crop_letterbox_output
    from utils.patch_diagnostic import BlendMap, tile_boxes
    from utils.restoration_probe import restore_probe
    from train_backend_adaptation import tensor_digest

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    files = config['submission']['models']
    model, _ = load_backend(files['backend']['path'], config, device)
    restorer = load_rgb_restorer(files['restorer']['path'], device).eval().requires_grad_(False)
    if restorer.num_steps != 16 or restorer.kappa != .03:
        raise ValueError('D5a必须保持16步/kappa.03首步恢复')
    model.eval().requires_grad_(False)
    return SimpleNamespace(np=np, torch=torch, cv2=cv2, model=model, restorer=restorer,
        inference_state=_inference_state, tensor_from_rgb=tensor_from_rgb,
        boundary_grid=boundary_grid, crop_letterbox_output=crop_letterbox_output,
        BlendMap=BlendMap, tile_boxes=tile_boxes, restore_probe=restore_probe,
        state_sha=lambda m: tensor_digest(m.state_dict().items()))


def capture_native_boundary(rgb, config, ops, *, device, source_seed):
    np, torch = ops.np, ops.torch
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('需要原尺寸uint8 RGB')
    blend = ops.BlendMap(rgb.shape[:2])
    windows = []
    with ops.inference_state(ops.model, ops.restorer, device):
        for index, box in enumerate(ops.tile_boxes(rgb.shape[:2], 1024, .25)):
            y, x, y1, x1 = box
            tensor, ph, pw = ops.tensor_from_rgb(rgb[y:y1, x:x1], device)
            proof = {}
            seed = source_seed + 100000 + 1024 * 1000 + index
            restored, _ = ops.restore_probe(ops.restorer, tensor, seed, 'zero', audit=proof)
            if (proof.get('q_sample_calls') != 1 or proof.get('noise_nonzero') != 0
                    or proof.get('initial_equals_condition') is not True
                    or proof.get('initial_max_abs') != 0.0 or proof.get('step') != 16):
                raise RuntimeError('真实局部D5a零噪声证明失败')
            grid = ops.boundary_grid(ops.model, restored, config)
            native = ops.crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).float().cpu()[0, 0].numpy()
            if native.dtype != np.float32 or not np.isfinite(native).all() or np.any((native < 0) | (native > 1)):
                raise ValueError('真实N1局部边界概率无效')
            blend.add(box, native)
            windows.append(dict(window=index, box=list(box), zero_noise=True))
    boundary, receipt = blend.finish()
    return np.ascontiguousarray(boundary, dtype=np.float32), windows, receipt


def run_cache(context, *, device='cuda'):
    import numpy as np
    from utils.image_io import read_image
    output = artifact(context, 'n1_cache')
    if output.exists():
        raise FileExistsError('Use a fresh N1 cache directory')
    target_path, targets = prepare_targets(context)
    cfg = cache_config(context)
    write(Path(context['run_dir']) / 'configs/n1_cache.json', cfg, frozen=True)
    ops = production_ops(cfg, device)
    output.mkdir(parents=True)
    document = dict(format='boundary_naf_real_heatmaps_v1', complete=False,
        no_test_training=True, local_restoration=dict(mode='zero', model='restorer'),
        models=cfg['submission']['models'], processed_gt_manifest_sha256=file_sha(target_path), files=[])
    for index, row in enumerate(targets['rows']):
        rgb = read_image(row['image'])
        if rgb is None or list(rgb.shape[:2]) != row['shape']:
            raise ValueError('Training image/target shape mismatch')
        rgb = np.ascontiguousarray(rgb[..., ::-1])
        boundary, windows, blend = capture_native_boundary(rgb, cfg, ops, device=device,
                                                          source_seed=314159+2000000+index)
        path = output / (row['source'] + '.npz')
        np.savez_compressed(path, boundary=boundary)
        document['files'].append(dict(source=row['source'], shape=row['shape'], heat_path=path.name,
            heat_sha256=file_sha(path), **row['hashes'], window_count=len(windows)))
        write(output / 'manifest.json', document)
    document.update(complete=True, source_count=len(document['files']))
    write(output / 'manifest.json', document)
    return dict(complete=True, stage='n1_cache', manifest=str(output / 'manifest.json'),
                source_count=len(document['files']))


def training_config(context):
    from utils.config import load_config

    options = context.get('settings', {}).get('stages', {}).get('naf_train', {})
    if not isinstance(options, dict) or set(options) - {'config', 'seed', 'training'}:
        raise ValueError('naf_train仅可显式指定config/seed/training')
    recipe = Path(options.get('config', 'config/train/boundary_width.yaml'))
    if not recipe.is_absolute():
        recipe = Path(context['project_root']) / recipe
    config = load_config(str(recipe))
    target_path, targets = prepare_targets(context)
    if (targets['target_statistics']['source_kind'] == 'model_teacher'
            and targets['target_statistics']['valid_positive_pixels'] == 0):
        raise ValueError('Teacher NAF全部有效正边界为0；仅封装统计，禁止负样本单边训练，需明确新的边界合同')
    heat_path = artifact(context, 'n1_cache') / 'manifest.json'
    heat, models = read(heat_path), model_files(context)
    if (heat.get('format') != 'boundary_naf_real_heatmaps_v1'
            or heat.get('complete') is not True or heat.get('no_test_training') is not True
            or heat.get('models') != models
            or heat.get('processed_gt_manifest_sha256') != file_sha(target_path)
            or heat.get('local_restoration') != dict(mode='zero', model='restorer')):
        raise ValueError('缓存与当前新权重/GT/真实N1合同不一致')
    rows = heat.get('files', [])
    expected = {row['source']: row for row in targets['rows']}
    if len(rows) != 32 or {r['source'] for r in rows} != set(expected):
        raise ValueError('N1缓存必须覆盖同32源且无重复')
    for row in rows:
        if any(row.get(k) != v for k, v in expected[row['source']]['hashes'].items()):
            raise ValueError('N1缓存源哈希不一致')
        path = (heat_path.parent / row['heat_path']).resolve()
        if not path.is_relative_to(heat_path.parent.resolve()) or file_sha(path) != row['heat_sha256']:
            raise ValueError('N1缓存内容变化或越界')
    config['paths']['project_root'] = context['project_root']
    cfg = config['boundary_naf']
    cfg.update(output_dir=str(artifact(context, 'naf')), manifest=str(target_path),
        real_input_manifest=str(heat_path), real_input_models={k: {'sha256': v['sha256']} for k, v in models.items()})
    if targets['parent_format'] == 'teacher_supervision_v1':
        cfg['data']['teacher_boundary_policy'] = targets['teacher_boundary_policy']
    # 固定训练小图保留；旧测试控制绑定旧模型，不能接入新链冒充e0精确相同。
    cfg['monitor'].update(pair_indices=[0, 5, 17, 29, 41, 63, 9921], control_manifests=[],
                          source_names=[], roi_spec=None, full_readout_steps=[])
    if 'seed' in options:
        if type(options['seed']) is not int or options['seed'] < 0:
            raise ValueError('NAF seed必须非负整数')
        cfg['seed'] = options['seed']
    cfg['training'].update(options.get('training', {}))
    for key in ('num_workers', 'torch_threads', 'opencv_threads'):
        value = cfg['training'][key]
        if type(value) is not int or value < (0 if key != 'torch_threads' else 1):
            raise ValueError('NAF线程/worker设置必须是有效整数: ' + key)
    cfg['current_training'] = dict(format=FORMAT, adapter_sha256=file_sha(__file__),
        recipe_path=str(recipe.resolve()), recipe_file_sha256=file_sha(recipe),
        cache_manifest_sha256=file_sha(heat_path), target_manifest_sha256=file_sha(target_path),
        monitor_policy='fixed training pairs only; no old-model test controls; no change to training loss/draw')
    from train_boundary_rgb import validate_config
    validate_config(config)
    return config


def run_train(context, *, device='cuda', resume=None):
    import cv2
    import numpy as np
    import torch
    import train_boundary_rgb as trainer

    config = training_config(context)
    output = artifact(context, 'naf')
    if resume is not None and Path(resume).resolve() != output / 'latest.pt':
        raise ValueError('NAF恢复只接受同run的latest.pt')
    write(Path(context['run_dir']) / 'configs/naf_train.json', config, frozen=True)
    cfg = config['boundary_naf']
    torch.set_num_threads(int(cfg['training']['torch_threads']))
    cv2.setNumThreads(int(cfg['training']['opencv_threads']))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    result = trainer.train(config, output, device=device, resume=resume)
    if result.get('complete') is not True or result.get('global_step') != cfg['training']['main_updates']:
        raise RuntimeError('NAF尚未完成原定1000更新')
    return dict(result, format=FORMAT, stage='naf_train', checkpoint=str(output / 'latest.pt'),
                checkpoint_sha256=file_sha(output / 'latest.pt'))


def run_export(context):
    import torch
    from tools.infer_boundary_rgb import load_naf_checkpoint

    config = training_config(context)
    folder, output = artifact(context, 'naf'), artifact(context, 'naf_export')
    if output.exists():
        raise FileExistsError('不覆盖已导出的NAF')
    complete = read(folder / 'complete.json')
    checkpoint = folder / 'latest.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (complete.get('complete') is not True or complete.get('global_step') != config['boundary_naf']['training']['main_updates']
            or saved.get('global_step') != config['boundary_naf']['training']['main_updates'] or saved.get('config') != config['boundary_naf']
            or saved.get('model_sha256') != complete.get('model_sha256')):
        raise ValueError('NAF终态/实际配方/完成回执不一致')
    model, provenance = load_naf_checkpoint(checkpoint, 'cpu', expected_model_sha=saved['model_sha256'])
    del model
    keys = ('format', 'architecture', 'model_state_dict', 'model_sha256', 'global_step', 'config', 'input_contract')
    payload = {key: saved[key] for key in keys}
    payload['identity'] = {'numerical_contract': deepcopy(saved['identity']['numerical_contract'])}
    payload['current_training_export'] = dict(format=FORMAT, training_checkpoint_sha256=file_sha(checkpoint),
        completion_sha256=file_sha(folder / 'complete.json'), identity_sha256=saved['identity_sha256'],
        optimizer_discriminator_RNG_removed=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    torch.save(payload, temporary)
    restored, exported = load_naf_checkpoint(temporary, 'cpu', expected_model_sha=saved['model_sha256'])
    del restored
    if exported['architecture'] != provenance['architecture'] or exported['input_contract'] != provenance['input_contract']:
        raise RuntimeError('NAF导出重载合同变化')
    temporary.replace(output)
    result = dict(complete=True, format=FORMAT, stage='naf_export', path=str(output),
        file_sha256=file_sha(output), model_sha256=saved['model_sha256'], global_step=saved['global_step'],
        parameter_count=exported['parameter_count'], input_contract=exported['input_contract'],
        source_checkpoint_sha256=file_sha(checkpoint), no_training=True)
    write(output.parent.parent / 'export_receipt.json', result, frozen=True)
    return result


def run_stage(stage, context):
    context = checked_context(context)
    if stage == 'n1_cache':
        return run_cache(context)
    if stage == 'naf_train':
        # 编排器必须显式请求恢复；文件存在本身不是完成或可恢复的凭据。
        options = context.get('execution', {})
        return run_train(context, resume=options.get('resume'))
    if stage == 'naf_export':
        return run_export(context)
    raise ValueError('未知NAF阶段: ' + stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--context', required=True)
    args = parser.parse_args(argv)
    result = run_stage(args.stage, args.context)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
