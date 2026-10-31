# -*- coding: utf-8 -*-
"""分赛段独立推理：保留复赛默认，显式支持半决赛150图和局部恢复消融。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

# 同一文件既可在仓库tools/运行，也可原样复制为code_model/infer.py。
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tools' else HERE
sys.path.insert(0, str(ROOT))

from models.backend_adaptation import load_backend
from models.rgb_restoration import load_rgb_restorer
from tools import affinity_native_views as views
from tools.probe_affinity_marker_replay import build_marker_variant
from tools.semantic_crossover import revote_instances
from tools.semantic_marker_diagnostic import marker_stages
from train_backend_adaptation import tensor_digest
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.config import load_config
from utils.image_io import read_image, write_image
from utils.instance_area_policy import resolve_instance_area_policy
from utils.final_instance_cleanup import merge_final_instances
from utils.patch_diagnostic import phase_description
from utils.restoration_noise import NOISE_MODE, configure_restoration_noise, noise_protocol
from utils.submission_stage import dataset_contract

FORMAT = 'semifinal_standalone_inference_v1'
TERMINAL_FORMAT = 'semantic_grain_inference_v1'
FULL_NAMES = tuple(f'test_{i:03d}' for i in range(69, 169))
BASE_SEED = 314159


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def source_index(stem, dataset=None):
    """编号决定该赛段完整清单的位置；旧100默认与子集种子保持。"""
    return dataset_contract(dataset).source_index(stem)


def source_seed(stem, dataset=None):
    return BASE_SEED + source_index(stem, dataset)


def input_files(directory, only=None, dataset=None):
    contract = dataset_contract(dataset)
    files = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
    names = [p.stem for p in files]
    if len(set(names)) != len(names) or any(name not in contract.names for name in names):
        raise ValueError('Duplicate or unexpected test image names')
    if only is None:
        if set(names) != set(contract.names):
            raise ValueError(f'Full inference requires exactly {contract.count} images: '
                             f'test_{contract.first:03d} through test_{contract.last:03d}')
        return files
    if not only or len(set(only)) != len(only) or not set(only) <= set(names):
        raise ValueError('--only requires distinct available test image names')
    for name in only:
        source_index(name, dataset)
    return [p for p in files if p.stem in set(only)]


def verify_prediction(ids, classes, shape):
    if (ids.ndim != 2 or ids.shape != tuple(shape) or ids.dtype != np.uint16
            or int(ids.max()) > 65535
            or set(classes) != {str(int(i)) for i in np.unique(ids) if i > 0}
            or any(type(c) is not int or c not in (0, 1) for c in classes.values())):
        raise RuntimeError('Final uint16/instance-class contract differs')


def validate_config(config):
    """入口固定已评分几何契约；不要把文件路径调整误作模型超参调整。"""
    if config.get('sam2', {}).get('input_normalization') != 'legacy_none':
        raise ValueError('Reflect deployment requires explicit legacy_none normalization')
    infer = config['inference']
    expected = dict(threshold=.5, boundary_threshold=.65,
                    max_instance_id=65535, watershed_dilate_width=1, bridge_width=1,
                    marker_border_seal_width=2, marker_boundary_low_threshold=.45,
                    marker_boundary_reconstruction_steps=8, semantic_vote_mode='probability_mean',
                    semantic_vote_erode_width=0, semantic_vote_threshold=.5,
                    sem_edge_boost_alpha=0., sem_edge_merge_weight=0., center_seeds=False)
    if any(infer.get(key) != value for key, value in expected.items()):
        raise ValueError('Frozen reflect inference parameters differ')
    # 种子仍锁定旧部署50；最终小区处理必须由独立字段和显式动作说明。
    # 不再用可能已被覆盖的legacy别名判断两个阶段的实际取值。
    area = resolve_instance_area_policy(infer)
    if area['seed_min_area'] != 50:
        raise ValueError('Frozen reflect seed_min_area must remain 50')
    if (infer.get('marker_partition_restore', {}).get('enabled', False)
            or config['affinity_native'].get('overlap') != .25
            or config['submission'].get('marker_reflection_padding') != 32
            or config['submission'].get('native_sizes') != [1024]
            or config['submission'].get('seed') != BASE_SEED):
        raise ValueError('Expected reflect32, native1024, overlap.25, no partition restoration')
    if config['submission'].get('restoration_noise') != NOISE_MODE:
        raise ValueError('Standalone inference requires explicit restoration_noise=' + NOISE_MODE)
    dataset_contract(config['submission'].get('dataset'))
    local_restoration(config)
    fusion = dict(fusion_mode='gated', distance2_weight=.5, distance4_weight=.25,
                  support_threshold=.2, support_temperature=.05, short_reduction='top2',
                  short_softmax_temperature=.15)
    if any(config['affinity_deployment'].get(key) != value for key, value in fusion.items()):
        raise ValueError('Frozen affinity fusion parameters differ')


def local_restoration(config):
    """局部策略单独声明，全图恢复永远沿用原固定噪声。"""
    value = config['submission'].get('local_restoration', {'mode':'single', 'model':'restorer'})
    if not isinstance(value, dict) or set(value) != {'mode', 'model'}:
        raise ValueError('Expected explicit local_restoration mode/model')
    mode, model = value['mode'], value['model']
    candidates = {('single','restorer'):'N0', ('zero','restorer'):'N1', ('zero','local_restorer'):'N2'}
    if (mode, model) not in candidates:
        raise ValueError('Expected N0 single/shared, N1 zero/shared, or N2 zero/local')
    if model == 'local_restorer' and not config['submission']['models'].get('local_restorer'):
        raise ValueError('N2 requires a separately identified local_restorer')
    if model == 'restorer' and config['submission']['models'].get('local_restorer'):
        raise ValueError('Unused local_restorer model would obscure candidate identity')
    return dict(mode=mode, model=model, candidate=candidates[(mode, model)])


def decode_reflect(semantic, boundary, config, *, original_decode=None):
    """在真实watershed调用处仅替换种子，原障碍高度、过滤、重编号不变。"""
    original_decode = views.decode_native if original_decode is None else original_decode
    base = marker_stages(boundary.detach().float().cpu()[0, 0].numpy(), config)
    candidate = build_marker_variant(base, 'reflect', 32)
    if not np.array_equal(candidate['barrier_belt'], base['barrier_belt']):
        raise RuntimeError('Reflect changed the actual watershed barrier')
    if base['marker_count'] == 0:
        # 既有部署遇到无种子时输出空图；不额外创造实例。
        if candidate['marker_count'] != 0:
            raise RuntimeError('Reflect cannot inject seeds into the original no-watershed fallback')
        return original_decode(semantic, boundary, config)
    semantic_probability = semantic.detach().float().cpu().sigmoid()[0, 0].numpy()
    semantic_mask = semantic_probability > float(config['inference']['threshold'])
    expected_image = np.full((*semantic_mask.shape, 3), 128, np.uint8)
    expected_image[semantic_mask] = 200
    overlay = expected_image.copy(); overlay[base['barrier_belt'] > 0] = 255
    expected_image = cv2.addWeighted(expected_image, .7, overlay, .3, 0)
    real_ws = cv2.watershed
    calls = []

    def reflected_ws(image, markers):
        if calls or not np.array_equal(markers, base['marker_labels']):
            raise RuntimeError('Actual original watershed marker differs from frozen replay')
        if not np.array_equal(image, expected_image):
            raise RuntimeError('Actual original watershed elevation/barrier differs from frozen replay')
        frozen_image = image.copy()
        result = real_ws(image, candidate['marker_labels'].copy())
        if not np.array_equal(image, frozen_image):
            raise RuntimeError('Watershed changed its frozen elevation input')
        calls.append(True)
        return result

    with patch('utils.post_process.cv2.watershed', side_effect=reflected_ws):
        result = original_decode(semantic, boundary, config)
    if len(calls) != 1:
        raise RuntimeError('Expected exactly one original watershed call')
    return result


def runtime_record(*, restoration_noise_mode=NOISE_MODE):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                opencv=cv2.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                autocast=False, precision='FP32', input_normalization='legacy_none',
                restoration_noise=noise_protocol(restoration_noise_mode))


def configure_runtime():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def model_paths(config, model_dir):
    result = {}
    for name in ('backend', 'restorer', 'terminal_semantic', 'local_restorer'):
        item = config['submission']['models'].get(name)
        if item is None and name in ('terminal_semantic', 'local_restorer'):
            continue
        if not isinstance(item, dict) or not isinstance(item.get('file'), str):
            raise ValueError('Missing explicit model file/SHA: ' + name)
        path = (model_dir / item['file']).resolve()
        if not path.is_relative_to(model_dir) or not path.is_file():
            raise ValueError('Model file must exist inside --model-dir: ' + name)
        expected = item.get('sha256')
        if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
            raise RuntimeError('Model file SHA differs: ' + name)
        result[name] = path
    return result


def verify_source_manifest(root=ROOT):
    manifest = root / 'manifest.json'
    if not manifest.is_file():
        return dict(manifest_present=False, reason='Repository source execution; released bundle must include manifest.json')
    value = json.loads(manifest.read_text(encoding='utf-8'))
    rows = value.get('files')
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('Bundle source manifest files are missing')
    seen = set()
    for row in rows:
        name = row['path']
        path = (root / name).resolve()
        if name in seen or not path.is_relative_to(root) or not path.is_file() or sha(path) != row['sha256']:
            raise RuntimeError('Bundle source/model manifest differs: ' + name)
        seen.add(name)
    return dict(manifest_present=True, manifest_sha256=sha(manifest), checked_files=len(rows))


def local_module_sources():
    result = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == '.py' and path.is_relative_to(ROOT) and path.is_file():
            relative = path.relative_to(ROOT).as_posix()
            if relative.split('/')[0] not in {'output', 'outputs'}:
                result[relative] = sha(path)
    result[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha(__file__)
    # 驱动直接读取PTX，Python模块清单不会自动发现这两份协议资源。
    # 正式包必须携带它们，并同其他源码一样在运行前后核验实际文件摘要。
    for relative in ('utils/cuda/restore_noise.cu', 'utils/cuda/restore_noise.ptx'):
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError('Required restoration noise resource missing: ' + relative)
        result[relative] = sha(path)
    return dict(sorted(result.items()))


def load_models(config, paths, device):
    model, bundle = load_backend(str(paths['backend']), config, device)
    model.eval().requires_grad_(False)
    restorer = load_rgb_restorer(str(paths['restorer']), device).eval().requires_grad_(False)
    if restorer.num_steps != 16 or restorer.kappa != .03:
        raise RuntimeError('Expected unchanged D5a first_x0 schedule:16steps/kappa.03')
    # 在恢复器对象上显式启用一次，覆盖全图与原生裁块两条 restore_first 调用。
    # 不修改模型权重，不向旧训练入口注入推理策略。
    configure_restoration_noise(restorer, config['submission']['restoration_noise'])
    terminal = None
    if 'terminal_semantic' in paths:
        head_bundle = torch.load(paths['terminal_semantic'], map_location='cpu', weights_only=True)
        if (head_bundle.get('format') != TERMINAL_FORMAT or head_bundle.get('steps') != 64
                or head_bundle.get('architecture') != bundle['architecture']['semantic_decoder']
                or head_bundle.get('base_checkpoint_sha256') != bundle.get('source_checkpoint_sha256')
                or head_bundle.get('base_head_sha256') != tensor_digest(model.semantic_decoder.state_dict().items())):
            raise RuntimeError('Terminal s64 architecture/base identity differs')
        terminal = deepcopy(model.semantic_decoder).eval().requires_grad_(False)
        terminal.load_state_dict(head_bundle['head_state_dict'], strict=True)
        terminal.to(device)
    return model, restorer, terminal


def load_local_restorer(config, paths, device):
    """保留load_models旧三元组接口，不把局部模型注册为全图模型的子模块。"""
    if local_restoration(config)['model'] == 'restorer':
        return None
    local = load_rgb_restorer(str(paths['local_restorer']), device).eval().requires_grad_(False)
    if local.num_steps != 16 or local.kappa != .03:
        raise RuntimeError('Local D5a must retain 16steps/kappa.03 first_x0')
    return local


class RestorationPreview:
    """只供私有目检：实际局部修复图按相同窗权重拼接，不回流模型。"""
    def __init__(self):
        self.maps = None

    def __call__(self, *, source_shape, box, restored, ph, pw, **_):
        if self.maps is None:
            self.maps = [views.BlendMap(source_shape) for _ in range(3)]
        y, x, y1, x1 = box
        native = crop_letterbox_output(restored.float(), 1024, ph, pw,
                                       (y1-y, x1-x)).cpu()[0].numpy()
        for channel, blend in enumerate(self.maps):
            blend.add(box, native[channel])

    def finish(self, shape):
        if self.maps is None:
            raise RuntimeError('No actual restoration windows captured')
        value = np.stack([blend.finish()[0] for blend in self.maps], axis=-1)
        if value.shape != (*shape, 3):
            raise RuntimeError('Restoration preview shape differs')
        return value


def finalize_instance_area(result, config):
    """原图raw语义投票之后再合并小区，避免先丢成0或改变大区类别。

    legacy_drop已在watershed内部完成，原路径直接返回。新策略保留接管者
    的类别；若存在语义分数，只保留接管者原分数并明确标注，不冒充合并后的均值。
    """
    area = resolve_instance_area_policy(config['inference'])
    if area['final_small_instance_action'] == 'legacy_drop':
        return result
    ids, classes, audit = merge_final_instances(
        result['instances'], result['classes'], area['final_min_instance_area'],
        action=area['final_small_instance_action'])
    result = dict(result, instances=ids, classes=classes, final_area_cleanup=audit)
    result['phases'] = phase_description(ids, classes)
    if 'instance_scores' in result:
        result['instance_scores'] = {key: result['instance_scores'][key] for key in classes}
        result['instance_score_policy'] = 'original_target_score_inherited_not_recomputed'
    return result


@torch.no_grad()
def predict(model, restorer, terminal, path, config, device, *, local_restorer=None,
            fixed_audit=None, local_audit=None, analysis_callback=None):
    policy = local_restoration(config)
    if (policy['model'] == 'local_restorer') != (local_restorer is not None):
        raise ValueError('Actual local model does not match the declared candidate')
    original_decode = views.decode_native

    def reflected_decode(semantic, boundary, actual_config, *args, **kwargs):
        if args or kwargs:
            raise RuntimeError('Unexpected native geometry decode arguments')
        return decode_reflect(semantic, boundary, actual_config, original_decode=original_decode)

    with torch.autocast(device_type=torch.device(device).type, enabled=False), \
            patch.object(views, 'decode_native', side_effect=reflected_decode):
        rgb, predictions = views.predict_views(
            model, restorer, path, config, device,
            source_seed(path.stem, config['submission'].get('dataset')), [1024],
            local_restorer=local_restorer, local_noise_mode=policy['mode'],
            fixed_audit=fixed_audit, local_audit=local_audit, analysis_callback=analysis_callback)
        result = predictions['patch1024']
        if terminal is not None:
            # 上游恢复图语义仍由原头决定geometry；新头只在原图末端改类别。
            _, raw, ph, pw = prepare_image(path, 1024, device)
            logits = terminal(model.encoder(raw.float()), raw.float())
            if isinstance(logits, dict):
                logits = logits['semantic_logits']
            probability = crop_letterbox_output(logits.float(), 1024, ph, pw, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()
            result['classes'], result['instance_scores'] = revote_instances(result['instances'], probability)
            result['terminal_probability_sha256'] = hashlib.sha256(probability.tobytes()).hexdigest()
            result['phases'] = phase_description(result['instances'], result['classes'])
        result = finalize_instance_area(result, config)
    verify_prediction(result['instances'], result['classes'], rgb.shape[:2])
    return rgb, result


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Formal inference needs CUDA; CPU is limited to contract tests')
    started = time.time()
    configure_runtime()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    # 便携入口的根由源码位置决定，忽略工作站环境变量对历史项目根的注入。
    config['paths']['project_root'] = str(ROOT)
    if getattr(args, 'dataset', None):
        config['submission']['dataset'] = args.dataset
    validate_config(config)
    contract = dataset_contract(config['submission'].get('dataset'))
    policy = local_restoration(config)
    input_dir = Path(args.input_dir).resolve()
    files = input_files(input_dir, args.only, contract.name)
    model_dir = Path(args.model_dir).resolve() if args.model_dir else ROOT / 'model'
    paths = model_paths(config, model_dir)
    output = Path(args.output_dir).resolve()
    if output.exists() or output.is_relative_to(input_dir) or output.is_relative_to(model_dir):
        raise ValueError('Use a fresh output directory outside input/model directories')
    output.mkdir(parents=True)
    result_dir = output / 'results'; result_dir.mkdir()
    audit = output / 'audit'; audit.mkdir()
    status = dict(format=FORMAT, complete=False, training=False, optimizer_steps=0,
                  expected_images=len(files), completed_images=0)
    write(audit / 'status.json', status)
    try:
        manifest = verify_source_manifest()
        model, restorer, terminal = load_models(config, paths, 'cuda')
        local = load_local_restorer(config, paths, 'cuda')
        named_modules = [('backend', model), ('restorer', restorer)]
        if terminal is not None:
            named_modules.append(('terminal', terminal))
        if local is not None:
            named_modules.append(('local_restorer', local))
        modules = [module for _, module in named_modules]
        if any(m.training or any(p.requires_grad for p in m.parameters()) for root in modules for m in root.modules()):
            raise RuntimeError('All modules must be frozen and eval for inference')
        before = {name: tensor_digest(module.state_dict().items()) for name, module in named_modules}
        parameters = sum(p.numel() for module in modules for p in module.parameters())
        if parameters >= 500_000_000:
            raise RuntimeError('Total model parameters exceed competition limit')
        source_hashes = local_module_sources()
        runtime = runtime_record(restoration_noise_mode=restorer.inference_noise_mode)
        report = dict(format=FORMAT, complete=False, full100_complete=False, full150_complete=False,
                      dataset=contract.name, dataset_count=contract.count,
                      local_restoration=policy, training=False,
                      no_optimizer_steps=True, parameter_count=parameters, runtime=runtime,
                      source_manifest=manifest, imported_sources=source_hashes,
                      config_sha256=sha(config_path), config=config,
                      models={name: dict(file=str(path.relative_to(model_dir)), sha256=sha(path), bytes=path.stat().st_size) for name, path in paths.items()},
                      model_states=before, semantic='raw terminal s64' if terminal is not None else 'raw original R1',
                      instance_area_policy=resolve_instance_area_policy(config['inference']),
                      geometry='original R1 native1024, D5a first_x0, reflect32 markers only', images=[])
        for path in files:
            fixed, windows = {}, []
            collector = RestorationPreview() if getattr(args, 'save_analysis', False) else None
            rgb, result = predict(model, restorer, terminal, path, config, 'cuda',
                                  local_restorer=local, fixed_audit=fixed, local_audit=windows,
                                  analysis_callback=collector)
            ids, classes = result['instances'], result['classes']
            png = result_dir / (path.stem + '_inst.png')
            labels = result_dir / (path.stem + '_class.json')
            write_image(png, ids)
            write(labels, classes)
            reloaded = read_image(png, cv2.IMREAD_UNCHANGED)
            reload_classes = json.loads(labels.read_text(encoding='utf-8'))
            verify_prediction(reloaded, reload_classes, rgb.shape[:2])
            if not np.array_equal(reloaded, ids) or reload_classes != classes:
                raise RuntimeError('Saved PNG/JSON differs from actual inference')
            row = dict(source=path.stem, full_source_index=source_index(path.stem, contract.name),
                       seed=source_seed(path.stem, contract.name), fixed_paths=fixed, local_windows=windows,
                       input_sha256=sha(path), shape=list(rgb.shape[:2]), phases=result['phases'], blend=result['blend'],
                       png_sha256=sha(png), classes_sha256=sha(labels))
            if collector is not None:
                preview = collector.finish(rgb.shape[:2])
                analysis_dir = output / 'analysis'
                analysis_dir.mkdir(exist_ok=True)
                np.savez_compressed(analysis_dir / (path.stem + '.npz'), boundary=result['boundary'],
                                    restored=preview)
                row['analysis'] = 'analysis/' + path.stem + '.npz'
            if terminal is not None:
                row.update(instance_scores=result['instance_scores'], terminal_probability_sha256=result['terminal_probability_sha256'])
            if 'final_area_cleanup' in result:
                row['final_area_cleanup'] = result['final_area_cleanup']
                if 'instance_score_policy' in result:
                    row['instance_score_policy'] = result['instance_score_policy']
            report['images'].append(row)
            write(audit / 'report.json', report)
            status['completed_images'] = len(report['images']); write(audit / 'status.json', status)
            print('INFER', status['completed_images'], '/', len(files), path.stem, flush=True)
        expected = {p.stem + suffix for p in files for suffix in ('_inst.png', '_class.json')}
        if {p.name for p in result_dir.iterdir()} != expected:
            raise RuntimeError('Result directory does not contain exactly one pair per image')
        after = {name: tensor_digest(module.state_dict().items()) for name, module in named_modules}
        if (after != before
                or runtime_record(restoration_noise_mode=restorer.inference_noise_mode) != runtime
                or verify_source_manifest() != manifest):
            raise RuntimeError('Models/source manifest/precision changed during inference')
        for name, digest in source_hashes.items():
            if sha(ROOT / name) != digest:
                raise RuntimeError('Imported source changed during inference: ' + name)
        complete = len(files) == contract.count
        flags = dict(full100_complete=complete and contract.count == 100,
                     full150_complete=complete and contract.count == 150, full_dataset_complete=complete)
        report.update(complete=True, **flags, frozen_models_unchanged=True,
                      elapsed_seconds=time.time() - started)
        write(audit / 'report.json', report)
        status.update(complete=True, status='complete', dataset=contract.name, **flags)
        write(audit / 'status.json', status)
        print('SEMIFINAL_INFERENCE_COMPLETE', len(files), output, flush=True)
    except Exception as exc:
        status.update(status='failed', error=repr(exc)); write(audit / 'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, help='Only source test images are required')
    parser.add_argument('--output-dir', required=True, help='Fresh directory; creates results/ and audit/')
    parser.add_argument('--config', default=str(ROOT / 'config/inference/submission_reflect.yaml'))
    parser.add_argument('--model-dir', help='Defaults to code_model/model; model files remain SHA-verified')
    parser.add_argument('--only', nargs='+', help='Optional fixed-name short check, e.g. test_101 test_116')
    parser.add_argument('--dataset', choices=('quarter_final_100', 'semifinal_150'),
                        help='Explicit stage override; defaults to config or legacy 100-image stage')
    parser.add_argument('--save-analysis', action='store_true', help='Private boundary/restored preview arrays outside results/')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
