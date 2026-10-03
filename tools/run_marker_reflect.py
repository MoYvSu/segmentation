# -*- coding: utf-8 -*-
"""固定正式R1原生1024链，只替换watershed种子为32像素反射版本。

从私有receipt读取模型身份，不含具体图名、GT或逐图坐标。完整模式严格
复现100图旧输出；--indices仅作按完整索引取seed的一张或小样本短测。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from tools.semantic_crossover import revote_instances
from utils.patch_diagnostic import phase_description

FORMAT = 'marker_reflect_run_v1'
PADDING = 32
VIEW = 'patch1024'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf8')


def config_gate(actual, registered):
    """仅接受已登记的输出路径及显式legacy_none默认值差异。"""
    actual = json.loads(json.dumps(actual)); registered = json.loads(json.dumps(registered))
    actual_explicit = 'input_normalization' in actual['sam2']
    canonical_explicit = 'input_normalization' in registered['sam2']
    differences = []
    for path in (('backend_adaptation', 'output_dir'), ('paths', 'project_root')):
        a, b = actual[path[0]].get(path[1]), registered[path[0]].get(path[1])
        if a != b:
            differences.append(dict(path='.'.join(path), actual=a, registered=b,
                                    reason='非模型或推理参数；项目根另按当前ROOT核验'))
        actual[path[0]].pop(path[1], None); registered[path[0]].pop(path[1], None)
    for value in (actual, registered):
        value['sam2'].setdefault('input_normalization', 'legacy_none')
    if actual != registered:
        raise RuntimeError('Receipt differs from canonical R1 outside permitted path/default metadata')
    if actual['sam2']['input_normalization'] != 'legacy_none':
        raise RuntimeError('R1 requires legacy_none input normalization')
    # 等价默认值也写入回执，不能把它误报成输入归一化改变。
    differences.append(dict(path='sam2.input_normalization', effective='legacy_none',
                            reason='后端代码的同一显式默认值', actual_explicit=actual_explicit,
                            canonical_explicit=canonical_explicit))
    return differences


def validate_manifest(manifest, files, checkpoint_sha, expected_count=100):
    selected = [r for r in manifest.get('images', []) if r.get('view') == VIEW]
    records = {r['source']: r for r in selected}
    names = {p.stem for p in files}
    if (len(files) != expected_count or len(names) != expected_count
            or manifest.get('complete') is not True or manifest.get('epoch') != 20
            or manifest.get('precision') != 'FP32' or 1024 not in manifest.get('sizes', [])
            or manifest.get('checkpoint_sha256') != checkpoint_sha
            or len(selected) != expected_count or set(records) != names):
        raise RuntimeError('Original complete e20 FP32 1024 deployment cohort/identity differs')
    return records


def selected_indices(value, count):
    if value is None:
        return list(range(count))
    try:
        values = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise ValueError('indices must be comma-separated full-cohort integer indices') from exc
    if not values or len(set(values)) != len(values) or min(values) < 0 or max(values) >= count:
        raise ValueError('indices must be distinct full-cohort indices in range')
    return sorted(values)


def verify_prediction(ids, classes, shape):
    if (ids.ndim != 2 or ids.shape != tuple(shape) or ids.dtype != np.uint16
            or int(ids.max()) > 65535
            or set(classes) != {str(int(i)) for i in np.unique(ids) if i > 0}
            or any(type(c) is not int or c not in (0, 1) for c in classes.values())):
        raise RuntimeError('Final uint16/instance-class contract differs')


def load_prediction(folder, stem, shape):
    ids = cv2.imread(str(Path(folder) / (stem + '_inst.png')), cv2.IMREAD_UNCHANGED)
    if ids is None:
        raise RuntimeError('Missing reference/candidate PNG: ' + stem)
    classes = read(Path(folder) / (stem + '_class.json'))
    verify_prediction(ids, classes, shape)
    return ids, classes


def save_prediction(folder, stem, ids, classes):
    verify_prediction(ids, classes, ids.shape)
    png = Path(folder) / (stem + '_inst.png'); labels = Path(folder) / (stem + '_class.json')
    if not cv2.imwrite(str(png), ids):
        raise RuntimeError('Could not write uint16 PNG')
    write(labels, classes)
    actual, mapping = load_prediction(folder, stem, ids.shape)
    if not np.array_equal(actual, ids) or mapping != classes:
        raise RuntimeError('Saved candidate differs from in-memory final output')
    return dict(png_sha256=sha(png), classes_sha256=sha(labels))


def capture_original(views, model, restorer, path, config, device, seed):
    """透明观察一次原forward；恢复、CUDA插值、窗口拼接均由原49代码完成。"""
    original_decode = views.decode_native; original_vote = views.revote_instances
    real_ws = cv2.watershed; calls = []; votes = []

    def observed_decode(semantic, boundary, actual_config, *args, **kwargs):
        if calls:
            raise RuntimeError('Expected exactly one original native decode')
        ws_calls = []

        def observed_ws(image, markers):
            if ws_calls:
                raise RuntimeError('Expected exactly one original watershed call')
            before_image = image.copy(); before_marker = markers.copy()
            result = real_ws(image, markers)
            if not np.array_equal(image, before_image):
                raise RuntimeError('Original watershed changed its elevation input')
            ws_calls.append(dict(markers=before_marker, watershed=np.maximum(result, 0).copy(),
                                 elevation_sha256=array_sha(before_image)))
            return result

        with patch('utils.post_process.cv2.watershed', side_effect=observed_ws):
            ids, classes = original_decode(semantic, boundary, actual_config, *args, **kwargs)
        if len(ws_calls) != 1:
            raise RuntimeError('Original nonempty marker contract requires one watershed call')
        calls.append(dict(semantic=semantic.detach().float().cpu().clone(),
                          boundary=boundary.detach().float().cpu().clone(),
                          geometry_ids=ids.copy(), geometry_classes=deepcopy(classes), **ws_calls[0]))
        return ids, classes

    def observed_vote(ids, probability, *args, **kwargs):
        if votes:
            raise RuntimeError('Expected exactly one original raw semantic vote')
        votes.append(np.asarray(probability).copy())
        return original_vote(ids, probability, *args, **kwargs)

    with patch.object(views, 'decode_native', side_effect=observed_decode), \
            patch.object(views, 'revote_instances', side_effect=observed_vote):
        rgb, results = views.predict_views(model, restorer, path, config, device, seed, [1024])
    if len(calls) != 1 or len(votes) != 1 or set(results) != {VIEW}:
        raise RuntimeError('Original single-forward single-view capture contract differs')
    captured = calls[0]; captured.update(raw_probability=votes[0], baseline=results[VIEW])
    if not np.array_equal(captured['boundary'][0, 0].numpy(), results[VIEW]['boundary']):
        raise RuntimeError('Observed native boundary differs from original returned blend')
    return rgb, captured


def replay_candidates(captured, config, *, stability=False):
    """每图复刻真实旧marker/WS/rawvote，再只替换marker生成候选。"""
    semantic, boundary = captured['semantic'], captured['boundary']
    original = decode_marker_replay(semantic, boundary, config)
    if (not np.array_equal(original['instances'], captured['geometry_ids'])
            or original['classes'] != captured['geometry_classes']
            or not np.array_equal(original['actual_markers'], captured['markers'])
            or not np.array_equal(original['watershed'], captured['watershed'])
            or original['elevation_sha256'] != captured['elevation_sha256']):
        raise RuntimeError('CPU replay differs from actual original native decode/marker/elevation')
    classes, _ = revote_instances(original['instances'], captured['raw_probability'])
    if (not np.array_equal(original['instances'], captured['baseline']['instances'])
            or classes != captured['baseline']['classes']):
        raise RuntimeError('Baseline CPU rawvote differs from original full output')
    record = dict(original_marker_exact=True, original_watershed_exact=True,
                  original_elevation_exact=True, original_rawvote_exact=True,
                  original_hashes=original['hashes'], same_marker_gate=None)
    if stability:
        same = decode_marker_replay(semantic, boundary, config, original['actual_markers'])
        if (same['hashes'] != original['hashes'] or same['classes'] != original['classes']
                or not np.array_equal(same['instances'], original['instances'])):
            raise RuntimeError('Same-marker intervention must reproduce complete original output exactly')
        record['same_marker_gate'] = True
    variants = {}
    for padding in ((PADDING, 64) if stability else (PADDING,)):
        stage = build_marker_variant(original['original_stages'], 'reflect', padding)
        value = decode_marker_replay(semantic, boundary, config, stage['marker_labels'])
        if (value['elevation_sha256'] != original['elevation_sha256']
                or not np.array_equal(value['original_markers'], original['actual_markers'])
                or not np.array_equal(value['actual_markers'], stage['marker_labels'])
                or not np.array_equal(stage['barrier_belt'], original['original_stages']['barrier_belt'])):
            raise RuntimeError('Reflect candidate changed frozen original marker/barrier/elevation')
        value['geometry_classes'] = value['classes']
        value['classes'], _ = revote_instances(value['instances'], captured['raw_probability'])
        value['variant_stages'] = stage
        verify_prediction(value['instances'], value['classes'], boundary.shape[-2:])
        variants[str(padding)] = value
    if stability:
        a, b = variants['32'], variants['64']
        record['padding_stability'] = dict(paddings=[32, 64],
            marker_exact=bool(np.array_equal(a['actual_markers'], b['actual_markers'])),
            watershed_exact=bool(np.array_equal(a['watershed'], b['watershed'])),
            final_ids_exact=bool(np.array_equal(a['instances'], b['instances'])),
            classes_exact=a['classes'] == b['classes'],
            changed_id_pixels=int(np.count_nonzero(a['instances'] != b['instances'])),
            rule='fixed32 candidate irrespective of64 result; no padding selection')
    return original, variants, record


def precision_record():
    value = dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 matmul_precision=torch.get_float32_matmul_precision(), autocast=False,
                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if not value['cudnn_tf32'] or value['matmul_tf32'] or value['matmul_precision'] != 'highest':
        raise RuntimeError('Original FP32/CUDA precision contract differs')
    return value


def source_snapshot(output):
    """只封存已导入的项目代码；赛题receipt与推理产物保持私有且不进入源码。"""
    paths = {Path(__file__).resolve()}; virtual_paths = set()
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT):
                relative = path.relative_to(ROOT)
                if relative.parts[0] not in {'output', 'outputs'}:
                    # 动态封存模块可能声明虚拟__file__；只跳过明确不存在者。
                    # PermissionError等真实读写失败仍向上传递，不能静默漏保护。
                    try:
                        path.stat()
                    except FileNotFoundError:
                        virtual_paths.add(relative.as_posix())
                        continue
                    paths.add(path)
    paths.update(ROOT / 'outputs/affinity_balance/source' / name
                 for name in ('train_affinity_native.py', 'tools/affinity_native_views.py'))
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT); target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        hashes[relative.as_posix()] = sha(path)
        if sha(target) != hashes[relative.as_posix()]:
            raise RuntimeError('Source snapshot copy differs')
    return hashes, sorted(virtual_paths)


def run(args):
    from data.mim_dataset import list_images
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from train_backend_adaptation import tensor_digest
    from tools.run_affinity_sweep import runtime_contract
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Use server sam2_env CUDA; CPU is only for tests and marker replay')
    output = (ROOT / args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'outputs'):
        raise RuntimeError('A fresh ignored outputs directory is required')
    output.mkdir(parents=True)
    started = time.time()
    torch.set_num_threads(4); cv2.setNumThreads(2)
    write(output / 'status.json', dict(status='running', stage='preflight', training=False))
    try:
        receipt_path = (ROOT / args.receipt).resolve(); receipt = read(receipt_path)
        r1 = receipt['arms']['r1']; config = deepcopy(r1['config'])
        canonical = load_config(str(ROOT / 'config/train/affinity_balance_mixed.yaml'))
        differences = config_gate(config, canonical)
        if (receipt.get('complete') is not True or r1['epoch'] != 20
                or Path(project_path(config)).resolve() != ROOT
                or 'affinity_coverage' in config
                or config['affinity_native']['overlap'] != .25
                or config['backend_adaptation']['restoration']['inference_seed'] != 314159):
            raise RuntimeError('R1 receipt/config/root/native-seed identity differs')
        checkpoint = Path(r1['checkpoint']).resolve()
        if not checkpoint.is_relative_to(ROOT) or sha(checkpoint) != r1['checkpoint_sha256']:
            raise RuntimeError('R1 checkpoint path/SHA differs from actual receipt')
        restoration = config['backend_adaptation']['restoration']
        restore_path = Path(project_path(config, restoration['checkpoint'])).resolve()
        if (sha(restore_path) != receipt['restorer_checkpoint_sha256']
                or sha(restore_path) != restoration['checkpoint_sha256']):
            raise RuntimeError('D5a checkpoint differs')
        reference = (ROOT / args.reference).resolve(); cache = reference / VIEW
        manifest_path = reference / 'manifest.json'; manifest = read(manifest_path)
        files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
        if files != sorted(files):
            raise RuntimeError('Original complete source ordering differs')
        records = validate_manifest(manifest, files, r1['checkpoint_sha256'])
        expected = {p.stem + suffix for p in files for suffix in ('_inst.png', '_class.json')}
        if {p.name for p in cache.iterdir() if p.is_file()} != expected:
            raise RuntimeError('Original complete reference requires exactly200 output files')
        indices = selected_indices(args.indices, len(files)); selected = set(indices)
        precision = precision_record(); runtime = runtime_contract(config)
        if precision != receipt['precision'] or runtime != receipt.get('current_runtime', receipt.get('runtime')):
            raise RuntimeError('Actual migrated runtime/precision differs from validated capture')
        core = load_original_core(config); views = core.coverage_original_views
        model, bundle = load_backend(checkpoint, config, 'cuda')
        model.eval().requires_grad_(False); restorer = get_restorer(config, 'cuda')
        before = tensor_digest(model.state_dict().items()); restore_before = tensor_digest(restorer.state_dict().items())
        if (bundle['epoch'] != 20 or before != r1['model_state_sha256']
                or restore_before != receipt['restorer_state_sha256']):
            raise RuntimeError('Actual frozen R1/D5a state differs')
        if any(m.training for m in list(model.modules()) + list(restorer.modules())):
            raise RuntimeError('All inference modules must remain eval')
        parameters = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in restorer.parameters())
        if parameters >= 500_000_000:
            raise RuntimeError('Competition model parameter budget exceeded')
        target = output / 'deployment' / VIEW; target.mkdir(parents=True)
        maps = output / 'maps'; maps.mkdir()
        sources, virtual_paths = source_snapshot(output)
        report = dict(format=FORMAT, complete=False, training=False, no_optimizer_steps=True,
            official_accuracy=False, best_deployment_changed=False, submission_package_created=False,
            only_variable='reflect32 marker thinning extension at watershed seeds; all real barriers unchanged',
            reflection_padding=32, no_seams=True, config=config, canonical_config_differences=differences,
            receipt_sha256=sha(receipt_path), reference_manifest_sha256=sha(manifest_path),
            checkpoint_sha256=r1['checkpoint_sha256'], model_state_sha256=before,
            restorer_checkpoint_sha256=sha(restore_path), restorer_state_sha256=restore_before,
            precision=precision, runtime=runtime, sources=sources, parameter_count=parameters,
            skipped_nonexistent_virtual_module_paths=virtual_paths,
            complete_source_count=len(files), selected_full_indices=indices,
            first6_rule='full sorted cohort indices0..5, no image or output-based selection',
            images=[], caveat='Unlabeled mechanism/distribution check; no official score or deployment promotion')
        write(output / 'preflight.json', report)
        rows = []
        for index, path in enumerate(files):
            if index not in selected:
                continue
            seed = config['backend_adaptation']['restoration']['inference_seed'] + index
            rgb, captured = capture_original(views, model, restorer, path, config, 'cuda', seed)
            old, old_classes = load_prediction(cache, path.stem, rgb.shape[:2])
            actual = captured['baseline']; source_record = records[path.stem]
            if (not np.array_equal(old, actual['instances']) or old_classes != actual['classes']
                    or actual['phases'] != source_record['phases'] or actual['blend'] != source_record['blend']):
                raise RuntimeError('Actual original forward differs from formal PNG/JSON/phases/blend: ' + path.stem)
            original, variants, gates = replay_candidates(captured, config, stability=index < 6)
            candidate = variants['32']
            saved = save_prediction(target, path.stem, candidate['instances'], candidate['classes'])
            phases = phase_description(candidate['instances'], candidate['classes'])
            row = dict(source=path.stem, full_source_index=index, seed=seed, shape=list(rgb.shape[:2]),
                source_sha256=sha(path), view=VIEW, phases=phases, blend=actual['blend'], output=saved,
                reference=dict(png_sha256=sha(cache/(path.stem+'_inst.png')),
                               classes_sha256=sha(cache/(path.stem+'_class.json'))),
                baseline_final_ids_exact=True, baseline_rawvote_classes_exact=True,
                baseline_phases_blend_exact=True, replay=gates, candidate_hashes=candidate['hashes'],
                fixed_maps=dict(semantic_sha256=array_sha(captured['semantic'].numpy()),
                    boundary_sha256=array_sha(captured['boundary'].numpy()),
                    raw_probability_sha256=array_sha(captured['raw_probability'])),
                original_marker_count=original['original_stages']['marker_count'],
                candidate_marker_count=candidate['variant_stages']['marker_count'],
                original_phases=actual['phases'],
                final_ids_exact=bool(np.array_equal(old,candidate['instances'])),
                final_classes_exact=old_classes==candidate['classes'])
            if index < 6:
                row['reflect64_phases']=phase_description(variants['64']['instances'],variants['64']['classes'])
                row['reflect64_hashes']=variants['64']['hashes']
                row['reflect64_classes']=variants['64']['classes']
                np.savez_compressed(maps/(path.stem+'.npz'), semantic=captured['semantic'].numpy(),
                    boundary=captured['boundary'].numpy(), raw_probability=captured['raw_probability'],
                    original_markers=original['actual_markers'], original_watershed=original['watershed'],
                    original_instances=actual['instances'], reflect32_instances=candidate['instances'],
                    reflect64_instances=variants['64']['instances'],
                    reflect32_markers=candidate['actual_markers'], reflect32_watershed=candidate['watershed'],
                    reflect64_markers=variants['64']['actual_markers'], reflect64_watershed=variants['64']['watershed'])
            rows.append(row); report['images'].append(row)
            write(output/'report.json',report)
            write(output/'status.json',dict(status='running',stage='reflect_inference',
                completed_images=len(rows),expected_images=len(indices),training=False))
            print('REFLECT',len(rows),path.stem,'markers',row['original_marker_count'],row['candidate_marker_count'],flush=True)
            del rgb,captured,original,variants,candidate
        if (tensor_digest(model.state_dict().items())!=before
                or tensor_digest(restorer.state_dict().items())!=restore_before
                or precision_record()!=precision or runtime_contract(config)!=runtime):
            raise RuntimeError('Frozen models/precision/runtime changed during inference')
        for name,digest in sources.items():
            if sha(ROOT/name)!=digest or sha(output/'source'/name)!=digest:
                raise RuntimeError('Source changed during formal capture: '+name)
        actual_files={p.name for p in target.iterdir() if p.is_file()}
        expected_files={files[i].stem+suffix for i in indices for suffix in ('_inst.png','_class.json')}
        if actual_files!=expected_files or len(rows)!=len(indices):
            raise RuntimeError('Candidate output cohort is incomplete')
        report.update(complete=True,full100_complete=len(indices)==100,model_unchanged=True,
            restorer_unchanged=True,all_baselines_exact=True,elapsed_seconds=time.time()-started)
        write(output/'report.json',report)
        write(output/'deployment/manifest.json',dict(complete=True,full100_complete=len(indices)==100,
            checkpoint=str(checkpoint),checkpoint_sha256=r1['checkpoint_sha256'],epoch=20,sizes=[1024],
            images=rows,official_score=None,precision='FP32',marker_variant='reflect32',
            semantic='fixed full-image D5a geometry plus raw final vote',report_sha256=sha(output/'report.json')))
        write(output/'status.json',dict(status='complete',complete=True,full100_complete=len(indices)==100,
            training=False,completed_images=len(rows),no_optimizer_steps=True))
        print('MARKER_REFLECT_COMPLETE',len(rows),output,flush=True)
    except Exception as exc:
        write(output/'status.json',dict(status='failed',stage='error',training=False,error=repr(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',default='outputs/affinity_weak/capture.json')
    parser.add_argument('--reference',default='outputs/affinity_balance/mixed/deployment')
    parser.add_argument('--output',default='outputs/marker_trial')
    parser.add_argument('--indices',help='短测完整排序索引，例如0；不改变seed索引')
    run(parser.parse_args())


if __name__=='__main__':
    main()
