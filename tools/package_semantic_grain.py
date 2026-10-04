# -*- coding: utf-8 -*-
"""固定reflect实例，用s64重投票；产出内部结果缓存，完整复赛包另需code_model。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.semantic_crossover import _archive_stems, _cell_row, _read_prediction, revote_instances
from utils.config import load_config, project_path

FORMAT = 'semantic_grain_submission_v1'
HEAD_FORMAT = 'semantic_grain_terminal_only_v1'
PINS = dict(
    checkpoint_sha256='43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434',
    head_checkpoint_sha256='19433e78379026458566c06d3ca5eb2a14195b9c6fd9ce3e7a1b635adba9b67e',
    training_report_sha256='590581824a1ba3ce2d9d57e610b5a34aa4c1e08a714e07506a4d62019419692d',
    scored_zip_sha256='c812fbf32a87ae3a22973b2cd5548a4ab4bf9643dfc4b3136a584942d54da1e2',
    base_state_sha256='6ae2866ff8b76137dba8e3282f1f54de7a485e975105a453521ed7133906a3f4',
    initial_head_sha256='be3686f15c2f42703157234f584bc2f4f83f8dc8d57d8c8fc80729cced8eb11e',
    final_head_sha256='a6d5eeb82d31a3a211f99061f739de99a40b07c336d9eb241375884349703aab')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def require_sha(path, expected):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError('Protected input SHA differs: ' + str(path))
    return actual


def sealed_input_sha(protected_inputs, path, project_root):
    """从已封存训练回执找项目内输入；允许项目根迁移，歧义时严格拒绝。"""
    path = Path(path).resolve(); project_root = Path(project_root).resolve()
    if not path.is_relative_to(project_root):
        raise RuntimeError('Sealed input must remain inside configured project')
    relative = path.relative_to(project_root).as_posix()
    matches = [digest for source, digest in protected_inputs.items()
               if source.replace('\\', '/').endswith('/'+relative)]
    if len(matches) != 1:
        raise RuntimeError('Unique training-sealed input identity missing: '+relative)
    return matches[0]


def validate_recipe(cfg):
    expected = dict(format=FORMAT, steps=64, input_size=1024, expected_images=100,
                    conservative_parameters=91253055, monitor_seed=20261004, random_monitor_images=4, **PINS)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered s64 terminal-only submission identity differs')


def validate_training(report, state, architecture, cfg):
    """只接受完整正式64步及同一训练身份；不把checkpoint的complete字段当作充分证据。"""
    identity = report.get('identity', {})
    if (report.get('format') != 'semantic_grain_fit_v1' or report.get('complete') is not True
            or report.get('steps') != 64 or report.get('optimizer_updates') != 64
            or report.get('effective_head_views') != 128 or report.get('final_sources') != 40
            or report.get('final_crop_views') != 64 or report.get('training') is not False
            or report.get('final_head_sha256') != cfg['final_head_sha256']
            or report.get('strict_reload_final40_exact') is not True
            or report.get('strict_reload_crop_scope_exact') is not True
            or report.get('upstream_frozen_exact') is not True or report.get('fixed_png_bytes_exact') is not True
            or identity.get('mode') != 'formal' or identity.get('total_steps') != 64
            or identity.get('base_checkpoint_sha256') != cfg['checkpoint_sha256']
            or identity.get('base_state_sha256') != cfg['base_state_sha256']
            or identity.get('initial_head_sha256') != cfg['initial_head_sha256']
            or state.get('format') != HEAD_FORMAT or state.get('step') != 64
            or state.get('total_steps') != 64 or state.get('terminal_only') is not True
            or state.get('do_not_replace_geometry_upstream') is not True
            or state.get('identity') != identity or state.get('architecture') != architecture
            or state.get('snapshots') != report.get('snapshots')
            or not isinstance(state.get('semantic_state_dict'), dict)):
        raise RuntimeError('Complete exact-identity s64 terminal-only checkpoint/report required')


def select_sources(stems, training, limit):
    if limit not in (None, 8):
        raise ValueError('Only the complete100 run or predefined --limit 8 smoke is permitted')
    rows = training['snapshots']['64']['rows']
    tests = sorted(name for name, row in rows.items() if row['kind'] == 'test')
    if len(tests) != 8 or not set(tests).issubset(stems):
        raise RuntimeError('Original predefined eight test sources differ')
    return tests if limit == 8 else list(stems), tests


def archive_contract(path, expected_count):
    with zipfile.ZipFile(path) as archive:
        stems = _archive_stems(archive)
        if len(stems) != expected_count or archive.testzip() is not None:
            raise RuntimeError('ZIP cohort count or CRC differs')
        for name in stems:
            _read_prediction(archive, name)
    return stems


def require_baseline(ids, probability, classes, expected_probability_sha):
    """末端零变化契约：同原始概率且最终每个ID类别恰与已评分包一致。"""
    # run_marker_reflect.array_sha的实际定义仅hash contiguous字节，不含dtype/shape。
    probability_sha = hashlib.sha256(np.ascontiguousarray(probability).tobytes()).hexdigest()
    actual, scores = revote_instances(ids, probability)
    if probability_sha != expected_probability_sha or actual != classes:
        raise RuntimeError('Original whole raw probability/final classes do not reproduce scored reflect ZIP')
    return scores


def write_package(source, output, classes_by_source):
    """只写已完成子集，PNG字节直接复制；严格拒绝JSON与实例ID错配。"""
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as target:
        for stem in sorted(classes_by_source):
            ids, old, _ = _read_prediction(original, stem)
            new = classes_by_source[stem]
            if (set(new) != set(old) or any(type(v) is not int or v not in (0, 1) for v in new.values())):
                raise ValueError('New classes must match all unchanged mask IDs with integer0/1')
            target.writestr(stem+'_inst.png', original.read(stem+'_inst.png'))
            target.writestr(stem+'_class.json', json.dumps(new, ensure_ascii=False, indent=2).encode('utf-8'))
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output) as result:
        if result.testzip() is not None or _archive_stems(result) != sorted(classes_by_source):
            raise RuntimeError('Written final package cohort/CRC differs')
        for stem in classes_by_source:
            _read_prediction(result, stem)
            if (result.read(stem+'_inst.png') != original.read(stem+'_inst.png')
                    or json.loads(result.read(stem+'_class.json')) != classes_by_source[stem]):
                raise RuntimeError('Final package changed PNG bytes or class mapping')
    return dict(path=str(output), sha256=sha(output), bytes=Path(output).stat().st_size,
                files=2*len(classes_by_source), mask_bytes_identical=True)


def logits_of(output):
    return output['semantic_logits'] if isinstance(output, dict) else output


def run(args):
    config = load_config(args.config)
    cfg = config['semantic_grain_submission']; validate_recipe(cfg)
    if not args.run:
        print(json.dumps(dict(format=FORMAT, status='plan_only', training=False, optimizer_updates=0,
            checkpoint=cfg['checkpoint'], head_checkpoint=cfg['head_checkpoint'], steps=64,
            output_dir=cfg['output_dir'], images=8 if args.limit == 8 else 100,
            fixed_geometry='Original scored reflect32 PNG bytes', input='Raw whole letterbox1024',
            forwards='One shared encoder, original semantic and copied s64 semantic; no affinity/D5a/watershed',
            baseline_gate='Every original raw probability SHA and final class exact',
            candidate_gate='Strict reloaded head exact for all images, original eight s64 replay exact',
            conservative_parameters=91253055, official_submission=False, best_deployment_changed=False), indent=2))
        return
    from models.backend_adaptation import load_backend
    from tools.probe_semantic_object_fit import frozen_check, preview, probability
    from tools.probe_semantic_reflect import protected_sources
    from tools.run_marker_reflect import precision_record
    from train_backend_adaptation import tensor_digest
    from utils.affinity_deployment import prepare_image
    from utils.semantic_object_fit import array_sha
    from data.mim_dataset import list_images

    if args.limit not in (None, 8):
        raise ValueError('Only complete100 or predefined8 smoke permitted')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual sam2_env CUDA required')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    resolve = lambda p: Path(project_path(config, p)).resolve()
    out = resolve(args.output or cfg['output_dir'])
    if not out.is_relative_to((ROOT/'outputs').resolve()) or out.exists():
        raise RuntimeError('Fresh private outputs directory required')
    out.mkdir(parents=True); started = time.time(); protected = {}; hook = None
    def protect(path, expected=None):
        path = resolve(path); digest = sha(path)
        if expected is not None and digest != expected:
            raise RuntimeError('Protected input SHA differs: '+str(path))
        protected[str(path)] = digest
        return path
    try:
        write(out/'status.json', dict(status='running', stage='identity', training=False))
        training_root = resolve(cfg['training_dir'])
        training = read(protect(training_root/'report.json', cfg['training_report_sha256']))
        sealed_inputs = training['identity']['protected_inputs']
        capture_pin = sealed_input_sha(sealed_inputs, resolve(cfg['capture']), resolve('.'))
        receipt_pin = sealed_input_sha(sealed_inputs, resolve(cfg['input_receipt']), resolve('.'))
        capture = read(protect(cfg['capture'], capture_pin))
        receipt = read(protect(cfg['input_receipt'], receipt_pin))
        source_zip = protect(receipt['scored_zip']['path'], cfg['scored_zip_sha256'])
        if receipt['scored_zip']['sha256'] != cfg['scored_zip_sha256']:
            raise RuntimeError('Scored ZIP input registration differs')
        marker = read(protect(receipt['reflect_report']['path'], receipt['reflect_report']['sha256']))
        if (marker.get('format') != 'marker_reflect_run_v1' or marker.get('complete') is not True
                or marker.get('full100_complete') is not True or marker.get('reflection_padding') != 32
                or marker.get('no_seams') is not True or marker.get('checkpoint_sha256') != cfg['checkpoint_sha256']):
            raise RuntimeError('Complete scored reflect32 deployment report required')
        stems = archive_contract(source_zip, 100)
        marker_rows = {r['source']: r for r in marker['images']}
        files = sorted(Path(p) for p in list_images(project_path(config, config['inference']['test_dir'])))
        if len(files) != 100 or [p.stem for p in files] != stems or set(marker_rows) != set(stems) or len(marker['images']) != 100:
            raise RuntimeError('Actual100 source images/scored ZIP/reflect report cohort differs')
        paths = {p.stem: p for p in files}
        selected, tests = select_sources(stems, training, args.limit)
        checkpoint = protect(cfg['checkpoint'], cfg['checkpoint_sha256'])
        state_path = protect(cfg['head_checkpoint'], cfg['head_checkpoint_sha256'])
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        model, bundle = load_backend(checkpoint, config, 'cuda'); model.eval().requires_grad_(False)
        validate_training(training, state, bundle['architecture']['semantic_decoder'], cfg)
        initial = tensor_digest(model.semantic_decoder.state_dict().items())
        before = tensor_digest(model.state_dict().items())
        if bundle['epoch'] != 20 or model.semantic_lora is not None or before != cfg['base_state_sha256'] or initial != cfg['initial_head_sha256']:
            raise RuntimeError('Original frozen R1/base semantic identity differs')
        head = deepcopy(model.semantic_decoder).eval().requires_grad_(False)
        head.load_state_dict(state['semantic_state_dict'], strict=True)
        reloaded = deepcopy(head).eval().requires_grad_(False)
        reloaded.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=False)['semantic_state_dict'], strict=True)
        final = tensor_digest(head.state_dict().items())
        if final != cfg['final_head_sha256'] or tensor_digest(reloaded.state_dict().items()) != final:
            raise RuntimeError('Final copied head strict state digest differs')
        if sum(p.numel() for p in head.parameters()) != 6302010 or sum(p.numel() for p in model.parameters())+3283651+6302010 != cfg['conservative_parameters']:
            raise RuntimeError('Conservative deployed parameter ledger differs')
        precision = precision_record()
        if precision != training['identity']['precision'] or precision != marker['precision']:
            raise RuntimeError('Actual original FP32 precision differs')
        config_sources = {}; p = Path(args.config).resolve(); visited = set()
        while True:
            if p in visited or not p.is_relative_to(ROOT):
                raise RuntimeError('Invalid inherited YAML path')
            visited.add(p); protect(p); key = p.relative_to(ROOT).as_posix()
            expected = training['inherited_config_sources'].get(key)
            if expected is not None:
                require_sha(p, expected)
            target = out/'config_source'/key; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
            config_sources[key] = sha(p)
            parent = (yaml.safe_load(p.read_text(encoding='utf-8')) or {}).get('_base')
            if not parent: break
            p = (p.parent/parent).resolve()
        code, virtual = protected_sources(out)
        for key, digest in code.items():
            expected = training['identity']['protected_sources'].get(key)
            if expected is not None and expected != digest:
                raise RuntimeError('Loaded training/inference shared source differs: '+key)
        def forbidden(*_):
            raise RuntimeError('Affinity forward forbidden in semantic terminal-only packaging')
        hook = model.affinity_decoder.register_forward_pre_hook(forbidden)
        other = sorted(set(stems)-set(tests)); rng = random.Random(cfg['monitor_seed'])
        monitors = set(tests) | set(rng.sample(other, min(cfg['random_monitor_images'], len(other))))
        write(out/'config.json', config)
        report = dict(format=FORMAT, complete=False, training=False, optimizer_updates=0,
            official_submission=False, official_scores=None, best_deployment_changed=False,
            scope='Only terminal classes; frozen scored reflect32 geometry and original upstream semantic',
            steps=64, conservative_parameters=cfg['conservative_parameters'], selected_sources=selected,
            full100_complete=False, precision=precision, protected_sources=code, virtual_module_paths=virtual,
            inherited_config_sources=config_sources, head_state_sha256=final, base_state_sha256=before,
            original_head_sha256=initial, monitoring=dict(fixed_eight=tests, seed=cfg['monitor_seed'],
                random_nonmonitor=sorted(monitors-set(tests))), images=[])
        rows = []; classes_by_source = {}
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False), \
                patch('cv2.watershed', side_effect=RuntimeError('No watershed permitted')), \
                patch('models.backend_adaptation.restore_first', side_effect=RuntimeError('No D5a permitted')), \
                zipfile.ZipFile(source_zip) as original:
            for name in selected:
                expected = marker_rows[name]
                source = protect(paths[name], expected['source_sha256'])
                ids, old_classes, areas = _read_prediction(original, name)
                png = original.read(name+'_inst.png'); png_sha = hashlib.sha256(png).hexdigest()
                if (png_sha != expected['output']['png_sha256']
                        or hashlib.sha256(original.read(name+'_class.json')).hexdigest() != expected['output']['classes_sha256']):
                    raise RuntimeError('Scored ZIP PNG/classes bytes differ from reflect capture')
                rgb, image, ph, pw = prepare_image(source, 1024, 'cuda')
                image = image.float()
                record = dict(shape=list(rgb.shape[:2]), pad_h=ph, pad_w=pw)
                if ids.shape != tuple(record['shape']) or record['shape'] != expected['shape']:
                    raise RuntimeError('Source/native output shape differs')
                features = model.encoder(image)
                feature_sha = tensor_digest([(str(i), v) for i, v in enumerate(features)])
                if name in tests:
                    cached = capture['captures'][name]
                    if (list(image.stride()) != cached['input_stride'] or feature_sha != cached['feature_sha256']
                            or [list(v.stride()) for v in features] != cached['feature_strides']
                            or record['shape'] != cached['shape']):
                        raise RuntimeError('Predeclared test raw input/features/cache identity differs')
                _, old_p = probability(logits_of(model.semantic_decoder(features, image)), record)
                old_scores = require_baseline(ids, old_p, old_classes, expected['fixed_maps']['raw_probability_sha256'])
                _, p_new = probability(logits_of(head(features, image)), record)
                new_classes, scores = revote_instances(ids, p_new)
                _, replay = probability(logits_of(reloaded(features, image)), record)
                replay_classes, replay_scores = revote_instances(ids, replay)
                if not np.array_equal(p_new, replay) or new_classes != replay_classes or scores != replay_scores:
                    raise RuntimeError('Reloaded new terminal head probability/classes/scores differ')
                p_sha = hashlib.sha256(p_new.tobytes()).hexdigest()
                if name in tests:
                    for step, value, classes in (('0', old_p, old_classes), ('64', p_new, new_classes)):
                        trained = training['snapshots'][step]['rows'][name]
                        trained_class_path = protect(training_root/f's{int(step):02d}'/name/(name+'_class.json'))
                        receipt_classes = dict(old_classes)
                        for key in trained['changed_ids']:
                            receipt_classes[key] = 1-receipt_classes[key]
                        if (hashlib.sha256(value.tobytes()).hexdigest() != trained['probability_sha256']
                                or png_sha != trained['png_sha256'] or read(trained_class_path) != classes
                                or classes != receipt_classes):
                            raise RuntimeError('Existing fixed-eight training final probability/classes/PNG replay differs')
                folder = out/'results'; folder.mkdir(exist_ok=True)
                (folder/(name+'_inst.png')).write_bytes(png); write(folder/(name+'_class.json'), new_classes)
                cell = _cell_row(new_classes, scores, areas, old_classes)
                changes = [dict(id=k, area=areas[k], original=old_classes[k], current=new_classes[k],
                    original_score=old_scores[k], current_score=scores[k]) for k in new_classes if new_classes[k] != old_classes[k]]
                row = dict(source=name, source_sha256=sha(source), shape=record['shape'], pad_h=ph, pad_w=pw,
                    input_sha256=array_sha(image), input_stride=list(image.stride()), feature_sha256=feature_sha,
                    feature_strides=[list(v.stride()) for v in features], png_sha256=png_sha,
                    classes_sha256=sha(folder/(name+'_class.json')),
                    original_probability_sha256=expected['fixed_maps']['raw_probability_sha256'],
                    probability_sha256=p_sha, strict_reload_exact=True, original_deployment_exact=True,
                    fixed_eight_training_exact=name in tests, prediction=cell, changes=changes,
                    changed_size_buckets={label: dict(count=sum(lo <= r['area'] and (hi is None or r['area'] < hi) for r in changes),
                        pixels=sum(r['area'] for r in changes if lo <= r['area'] and (hi is None or r['area'] < hi)))
                        for lo, hi, label in ((0, 200, 'lt200'), (200, 1000, '200to999'), (1000, 10000, '1000to9999'), (10000, None, 'ge10000'))})
                if name in monitors:
                    preview(out/'monitor'/(name+'_new.png'), rgb, ids, new_classes, p_new, 's64', 320)
                    preview(out/'monitor'/(name+'_old.png'), rgb, ids, old_classes, old_p, 'reflect R1', 320)
                rows.append(row); classes_by_source[name] = new_classes
                write(out/'rows.json', rows)
                write(out/'status.json', dict(status='running', stage='terminal_vote', completed_images=len(rows), expected_images=len(selected), training=False))
                print('SEMANTIC_GRAIN_PACKAGE', len(rows), name, cell['changed_instances'], flush=True)
        frozen_check(model, before); frozen_check(head, final); frozen_check(reloaded, final)
        if precision_record() != precision:
            raise RuntimeError('Inference precision changed')
        for path, digest in protected.items(): require_sha(path, digest)
        for key, digest in code.items():
            require_sha(ROOT/key, digest); require_sha(out/'source'/key, digest)
        for key, digest in config_sources.items(): require_sha(out/'config_source'/key, digest)
        # 仅预测的兼容ZIP是中间产物；正式复赛交付须封装results/与完整code_model/。
        package = None if args.limit == 8 else write_package(source_zip, out/'grain64_results.zip', classes_by_source)
        report.update(complete=True, full100_complete=len(selected)==100, images=rows, package=package,
            protected_inputs=protected, protected_inputs_exact=True, model_frozen_exact=True,
            head_frozen_exact=True, fixed_png_bytes_exact=True, all_baselines_exact=True,
            strict_reload_all_exact=True, fixed_eight_replay_exact=True, encoder_calls=len(selected),
            original_head_calls=len(selected), candidate_head_calls=2*len(selected), affinity_calls=0,
            D5a_calls=0, watershed_calls=0, elapsed_seconds=time.time()-started,
            summary={field:sum(r['prediction'][field] for r in rows) for field in
                ('instances','ferrite','pearlite','ferrite_pixels','covered_pixels','changed_instances','changed_pixels')})
        write(out/'report.json', report); write(out/'status.json', dict(status='complete', training=False, images=len(selected)))
        print('SEMANTIC_GRAIN_PACKAGE_COMPLETE', json.dumps(package), flush=True)
        return report
    except Exception as error:
        write(out/'status.json', dict(status='failed', training=False, error=repr(error)))
        raise
    finally:
        if hook is not None: hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/inference/semantic_grain_submission.yaml')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int, choices=(8,))
    parser.add_argument('--output', help='Fresh ignored output override; smoke uses separate directory')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
