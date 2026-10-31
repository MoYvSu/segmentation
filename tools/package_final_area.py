# -*- coding: utf-8 -*-
"""从既有完整结果 ZIP 执行最终小区合并；纯 CPU，不重新推理或提交。

源报告提供 ZIP、逐文件哈希与原尺寸；源快照配置的完整 _base 链必须匹配
源源码 manifest，才能据此声明源种子面积。候选配置与源快照分开核验。
审计留在包外，ZIP 仅包含 results/ 下的 150 对 PNG/JSON。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import zipfile

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.final_instance_cleanup import ACTION, merge_final_instances
from utils.instance_area_policy import resolve_instance_area_policy

SOURCES = tuple('test_%03d' % number for number in range(169, 319))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def strict_json(payload):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=invalid)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def config_chain(path, manifest=None):
    """逐文件留证；源配置必须来自对应冻结 manifest，候选只记录新链。"""
    path = Path(path).resolve()
    # 本仓库配置以 config/ 为根；允许任意位置的完整轻量源码快照。
    require('config' in path.parts, 'Configuration must live under its snapshot config/ directory')
    config_index = len(path.parts) - 1 - list(reversed(path.parts)).index('config')
    snapshot = Path(*path.parts[:config_index])
    rows, seen = [], set()
    while True:
        require(path not in seen, 'Cyclic config inheritance')
        seen.add(path)
        try:
            relative = path.relative_to(snapshot).as_posix()
        except ValueError as error:
            raise ValueError('Config inheritance escapes source snapshot') from error
        digest = sha(path)
        if manifest is not None:
            require(manifest.get(relative) == digest, 'Source frozen config SHA differs: ' + relative)
        rows.append(dict(path=relative, sha256=digest))
        value = yaml.safe_load(path.read_text(encoding='utf8')) or {}
        require(isinstance(value, dict), 'Configuration must be an object')
        base = value.get('_base')
        if base is None:
            break
        require(isinstance(base, str) and bool(base), 'Expected one _base config path')
        path = (path.parent / base).resolve()
    return rows


def validate_source_report(report, input_sha):
    require(isinstance(report, dict), 'Source report must be an object')
    require(report.get('complete') is True and report.get('sha256') == input_sha,
            'Source report is incomplete or does not match input ZIP SHA256')
    require(report.get('pairs') == 150 and report.get('files') == 300,
            'Source report must contain exactly 150 pairs / 300 files')
    for key in ('only_results', 'CRC_all_passed', 'member_sha256_exact', 'native_sizes_uint16_and_ID_class_pairs_exact'):
        require(report.get(key) is True, 'Source report gate missing: ' + key)
    rows = report.get('files_sha')
    require(isinstance(rows, list) and len(rows) == 150, 'Missing 150 source member hash/shape records')
    registry = {}
    for row in rows:
        require(isinstance(row, dict) and row.get('source') in SOURCES, 'Invalid source name in report')
        source = row['source']
        require(source not in registry, 'Duplicate source in report: ' + source)
        shape = row.get('shape')
        require(isinstance(shape, list) and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape),
                'Invalid native shape: ' + source)
        for key in ('png_sha256', 'classes_sha256'):
            require(isinstance(row.get(key), str) and re.fullmatch('[0-9a-f]{64}', row[key]), 'Invalid member SHA')
        registry[source] = row
    require(set(registry) == set(SOURCES), 'Source report cohort must be exactly test_169 through test_318')
    return registry


def archive_members(archive):
    """拒绝越界路径、符号链接、混合根目录、重复成员和任何额外文件。"""
    members, prefixes = {}, set()
    for info in archive.infolist():
        name = info.filename
        require(not info.is_dir() and not info.flag_bits & 1, 'Directories/encryption are not allowed in result ZIP')
        require(not name.startswith('/') and '\\' not in name and ':' not in name and '\x00' not in name
                and all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe ZIP member: ' + name)
        require((info.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlinks are not allowed')
        match = re.fullmatch(r'(results/)?(test_[0-9]{3})(_inst\.png|_class\.json)', name)
        require(match is not None and match.group(2) in SOURCES, 'Unexpected result ZIP member: ' + name)
        prefix, source, suffix = match.groups()
        key = source + suffix
        require(key not in members, 'Duplicate result ZIP member: ' + key)
        require(info.file_size <= 64 * 1024 * 1024, 'Oversized result ZIP member')
        members[key] = name
        prefixes.add(prefix or '')
    expected = {source + suffix for source in SOURCES for suffix in ('_inst.png', '_class.json')}
    require(set(members) == expected and len(prefixes) == 1, 'ZIP must contain exactly 150 pairs under a single result root')
    require(archive.testzip() is None, 'ZIP CRC failure')
    return members


def decode_prediction(png, class_bytes, shape, source):
    require(len(png) >= 33 and png[:8] == b'\x89PNG\r\n\x1a\n' and png[8:16] == b'\x00\x00\x00\rIHDR'
            and png[24:26] == bytes((16, 0)), source + ': expected 16-bit single-channel PNG')
    width, height = struct.unpack('>II', png[16:24])
    require([height, width] == shape, source + ': PNG original dimensions differ')
    ids = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    require(ids is not None and ids.ndim == 2 and ids.dtype == np.uint16 and list(ids.shape) == shape,
            source + ': decoded uint16/native shape contract failed')
    classes = strict_json(class_bytes.decode('utf8'))
    require(isinstance(classes, dict), source + ': expected class JSON object')
    require(all(isinstance(k, str) and k.isascii() and k.isdecimal() and str(int(k)) == k
                and 1 <= int(k) <= 65535 for k in classes), source + ': invalid class key')
    require(all(type(v) is int and v in (0, 1) for v in classes.values()), source + ': class must be integer 0/1')
    require(set(classes) == {str(int(i)) for i in np.unique(ids) if i}, source + ': PNG/JSON IDs differ')
    return ids, classes


def _stats(ids, classes):
    sizes = np.bincount(ids.ravel(), minlength=65536)
    result = {'instances': len(classes), 'ID0_pixels': int(sizes[0])}
    for name, value in (('F', 1), ('P', 0)):
        areas = [int(sizes[int(key)]) for key, phase in classes.items() if phase == value]
        result[name] = dict(instances=len(areas), pixels=sum(areas), mean_area=sum(areas) / len(areas) if areas else None)
    return result


def _upstream_config(config):
    """缓存只允许改变末端面积声明；不得冒称执行了新的模型或阈值。"""
    result = copy.deepcopy(config)
    infer = result.get('inference', {})
    for key in ('seed_min_area', 'final_min_instance_area', 'final_small_instance_action'):
        infer.pop(key, None)
    result.get('paths', {}).pop('project_root', None)
    return result


def package_archive(input_zip, source_report, source_config, source_manifest, output_zip, audit_dir, *,
                    candidate_config=None, final_min_instance_area=None, seed_min_area=None,
                    expected_input_sha256=None):
    started = time.perf_counter()
    input_zip, output_zip, audit_dir = Path(input_zip).resolve(), Path(output_zip).resolve(), Path(audit_dir).resolve()
    require(input_zip != output_zip, 'Cannot overwrite input ZIP')
    temporary = output_zip.with_name(output_zip.name + '.partial')
    require(not output_zip.exists() and not temporary.exists() and not audit_dir.exists(),
            'Refusing to overwrite existing output ZIP, partial ZIP or audit directory')
    require(output_zip not in (Path(source_report).resolve(), Path(source_config).resolve(), Path(source_manifest).resolve()),
            'Output must not overwrite source provenance')
    input_sha = sha(input_zip)
    if expected_input_sha256 is not None:
        require(input_sha == expected_input_sha256, 'Input ZIP differs from explicitly expected SHA256')
    report = strict_json(Path(source_report).read_text(encoding='utf8'))
    registry = validate_source_report(report, input_sha)
    manifest = strict_json(Path(source_manifest).read_text(encoding='utf8'))
    require(isinstance(manifest, dict), 'Source manifest must be an object')
    source_chain = config_chain(source_config, manifest)
    source_value = load_config(str(source_config))
    source_policy = resolve_instance_area_policy(source_value.get('inference', {}))
    if seed_min_area is not None:
        require(type(seed_min_area) is int and seed_min_area == source_policy['seed_min_area'],
                'Cached predictions cannot change source seed_min_area')
    if candidate_config is not None:
        candidate_chain = config_chain(candidate_config)
        candidate_value = load_config(str(candidate_config))
        candidate_policy = resolve_instance_area_policy(candidate_value.get('inference', {}))
        require(candidate_policy['final_small_instance_action'] == ACTION, 'Candidate must select neighbor_merge_v1')
        require(candidate_policy['seed_min_area'] == source_policy['seed_min_area'], 'Candidate changes cached source seed area')
        require(_upstream_config(source_value) == _upstream_config(candidate_value),
                'Candidate changes upstream settings; cached ZIP only supports the final area operation')
        require(final_min_instance_area is None or final_min_instance_area == candidate_policy['final_min_instance_area'],
                'Explicit final area differs from candidate config')
        final_min_instance_area = candidate_policy['final_min_instance_area']
    else:
        candidate_chain = None
    require(type(final_min_instance_area) is int and final_min_instance_area >= 0, 'Final area must be a nonnegative integer')
    # 已被原流程删除的实例不能从 ZIP 恢复；源末端阈值以下的试验不成立。
    require(final_min_instance_area == 0 or final_min_instance_area >= source_policy['final_min_instance_area'],
            'Final threshold below source cleanup cannot be evaluated from cached ZIP')
    audit_dir.mkdir(parents=True, exist_ok=False)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with zipfile.ZipFile(input_zip) as original, zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED) as output:
            members = archive_members(original)
            for source in SOURCES:
                tick = time.perf_counter()
                png = original.read(members[source + '_inst.png'])
                labels = original.read(members[source + '_class.json'])
                expected = registry[source]
                require(hashlib.sha256(png).hexdigest() == expected['png_sha256']
                        and hashlib.sha256(labels).hexdigest() == expected['classes_sha256'],
                        'Input member bytes differ from source report: ' + source)
                before, classes = decode_prediction(png, labels, expected['shape'], source)
                after, new_classes, audit = merge_final_instances(before, classes, final_min_instance_area)
                # 恒等时保留原 PNG/JSON 字节，避免把编码差异误报为结果差异。
                if np.array_equal(before, after) and classes == new_classes:
                    new_png, new_json = png, labels
                else:
                    ok, encoded = cv2.imencode('.png', after)
                    require(ok, 'PNG encode failed: ' + source)
                    new_png = encoded.tobytes()
                    new_json = (json.dumps(new_classes, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode('utf8')
                check_ids, check_classes = decode_prediction(new_png, new_json, expected['shape'], source)
                require(np.array_equal(check_ids, after) and check_classes == new_classes, 'Output serialization differs: ' + source)
                png_sha, json_sha = hashlib.sha256(new_png).hexdigest(), hashlib.sha256(new_json).hexdigest()
                output.writestr('results/' + source + '_inst.png', new_png)
                output.writestr('results/' + source + '_class.json', new_json)
                audit.update(source=source, shape=expected['shape'], before=_stats(before, classes), after=_stats(after, new_classes),
                             png_sha256=png_sha, classes_sha256=json_sha, seconds=time.perf_counter() - tick)
                write_json(audit_dir / (source + '.json'), audit)
                rows.append(audit)
        with zipfile.ZipFile(temporary) as completed:
            members = archive_members(completed)
            require(all(name.startswith('results/') for name in members.values()), 'Output must be results-only')
            for row in rows:
                source = row['source']
                png, labels = completed.read(members[source + '_inst.png']), completed.read(members[source + '_class.json'])
                require(hashlib.sha256(png).hexdigest() == row['png_sha256']
                        and hashlib.sha256(labels).hexdigest() == row['classes_sha256'], 'Output member SHA mismatch')
                decode_prediction(png, labels, row['shape'], source)
        require(sha(input_zip) == input_sha, 'Input ZIP changed during candidate generation')
        # link 原子创建全新文件名；Linux rename 会覆盖目标，不能用于此处。
        os.link(temporary, output_zip)
        temporary.unlink()
        seconds = time.perf_counter() - started
        receipt = dict(format='final_area_cached_package_v1', complete=True, cpu_only=True, model_forward_performed=False,
            source_archive_sha256=input_sha, archive_sha256=sha(output_zip), archive_bytes=output_zip.stat().st_size,
            source_report_sha256=sha(source_report), source_manifest_sha256=sha(source_manifest),
            cleanup_source_sha256=sha(ROOT / 'utils' / 'final_instance_cleanup.py'),
            package_tool_sha256=sha(Path(__file__)),
            source_config_chain=source_chain, source_area_policy=source_policy, candidate_config_chain=candidate_chain,
            seed_min_area=source_policy['seed_min_area'], final_min_instance_area=final_min_instance_area,
            final_small_instance_action=ACTION, cached_seed_changed=False, source_input_unchanged=True,
            semantic_policy='target class inherited; no raw probability revote', dataset='semifinal_150',
            pairs=150, files=300, only_results=True, no_audit_json_in_zip=True, CRC_all_passed=True,
            native_sizes_uint16_and_ID_class_pairs_exact=True, member_sha256_exact=True,
            merged_instances=sum(row['merged_count'] for row in rows), changed_pixels=sum(row['changed_pixels'] for row in rows),
            total_seconds=seconds, mean_image_seconds=sum(row['seconds'] for row in rows) / len(rows),
            model_state_sha256=report.get('model_state_sha256'), checkpoint_file_sha256=report.get('checkpoint_file_sha256'),
            data_manifest_sha256=report.get('data_manifest_sha256'), images=rows)
        write_json(audit_dir / 'complete.json', receipt)
        return receipt
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        write_json(audit_dir / 'failed.json', dict(complete=False, error_type=type(error).__name__, error=str(error),
                                                  source_archive_sha256=input_sha))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-zip', 'source-report', 'source-config', 'source-manifest', 'output-zip', 'audit-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--candidate-config')
    parser.add_argument('--final-min-instance-area', type=int)
    parser.add_argument('--seed-min-area', type=int, help='Must equal the verified cached source; does not rerun seeds')
    parser.add_argument('--expected-input-sha256')
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    result = package_archive(**vars(args))
    print(json.dumps({key: result[key] for key in ('complete', 'archive_sha256', 'pairs', 'seed_min_area',
          'final_min_instance_area', 'merged_instances', 'total_seconds', 'mean_image_seconds')}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
