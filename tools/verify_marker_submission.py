# -*- coding: utf-8 -*-
"""CPU只读验证marker候选ZIP与完整推理报告；不推理、不打包、不上传。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile

import cv2
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def strict_json(payload):
    return json.loads(payload, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def report_rows(report):
    """名单仅取自真实完整报告，禁止另造比赛图名或按ZIP补齐缺项。"""
    flags = ('complete', 'full100_complete', 'all_baselines_exact', 'model_unchanged',
             'restorer_unchanged', 'no_optimizer_steps', 'no_seams')
    require(isinstance(report, dict), 'Report must be a JSON object')
    require(all(report.get(key) is True for key in flags), 'Incomplete or unfrozen marker run report')
    require(report.get('format') == 'marker_reflect_run_v1'
            and report.get('reflection_padding') == 32
            and type(report.get('reflection_padding')) is int, 'Expected fixed reflect32 run format')
    require(report.get('training') is False and report.get('best_deployment_changed') is False,
            'Report must describe a frozen inference candidate')
    require(type(report.get('parameter_count')) is int and 0 < report['parameter_count'] < 500_000_000,
            'Invalid competition model parameter count')
    require(report.get('complete_source_count') == 100
            and report.get('selected_full_indices') == list(range(100)), 'Incomplete full cohort indices')
    rows = report.get('images')
    require(isinstance(rows, list) and len(rows) == 100, 'Exactly100 report rows required')
    names = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), 'Invalid report row')
        source = row.get('source')
        require(isinstance(source, str) and bool(source) and source not in ('.', '..')
                and not any(c in source for c in '/\\\x00:'), 'Invalid flat source name')
        require(source not in names, 'Duplicate source in report: ' + source)
        names.add(source)
        require(row.get('full_source_index') == index and type(row.get('full_source_index')) is int
                and row.get('view') == 'patch1024', 'Report full-cohort order or view differs')
        require(all(row.get(key) is True for key in ('baseline_final_ids_exact',
                'baseline_rawvote_classes_exact', 'baseline_phases_blend_exact')), 'Per-source baseline gate failed')
        replay = row.get('replay', {})
        require(isinstance(replay, dict) and all(replay.get(key) is True for key in
                ('original_marker_exact', 'original_watershed_exact', 'original_elevation_exact',
                 'original_rawvote_exact')), 'Per-source actual marker/elevation/rawvote gate failed')
        shape = row.get('shape')
        require(isinstance(shape, list) and len(shape) == 2
                and all(type(v) is int and v > 0 for v in shape), 'Invalid original HW shape')
        output = row.get('output', {})
        require(isinstance(output, dict) and all(isinstance(output.get(key), str)
                and re.fullmatch('[0-9a-f]{64}', output[key]) for key in
                ('png_sha256', 'classes_sha256')), 'Invalid per-source output SHA256')
    return rows


def validate_prediction(png, json_bytes, shape, source):
    require(len(png) >= 33 and png[:8] == b'\x89PNG\r\n\x1a\n'
            and png[8:12] == b'\x00\x00\x00\r' and png[12:16] == b'IHDR'
            and png[24:26] == bytes((16, 0)), source + ': expected16-bit grayscale PNG IHDR')
    width, height = struct.unpack('>II', png[16:24])
    require([height, width] == shape, source + ': PNG IHDR differs from original shape')
    ids = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    require(ids is not None and ids.ndim == 2 and ids.dtype == np.uint16
            and list(ids.shape) == shape and int(ids.max()) <= 65535,
            source + ': invalid decoded uint16 native instance image')
    classes = strict_json(json_bytes.decode('utf8'))
    require(isinstance(classes, dict), source + ': class JSON must be an object')
    require(all(isinstance(key, str) and key.isascii() and key.isdecimal()
                and str(int(key)) == key and 1 <= int(key) <= 65535 for key in classes),
            source + ': class IDs must be canonical positive integer keys')
    require(all(type(value) is int and value in (0, 1) for value in classes.values()),
            source + ': classes must be integer0/1')
    positive = np.unique(ids[ids > 0])
    require(set(classes) == {str(int(value)) for value in positive},
            source + ': positive instance IDs and class keys differ')
    return dict(shape=shape, dtype='uint16', maximum_id=int(ids.max()), instances=len(positive),
                ferrite_instances=sum(c == 1 for c in classes.values()),
                pearlite_instances=sum(c == 0 for c in classes.values()))


def verify_archive(archive_path, report_path):
    archive_path, report_path = Path(archive_path), Path(report_path)
    report = strict_json(report_path.read_text(encoding='utf8')); rows = report_rows(report)
    expected = {r['source'] + suffix for r in rows for suffix in ('_inst.png', '_class.json')}
    checked = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == 200 and len(set(names)) == 200 and set(names) == expected,
                'ZIP must contain exactly200 unique flat report-matching members')
        require(all('/' not in name and '\\' not in name and not entry.is_dir()
                    for name, entry in zip(names, archive.infolist())), 'ZIP members must be flat files')
        require(archive.testzip() is None, 'ZIP CRC failed')
        for row in rows:
            source = row['source']; png = archive.read(source + '_inst.png')
            labels = archive.read(source + '_class.json')
            require(hashlib.sha256(png).hexdigest() == row['output']['png_sha256']
                    and hashlib.sha256(labels).hexdigest() == row['output']['classes_sha256'],
                    'ZIP member bytes differ from candidate report: ' + source)
            checked.append(dict(source=source, **validate_prediction(png, labels, row['shape'], source),
                                **row['output']))
    return dict(format='marker_submission_verification_v1', passed=True, cpu_only=True,
        inference_performed=False, package_created=False, uploaded=False, images=100, files=200,
        flat=True, unique_members=True, crc_ok=True, report_flags_exact=True,
        member_sha256_exact=True, uint16_native_shapes_exact=True, instance_class_ids_exact=True,
        archive_sha256=sha(archive_path), report_sha256=sha(report_path),
        checkpoint_sha256=report.get('checkpoint_sha256'),
        model_state_sha256=report.get('model_state_sha256'),
        restorer_state_sha256=report.get('restorer_state_sha256'), checked=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(output not in {Path(args.archive).resolve(), Path(args.report).resolve()},
            'Verification receipt must not overwrite archive or report')
    receipt = verify_archive(args.archive, args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print('MARKER_SUBMISSION_VERIFIED', receipt['images'], receipt['archive_sha256'], flush=True)


if __name__ == '__main__':
    main()
