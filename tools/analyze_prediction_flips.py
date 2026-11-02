# -*- coding: utf-8 -*-
"""比较两组最终实例预测并原字节组包；不读取原图、模型或测试GT。

F=1、P=0；实例ID0另记背景。变化率不是准确率或独立语义头效果。
纯输出算术沿用global_zero/analyze.py；90%对应之外的实例另列，避免隐藏吸收。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import zipfile

import cv2
import numpy as np

MAX_PIXELS = 6_000_000
MAX_PNG_BYTES = 64 * 2**20
MAX_JSON_BYTES = 4 * 2**20
NAME = re.compile(r'(test_[0-9]{3,4})(_inst\.png|_class\.json)\Z')
DETAIL_LIMIT = 64


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate class JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def decode_prediction(png, class_bytes):
    if not (33 <= len(png) <= MAX_PNG_BYTES and png[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'):
        raise ValueError('Expected a bounded PNG with a standard IHDR')
    width, height = struct.unpack('>II', png[16:24])
    if height < 1 or width < 1 or height * width > MAX_PIXELS or png[24:26] != bytes((16, 0)):
        raise ValueError('Expected native single-channel uint16 PNG within the pixel bound')
    ids = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if ids is None or ids.dtype != np.uint16 or ids.ndim != 2 or ids.shape != (height, width):
        raise ValueError('Decoded prediction must be single-channel uint16 at its declared shape')
    if not 0 < len(class_bytes) <= MAX_JSON_BYTES:
        raise ValueError('Class JSON exceeds its size bound')
    classes = json.loads(class_bytes.decode('utf-8'), object_pairs_hook=unique_object,
                         parse_constant=_invalid_constant)
    if not isinstance(classes, dict) or any(
        not isinstance(key, str) or not key.isascii() or re.fullmatch(r'[1-9][0-9]{0,4}', key) is None
        or int(key) > 65535 or type(value) is not int or value not in (0, 1)
        for key, value in classes.items()
    ):
        raise ValueError('Classes require canonical positive IDs <=65535 and integer P0/F1')
    if set(classes) != {str(int(value)) for value in np.unique(ids) if value}:
        raise ValueError('Class keys must match every positive instance ID exactly')
    return ids, classes


class PredictionStore:
    """ZIP或results目录，只逐图读取；目录外的audit不进入结果集合。"""
    def __init__(self, path, expected_count=150, expected_sha256=None):
        if type(expected_count) is not int or expected_count < 1:
            raise ValueError('expected_count must be a positive integer')
        self.path = Path(path).resolve()
        self.expected_count, self.zip = expected_count, None
        self.member_sha256 = {}
        self.archive_sha256 = None
        if self.path.is_file():
            self.archive_sha256 = file_sha(self.path)
            if expected_sha256 is not None and self.archive_sha256 != expected_sha256.lower():
                raise ValueError('Baseline archive SHA256 differs')
            self.zip = zipfile.ZipFile(self.path)
            try:
                infos = self.zip.infolist()
                if len(infos) != 2 * expected_count or len({i.filename for i in infos}) != len(infos):
                    raise ValueError('ZIP needs exactly two unique members per expected source')
                files = {}
                for info in infos:
                    if (not info.filename.startswith('results/') or info.is_dir() or info.flag_bits & 1
                            or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)):
                        raise ValueError('ZIP permits only regular unencrypted results files')
                    name = info.filename.removeprefix('results/')
                    match = NAME.fullmatch(name)
                    limit = MAX_PNG_BYTES if name.endswith('_inst.png') else MAX_JSON_BYTES
                    if match is None or not 0 < info.file_size <= limit:
                        raise ValueError('Unexpected result name or uncompressed member size')
                    files[name] = info.filename
                self.files = files
            except BaseException:
                self.zip.close()
                raise
        elif self.path.is_dir():
            if expected_sha256 is not None:
                raise ValueError('Archive SHA256 can only bind a ZIP baseline')
            self.path = (self.path / 'results') if (self.path / 'results').is_dir() else self.path
            self.files = self._directory_files()
        else:
            raise FileNotFoundError(self.path)
        self.names = sorted({NAME.fullmatch(name).group(1) for name in self.files})
        required = {name + suffix for name in self.names for suffix in ('_inst.png', '_class.json')}
        if len(self.names) != expected_count or set(self.files) != required:
            self.close()
            raise ValueError('Result sources must have exactly one PNG and one class JSON')

    def _directory_files(self):
        files = {}
        for path in self.path.iterdir():
            if not path.is_file() or path.is_symlink() or NAME.fullmatch(path.name) is None:
                raise ValueError('Results directory contains an extra, nested or symlink entry')
            limit = MAX_PNG_BYTES if path.name.endswith('_inst.png') else MAX_JSON_BYTES
            if not 0 < path.stat().st_size <= limit:
                raise ValueError('Prediction file exceeds its size bound')
            files[path.name] = path
        return files

    def read(self, name):
        try:
            data = self.zip.read(self.files[name]) if self.zip is not None else self.files[name].read_bytes()
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError('Prediction read/CRC verification failed: ' + name) from exc
        key, digest = 'results/' + name, hashlib.sha256(data).hexdigest()
        if key in self.member_sha256 and self.member_sha256[key] != digest:
            raise RuntimeError('Prediction changed during analysis: ' + name)
        self.member_sha256[key] = digest
        return data

    def prediction(self, source):
        png, classes = self.read(source + '_inst.png'), self.read(source + '_class.json')
        return (*decode_prediction(png, classes), png, classes)

    def verify_unchanged(self):
        if len(self.member_sha256) != 2 * self.expected_count:
            raise RuntimeError('Not every result member has been validated')
        if self.zip is not None:
            if file_sha(self.path) != self.archive_sha256:
                raise RuntimeError('Prediction archive changed during analysis')
        else:
            if set(self._directory_files()) != set(self.files):
                raise RuntimeError('Prediction directory inventory changed')
            for name, path in self.files.items():
                if file_sha(path) != self.member_sha256['results/' + name]:
                    raise RuntimeError('Prediction file changed during analysis: ' + name)

    def identity(self):
        return dict(path=str(self.path), archive_sha256=self.archive_sha256, sources=self.expected_count,
                    member_manifest_sha256=canonical_sha(self.member_sha256),
                    member_sha256=dict(sorted(self.member_sha256.items())))

    def close(self):
        if self.zip is not None:
            self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def phase(ids, classes):
    lookup = np.full(65536, 2, np.uint8)
    for key, value in classes.items():
        lookup[int(key)] = value
    return lookup[ids]


def rates(counts):
    keys = ('F_to_P_pixels', 'P_to_F_pixels', 'any_FP_flip_pixels')
    return dict(all_pixels={key: divide(counts[key], counts['total_pixels']) for key in keys},
        common_foreground={key: divide(counts[key], counts['common_foreground_pixels']) for key in keys},
        conditional_on_old_phase=dict(F_to_P=divide(counts['F_to_P_pixels'], counts['old_F_pixels']),
            P_to_F=divide(counts['P_to_F_pixels'], counts['old_P_pixels'])),
        background=dict(positive_to_zero_on_all=divide(counts['positive_to_zero_pixels'], counts['total_pixels']),
            zero_to_positive_on_all=divide(counts['zero_to_positive_pixels'], counts['total_pixels']),
            positive_to_zero_on_old_foreground=divide(counts['positive_to_zero_pixels'], counts['old_foreground_pixels']),
            zero_to_positive_on_new_foreground=divide(counts['zero_to_positive_pixels'], counts['new_foreground_pixels'])))


def flip_metrics(old, oldclasses, new, newclasses):
    a, b = phase(old, oldclasses), phase(new, newclasses)
    fp, pf = (a == 1) & (b == 0), (a == 0) & (b == 1)
    counts = dict(total_pixels=int(a.size), common_foreground_pixels=int(((a < 2) & (b < 2)).sum()),
        old_foreground_pixels=int((a < 2).sum()), new_foreground_pixels=int((b < 2).sum()),
        old_F_pixels=int((a == 1).sum()), old_P_pixels=int((a == 0).sum()),
        new_F_pixels=int((b == 1).sum()), new_P_pixels=int((b == 0).sum()),
        F_to_P_pixels=int(fp.sum()), P_to_F_pixels=int(pf.sum()), any_FP_flip_pixels=int(fp.sum() + pf.sum()),
        positive_to_zero_pixels=int(((a < 2) & (b == 2)).sum()),
        zero_to_positive_pixels=int(((a == 2) & (b < 2)).sum()))
    return dict(counts=counts, rates=rates(counts))


def boundaries(ids):
    result = np.zeros(ids.shape, bool)
    x = (ids[:, :-1] != ids[:, 1:]) & ((ids[:, :-1] > 0) | (ids[:, 1:] > 0))
    y = (ids[:-1] != ids[1:]) & ((ids[:-1] > 0) | (ids[1:] > 0))
    result[:, :-1] |= x
    result[:, 1:] |= x
    result[:-1] |= y
    result[1:] |= y
    return result


def geometry(old, oldclasses, new, newclasses):
    a, b = boundaries(old), boundaries(new)
    xor, union = int((a ^ b).sum()), int((a | b).sum())
    anear = cv2.dilate(a.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    bnear = cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    result = dict(old_boundary_pixels=int(a.sum()), new_boundary_pixels=int(b.sum()),
        boundary_symmetric_difference_pixels=xor, boundary_union_pixels=union,
        boundary_xor_on_union=divide(xor, union), boundary_xor_on_all_pixels=xor / old.size,
        old_boundary_retained_r1=divide(int((a & bnear).sum()), int(a.sum())),
        new_boundary_with_old_r1=divide(int((b & anear).sum()), int(b.sum())))
    del a, b, anear, bnear
    # 只编码实际像素对并unique；绝不分配65536×65536的稠密矩阵。
    encoded = old.astype(np.uint32)
    encoded *= np.uint32(65536)
    encoded += new
    pairs, amounts = np.unique(encoded, return_counts=True)
    del encoded
    osizes = np.bincount(old.ravel(), minlength=65536)
    nsizes = np.bincount(new.ravel(), minlength=65536)
    forward, reverse, splits, merges = (defaultdict(list) for _ in range(4))
    stable = []
    for pair, amount in zip(pairs, amounts):
        left, right, amount = int(pair) // 65536, int(pair) % 65536, int(amount)
        forward[left].append((right, amount))
        reverse[right].append((left, amount))
        if left > 0 and right > 0:
            if amount >= max(32, .05 * osizes[left]):
                splits[left].append(right)
            if amount >= max(32, .05 * nsizes[right]):
                merges[right].append(left)
            if amount >= .9 * osizes[left] and amount >= .9 * nsizes[right]:
                stable.append(dict(old_id=left, new_id=right, overlap_pixels=amount,
                    old_coverage=amount / int(osizes[left]), new_coverage=amount / int(nsizes[right]),
                    old_class=oldclasses[str(left)], new_class=newclasses[str(right)]))
    fp = sum(row['old_class'] == 1 and row['new_class'] == 0 for row in stable)
    pf = sum(row['old_class'] == 0 and row['new_class'] == 1 for row in stable)
    result.update(partition_equivalent_up_to_relabeling=bool(np.array_equal(old == 0, new == 0)
        and all(len(forward[int(k)]) == 1 for k in oldclasses)
        and all(len(reverse[int(k)]) == 1 for k in newclasses)),
        spatial_transition_definition='overlap>=32px AND >=5% of respective region; not an error rate',
        split_old_regions=sum(len(v) > 1 for v in splits.values()),
        merged_new_regions=sum(len(v) > 1 for v in merges.values()),
        stable_correspondences=dict(definition='one overlap covers >=90% of both regions', count=len(stable),
            F_to_P=fp, P_to_F=pf, flip_rate=divide(fp + pf, len(stable)), pairs=stable))

    def excluded(classes, destination_classes, sizes, destination_sizes, overlaps, included):
        rows = []
        for key in classes:
            label = int(key)
            if label in included:
                continue
            destination, amount = max(overlaps[label], key=lambda v: (v[1], -v[0]))
            coverage = amount / int(sizes[label])
            dest_coverage = amount / int(destination_sizes[destination]) if destination else None
            rows.append(dict(id=label, area=int(sizes[label]), phase=classes[key], dominant_id=destination,
                dominant_phase=destination_classes[str(destination)] if destination else None,
                dominant_coverage=coverage, dominant_destination_coverage=dest_coverage,
                absorbed_into_larger_region=bool(destination and coverage >= .9 and dest_coverage < .9),
                dominant_phase_changed=bool(destination and classes[key] != destination_classes[str(destination)])))
        return dict(count=len(rows), absorbed_into_larger_region=sum(r['absorbed_into_larger_region'] for r in rows),
                    dominant_phase_changed=sum(r['dominant_phase_changed'] for r in rows),
                    details=rows[:DETAIL_LIMIT], detail_limit=DETAIL_LIMIT, details_truncated=len(rows) > DETAIL_LIMIT)

    result['excluded_correspondences'] = dict(
        old=excluded(oldclasses, newclasses, osizes, nsizes, forward, {r['old_id'] for r in stable}),
        new=excluded(newclasses, oldclasses, nsizes, osizes, reverse, {r['new_id'] for r in stable}),
        interpretation='Outside the 90% stable subset; absorption/dominant-phase changes are not truth errors')
    return result


def composition(ids, classes):
    sizes = np.bincount(ids.ravel(), minlength=65536)
    result = dict(total_instances=len(classes))
    for value, label in ((0, 'P'), (1, 'F')):
        areas = sizes[np.array([int(k) for k, v in classes.items() if v == value], np.int64)]
        result[label] = dict(instances=int(areas.size), pixels=int(areas.sum()))
    return result


def compare_predictions(old, oldclasses, new, newclasses):
    if old.dtype != np.uint16 or new.dtype != np.uint16 or old.ndim != 2 or old.shape != new.shape:
        raise ValueError('Compared predictions must share native shape and uint16 dtype')
    if not 0 < old.size <= MAX_PIXELS:
        raise ValueError('Prediction exceeds native pixel bound')
    return dict(flip=flip_metrics(old, oldclasses, new, newclasses), geometry=geometry(old, oldclasses, new, newclasses),
                baseline=composition(old, oldclasses), candidate=composition(new, newclasses))


def aggregate(rows):
    counts = {key: sum(row['flip']['counts'][key] for row in rows) for key in rows[0]['flip']['counts']}
    macro, macro_counts = {}, {}
    for group, fields in rows[0]['flip']['rates'].items():
        macro[group], macro_counts[group] = {}, {}
        for key in fields:
            values = [row['flip']['rates'][group][key] for row in rows if row['flip']['rates'][group][key] is not None]
            macro[group][key] = sum(values) / len(values) if values else None
            macro_counts[group][key] = len(values)
    stable = {key: sum(row['geometry']['stable_correspondences'][key] for row in rows)
              for key in ('count', 'F_to_P', 'P_to_F')}
    stable['flip_rate'] = divide(stable['F_to_P'] + stable['P_to_F'], stable['count'])
    geometry_counts = {key: sum(row['geometry'][key] for row in rows) for key in (
        'split_old_regions', 'merged_new_regions', 'boundary_symmetric_difference_pixels', 'boundary_union_pixels')}
    geometry_counts['boundary_xor_on_union'] = divide(geometry_counts['boundary_symmetric_difference_pixels'],
                                                     geometry_counts['boundary_union_pixels'])
    geometry_counts['boundary_xor_on_all_pixels'] = divide(geometry_counts['boundary_symmetric_difference_pixels'], counts['total_pixels'])
    return dict(counts=counts, pixel_weighted_rates=rates(counts), macro_image_rates=macro,
        macro_rate_valid_images=macro_counts, stable_correspondences=stable, geometry=geometry_counts,
        excluded_correspondences={side: {key: sum(row['geometry']['excluded_correspondences'][side][key] for row in rows)
            for key in ('count', 'absorbed_into_larger_region', 'dominant_phase_changed')} for side in ('old', 'new')},
        composition={side: dict(total_instances=sum(row[side]['total_instances'] for row in rows),
            **{label: {key: sum(row[side][label][key] for row in rows) for key in ('instances', 'pixels')}
               for label in ('F', 'P')}) for side in ('baseline', 'candidate')})


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def analyze_predictions(baseline, candidate, output_dir, expected_count=150, baseline_sha256=None):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Analysis output directory must be fresh or empty')
    with PredictionStore(baseline, expected_count, baseline_sha256) as old, PredictionStore(candidate, expected_count) as new:
        if old.names != new.names:
            raise ValueError('Baseline and candidate source sets differ')
        # 避免输出写入输入results目录、改变预测集合。
        for store in (old, new):
            if store.zip is None and output.is_relative_to(store.path):
                raise ValueError('Analysis output must lie outside prediction results')
        output.mkdir(parents=True, exist_ok=True)
        folder = output / 'per_source'
        folder.mkdir()
        rows, flat = [], []
        for source in old.names:
            a, ac, _, _ = old.prediction(source)
            b, bc, _, _ = new.prediction(source)
            row = dict(source=source, shape=list(a.shape), **compare_predictions(a, ac, b, bc))
            row['baseline_member_sha256'] = {suffix: old.member_sha256['results/' + source + suffix]
                                            for suffix in ('_inst.png', '_class.json')}
            row['candidate_member_sha256'] = {suffix: new.member_sha256['results/' + source + suffix]
                                             for suffix in ('_inst.png', '_class.json')}
            _write_json(folder / (source + '.json'), row)
            rows.append(row)
            stable = row['geometry']['stable_correspondences']
            flat.append(dict(source=source, height=a.shape[0], width=a.shape[1], **row['flip']['counts'],
                FP_flip_on_all=row['flip']['rates']['all_pixels']['any_FP_flip_pixels'],
                FP_flip_on_common_foreground=row['flip']['rates']['common_foreground']['any_FP_flip_pixels'],
                stable_instances=stable['count'], stable_F_to_P=stable['F_to_P'], stable_P_to_F=stable['P_to_F'],
                old_excluded=row['geometry']['excluded_correspondences']['old']['count'],
                new_excluded=row['geometry']['excluded_correspondences']['new']['count'],
                split_old_regions=row['geometry']['split_old_regions'], merged_new_regions=row['geometry']['merged_new_regions']))
        old.verify_unchanged()
        new.verify_unchanged()
        with (output / 'per_source.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        summary = dict(format='prediction_flips_v1', complete=True, source_count=len(rows),
            baseline=old.identity(), candidate=new.identity(), aggregate=aggregate(rows),
            source_names=old.names, no_GT=True, no_model_forward=True, predictions_unchanged=True,
            interpretation='Prediction changes from geometry and class voting together; not accuracy, official mIoU, or isolated semantic-head effects.',
            conventions=dict(F=1, P=0, background='instance ID0; excluded from F/P flips and reported separately',
                phase_flip='F-to-P or P-to-F on common positive foreground; all-pixel and common-foreground denominators both reported',
                stable_instances='>=90% mutual overlap, not necessarily identical geometry; excluded regions reported separately',
                macro='mean of per-image rates; zero-denominator images omitted with counts recorded',
                input_identity='same source-name set and native prediction shapes; raw-input SHA binding belongs inference reports',
                geometry='split/merge and boundary changes are diagnostics, never error rates'))
        _write_json(output / 'summary.json', summary)
        return summary


def pack_results(results, output_zip, expected_count=150):
    """只复制最终PNG/JSON原字节；不复票、不重新编号、不运行面积合并。"""
    target = Path(output_zip).resolve()
    if target.exists():
        raise FileExistsError('Will not overwrite an existing result archive')
    temporary = target.with_name(target.name + '.tmp')
    if temporary.exists():
        raise FileExistsError('Preserve existing temporary archive')
    with PredictionStore(results, expected_count) as source:
        if source.zip is None and target.is_relative_to(source.path):
            raise ValueError('Archive output must lie outside prediction results')
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in source.names:
                _, _, png, classes = source.prediction(name)
                for suffix, data in (('_inst.png', png), ('_class.json', classes)):
                    info = zipfile.ZipInfo('results/' + name + suffix, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    archive.writestr(info, data)
        source.verify_unchanged()
        with PredictionStore(temporary, expected_count) as verified:
            for name in verified.names:
                verified.prediction(name)
            verified.verify_unchanged()
            if verified.member_sha256 != source.member_sha256:
                raise RuntimeError('Packed result member bytes differ from source')
            receipt = dict(format='prediction_result_package_v1', complete=True, source_count=expected_count,
                members=2 * expected_count, only_results=True, CRC_all_passed=True,
                native_uint16_ID_class_contract=True, member_manifest_sha256=canonical_sha(verified.member_sha256),
                member_sha256=dict(sorted(verified.member_sha256.items())))
        temporary.replace(target)
    return dict(receipt, path=str(target), bytes=target.stat().st_size, sha256=file_sha(target),
                no_model_forward=True, no_GT=True, final_outputs_copied_without_postprocessing=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline')
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-count', type=int, default=150)
    parser.add_argument('--baseline-sha256')
    parser.add_argument('--pack-zip')
    args = parser.parse_args(argv)
    if args.baseline is None and (args.pack_zip is None or args.baseline_sha256 is not None):
        parser.error('Provide --baseline for comparison, or --pack-zip for packaging only')
    output = Path(args.output_dir).resolve()
    summary = None
    if args.baseline is not None:
        summary = analyze_predictions(args.baseline, args.candidate, output, args.expected_count, args.baseline_sha256)
    if args.pack_zip is not None:
        if summary is None:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('Package audit output must be fresh or empty')
            with PredictionStore(args.candidate, args.expected_count) as source:
                if source.zip is None and output.is_relative_to(source.path):
                    raise ValueError('Package audit output must lie outside prediction results')
            output.mkdir(parents=True, exist_ok=True)
        receipt = pack_results(args.candidate, args.pack_zip, args.expected_count)
        _write_json(output / 'package_audit.json', receipt)
    print(json.dumps(dict(complete=True, output_dir=str(output), source_count=args.expected_count,
                         compared=summary is not None, packaged=args.pack_zip is not None), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
