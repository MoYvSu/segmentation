# -*- coding: utf-8 -*-
"""训练源RGB扰动后的NAF输入诊断；仅CPU统计/缩略图，不训练、不读测试GT。

先整图T8标准化再比较固定GT集合。计数是监督覆盖诊断，不是分割准确率。
CLI允许预检源子集；输入与输出均应位于私有忽略目录。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.boundary_naf import load_processed_binary
from utils.boundary_input import normalize_boundary_input
from utils.image_io import read_image

FORMAT = 'boundary_naf_perturbed_heatmaps_v1'
NORMALIZATION = {'mode': 'skeleton_dilate', 'kernel_size': 8}


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _binary(value, name, shape=None):
    value = np.asarray(value)
    if (value.ndim != 2 or not value.size or value.dtype.kind not in 'bifu'
            or not np.isfinite(value).all() or not np.isin(value, (0, 1)).all()
            or (shape is not None and value.shape != shape)):
        raise ValueError(name+' must be aligned nonempty binary HW')
    return value.astype(bool, copy=False)


def _probability(value, name, shape=None):
    value = np.asarray(value)
    if (value.ndim != 2 or not value.size or value.dtype.kind not in 'bifu'
            or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
            or (shape is not None and value.shape != shape)):
        raise ValueError(name+' must be aligned finite HW in [0,1]')
    return np.ascontiguousarray(value, dtype=np.float32)


def _mask_count(mask, region):
    return int(np.count_nonzero(mask & region))


def _width_summary(binary):
    from skimage.morphology import skeletonize
    skeleton = skeletonize(binary)
    # 显式图框0，使全前景输入也有有限距离；此处不复用历史剔叉口口径。
    distance = cv2.distanceTransform(np.pad(binary.astype(np.uint8), 1),
                                    cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    values = 2 * distance[skeleton]
    return dict(skeleton_points=int(values.size),
                median=float(np.median(values)) if values.size else None,
                p90=float(np.quantile(values, .9)) if values.size else None,
                over16_points=int(np.count_nonzero(values > 16)),
                over16_fraction=float(np.mean(values > 16)) if values.size else None,
                definition='2*EDT at all whole-source skeleton points; includes junction/frame, not physical width',
                historical_filtered_width_comparable=False)


def analyze_pair(clean_boundary, variant_boundary, target, valid, original, filled,
                 *, tolerance=2, orientation_radius=3, return_masks=False):
    """比较整图clean/variant，返回JSON摘要；可额外返回同坐标诊断mask。

    target/valid应由现有load_processed_binary生成，不能把输入热图用于改GT。
    positive/negative与build_tolerant_targets同口径；original/filled只分栏统计。
    """
    if type(tolerance) is not int or tolerance < 0 or type(orientation_radius) is not int or orientation_radius < 1:
        raise ValueError('Invalid native tolerance/orientation radius')
    clean = _probability(clean_boundary, 'clean boundary')
    variant = _probability(variant_boundary, 'variant boundary', clean.shape)
    target = _binary(target, 'target', clean.shape)
    valid = _binary(valid, 'valid', clean.shape)
    original = _binary(original, 'original', clean.shape)
    filled = _binary(filled, 'filled', clean.shape)
    unknown = ~(original | filled)
    if np.any(original & filled) or np.any(valid & unknown) or np.any(target & unknown):
        raise ValueError('GT provenance overlaps or unknown pixels enter supervision')
    clean_T, clean_receipt = normalize_boundary_input(clean, settings=NORMALIZATION)
    variant_T, variant_receipt = normalize_boundary_input(variant, settings=NORMALIZATION)
    radius = max(tolerance, orientation_radius)
    trusted = cv2.erode(valid.astype(np.uint8), np.ones((2*radius+1,)*2, np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    positive = target & trusted
    near_target = cv2.dilate(target.astype(np.uint8), np.ones((2*tolerance+1,)*2, np.uint8)) > 0
    negative = ~near_target & trusted
    clean_near = cv2.dilate(clean_T.astype(np.uint8), np.ones((2*tolerance+1,)*2, np.uint8)) > 0
    variant_near = cv2.dilate(variant_T.astype(np.uint8), np.ones((2*tolerance+1,)*2, np.uint8)) > 0
    masks = dict(positive=positive, negative=negative, trusted=trusted,
        natural_missing=positive & ~clean_near,
        new_missing=positive & clean_near & ~variant_near,
        recovered_old_missing=positive & ~clean_near & variant_near,
        unchanged_supported=positive & clean_near & variant_near,
        still_missing=positive & ~clean_near & ~variant_near,
        actual_variant_missing=positive & ~variant_near,
        clean_interior_input=negative & clean_T,
        variant_interior_input=negative & variant_T,
        original=original, filled=filled, unknown=unknown,
        clean_T8=clean_T, variant_T8=variant_T)
    fields = ('positive', 'negative', 'trusted', 'natural_missing', 'new_missing',
              'recovered_old_missing', 'unchanged_supported', 'still_missing',
              'actual_variant_missing', 'clean_interior_input', 'variant_interior_input')
    by_provenance = {}
    for name, region in (('all', np.ones(clean.shape, bool)), ('original', original),
                         ('filled', filled), ('unknown', unknown)):
        counts = {key: _mask_count(masks[key], region) for key in fields}
        counts.update(pixels=int(region.sum()), valid_pixels=_mask_count(valid, region),
            clean_raw_boundary_pixels=_mask_count(clean >= .5, region),
            variant_raw_boundary_pixels=_mask_count(variant >= .5, region),
            clean_T8_pixels=_mask_count(clean_T, region), variant_T8_pixels=_mask_count(variant_T, region))
        counts['new_missing_fraction_of_positive'] = counts['new_missing']/counts['positive'] if counts['positive'] else None
        by_provenance[name] = counts
    assert not by_provenance['unknown']['positive'] and not by_provenance['unknown']['negative']
    result = dict(format='naf_perturbation_pair_probe_v1', shape=list(clean.shape),
        normalization=dict(NORMALIZATION), clean_normalization=clean_receipt,
        variant_normalization=variant_receipt, tolerance=tolerance, orientation_radius=orientation_radius,
        by_provenance=by_provenance,
        widths=dict(clean_raw=_width_summary(clean >= .5), variant_raw=_width_summary(variant >= .5),
                    clean_T8=_width_summary(clean_T), variant_T8=_width_summary(variant_T)),
        normalized_input_changed_pixels=int(np.count_nonzero(clean_T != variant_T)),
        target_sha256=array_sha(target), valid_sha256=array_sha(valid),
        positive_sha256=array_sha(positive), negative_sha256=array_sha(negative),
        count_scope='whole source at native pixels before training crop/D4, not accumulated training visits',
        GT_sets_depend_only_on_target_valid=True,
        target_contract_scope='same target/valid arrays used once; caller must separately bridge dataset target/valid',
        new_missing_definition='trusted GT positive with clean T8 radius2 support but no variant T8 radius2 support',
        counts_are_not_gradient_mass_or_accuracy=True, no_test_GT=True,
        no_training=True, no_model_forward=True)
    return (result, masks) if return_masks else result


def coverage_gradient_probe(logits, target, valid, masks, *, tolerance=2, orientation_radius=3):
    """只对传入logits分解旧coverage；不加载模型或更新参数。

    可传实际G输出的detach副本，梯度只用于确认新增缺口并非监督盲区。
    每个子集仍除完整Npos，不能用独立分母悄悄增权。
    """
    import torch
    from torch.nn import functional as F
    from utils.boundary_rgb_training import build_tolerant_targets, directional_coverage_logits
    if (not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[:2] != (1, 1)
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
            or tuple(logits.shape[-2:]) != tuple(np.asarray(target).shape)):
        raise ValueError('Expected one source/crop logits as B1HW')
    def tensor(value):
        return torch.as_tensor(value, device=logits.device, dtype=torch.float32)[None, None]
    target_tensor, valid_tensor = tensor(target), tensor(valid)
    prepared = build_tolerant_targets(target_tensor, valid_tensor,
        tolerance=tolerance, orientation_radius=orientation_radius)
    denominator = int(prepared['positive'].sum())
    value = logits.detach().clone().requires_grad_(True)
    peak = directional_coverage_logits(value, prepared['normal_index'], prepared['valid'],
                                      tolerance, prepared['fallback'])
    output = {}
    for name in ('natural_missing', 'new_missing', 'unchanged_supported', 'actual_variant_missing'):
        mask = tensor(masks[name]).bool() & prepared['positive']
        part = F.softplus(-peak[mask]).sum()/max(1, denominator)
        gradient = torch.autograd.grad(part, value, retain_graph=True)[0]
        output[name] = dict(points=int(mask.sum()), global_positive_denominator=denominator,
            coverage_contribution=float(part.detach()), logit_gradient_L1=float(gradient.abs().sum()),
            logit_gradient_L2=float(gradient.norm()), nonzero_logit_gradient_pixels=int((gradient != 0).sum()))
    return dict(cohorts=output, no_model_loaded=True, no_parameter_gradient_claim=True,
                no_optimizer_update=True, loss_recipe_unchanged=True)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _bound_path(value, base, *, expected=None):
    if isinstance(value, dict):
        if expected is not None and value.get('sha256', expected) != expected:
            raise ValueError('Conflicting declared SHA256 bindings')
        expected = value.get('sha256', expected); value = value.get('path')
    if not isinstance(value, str) or not value:
        raise ValueError('Explicit path required')
    path = Path(value); path = path if path.is_absolute() else base/path
    path = path.resolve()
    if expected is not None and file_sha(path) != expected:
        raise ValueError('SHA256 differs: '+str(path))
    return path


def _row_list(document):
    rows = document.get('rows', document.get('files'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('A nonempty source row list is required')
    result = {}
    for row in rows:
        name = row.get('source') if isinstance(row, dict) else None
        if not isinstance(name, str) or re.fullmatch(r'train_[0-9]{3,4}', name) is None or name in result:
            raise ValueError('Unique train-only source rows are required')
        result[name] = row
    return result


def _load_heat(row, base, shape, pins):
    path = _bound_path(row.get('heat_path'), base, expected=row.get('heat_sha256'))
    if not isinstance(row.get('heat_sha256'), str):
        raise ValueError('Heat file SHA required')
    pins[str(path)] = file_sha(path)
    with np.load(path, allow_pickle=False) as arrays:
        if 'boundary' not in arrays.files:
            raise ValueError('Heat NPZ needs boundary')
        value = arrays['boundary'].copy()
    if value.shape != shape or ('shape' in row and list(row['shape']) != list(shape)):
        raise ValueError('Native heatmap shape differs')
    if row.get('representation') == 'thresholded_binary':
        value = _binary(value, 'cached binary boundary', shape)
        if row.get('threshold') != .5 or array_sha(value) != row.get('binary_sha256'):
            raise ValueError('Binary threshold/SHA differs')
    else:
        value = _probability(value, 'cached continuous boundary', shape)
    return value


def _gallery(rgb, clean, variant, masks, source, profile, output):
    h, w = rgb.shape[:2]
    center = (max(0, (w-768)//2), max(0, (h-768)//2), min(w, (w+768)//2), min(h, (h+768)//2))
    artifacts = []
    for label, box in (('whole', (0, 0, w, h)), ('center', center)):
        x0, y0, x1, y1 = box; sl = np.s_[y0:y1, x0:x1]
        ref = cv2.cvtColor(rgb[sl], cv2.COLOR_RGB2BGR)
        diagnostics = ref.copy()
        for key, color in (('natural_missing', (20, 20, 230)), ('new_missing', (0, 215, 255)),
                           ('recovered_old_missing', (0, 210, 0))):
            marker = cv2.dilate(masks[key][sl].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            diagnostics[marker] = color
        def heat(value):
            return cv2.applyColorMap(np.rint(np.clip(value[sl], 0, 1)*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        panels = [ref, heat(clean), heat(variant), heat(masks['clean_T8']), heat(masks['variant_T8']), diagnostics]
        scale = min(240/ref.shape[1], 280/ref.shape[0], 1.)
        tw, th = max(1, round(ref.shape[1]*scale)), max(1, round(ref.shape[0]*scale))
        canvas = np.full((th+72, tw*6, 3), 247, np.uint8)
        titles = ('RGB', 'Clean raw B', 'Perturbed raw B', 'Clean T8', 'Perturbed T8', 'GT support changes')
        for i, (panel, title) in enumerate(zip(panels, titles)):
            canvas[72:, i*tw:(i+1)*tw] = cv2.resize(panel, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.putText(canvas, title, (i*tw+3, 61), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, source+' | '+profile+' | '+label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Red old missing / yellow newly missing / green newly supported. Training GT; not accuracy.',
                    (5, 37), cv2.FONT_HERSHEY_SIMPLEX, .37, (20, 20, 20), 1, cv2.LINE_AA)
        path = output/(source+'_'+profile+'_'+label+'.jpg')
        ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if not ok:
            raise RuntimeError('Gallery encode failed')
        path.write_bytes(encoded.tobytes())
        artifacts.append(dict(path=str(path), sha256=file_sha(path), crop_xyxy=list(box),
                              display_only_resize=True, native_measurement_unchanged=True))
    return artifacts


def run(perturbation_manifest, processed_gt_manifest, output_dir, *, clean_real_input_manifest=None, sources=None):
    manifest_path = Path(perturbation_manifest).resolve()
    document = _read(manifest_path)
    if (document.get('format') != FORMAT or document.get('complete') is not True
            or document.get('no_test_training') is not True):
        raise ValueError('Complete no-test perturbed-heatmap manifest required')
    base = manifest_path.parent
    gt_path = Path(processed_gt_manifest).resolve()
    declared_gt = _bound_path(document.get('processed_gt_manifest'), base,
        expected=document.get('processed_gt_manifest_sha256'))
    if file_sha(gt_path) != file_sha(declared_gt):
        raise ValueError('Processed GT manifest binding differs')
    clean_path = _bound_path(document.get('clean_real_input_manifest'), base,
        expected=document.get('clean_real_input_manifest_sha256'))
    if clean_real_input_manifest is not None and file_sha(clean_real_input_manifest) != file_sha(clean_path):
        raise ValueError('Clean N1 manifest binding differs')
    clean_doc = _read(clean_path)
    if (clean_doc.get('format') != 'boundary_naf_real_heatmaps_v1' or clean_doc.get('complete') is not True
            or clean_doc.get('no_test_training') is not True):
        raise ValueError('Complete no-test clean N1 manifest required')
    if clean_doc.get('local_restoration') != {'mode': 'zero', 'model': 'restorer'}:
        raise ValueError('Clean input must use fixed N1 zero/restorer local restoration')
    for key in ('local_restoration', 'models'):
        if key in document and document[key] != clean_doc.get(key):
            raise ValueError('Frozen front-end provenance differs: '+key)
    gt_rows, clean_rows = _row_list(_read(gt_path)), _row_list(clean_doc)
    variants = document.get('variants')
    if not isinstance(variants, list) or not variants:
        raise ValueError('Named variant list required')
    names, selected = set(), set(sources) if sources is not None else None
    if selected is not None and (not selected or not selected <= set(gt_rows)):
        raise ValueError('Requested training-source subset differs')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True); (output/'gallery').mkdir()
    pins = {str(p): file_sha(p) for p in (manifest_path, gt_path, clean_path)}
    results, gallery = [], []
    for variant in variants:
        profile = variant.get('id')
        if not isinstance(profile, str) or re.fullmatch(r'[a-z0-9_]+', profile) is None or profile in names:
            raise ValueError('Unique safe variant id required')
        names.add(profile); var_rows = _row_list(variant)
        if not set(var_rows) <= set(gt_rows) & set(clean_rows):
            raise ValueError('Perturbed source absent from GT or clean manifest')
        chosen = sorted(set(var_rows) if selected is None else selected)
        if not set(chosen) <= set(var_rows):
            raise ValueError('Requested source absent from a variant')
        for source in chosen:
            gt_row, clean_row, row = gt_rows[source], clean_rows[source], var_rows[source]
            record = {}
            for key in ('image', 'gt', 'classes'):
                expected = gt_row.get('hashes', gt_row).get(key+'_sha256')
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError('Processed source SHA required')
                path = _bound_path(gt_row.get(key), gt_path.parent, expected=expected)
                record[key] = path; pins[str(path)] = file_sha(path)
                for heat_row in (clean_row, row):
                    if heat_row.get(key+'_sha256') != expected:
                        raise ValueError('Heatmap source provenance differs: '+source+'/'+key)
            target, valid, gt_metadata = load_processed_binary(record['gt'], record['classes'], boundary_dilation=0, valid_margin=4)
            with np.load(record['gt'], allow_pickle=False) as payload:
                original, filled = payload['original_covered'].copy(), payload['filled'].copy()
            clean = _load_heat(clean_row, clean_path.parent, target.shape, pins)
            perturbed = _load_heat(row, base, target.shape, pins)
            result, masks = analyze_pair(clean, perturbed, target, valid, original, filled, return_masks=True)
            result.update(source=source, variant=profile, GT_metadata=gt_metadata,
                          continuous_variant_not_available=row.get('representation') == 'thresholded_binary')
            _write(output/(source+'_'+profile+'.json'), result); results.append(result)
            rgb = read_image(record['image'], cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape != (*target.shape, 3):
                raise ValueError('Original RGB shape differs')
            gallery.extend(_gallery(rgb[..., ::-1], clean, perturbed, masks, source, profile, output/'gallery'))
    for path, digest in pins.items():
        if file_sha(path) != digest:
            raise RuntimeError('Input changed during diagnostic: '+path)
    summary = dict(format='naf_perturbation_probe_v1', complete=True, source_pins=pins, rows=results,
        gallery=gallery, no_training=True, no_model_forward=True, no_loss_change=True,
        whole_source_counts_not_training_visit_counts=True,
        front_end_identity_scope='clean N1 restoration mode checked; declared perturbation model/mode identities compared when supplied',
        no_test_GT=True, not_accuracy=True, no_test_epoch_selection=True)
    _write(output/'summary.json', summary)
    body = '<h1>Train-only NAF input perturbation</h1><p>Counts diagnose supervision coverage, not accuracy.</p>'
    for item in gallery:
        body += '<img style="max-width:100%" src="gallery/'+html.escape(Path(item['path']).name)+'">'
    (output/'index.html').write_text(body, encoding='utf8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perturbation-manifest', required=True)
    parser.add_argument('--processed-gt-manifest', required=True)
    parser.add_argument('--clean-real-input-manifest')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sources', nargs='+')
    args = parser.parse_args(argv)
    result = run(**vars(args))
    print(json.dumps(dict(complete=True, rows=len(result['rows']), no_training=True)))


if __name__ == '__main__':
    main()
