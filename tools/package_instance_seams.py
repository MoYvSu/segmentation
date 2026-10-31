# -*- coding: utf-8 -*-
"""核验已评分最高包，执行唯一细缝规则；完整验收后保存候选，不上传平台。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assign_instance_seams import assign_instance_seams
from tools.render_backend_ablation import overlay, text_line, write_png
from utils.config import load_config, project_path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate class-map key')
        result[key] = value
    return result


def read_png(raw, shape):
    if (len(raw) < 29 or raw[:8] != b'\x89PNG\r\n\x1a\n'
            or raw[12:16] != b'IHDR' or raw[24:26] != bytes((16, 0))):
        raise ValueError('Expected actual 16-bit grayscale PNG')
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None or a.dtype != np.uint16 or a.ndim != 2 or a.shape != tuple(shape):
        raise ValueError('PNG dtype or original source shape differs')
    return a


def phase_areas(ids, classes):
    sizes = np.bincount(ids.ravel(), minlength=65536)
    result = {}
    for name, phase in (('P', 0), ('F', 1)):
        chosen = [int(k) for k, v in classes.items() if v == phase]
        area = int(sizes[chosen].sum()) if chosen else 0
        result[name] = dict(instances=len(chosen), pixels=area,
                            mean_area=area / len(chosen) if chosen else None)
    result['ID0_pixels'] = int(sizes[0])
    return result


def gallery(path, source, before, after, classes):
    image = cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != before.shape:
        raise RuntimeError('Gallery original source differs')
    # 固定中央窗口；不根据变化或疑似错误选位置。
    h, w = before.shape; side = min(600, h, w)
    y, x = (h-side)//2, (w-side)//2
    sl = np.s_[y:y+side, x:x+side]
    changed = (before == 0) & (after > 0)
    marked = image.copy(); marked[changed] = (35, 30, 245)
    integer_classes = {int(k):v for k,v in classes.items()}
    panels = [('Original / fixed centre', image[sl]),
              ('Highest mixed balance', overlay(image, before, integer_classes)[sl]),
              ('Same-phase seam assignment', overlay(image, after, integer_classes)[sl]),
              ('Assigned pixels / red', marked[sl])]
    width = 350; canvas = np.full((width+28, width*4, 3), 255, np.uint8)
    for index, (title, panel) in enumerate(panels):
        canvas[28:,index*width:(index+1)*width] = cv2.resize(panel, (width,width), interpolation=cv2.INTER_AREA)
        text_line(canvas, title, (index*width+4,18), width-8)
    write_png(path, canvas)


def run(config, input_archive, validation_path, output):
    opt = config['instance_seam_assignment']; archive_path = Path(input_archive)
    if opt['version'] != 'same_phase_nearest_original_v1' or opt['enabled'] is not True:
        raise ValueError('Only the predeclared enabled seam candidate is accepted')
    core_hash = sha(ROOT/'tools/assign_instance_seams.py')
    digest = sha(archive_path)
    if digest != opt['input_archive_sha256']:
        raise RuntimeError('Input is not the registered highest mixed balance package')
    validation = json.loads(Path(validation_path).read_text(encoding='utf8'))
    if (validation['complete'] is not True or validation['archive_sha256'] != digest
            or validation['checkpoint_sha256'] != opt['checkpoint_sha256'] or validation['epoch'] != opt['epoch']
            or validation['inference'] != 'native1024' or validation['images'] != opt['expected_images']):
        raise RuntimeError('Registered original package provenance differs')
    registry = {r['source']: r for r in validation['rows']}
    if len(registry) != opt['expected_images']:
        raise RuntimeError('Original shape registry is incomplete')
    source_dir = Path(project_path(config,config['inference']['test_dir']))
    sources = {p.stem:p for p in source_dir.iterdir() if p.is_file()
               and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')}
    if set(sources) != set(registry):
        raise RuntimeError('Actual semifinal source cohort differs')
    selected = list(opt['gallery_images'])
    remaining = sorted(set(registry)-set(selected))
    selected += [str(v) for v in np.random.default_rng(opt['gallery_random_seed']).choice(remaining,size=2,replace=False)]
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    result_dir = output/'results'; result_dir.mkdir(); (output/'gallery').mkdir()
    rows = []; file_hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        expected = {s+suffix for s in registry for suffix in ('_inst.png','_class.json')}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected or archive.testzip() is not None:
            raise RuntimeError('Input package members or CRC differ')
        for stem in sorted(registry):
            png_name, class_name = stem+'_inst.png',stem+'_class.json'
            png_bytes, class_bytes = archive.read(png_name), archive.read(class_name)
            if hashlib.sha256(png_bytes).hexdigest() != registry[stem]['png_sha256']:
                raise RuntimeError('Input PNG differs from registered scored package')
            # 源尺寸实时核验；赛题GT从不进入本入口。
            source_rgb = cv2.imdecode(np.fromfile(sources[stem],np.uint8),cv2.IMREAD_COLOR)
            if source_rgb is None or list(source_rgb.shape[:2]) != registry[stem]['shape']:
                raise RuntimeError('Actual original source dimensions differ')
            shape = source_rgb.shape[:2]; del source_rgb
            before = read_png(png_bytes,shape)
            classes = json.loads(class_bytes.decode('utf8'),object_pairs_hook=unique_json)
            disabled,_ = assign_instance_seams(before,classes,enabled=False)
            if not np.array_equal(before,disabled):
                raise RuntimeError('Disabled final output compatibility failed')
            after,receipt = assign_instance_seams(before,classes)
            positive = before>0
            if (not np.array_equal(before[positive],after[positive])
                    or not np.array_equal(np.unique(before[positive]),np.unique(after[after>0]))):
                raise RuntimeError('Positive pixels or ID set changed')
            ok,data = cv2.imencode('.png',after)
            if not ok:
                raise RuntimeError('PNG encoding failed')
            candidate_bytes = data.tobytes()
            if not np.array_equal(read_png(candidate_bytes,shape),after):
                raise RuntimeError('Saved PNG decode differs')
            (result_dir/png_name).write_bytes(candidate_bytes)
            (result_dir/class_name).write_bytes(class_bytes)
            # class JSON保持原始字节，不重投票或重排ID。
            for name in (png_name,class_name): file_hashes[name] = sha(result_dir/name)
            phases_before = phase_areas(before,classes); phases_after = phase_areas(after,classes)
            row = dict(source=stem,shape=list(shape),source_sha256=sha(sources[stem]),
                       baseline_png_sha256=hashlib.sha256(png_bytes).hexdigest(),
                       candidate_png_sha256=file_hashes[png_name],
                       classes_sha256=file_hashes[class_name],class_bytes_exact=True,
                       positive_pixels_exact=True,ID_set_exact=True,disabled_exact=True,
                       before=phases_before,after=phases_after,assignment=receipt)
            rows.append(row)
            if stem in selected: gallery(output/'gallery'/f'{stem}.png',sources[stem],before,after,classes)
            print(stem,receipt.get('assigned_pixels','complete'),flush=True)
    totals = {a:{p:sum(r[a][p]['pixels'] for r in rows) for p in ('F','P')} for a in ('before','after')}
    ratios = [r['after']['F']['pixels']/r['before']['F']['pixels']-1 for r in rows if r['before']['F']['pixels']>0]
    summary = dict(images=len(rows),instances=sum(r['before']['F']['instances']+r['before']['P']['instances'] for r in rows),
                   F_instances=sum(r['before']['F']['instances'] for r in rows),phase_pixels=totals,
                   F_mean_area_change_median=float(np.median(ratios)),
                   F_mean_area_change_min=float(min(ratios)),F_mean_area_change_max=float(max(ratios)),
                   total_assigned_pixels=sum(int(((r['after']['F']['pixels']+r['after']['P']['pixels'])
                                                  -(r['before']['F']['pixels']+r['before']['P']['pixels']))) for r in rows),
                   class_bytes_all_exact=True,positive_pixels_all_exact=True,ID_sets_all_exact=True,
                   scope='unlabelled deployment distribution; area changes are not area accuracy')
    candidate_zip = output/'seams1024.zip'
    with zipfile.ZipFile(candidate_zip,'x',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(file_hashes): archive.write(result_dir/name,name)
    with zipfile.ZipFile(candidate_zip) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(file_hashes):
            raise RuntimeError('Candidate ZIP CRC/member contract failed')
        for name,expected_sha in file_hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_sha:
                raise RuntimeError('Candidate ZIP content differs from audited result')
    if sha(archive_path) != digest or sha(ROOT/'tools/assign_instance_seams.py') != core_hash:
        raise RuntimeError('Baseline archive or sealed core changed during processing')
    receipt = dict(complete=True,config=config,checkpoint_sha256=opt['checkpoint_sha256'],epoch=opt['epoch'],
                   input_archive_sha256=digest,reference_validation_sha256=sha(validation_path),
                   candidate_archive_sha256=sha(candidate_zip),candidate_archive_bytes=candidate_zip.stat().st_size,
                   images=len(rows),files=len(file_hashes),files_sha256=file_hashes,
                   summary=summary,rows=rows,selected_gallery=selected,
                   core_sha256=core_hash,package_tool_sha256=sha(__file__),
                   official_score=None,submitted=False,model_parameters_changed=False,no_training=True,
                   rule='original ID0, inside frame, homogeneous positive 8-neighbour phase; nearest original positive pixel, one pass')
    (output/'report.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    links=''.join(f'<h3>{html.escape(s)}</h3><img width="1400" src="gallery/{html.escape(s)}.png">' for s in selected)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Instance seams</title>'
                                    '<p>Fixed central crop. Red: assigned original ID0 only. No test GT.</p>'+links,encoding='utf8')
    print(json.dumps(dict(summary=summary,archive=str(candidate_zip),sha256=receipt['candidate_archive_sha256']),ensure_ascii=False))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/inference/instance_seams.yaml')
    parser.add_argument('--input-zip')
    parser.add_argument('--validation',default='output/balance_analysis/mixed_validation.json')
    parser.add_argument('--output')
    args=parser.parse_args();cfg=load_config(args.config);opt=cfg['instance_seam_assignment']
    run(cfg,args.input_zip or project_path(cfg,opt['input_archive']),args.validation,
        args.output or project_path(cfg,opt['output_dir']))
