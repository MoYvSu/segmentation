# -*- coding: utf-8 -*-
"""冻结 N0 的原生 RGB 模糊诊断；先精确零退化验收，再运行两档模糊。

仅按既有私有清单选训练源，不读取 GT 标签、不训练、不执行全图语义或实例读出。
所有输出应放在被忽略的私有目录。sigma 的单位是原图像素，不是网络网格。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


SELECTION = {'train_gt': (0, 10, 21, 31), 'train_extra': (0, 6, 12, 18)}
FORMAT = 'boundary_rgb_probe_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def select_sources(samples, reference):
    """顺序来自旧清单，不能按本次预测或同名文件重新分配源图索引。"""
    if samples.get('complete') is not True or reference.get('complete') is not True:
        raise ValueError('Both historical manifests must be complete')
    prior = {row['source']: row for row in reference['files']}
    selected = []
    for domain, indices in SELECTION.items():
        pool = [row for row in samples['files'] if row['domain'] == domain]
        if len(pool) <= max(indices):
            raise ValueError('Historical training pool is smaller than the fixed selection')
        for index in indices:
            row = dict(pool[index])
            old = prior[row['source']]
            for name in ('domain', 'source_index', 'input_sha256', 'source_shape', 'window', 'box', 'roi_box'):
                if row[name] != old[name]:
                    raise ValueError('Historical source/geometry differs: '+row['source']+'/'+name)
            if row['domain'] not in SELECTION or row['roi_box_format'] != 'yxyx':
                raise ValueError('Only original training-source native yxyx ROIs are allowed')
            if old['source_path'] != row['path']:
                raise ValueError('Historical source path differs')
            row.update(selection_index=index, reference=old)
            selected.append(row)
    if len(selected) != 8 or len({r['input_sha256'] for r in selected}) != 8:
        raise ValueError('Eight distinct original source images are required')
    return selected


def blur_rgb(rgb, sigma):
    """完整 uint8 RGB 上模糊，再由调用方裁窗；零档直接复制，不经过滤波。"""
    import cv2
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected original uint8 RGB HWC')
    if sigma not in (0, 2, 4):
        raise ValueError('This controlled probe fixes sigma to 0, 2 or 4 native pixels')
    if sigma == 0:
        return rgb.copy(), dict(sigma_native=0, kernel=[1, 1], operation='exact_uint8_copy', border='not_applicable')
    radius = int(math.ceil(3 * sigma))
    kernel = 2 * radius + 1
    result = cv2.GaussianBlur(rgb, (kernel, kernel), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT)
    assert result.shape == rgb.shape and result.dtype == np.uint8
    return result, dict(sigma_native=sigma, kernel=[kernel, kernel], truncate_sigma=3,
        operation='cv2.GaussianBlur_full_source_uint8_RGB_to_uint8', border='BORDER_REFLECT',
        quantization='OpenCV uint8 GaussianBlur output; no external float cast or rescale', opencv=cv2.__version__)


@contextmanager
def capture_actual_noise(restorer):
    """只记录 q_sample 已收到的噪声字节；不改张量、不额外采样，退出时恢复方法。"""
    old = restorer.q_sample
    existed = 'q_sample' in restorer.__dict__
    previous = restorer.__dict__.get('q_sample')
    observations = []

    def observed(*args, **kwargs):
        noise = kwargs.get('noise')
        if noise is None or kwargs.get('generator') is not None:
            raise RuntimeError('Frozen Philox path must explicitly pass its actual noise tensor')
        array = noise.detach().cpu().numpy()
        if array.dtype != np.float32 or array.shape != (1, 3, 1024, 1024):
            raise RuntimeError('Unexpected actual N0 noise shape/dtype')
        observations.append(dict(sha256=array_sha(array), shape=list(array.shape), dtype=str(array.dtype),
                                 mode=getattr(restorer, 'inference_noise_mode', None)))
        return old(*args, **kwargs)

    restorer.q_sample = observed
    try:
        yield observations
    finally:
        if existed:
            restorer.q_sample = previous
        else:
            delattr(restorer, 'q_sample')
        if restorer.q_sample != old:
            raise RuntimeError('The original q_sample method was not restored')


def check_window(audit, previous, seeds, box):
    if seeds != previous['seed_audit'] or list(box) != previous['box']:
        raise RuntimeError('Original full-image window seed/order/box changed')
    for key in ('input_tensor_sha256', 'restored_tensor_sha256', 'grid512_sha256', 'boundary1024_sha256'):
        if audit[key] != previous[key]:
            raise RuntimeError('Zero degradation differs from historical window: '+key)


def render_gallery(manifest, output):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    gallery = output/'gallery'; gallery.mkdir(exist_ok=True)
    groups = {}
    for row in manifest['files']:
        groups.setdefault(row['source'], {})[row['sigma_native']] = row
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    def heat(value):
        return cv2.applyColorMap(np.rint(np.clip(value,0,1)*255).astype(np.uint8), cv2.COLORMAP_MAGMA)[:,:,::-1]

    files = []
    for source, modes in groups.items():
        with np.load(modes[0]['path'], allow_pickle=False) as f:
            original = f['rgb512'].copy(); base = f['boundary512'].copy()
        for sigma in (2,4):
            with np.load(modes[sigma]['path'], allow_pickle=False) as f:
                rgb, boundary = f['rgb512'].copy(), f['boundary512'].copy()
            difference = boundary-base
            delta = np.zeros((*difference.shape,3), np.uint8)
            delta[:,:,0] = np.rint(np.clip(difference,0,1)*255).astype(np.uint8)
            delta[:,:,2] = np.rint(np.clip(-difference,0,1)*255).astype(np.uint8)
            crossing = original.copy()
            crossing[(base>=.65)&(boundary<.45)] = [0,150,255]
            crossing[(base<.2)&(boundary>=.65)] = [255,70,0]
            panels = [('Original RGB',original),(f'Full-RGB blur sigma={sigma}',rgb),
                      ('N0 original response [0,1]',heat(base)),('N0 blurred response [0,1]',heat(boundary)),
                      ('Signed delta: red up / blue down (scale 1)',delta),
                      ('Blue: .65 -> <.45 / orange: <.2 -> .65',crossing)]
            tile=320;canvas=Image.new('RGB',(tile*3,(tile+29)*2+62),'white');draw=ImageDraw.Draw(canvas)
            draw.text((6,5),f'{source} | native512 ROI | sigma={sigma} native px | same N0 / exact noise',fill='black',font=font)
            draw.text((6,27),'Unlabeled model-response comparison. Threshold changes are NOT missing-edge accuracy.',fill='black',font=font)
            for i,(label,array) in enumerate(panels):
                x,y=(i%3)*tile,58+(i//3)*(tile+29)
                draw.text((x+4,y),label,fill='black',font=font)
                canvas.paste(Image.fromarray(np.ascontiguousarray(array)).resize((tile,tile),Image.Resampling.BILINEAR),(x,y+25))
            path=gallery/f'{source}_sigma{sigma}.jpg';canvas.save(path,quality=95)
            files.append(dict(source=source,sigma_native=sigma,path=str(path),sha256=sha(path)))
    write(output/'gallery_manifest.json',dict(files=files,all_eight_sources=True,scale=[0,1],script_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('code-root','baseline-audit','model-dir','sample-manifest','reference-manifest','window-tool','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('zero','blur'),default='zero')
    args=p.parse_args()
    args.code_root=args.code_root.resolve();args.output=args.output.resolve()
    reference=json.loads(args.reference_manifest.read_text(encoding='utf-8'))
    samples=json.loads(args.sample_manifest.read_text(encoding='utf-8'))
    rows=select_sources(samples,reference)
    if sha(args.window_tool)!=reference['capture_script_sha256']:
        raise RuntimeError('Window helper is not the original sealed capture implementation')
    helper_spec=importlib.util.spec_from_file_location('_frozen_boundary_domain_rgb',args.window_tool)
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    for row in rows:
        if (Path(row['path']).stem!=row.get('file_stem',row['source'])
                or sha(row['path'])!=row['input_sha256']
                or sha(row['reference']['path'])!=row['reference']['sha256']):
            raise RuntimeError('Original RGB or reference cache SHA differs: '+row['source'])
    sys.path.insert(0,str(args.code_root))
    import cv2
    from tools import infer_semifinal as entry
    from tools.affinity_connectivity_views import _inference_state
    from train_backend_adaptation import tensor_digest
    from utils.image_io import read_image
    old=json.loads(args.baseline_audit.read_text(encoding='utf-8'));config=old['config']
    if Path(config['paths']['project_root']).resolve()!=args.code_root or config!=reference['config']:
        raise RuntimeError('N0 source/config differs from the sealed reference')
    for name,expected in old['imported_sources'].items():
        if sha(args.code_root/name)!=expected:raise RuntimeError('Frozen source differs: '+name)
    source_manifest=entry.verify_source_manifest(args.code_root)
    for value in old['models'].values():
        if sha(args.model_dir/value['file'])!=value['sha256']:raise RuntimeError('Frozen weight file differs')
    receipt_path=args.output/'capture_manifest.json'
    if args.phase=='zero':
        if receipt_path.exists():raise FileExistsError('Zero phase requires a fresh output without a capture manifest')
        receipt=None
    else:
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt['format']!=FORMAT or receipt.get('complete') or not receipt.get('zero_gate',{}).get('all_exact')
                or len(receipt['files'])!=8 or any(r['sigma_native']!=0 for r in receipt['files'])):
            raise RuntimeError('Blur phase requires the completed, unchanged eight-source exact zero gate')
        for key,path in [('capture_script_sha256',Path(__file__)),('window_helper_sha256',args.window_tool),
                         ('sample_manifest_sha256',args.sample_manifest),('reference_manifest_sha256',args.reference_manifest),
                         ('baseline_audit_sha256',args.baseline_audit)]:
            if receipt[key]!=sha(path):raise RuntimeError('Source/protocol changed between phases: '+key)
        for row in receipt['files']:
            if sha(row['path'])!=row['sha256']:raise RuntimeError('Zero-phase cache changed')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPU occupied; do not collide with another job: '+active)
    if shutil.disk_usage(args.output.parent).free < 1 << 30:raise RuntimeError('At least 1 GiB free space is required')
    entry.validate_config(config);entry.configure_runtime()
    model,restorer,terminal=entry.load_models(config,entry.model_paths(config,args.model_dir.resolve()),'cuda')
    if terminal is not None or entry.local_restoration(config)['candidate']!='N0':
        raise RuntimeError('Only original N0 local restoration is permitted')
    states=dict(backend=tensor_digest(model.state_dict().items()),restorer=tensor_digest(restorer.state_dict().items()))
    if states!=old['model_states'] or states!=reference['model_states']:raise RuntimeError('Loaded states differ')
    runtime=entry.runtime_record()
    if runtime!=reference['runtime']:raise RuntimeError('Runtime differs from exact historical capture')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'source').mkdir(exist_ok=True)
    copy_path=args.output/'source'/Path(__file__).name
    if copy_path.resolve()!=Path(__file__).resolve():shutil.copyfile(__file__,copy_path)
    if receipt is None:
        write(args.output/'selected_sources.json',dict(selection={k:list(v) for k,v in SELECTION.items()},files=rows))
        receipt=dict(format=FORMAT,complete=False,training=False,gt_labels_read=False,test_sources=False,
            global_semantic_forward=False,final_decode=False,source_root=str(args.code_root),source_manifest=source_manifest,
            capture_script_sha256=sha(__file__),window_helper_sha256=sha(args.window_tool),
            sample_manifest_sha256=sha(args.sample_manifest),reference_manifest_sha256=sha(args.reference_manifest),
            baseline_audit_sha256=sha(args.baseline_audit),config=config,models=old['models'],model_states=states,runtime=runtime,
            selection={k:list(v) for k,v in SELECTION.items()},source_count=8,planned_rois=24,
            roi_box_format='yxyx',noise_check='actual q_sample noise tensor SHA, shape and mode per original window',
            degradation='full source uint8 RGB Gaussian blur first; crop original windows second',
            files=[],phases=[],zero_gate=dict(all_exact=False))
    elif receipt['runtime']!=runtime or receipt['model_states']!=states:
        raise RuntimeError('Runtime or model changed between phases')
    zero_rows={r['source']:r for r in receipt['files'] if r['sigma_native']==0}
    start=time.time()
    for sigma in ((0,) if args.phase=='zero' else (2,4)):
        target_dir=args.output/f'sigma{sigma}';target_dir.mkdir()
        for row in rows:
            original=cv2.cvtColor(read_image(row['path']),cv2.COLOR_BGR2RGB)
            rgb,degradation=blur_rgb(original,sigma)
            primary,box,roi,windows=helper.roi_window_plan(rgb.shape[:2],row)
            prior=row['reference'];expected_windows={r['window']:r for r in prior['windows']}
            if ([i for i,_ in windows]!=[r['window'] for r in prior['windows']]
                    or list(roi)!=prior['roi_box'] or primary!=prior['window']):
                raise RuntimeError('Original ROI/window plan changed')
            baseline_windows={} if sigma==0 else {r['window']:r for r in zero_rows[row['source']]['windows']}
            blend=helper.RoiBlendMap(rgb.shape[:2],roi);audits=[]
            with _inference_state(model,restorer,'cuda'):
                for index,window_box in windows:
                    seeds=helper.seed_for(row,index)
                    with capture_actual_noise(restorer) as noise:
                        arrays,audit=helper.capture_one(model,restorer,rgb,window_box,config,seeds['local_seed'])
                    if len(noise)!=1:raise RuntimeError('Expected exactly one first-step q_sample noise tensor')
                    if sigma==0:
                        check_window(audit,expected_windows[index],seeds,window_box)
                    elif (noise[0]!=baseline_windows[index]['actual_noise'] or seeds!=baseline_windows[index]['seed_audit']):
                        raise RuntimeError('Blur and original did not receive identical actual injected noise')
                    blend.add(window_box,arrays['boundary1024'])
                    audits.append(dict(window=index,box=list(window_box),seed_audit=seeds,actual_noise=noise[0],**audit))
            boundary,blend_audit=blend.finish()
            if blend_audit!=prior['blend_audit']:raise RuntimeError('Original blend weights/overlap changed')
            bridge=None
            if sigma==0:
                with np.load(prior['path'],allow_pickle=False) as cached:
                    expected=cached['stitched_boundary512']
                    exact=np.array_equal(boundary,expected)
                    bridge=dict(exact=exact,max_absolute_difference=float(np.abs(boundary-expected).max()),reference_sha256=prior['sha256'])
                if not exact:
                    write(args.output/'zero_failure.json',dict(source=row['source'],bridge=bridge,windows=audits))
                    raise RuntimeError('Zero degradation is not exact; stop without running blur')
            y,x,y1,x1=roi;path=target_dir/f"{row['source']}.npz"
            np.savez_compressed(path,rgb512=rgb[y:y1,x:x1],reference_rgb512=original[y:y1,x:x1],
                                boundary512=boundary,roi_box=np.asarray(roi,np.int32))
            receipt['files'].append(dict(source=row['source'],domain=row['domain'],selection_index=row['selection_index'],
                source_index=row['source_index'],source_path=row['path'],source_shape=list(rgb.shape[:2]),input_sha256=row['input_sha256'],
                sigma_native=sigma,mode=f'sigma{sigma}',degradation=degradation,full_degraded_rgb_sha256=array_sha(rgb),
                roi_box=list(roi),roi_box_format='yxyx',path=str(path),sha256=sha(path),boundary_sha256=array_sha(boundary),
                windows=audits,blend_audit=blend_audit,bridge=bridge))
            write(receipt_path,receipt)
            print(f'CAPTURE {args.phase} {row["source"]} sigma={sigma} windows={len(windows)} exact_noise=True',flush=True)
    after=dict(backend=tensor_digest(model.state_dict().items()),restorer=tensor_digest(restorer.state_dict().items()))
    if after!=states:raise RuntimeError('Model states changed during the diagnostic')
    for name,expected in old['imported_sources'].items():
        if sha(args.code_root/name)!=expected:raise RuntimeError('Sealed source changed during capture')
    if entry.verify_source_manifest(args.code_root)!=source_manifest:raise RuntimeError('Sealed source manifest changed')
    for value in old['models'].values():
        if sha(args.model_dir/value['file'])!=value['sha256']:raise RuntimeError('Weight file changed during capture')
    receipt['phases'].append(dict(phase=args.phase,elapsed_seconds=time.time()-start,frozen_models_unchanged=True,
                                  sealed_source_unchanged=True,q_sample_method_restored=True))
    if args.phase=='zero':
        receipt['zero_gate']=dict(all_exact=True,source_count=8,window_count=sum(len(r['windows']) for r in receipt['files']),
                                  max_absolute_difference=0.0,requires_separate_blur_phase=True)
    else:
        receipt.update(complete=True,frozen_models_unchanged=True,sealed_source_unchanged=True,all_actual_noise_identical=True)
    write(receipt_path,receipt)
    if args.phase=='blur':render_gallery(receipt,args.output)
    print('PHASE_COMPLETE',args.phase,len(receipt['files']),time.time()-start,flush=True)


if __name__=='__main__':
    main()
