# -*- coding: utf-8 -*-
"""冻结N1局部D5a恢复RGB，原生拼接后供NAF裁块；不加载分割模型或训练。

请用独立Python进程运行本文件，避免把当前工作区模块混入封存N1源码。
浮点RGB无量化；训练32源与四个固定目检源分别写manifest。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import torch

FORMAT = 'boundary_rgb_restored_v1'
COORDINATES = 'native1024_overlap025_letterbox_reflect_blend_before_crop_D4'
MIN_FREE = 3 * 1024 ** 3
MAX_OUTPUT = int(2.5 * 1024 ** 3)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_state_sha(module):
    """与N1 manifest的train_backend_adaptation.tensor_digest完全相同。"""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_rgb(rgb, shape=None):
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError('原始RGB必须是uint8 HWC三通道')
    if shape is not None and list(value.shape[:2]) != list(shape[:2]):
        raise ValueError('原始RGB与登记形状不符')
    return np.ascontiguousarray(value)


def reference_hashes(window):
    first = window.get('input_tensor_sha256', window.get('input_sha256'))
    second = window.get('restored_tensor_sha256', window.get('restored_sha256'))
    if any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in (first, second)):
        raise ValueError('缺少原N1逐窗输入/恢复哈希')
    return first, second


def restore_native_rgb(rgb, windows, restorer, ops, *, device='cuda', native_size=1024,
                       overlap=.25, repeat_first=False):
    """同原N1窗序与算术；只调用D5a，不调用encoder/decoder/WS。"""
    rgb = validate_rgb(rgb)
    boxes = ops.tile_boxes(rgb.shape[:2], native_size, overlap)
    if len(boxes) != len(windows):
        raise ValueError('原N1窗数量与当前tile_boxes不同')
    blends = [ops.BlendMap(rgb.shape[:2]) for _ in range(3)]
    proofs = []
    for index, (box, prior) in enumerate(zip(boxes, windows)):
        if list(box) != prior['box'] or int(prior.get('window', index)) != index:
            raise ValueError('原N1窗坐标/顺序改变')
        y, x, y1, x1 = box
        selected = np.ascontiguousarray(rgb[y:y1, x:x1])
        expected_input, expected_restored = reference_hashes(prior)
        if 'rgb1024_sha256' in prior and array_sha(selected) != prior['rgb1024_sha256']:
            raise ValueError('原始uint8窗RGB改变；rgb1024不是恢复后数组')
        image, ph, pw = ops.tensor_from_rgb(selected, device)
        if image.dtype != torch.float32 or tuple(image.shape) != (1, 3, native_size, native_size):
            raise ValueError('恢复输入shape/dtype不符')
        input_digest = array_sha(image.detach().cpu().numpy())
        if input_digest != expected_input:
            raise RuntimeError('当前原生输入未逐值复现N1: window ' + str(index))
        audit = {}
        restored, _ = ops.restore_probe(restorer, image, int(prior['seed']), 'zero', audit=audit)
        restored_digest = array_sha(restored.detach().cpu().numpy())
        if restored_digest != expected_restored:
            raise RuntimeError('当前D5a恢复未逐值复现N1: window ' + str(index))
        if (audit.get('q_sample_calls') != 1 or audit.get('noise_nonzero') != 0
                or audit.get('initial_equals_condition') is not True or audit.get('step') != 16):
            raise RuntimeError('D5a未实际使用原N1零噪声首步')
        repeated = None
        if repeat_first and index == 0:
            again, _ = ops.restore_probe(restorer, image, int(prior['seed']), 'zero')
            repeated = bool(torch.equal(restored, again))
            if not repeated:
                raise RuntimeError('同一D5a零噪声窗重推不一致')
        native = ops.crop_letterbox_output(restored.float(), native_size, ph, pw,
                                           (y1-y, x1-x)).detach().cpu()[0].numpy()
        if native.shape != (3, y1-y, x1-x) or native.dtype != np.float32 or not np.isfinite(native).all():
            raise RuntimeError('逆letterbox的恢复RGB不符合原生契约')
        for channel, blend in enumerate(blends):
            blend.add(box, native[channel])
        proofs.append(dict(window=index, box=list(box), seed=int(prior['seed']), pad_h=int(ph), pad_w=int(pw),
            raw_RGB_sha256=array_sha(selected), input_tensor_sha256=input_digest,
            restored_tensor_sha256=restored_digest, actual_local_zero=audit,
            same_window_repeat_exact=repeated))
    channels = [blend.finish() for blend in blends]
    value = np.ascontiguousarray(np.stack([a[0] for a in channels], axis=-1), dtype=np.float32)
    if value.shape != rgb.shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise RuntimeError('拼接RGB必须为原生float32 HWC [0,1]')
    return value, proofs, [a[1] for a in channels]


def save_array(path, value):
    """独立重读，保留完整float32位，不以PNG/float16额外量化。"""
    path = Path(path)
    if path.exists() or value.dtype != np.float32 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError('cache需fresh float32 HWC')
    temporary = path.with_suffix('.npy.tmp')
    with temporary.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
    reloaded = np.load(temporary, mmap_mode='r', allow_pickle=False)
    if not np.array_equal(reloaded, value) or reloaded.dtype != np.float32:
        raise RuntimeError('恢复cache重读不一致')
    del reloaded
    temporary.replace(path)
    return dict(restored_path=str(path.resolve()), restored_sha256=file_sha(path),
                restored_array_sha256=array_sha(value), shape=list(value.shape), dtype='float32',
                value_range=[float(value.min()), float(value.max())])


def load_frozen_ops(code_root, device):
    root = Path(code_root).resolve()
    # 本工具必须独立进程运行。拒绝把已经导入的其它checkout静默当封存源码。
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in {'data', 'models', 'utils'}:
            continue
        path = getattr(module, '__file__', None)
        if path and not Path(path).resolve().is_relative_to(root):
            raise RuntimeError('请用独立Python进程运行；已有外部模块: ' + name)
    sys.path.insert(0, str(root))
    image_module = importlib.import_module('data.dataset')
    rgb_module = importlib.import_module('models.rgb_restoration')
    probe_module = importlib.import_module('utils.restoration_probe')
    crop_module = importlib.import_module('utils.affinity_deployment')
    blend_module = importlib.import_module('utils.patch_diagnostic')
    for module in (image_module, rgb_module, probe_module, crop_module, blend_module):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('实际模块不属于封存N1源码')
    def tensor_from_rgb(rgb, actual_device=device):
        value, _, ph, pw = image_module.letterbox(rgb, 1024)
        tensor = torch.from_numpy(value).permute(2, 0, 1).float().unsqueeze(0).to(actual_device)
        if rgb.dtype == np.uint8:
            tensor = tensor / 255.
        return tensor, ph, pw
    return SimpleNamespace(load_restorer=rgb_module.load_rgb_restorer,
        tensor_from_rgb=tensor_from_rgb, restore_probe=probe_module.restore_probe,
        crop_letterbox_output=crop_module.crop_letterbox_output,
        BlendMap=blend_module.BlendMap, tile_boxes=blend_module.tile_boxes)


def sealed_sources(root, expected_digest):
    path = root/'manifest.json'
    if file_sha(path) != expected_digest:
        raise RuntimeError('封存N1源码manifest改变')
    rows = read(path)['files']
    for row in rows:
        source = (root/row['path']).resolve()
        if not source.is_relative_to(root) or not source.is_file() or file_sha(source) != row['sha256']:
            raise RuntimeError('封存N1源码/资源改变: ' + row['path'])
    return dict(manifest_sha256=file_sha(path), checked_files=len(rows))


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def cache(args):
    output, code = args.output_root.resolve(), args.code_root.resolve()
    if output.exists():
        raise FileExistsError('缓存目录必须fresh: ' + str(output))
    heat, controls = read(args.heat_manifest), read(args.control_manifest)
    if (heat.get('complete') is not True or controls.get('complete') is not True
            or heat['local_restoration'] != dict(mode='zero', model='restorer')
            or controls['local_restoration'] != heat['local_restoration']):
        raise ValueError('只接受完整N1零噪声参考')
    if len(heat['files']) != 32 or {r['source'] for r in controls['files']} != {'test_101','test_116','test_166','test_169'}:
        raise ValueError('需要固定32训练源和4个已有monitor')
    if (heat['model_states']['restorer'] != controls['model_states']['restorer']
            or heat['models']['restorer']['sha256'] != controls['models']['restorer']['sha256']
            or file_sha(args.model) != heat['models']['restorer']['sha256']):
        raise RuntimeError('D5a权重/状态参考不同')
    source_proof = sealed_sources(code, heat['source_manifest']['manifest_sha256'])
    if code != Path(heat['source_root']).resolve() or controls['source_manifest'] != heat['source_manifest']:
        raise RuntimeError('参考N1的源码身份不同')
    projected = sum(int(r['shape'][0])*int(r['shape'][1])*12 + 512 for r in [*heat['files'],*controls['files']])
    if projected > MAX_OUTPUT or shutil.disk_usage(nearest_existing(output)).free-projected < MIN_FREE:
        raise RuntimeError('float32 cache将超出2.5GiB预算或低于3GiB保留空间')
    protected = {str(args.heat_manifest.resolve()):file_sha(args.heat_manifest),
                 str(args.control_manifest.resolve()):file_sha(args.control_manifest),
                 str(args.model.resolve()):file_sha(args.model)}
    ops = load_frozen_ops(code, args.device)
    runtime = heat['runtime']
    torch.set_num_threads(int(runtime['torch_threads'])); cv2.setNumThreads(int(runtime['opencv_threads']))
    torch.backends.cudnn.allow_tf32 = runtime['cudnn_tf32']
    torch.backends.cuda.matmul.allow_tf32 = runtime['matmul_tf32']
    torch.set_float32_matmul_precision(runtime['matmul_precision'])
    torch.backends.cudnn.benchmark = runtime['cudnn_benchmark']
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(runtime['deterministic_algorithms'])
    restorer = ops.load_restorer(str(args.model), args.device).eval().requires_grad_(False)
    expected_state = heat['model_states']['restorer']
    if tensor_state_sha(restorer) != expected_state or restorer.num_steps != 16 or restorer.kappa != .03:
        raise RuntimeError('实际D5a模型与N1不符')
    output.mkdir(parents=True); start = time.perf_counter()
    restoration = dict(model_file_sha256=file_sha(args.model), model_state_sha256=expected_state,
        noise_mode='zero',steps=16,output='first',native_size=1024,overlap=.25,
        cache_dtype='float32',quantization='none',blend='three-channel frozen BlendMap weighted mean')
    common = dict(format=FORMAT, complete=False, coordinate_contract=COORDINATES, restoration=restoration,
        provenance=dict(heat_manifest_sha256=protected[str(args.heat_manifest.resolve())],
            control_manifest_sha256=protected[str(args.control_manifest.resolve())],sealed_source=source_proof,
            cache_script_sha256=file_sha(__file__),original_runtime=runtime),
        purpose='N1 actual local D5a tensors stitched as conditioning RGB; not one historical full encoder input',
        no_backend_model_loaded=True,no_training=True,rows=[])
    manifests = {domain:dict(common, rows=[]) for domain in ('train','monitor_test')}
    done = 0
    with torch.inference_mode(), torch.autocast(device_type=torch.device(args.device).type, enabled=False):
        for domain, references in [('train',heat['files']),('monitor_test',controls['files'])]:
            folder=output/('train' if domain=='train' else 'monitor');folder.mkdir()
            for row in references:
                name=row['source']
                if not re.fullmatch(('train' if domain=='train' else 'test')+r'_\d{3}',name):
                    raise ValueError('源命名/训练与目检分组不符')
                if domain=='train':
                    image_path=Path(row['pool_match']['path'])
                    if file_sha(image_path)!=row['image_sha256']:
                        raise RuntimeError('训练原图字节改变: '+name)
                    rgb=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR)
                    if rgb is None:raise ValueError('原图无法解码')
                    rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB);windows=row['windows'];image_sha=row['image_sha256']
                    original_provenance=dict(image_path=str(image_path),image_file_verified=True)
                else:
                    if file_sha(row['path'])!=row['sha256']:raise RuntimeError('原monitor缓存改变')
                    with np.load(row['path'],allow_pickle=False) as arrays:rgb=arrays['rgb'].copy()
                    windows=row['local_windows'];image_sha=row['input_sha256']
                    original_provenance=dict(cache_path=row['path'],cache_sha256=row['sha256'],
                        image_file_verified=False,image_file_hash_from_original_N1_receipt=True)
                rgb=validate_rgb(rgb,row['shape'])
                value,proofs,blend=restore_native_rgb(rgb,windows,restorer,ops,device=args.device,repeat_first=done==0)
                saved=save_array(folder/(name+'.npy'),value)
                record=dict(source=name,domain=domain,image_sha256=image_sha,
                    original_rgb_array_sha256=array_sha(rgb),**saved,windows=proofs,blend_audit=blend,
                    original_provenance=original_provenance)
                manifests[domain]['rows'].append(record);done+=1
                write(output/('train_manifest.json' if domain=='train' else 'monitor_manifest.json'),manifests[domain])
                write(output/'progress.json',dict(completed=done,total=36,source=name,domain=domain,
                    free_gib=shutil.disk_usage(output).free/1024**3))
                print(json.dumps(dict(completed=done,total=36,source=name,windows=len(proofs))),flush=True)
                del value,rgb
                size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                if size>MAX_OUTPUT or shutil.disk_usage(output).free<MIN_FREE:
                    raise RuntimeError('缓存预算/快盘保留空间触发保护，保留现场')
    if tensor_state_sha(restorer)!=expected_state or any(file_sha(p)!=digest for p,digest in protected.items()):
        raise RuntimeError('生成过程中冻结资产改变')
    if sealed_sources(code,source_proof['manifest_sha256'])!=source_proof:
        raise RuntimeError('生成过程中封存源码改变')
    for domain, manifest in manifests.items():
        manifest['complete']=True
        write(output/('train_manifest.json' if domain=='train' else 'monitor_manifest.json'),manifest)
    complete=dict(complete=True,train_sources=32,monitor_sources=4,window_hashes_exact=True,
        restorer_unchanged=True,sealed_source_unchanged=True,no_backend_model_loaded=True,no_training=True,
        train_manifest_sha256=file_sha(output/'train_manifest.json'),monitor_manifest_sha256=file_sha(output/'monitor_manifest.json'),
        elapsed_seconds=time.perf_counter()-start,free_gib=shutil.disk_usage(output).free/1024**3)
    write(output/'complete.json',complete);print(json.dumps(complete),flush=True)
    return complete


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('code-root','heat-manifest','control-manifest','model','output-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    try:
        return cache(args)
    except Exception:
        if args.output_root.exists() and not (args.output_root/'complete.json').exists():
            write(args.output_root/'failure.json',dict(complete=False,traceback=traceback.format_exc(),no_training=True))
        raise


if __name__=='__main__':
    main()
