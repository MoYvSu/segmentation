# -*- coding: utf-8 -*-
"""构建 results/code_model 复赛包；默认只列计划，显式执行才导出模型或压缩。

只封装声明的推理状态和逐项 SHA 验证的源码，不复制训练缓存、优化器、
原图或用户环境。FP32 参数逐张量精确保留，不做量化或半精度转换。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.submission_stage import dataset_contract

NAMES = dataset_contract().names
MODEL_NAMES = dict(backend='backend.pt', restorer='restorer.pt', terminal_semantic='semantic_s64.pt')
LOCAL_RESTORER_NAME = 'local_restorer.pt'
SOURCE_DIRS = {'models', 'tools', 'utils', 'data', 'losses', 'segment-anything-2', 'config'}
NOISE_RESOURCES = {'utils/cuda/restore_noise.cu', 'utils/cuda/restore_noise.ptx'}
FORBIDDEN_PARTS = {'.git', '.github', '__pycache__', 'output', 'outputs', 'checkpoints',
                   'weights', 'cache', 'raw', 'test', 'unlabeled', 'node_modules'}
REQUIRED_PACKAGES = ('torch', 'torchvision', 'numpy', 'Pillow',
                     'scipy', 'scikit-image', 'PyYAML', 'hydra-core', 'omegaconf',
                     'iopath', 'tqdm', 'scikit-learn', 'joblib', 'matplotlib')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def safe_relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and '\x00' not in name, 'Unsafe relative path: ' + repr(name))
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')),
            'Unsafe relative path: ' + name)
    return path


def no_symlink(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink not allowed: ' + str(path))
    return path.resolve()


def state_digest(state):
    """与 train_backend_adaptation.tensor_digest 相同，另外检查 dtype 和 shape。"""
    import torch
    require(isinstance(state, dict) and state, 'Missing nonempty tensor state dictionary')
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(name, str) and torch.is_tensor(value), 'State must contain named tensors only')
        require(value.layout == torch.strided, 'Unsupported sparse/quantized tensor: ' + name)
        if value.is_floating_point():
            require(value.dtype == torch.float32, 'Inference export requires unchanged FP32: ' + name)
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def exact_state(original, exported):
    import torch
    require(set(original) == set(exported), 'Exported tensor keys differ')
    for name, value in original.items():
        copied = exported[name]
        require(value.dtype == copied.dtype and value.shape == copied.shape
                and torch.equal(value.detach().cpu(), copied.detach().cpu()),
                'Exported tensor dtype/shape/value differs: ' + name)
    require(state_digest(original) == state_digest(exported), 'Exported tensor digest differs')


def export_checkpoint(source, destination, kind, *, base_sha=None, base_head_sha=None):
    """受信的本地 checkpoint -> 最小严格可加载格式；重载后核全部数值。"""
    import torch
    source, destination = no_symlink(source), no_symlink(destination)
    require(source.is_file() and not destination.exists(), 'Source missing or export already exists')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    require(isinstance(payload, dict), 'Checkpoint must be an object')
    original_sha = sha(source)
    if kind == 'backend':
        require(payload.get('format') == 'phase_affinity_fused_v1', 'Expected complete fused backend')
        architecture = deepcopy(payload['architecture'])
        require(architecture['sam2'].get('input_normalization', 'legacy_none') == 'legacy_none',
                'Unexpected backend normalization')
        architecture['sam2']['sam2_repo_path'] = 'segment-anything-2'
        require(not Path(architecture['sam2']['config_file']).is_absolute()
                and ':' not in architecture['sam2']['config_file'], 'SAM2 config must be package-relative')
        state_key = 'model_state_dict'
        result = dict(format=payload['format'], architecture=architecture,
                      epoch=int(payload.get('epoch', -1)), model_state_dict=payload[state_key],
                      source_checkpoint_sha256=original_sha)
    elif kind == 'restorer':
        require(payload.get('format') == 'rgb_restoration_diffusion_v1', 'Expected D5a diffusion checkpoint')
        require(payload['model_config'].get('architecture') == 'pixel_diffusion_nafnet',
                'Unexpected D5a architecture')
        state_key = 'model'
        result = dict(format=payload['format'], model_config=deepcopy(payload['model_config']),
                      model=payload[state_key], source_checkpoint_sha256=original_sha)
    elif kind == 'terminal_semantic':
        require(payload.get('format') == 'semantic_grain_terminal_only_v1'
                and payload.get('terminal_only') is True
                and payload.get('do_not_replace_geometry_upstream') is True
                and payload.get('complete') is True and payload.get('step') == 64,
                'Expected completed terminal-only s64 checkpoint')
        require(re.fullmatch('[0-9a-f]{64}', str(base_sha)) is not None
                and re.fullmatch('[0-9a-f]{64}', str(base_head_sha)) is not None,
                'Original base checkpoint and head digests required')
        require(payload.get('identity', {}).get('base_checkpoint_sha256') == base_sha
                and payload['identity'].get('initial_head_sha256') == base_head_sha,
                'Terminal checkpoint original base/head identity differs')
        state_key = 'semantic_state_dict'
        result = dict(format='semantic_grain_inference_v1', architecture=deepcopy(payload['architecture']),
                      head_state_dict=payload[state_key], steps=64, terminal_only=True,
                      do_not_replace_geometry_upstream=True, base_checkpoint_sha256=base_sha,
                      base_head_sha256=base_head_sha, source_checkpoint_sha256=original_sha)
    else:
        raise ValueError('Unknown checkpoint kind: ' + str(kind))
    original = payload[state_key]
    original_digest = state_digest(original)
    export_key = 'head_state_dict' if kind == 'terminal_semantic' else state_key
    result[export_key] = {name: value.detach().cpu().contiguous().clone() for name, value in original.items()}
    result['state_sha256'] = original_digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    replay = torch.load(destination, map_location='cpu', weights_only=True)
    exact_state(original, replay[export_key])
    require(set(replay) == set(result), 'Export checkpoint schema replay differs')
    return dict(kind=kind, file=destination.name, original_sha256=original_sha, sha256=sha(destination),
                original_bytes=source.stat().st_size, bytes=destination.stat().st_size,
                state_sha256=original_digest, tensor_count=len(original),
                tensor_elements=sum(value.numel() for value in original.values()),
                fp32_exact=True, optimizer_rng_training_records_removed=True)


def build_models(code_model, backend, restorer, terminal=None, *, local_restorer=None):
    import torch
    code_model = no_symlink(code_model)
    require(not (code_model / 'inference_export.json').exists(), 'Model export receipt already exists')
    backend_payload = torch.load(no_symlink(backend), map_location='cpu', weights_only=False)
    backend_state = backend_payload['model_state_dict']
    prefix = 'semantic_decoder.'
    base_head = {key[len(prefix):]: value for key, value in backend_state.items() if key.startswith(prefix)}
    base_head_sha = state_digest(base_head)
    original_backend_sha = sha(backend)
    if terminal is not None:
        head_payload = torch.load(no_symlink(terminal), map_location='cpu', weights_only=False)
        require(head_payload['architecture'] == backend_payload['architecture']['semantic_decoder'],
                'Terminal architecture differs from original backend semantic decoder')
        require(set(head_payload['semantic_state_dict']) == set(base_head)
                and all(head_payload['semantic_state_dict'][key].shape == value.shape
                        for key, value in base_head.items()), 'Terminal head shape/key ownership differs')
    del backend_payload, backend_state, base_head
    model_dir = code_model / 'model'
    rows = [export_checkpoint(backend, model_dir / MODEL_NAMES['backend'], 'backend'),
            export_checkpoint(restorer, model_dir / MODEL_NAMES['restorer'], 'restorer')]
    if terminal is not None:
        rows.append(export_checkpoint(terminal, model_dir / MODEL_NAMES['terminal_semantic'], 'terminal_semantic',
                                      base_sha=original_backend_sha, base_head_sha=base_head_sha))
    models = {row['kind']: row for row in rows}
    extra_restorer_elements = 0
    if local_restorer is not None:
        # 路径职责是 local_restorer，checkpoint 的严格加载格式仍是 restorer。
        local_row = export_checkpoint(local_restorer, model_dir / LOCAL_RESTORER_NAME, 'restorer')
        models['local_restorer'] = local_row
        # 包含 buffer 的状态元素总数是保守上界，避免遗漏第二份恢复器。
        extra_restorer_elements = local_row['tensor_elements']
    conservative_total = (91253055 if terminal else 84951045) + extra_restorer_elements
    require(conservative_total < 500_000_000, 'Model parameter constraint failed')
    receipt = dict(format='semifinal_model_export_v1', complete=True, floating_precision='FP32',
                   models=models, original_base_head_sha256=base_head_sha,
                   parameter_ledger=dict(fixed_deployment=84951045, additional_terminal=6302010 if terminal else 0,
                                         additional_local_restorer=extra_restorer_elements,
                                         conservative_total=conservative_total,
                                         constraint='total < 500000000',
                                         basis='Measured deployment models; terminal head counted separately; '
                                               'optional local restorer includes all tensor-state elements'))
    write_json(code_model / 'inference_export.json', receipt)
    return receipt


def source_rows(manifest):
    rows = manifest.get('files')
    require(isinstance(rows, list) and rows, 'Explicit nonempty runtime source SHA manifest required')
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Runtime source SHA256 required')
        path = safe_relative(row.get('path'))
        require(str(path) not in seen, 'Duplicate source manifest path')
        seen.add(str(path))
        require(not set(path.parts) & FORBIDDEN_PARTS and not path.name.endswith('_class.json'),
                'Artifact/cache/git source forbidden: ' + str(path))
        require((len(path.parts) == 1 and path.suffix == '.py') or path.parts[0] in SOURCE_DIRS,
                'Runtime source outside code directories: ' + str(path))
        require(path.suffix in {'.py', '.yaml', '.yml', '.json', '.txt'}
                or str(path) in NOISE_RESOURCES
                or (path.parts[0] == 'segment-anything-2' and path.name.startswith('LICENSE')),
                'Unexpected binary/non-runtime source: ' + str(path))
    require('tools/infer_semifinal.py' in seen, 'Real standalone inference module missing')
    require(any(path.startswith('segment-anything-2/sam2/') and path.endswith('.py') for path in seen)
            and any(path.startswith('segment-anything-2/sam2/') and path.endswith(('.yaml', '.yml')) for path in seen),
            'SAM2 source and packaged Hydra configs missing')
    if 'utils/restoration_noise.py' in seen:
        require(NOISE_RESOURCES <= seen,
                'Versioned restoration noise requires both CUDA source and PTX resources')
    return rows


def copy_runtime_sources(source_root, manifest, code_model):
    source_root, code_model = no_symlink(source_root), no_symlink(code_model)
    rows = source_rows(manifest)
    inference_source = source_root / 'tools/infer_semifinal.py'
    if re.search(r'^\s*(?:from\s+utils\.submission_stage\s+import\b|import\s+utils\.submission_stage\b)',
                 inference_source.read_text(encoding='utf8'), flags=re.MULTILINE):
        require('utils/submission_stage.py' in {row['path'] for row in rows},
                'Runtime inference requires utils/submission_stage.py in the source manifest')
    for row in rows:
        relative = safe_relative(row['path'])
        requested = source_root / str(relative)
        if relative.parts[0] == 'segment-anything-2':
            # 服务器 SAM2 常是已存在的代码目录软链接；只读其实名源树并复制普通文件。
            # 最终 code_model/ 和 ZIP 仍禁止任何软链接。
            sam2_root = (source_root / 'segment-anything-2').resolve()
            source = requested.resolve()
            allowed = source.is_relative_to(sam2_root)
        else:
            source = no_symlink(requested)
            allowed = source.is_relative_to(source_root)
        require(allowed and source.is_file() and sha(source) == row['sha256'],
                'Runtime source SHA/source root differs: ' + row['path'])
        if row['path'] == 'config/inference/submission_reflect.yaml':
            # 此模板按模型导出 SHA 生成，不把占位符配置原样放入正式工程。
            continue
        destination = code_model / row['path']
        require(not destination.exists(), 'Runtime destination already exists: ' + row['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(sha(destination) == row['sha256'], 'Copied runtime source SHA differs')
    return rows


def pinned_requirements(environment):
    packages = environment.get('packages', {})
    require(isinstance(packages, dict) and all(name in packages for name in REQUIRED_PACKAGES),
            'Verified precise runtime package versions are incomplete')
    opencv = [name for name in ('opencv-python', 'opencv-contrib-python') if name in packages]
    require(len(opencv) == 1, 'Exactly one measured OpenCV distribution required')
    require(re.fullmatch(r'3\.\d+\.\d+', str(environment.get('python'))) is not None,
            'Exact measured Python version required')
    names = (*REQUIRED_PACKAGES, *opencv)
    for name in names:
        require(re.fullmatch(r'[0-9][A-Za-z0-9.+!_-]*', str(packages[name])) is not None,
                'Unsafe or imprecise package version: ' + name)
    return '# Python ' + environment['python'] + '\n' + '\n'.join(
        name + '==' + str(packages[name]) for name in names) + '\n'


def local_restoration_policy(config, models):
    """只核对路径职责与文件存在，不改变推理模式或训练配方。"""
    policy = config.get('submission', {}).get('local_restoration', {'mode': 'single', 'model': 'restorer'})
    require(isinstance(policy, dict), 'Local restoration policy must be an object')
    require(set(policy) == {'mode', 'model'}, 'Expected explicit local_restoration mode/model only')
    mode, model = policy['mode'], policy['model']
    require(mode in ('single', 'zero'), 'Unknown local restoration mode')
    require(model in ('restorer', 'local_restorer'), 'Unknown local restoration model role')
    require((mode, model) in (('single', 'restorer'), ('zero', 'restorer'), ('zero', 'local_restorer')),
            'Expected N0 single/shared, N1 zero/shared, or N2 zero/local')
    require(bool(models.get(model)), 'Referenced local restoration model was not exported: ' + model)
    if model == 'restorer':
        require(not models.get('local_restorer'), 'Unused local_restorer model would obscure candidate identity')
    return mode, model


def build_code(code_model, source_root, runtime_manifest, template_config, environment):
    import yaml
    code_model = no_symlink(code_model)
    require(not (code_model / 'manifest.json').exists(), 'Code-model package already sealed')
    receipt = read_json(code_model / 'inference_export.json')
    require(receipt.get('complete') is True and receipt.get('format') == 'semifinal_model_export_v1',
            'Complete actual model export required before code sealing')
    environment = {key: environment[key] for key in ('python', 'packages', 'cuda', 'cudnn') if key in environment}
    # 仅列公开依赖版本，避免把调用者提供的无关环境变量/凭据写入提交工程。
    environment['packages'] = {key: value for key, value in environment.get('packages', {}).items()
                               if key in {*REQUIRED_PACKAGES, 'opencv-python', 'opencv-contrib-python'}}
    pinned = pinned_requirements(environment)
    config = yaml.safe_load(Path(template_config).read_text(encoding='utf8'))
    require(isinstance(config, dict) and '_base' not in config, 'Standalone resolved config required')
    contract = dataset_contract(config.get('submission', {}).get('dataset'))
    config.setdefault('paths', {})['project_root'] = 'auto'
    require(config.get('sam2', {}).get('sam2_repo_path') == 'segment-anything-2',
            'Portable SAM2 code path required')
    models = {}
    for kind in ('backend', 'restorer', 'terminal_semantic'):
        row = receipt['models'].get(kind)
        models[kind] = dict(file=row['file'], sha256=row['sha256']) if row else None
    local_row = receipt['models'].get('local_restorer')
    if local_row is not None:
        models['local_restorer'] = dict(file=local_row['file'], sha256=local_row['sha256'])
    config.setdefault('submission', {})['models'] = models
    local_mode, local_model = local_restoration_policy(config, models)
    rows = copy_runtime_sources(source_root, runtime_manifest, code_model)
    config_path = code_model / 'config/inference/submission_reflect.yaml'
    require(not config_path.exists(), 'Generated portable config collides with runtime manifest')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf8')
    launcher = '# -*- coding: utf-8 -*-\n"""复赛独立推理入口。"""\nfrom tools.infer_semifinal import main\n\nif __name__ == "__main__":\n    main()\n'
    (code_model / 'infer.py').write_text(launcher, encoding='utf8')
    (code_model / 'requirements.txt').write_text(pinned, encoding='utf8')
    torch_version = str(environment['packages']['torch'])
    cuda = str(environment.get('cuda', '请见实测环境回执'))
    stage_title = '半决赛' if contract.name == 'semifinal_150' else '复赛'
    example_names = (contract.names[0], contract.names[-1])
    readme = f'''# {stage_title}推理工程

团队名、团队编号和队长姓名：待参赛者填写。

本目录包含实际推理模型与源码。评测结果在压缩包的 `results/`，
评测程序只读取该目录；本工程供复核重现，不依赖原项目目录或训练缓存。

## 环境

实测 Python {environment['python']}，PyTorch {torch_version}，CUDA {cuda}。
请建立相同 Python 环境后执行 `pip install -r requirements.txt`。
若 PyTorch 带 CUDA 构建标记，请从对应官方 CUDA wheel 索引安装相同版本，
不要将 CPU 构建当作 GPU 复现环境。SAM2 源码及 Hydra 配置已包含在
`segment-anything-2/`，入口按相对路径导入，不下载用户缓存中的权重。

正式 reflect 配置显式采用 `cuda_philox4090_v1`：固定恢复噪声与像素的对应关系，
避免随机种子相同但显卡核心数量不同造成不同输入。随包 PTX 的最低目标为
compute_70（Volta 及更新架构），需要支持 CUDA 12.8 的 NVIDIA 驱动；
运行时由驱动加载，无需 NVCC、NVRTC 或 C++ 编译器。
普通训练入口保留历史噪声方式，噪声策略不改变模型权重。

## 模型与流程

- `model/backend.pt`：固定 R1 的完整 encoder/LoRA、语义与 affinity 状态。
- `model/restorer.pt`：D5a 首步 RGB 恢复器，保留原 FP32 权重。
'''
    if models['terminal_semantic']:
        readme += '- `model/semantic_s64.pt`：64 次更新的末端语义头，只改变最终实例类别投票，几何仍使用固定 R1/reflect 路径。\n'
    if local_row is not None:
        readme += '- `model/local_restorer.pt`：可供局部 affinity 路径使用的独立恢复器；全图语义仍使用 `model/restorer.pt`。\n'
    readme += f'''
局部 affinity 恢复策略为 `{local_mode}`，使用 `{local_model}` 模型；
`single` 保留规定的首步注入噪声，`zero` 使用零注入噪声。
该局部策略不改变全图语义路径的恢复模型与模式。

所有模型从原 checkpoint 去掉优化器、随机状态和训练记录后导出，未做
半精度/量化；逐张量数值、dtype、shape 和状态摘要均已精确核对。
保守总参数量为 {receipt['parameter_ledger']['conservative_total']:,}，
小于 500M。来源与导出 SHA256、模型状态摘要见 `inference_export.json`。
不加载 SAM2 原生 Mask Decoder 权重，不使用测试 GT。

## 推理

在本 `code_model/` 目录内执行：

```bash
python infer.py --input-dir /path/to/test_images --output-dir /path/to/reproduced --config config/inference/submission_reflect.yaml
```

本包数据阶段为 `{contract.name}`。输入为 `test_{contract.first:03d}` 至
`test_{contract.last:03d}` 的 {contract.count} 张原图，文件名保留三位编号。
输出的 `results/` 包含原尺寸、单通道 uint16 的实例 PNG 和 0/1 类别 JSON；
`audit/` 保存来源、权重、精度、随机种子及输出检查回执，不参与评分。
可使用 `--only {example_names[0]} {example_names[1]}` 做小样本复现，阶段内原始编号所确定的种子不变。
正式复现仍须跑完全部 {contract.count} 图。输出目录与输入目录应分开。

`manifest.json` 对推理代码、配置、依赖和模型逐项登记 SHA256。
当前官方黑盒分数及独立复现证据由参赛者在选定最终版本后补充，
本说明不将训练指标或目检结果描述为官方成绩。
'''
    (code_model / 'README.md').write_text(readme, encoding='utf8')
    write_json(code_model / 'environment.json', environment)
    files = []
    for path in sorted(code_model.rglob('*')):
        no_symlink(path)
        if path.is_file():
            files.append(dict(path=path.relative_to(code_model).as_posix(), sha256=sha(path), bytes=path.stat().st_size))
    result = dict(format='semifinal_code_model_manifest_v1', complete=True, files=files,
                  runtime_source_count=len(rows), floating_precision='FP32')
    write_json(code_model / 'manifest.json', result)
    return result


def validate_results(results, test_dir, dataset=None):
    import cv2
    import numpy as np
    from utils.image_io import read_image
    contract = dataset_contract(dataset)
    names = contract.names
    cohort = f'test_{contract.first:03d}..{contract.last:03d}'
    results, test_dir = no_symlink(results), no_symlink(test_dir)
    expected = {name + suffix for name in names for suffix in ('_inst.png', '_class.json')}
    actual = list(results.iterdir())
    require(len(actual) == 2 * contract.count and {path.name for path in actual} == expected
            and all(path.is_file() for path in actual),
            f'Results require exactly {contract.count} pairs {cohort}')
    originals = {}
    for path in test_dir.iterdir():
        no_symlink(path)
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}:
            require(path.stem not in originals, 'Duplicate original test image stem')
            originals[path.stem] = path
    require(set(originals) == set(names), 'Original test cohort must be exactly ' + cohort)
    rows = []
    for name in names:
        original = read_image(originals[name], cv2.IMREAD_UNCHANGED)
        require(original is not None and original.ndim in (2, 3), 'Unreadable original image: ' + name)
        png, labels = no_symlink(results / (name + '_inst.png')), no_symlink(results / (name + '_class.json'))
        data = png.read_bytes()
        require(len(data) >= 33 and data[:8] == b'\x89PNG\r\n\x1a\n'
                and data[8:16] == b'\x00\x00\x00\rIHDR' and data[24:26] == bytes((16, 0)),
                '16-bit single-channel PNG IHDR required: ' + name)
        width, height = struct.unpack('>II', data[16:24])
        ids = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        require(ids is not None and ids.ndim == 2 and ids.dtype == np.uint16
                and ids.shape == original.shape[:2] and (height, width) == ids.shape,
                'uint16/original shape contract failed: ' + name)
        classes = read_json(labels)
        require(isinstance(classes, dict) and all(isinstance(key, str) and key.isascii() and key.isdecimal()
                and str(int(key)) == key and 1 <= int(key) <= 65535 for key in classes),
                'Canonical positive instance IDs required: ' + name)
        require(all(type(value) is int and value in (0, 1) for value in classes.values()),
                'Integer class 0/1 only: ' + name)
        require(set(classes) == {str(int(value)) for value in np.unique(ids) if value > 0},
                'PNG/JSON positive instance IDs differ: ' + name)
        rows.append(dict(source=name, shape=list(ids.shape), png_sha256=sha(png),
                         classes_sha256=sha(labels), instances=len(classes)))
    return rows


def validate_code_model(code_model):
    import yaml
    code_model = no_symlink(code_model)
    manifest = read_json(code_model / 'manifest.json')
    require(manifest.get('format') == 'semifinal_code_model_manifest_v1' and manifest.get('complete') is True,
            'Complete code/model source seal required')
    rows = manifest.get('files')
    require(isinstance(rows, list) and rows, 'Empty code/model manifest')
    recorded = set()
    for row in rows:
        relative = str(safe_relative(row['path']))
        require(relative not in recorded, 'Duplicate packaged code/model path')
        recorded.add(relative)
        path = no_symlink(code_model / relative)
        require(path.is_file() and path.is_relative_to(code_model) and sha(path) == row['sha256'],
                'Code/model sealed SHA differs: ' + relative)
    actual = set()
    for path in code_model.rglob('*'):
        no_symlink(path)
        if path.is_file():
            actual.add(path.relative_to(code_model).as_posix())
    require(actual == recorded | {'manifest.json'}, 'Extra or missing code/model files')
    require({'infer.py', 'README.md', 'requirements.txt', 'environment.json', 'inference_export.json',
             'tools/infer_semifinal.py', 'config/inference/submission_reflect.yaml',
             'model/backend.pt', 'model/restorer.pt'} <= actual, 'Standalone inference/code/model contract incomplete')
    receipt = read_json(code_model / 'inference_export.json')
    require(receipt.get('complete') is True and receipt['parameter_ledger']['conservative_total'] < 500_000_000,
            'Missing real export or model parameter constraint failed')
    config = yaml.safe_load((code_model / 'config/inference/submission_reflect.yaml').read_text(encoding='utf8'))
    require(config['paths']['project_root'] == 'auto' and '_base' not in config, 'Config not independently portable')
    dataset_contract(config.get('submission', {}).get('dataset'))
    require({'backend', 'restorer'} <= set(receipt['models']), 'Required exported model roles missing')
    require(set(receipt['models']) <= {*MODEL_NAMES, 'local_restorer'}, 'Unknown packaged model role')
    require(set(config['submission']['models']) <= {*MODEL_NAMES, 'local_restorer'}, 'Unknown runtime model role')
    require({key for key, value in config['submission']['models'].items() if value is not None}
            == set(receipt['models']), 'Runtime and exported model roles differ')
    local_restoration_policy(config, config['submission']['models'])
    for kind, row in receipt['models'].items():
        filename = safe_relative(row['file'])
        require(len(filename.parts) == 1, 'Model file must be directly inside model/')
        require(row['kind'] == ('restorer' if kind == 'local_restorer' else kind),
                'Checkpoint format kind differs from model role')
        require(row['fp32_exact'] is True and row['optimizer_rng_training_records_removed'] is True,
                'Checkpoint export tensor verification missing')
        require(config['submission']['models'][kind] == dict(file=row['file'], sha256=row['sha256'])
                and sha(code_model / 'model' / row['file']) == row['sha256'], 'Runtime model identity differs')
    return manifest


def pack_submission(submission_dir, archive_path, test_dir, *, compression_level=9, dataset=None):
    import yaml
    submission_dir, archive_path = no_symlink(submission_dir), no_symlink(archive_path)
    require(compression_level in (6, 9), 'Use measured DEFLATE level 6 or 9')
    require(archive_path.suffix == '.zip' and not archive_path.exists()
            and not archive_path.is_relative_to(submission_dir), 'Archive must be new and outside submission tree')
    require({path.name for path in submission_dir.iterdir()} == {'results', 'code_model'},
            'Submission tree must contain only results/ and code_model/')
    validate_code_model(submission_dir / 'code_model')
    packaged_config = yaml.safe_load((submission_dir / 'code_model/config/inference/submission_reflect.yaml')
                                     .read_text(encoding='utf8'))
    contract = dataset_contract(packaged_config.get('submission', {}).get('dataset'))
    if dataset is not None:
        require(dataset_contract(dataset) == contract, 'Requested dataset differs from sealed inference config')
    predictions = validate_results(submission_dir / 'results', test_dir, dataset=contract.name)
    paths = []
    for path in sorted(submission_dir.rglob('*')):
        no_symlink(path)
        if path.is_file():
            relative = str(safe_relative(path.relative_to(submission_dir).as_posix()))
            require(PurePosixPath(relative).parts[0] in {'results', 'code_model'}, 'Invalid submission path')
            paths.append((path, relative))
    require(len(paths) == len({relative for _, relative in paths}), 'Duplicate ZIP member')
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=compression_level, allowZip64=True) as archive:
        for path, relative in paths:
            archive.write(path, relative)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.namelist() == [relative for _, relative in paths] and archive.testzip() is None,
                'ZIP entry or CRC verification failed')
        for path, relative in paths:
            require(hashlib.sha256(archive.read(relative)).hexdigest() == sha(path), 'ZIP member SHA differs')
    size = archive_path.stat().st_size
    return dict(format='semifinal_complete_submission_v1', complete=True, compression='DEFLATE',
                compression_level=compression_level, seconds=time.monotonic() - started,
                archive_bytes=size, archive_MiB=size / 1024 ** 2, archive_sha256=sha(archive_path),
                uncompressed_bytes=sum(path.stat().st_size for path, _ in paths), files=len(paths),
                result_pairs=len(predictions), dataset=contract.name, results=predictions,
                zip_crc_ok=True, all_member_sha_verified=True,
                size_comparison=[dict(limit_MiB=limit, fits=size <= limit * 1024 ** 2)
                                 for limit in (50, 100, 200, 500)],
                size_limits_verified=False, upload_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs/submission_grain')
    parser.add_argument('--backend-checkpoint')
    parser.add_argument('--restorer-checkpoint')
    parser.add_argument('--local-restorer-checkpoint', help='可选，仅局部 affinity 使用的独立恢复器')
    parser.add_argument('--terminal-checkpoint')
    parser.add_argument('--source-root', default=str(ROOT))
    parser.add_argument('--runtime-manifest')
    parser.add_argument('--config', default='config/inference/submission_reflect.yaml')
    parser.add_argument('--environment')
    parser.add_argument('--results-dir')
    parser.add_argument('--test-dir')
    parser.add_argument('--build-models', action='store_true')
    parser.add_argument('--build-code', action='store_true')
    parser.add_argument('--build', action='store_true', help='同时导出模型并封装代码')
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--archive')
    parser.add_argument('--compression-level', type=int, choices=(6, 9), default=9)
    args = parser.parse_args()
    import yaml
    template = yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    require(isinstance(template, dict), 'Submission config must be an object')
    contract = dataset_contract(template.get('submission', {}).get('dataset'))
    output = no_symlink(args.output_dir)
    require(any(output.is_relative_to(ROOT / name) for name in ('output', 'outputs')),
            'Real competition bundle must remain in ignored output/ or outputs/')
    submission, code_model = output / 'submission', output / 'submission/code_model'
    if not any((args.build, args.build_models, args.build_code, args.pack)):
        print(json.dumps(dict(status='plan_only', output=str(output), format='results/+code_model/',
                              models='Inference-only exact FP32; no optimizer/RNG/cache/pretrained duplicate',
                              stages=['--build-models', '--build-code', '--pack'],
                              expected_test_names=f'test_{contract.first:03d}..{contract.last:03d}',
                              dataset=contract.name, uploads=False), ensure_ascii=False, indent=2))
        return
    results = {}
    if args.build or args.build_models:
        require(args.backend_checkpoint and args.restorer_checkpoint, 'Both real backend and D5a checkpoints required')
        results['models'] = build_models(code_model, args.backend_checkpoint, args.restorer_checkpoint,
                                        args.terminal_checkpoint, local_restorer=args.local_restorer_checkpoint)
    if args.build or args.build_code:
        require(args.runtime_manifest and args.environment, 'Runtime source SHA manifest and measured environment required')
        results['code'] = build_code(code_model, args.source_root, read_json(args.runtime_manifest),
                                    args.config, read_json(args.environment))
    if args.pack:
        require(args.test_dir, 'Original test directory required for exact shape checks')
        packaged_config = yaml.safe_load((code_model / 'config/inference/submission_reflect.yaml').read_text(encoding='utf8'))
        package_contract = dataset_contract(packaged_config.get('submission', {}).get('dataset'))
        if args.results_dir:
            require(not (submission / 'results').exists(), 'Packaged results already exist')
            validate_results(args.results_dir, args.test_dir, dataset=package_contract.name)
            shutil.copytree(no_symlink(args.results_dir), submission / 'results')
        archive = args.archive or str(output / '复赛结果_待填写.zip')
        results['archive'] = pack_submission(submission, archive, args.test_dir,
                                             compression_level=args.compression_level)
        write_json(output / ('package_level' + str(args.compression_level) + '.json'), results['archive'])
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
