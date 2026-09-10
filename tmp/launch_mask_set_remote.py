# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

root = Path('/root/autodl-tmp/segmentationv2_maskset_20260910')
archive_path = Path('/root/autodl-tmp/mask_set_source_final.tar.gz')
manifest_path = Path('/root/autodl-tmp/mask_set_source_final_manifest.json')
# 初次失败probe的源码也保留，避免后续初始化修正使历史状态不可追溯。
original_archive = Path('/root/autodl-tmp/mask_set_source.tar.gz')
old_snapshot = root / 'outputs/20260910_mask_set_overfit_v1/source_snapshot.tar.gz'
if not old_snapshot.exists():
    shutil.copy2(original_archive, old_snapshot)
with tarfile.open(archive_path) as archive:
    for member in archive.getmembers():
        if not (root / member.name).resolve().is_relative_to(root.resolve()):
            raise ValueError(member.name)
    archive.extractall(root, filter='data')
manifest = json.loads(manifest_path.read_text())
for relative, expected in manifest.items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(relative)
(root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
stamp = datetime.now(ZoneInfo('Asia/Shanghai'))
relative_output = 'outputs/' + stamp.strftime('%Y%m%d_%H%M%S') + '_maskset'
output = root / relative_output
output.mkdir(exist_ok=False)
shutil.copy2(archive_path, output / 'source_snapshot.tar.gz')
shutil.copy2(manifest_path, output / 'source_manifest.json')
command = ['/root/miniconda3/envs/sam2_env/bin/python', '-u', 'train_mask_set.py',
           '--config', 'config/train/mask_set_clean60.yaml', '--output-dir', relative_output]
environment = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
with (output / 'run.log').open('xb') as log:
    process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = {'pid': process.pid, 'started_at': stamp.isoformat(), 'project_root': str(root),
          'output_dir': str(output), 'command': command, 'branch': 'codex/maskset-clean60',
          'base_head': '2dbc9bf', 'source_state': 'uncommitted; exact files in source_manifest.json',
          'ssl_sha256': 'e5b837fce1707a682d4a149645dd3a210556f737b8f5778d9d4c8b275a34cb49',
          'stages': {'head_warmup': 60, 'joint_lora': 60},
          'selection': 'minimum deterministic validation loss; restore warmup best',
          'initialization': 'same existing SSL LoRA; fresh random task decoder',
          'monitoring': 'none; user will request comparison after completion'}
(output / 'launch.json').write_text(json.dumps(record, indent=2))
(root / 'launch_mask_set.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
