# -*- coding: utf-8 -*-
from pathlib import Path
import hashlib
import json
import tarfile

base = Path('/root/autodl-tmp/segmentationv2_clean60_20260908_133740')
root = Path('/root/autodl-tmp/segmentationv2_maskset_20260910')
root.mkdir(exist_ok=False)
with tarfile.open('/root/autodl-tmp/mask_set_source.tar.gz') as archive:
    for member in archive.getmembers():
        if not (root / member.name).resolve().is_relative_to(root.resolve()):
            raise ValueError(member.name)
    archive.extractall(root, filter='data')
(root / 'outputs').mkdir()
links = {
    'weights': base / 'weights',
    'segment-anything-2': base / 'segment-anything-2',
    'data/raw': base / 'data/raw',
    'outputs/experiments': base / 'outputs/experiments',
    'outputs/20260908_133740_ssl60': base / 'outputs/20260908_133740_ssl60',
}
for relative, source in links.items():
    if not source.exists():
        raise FileNotFoundError(source)
    (root / relative).symlink_to(source.resolve(), target_is_directory=True)
manifest = json.loads(Path('/root/autodl-tmp/mask_set_source_manifest.json').read_text())
for relative, digest in manifest.items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
        raise RuntimeError(relative)
(root / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
print(root)
