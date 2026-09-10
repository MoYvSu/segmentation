# -*- coding: utf-8 -*-
"""本轮远端源码包，不打包数据和第三方权重。"""
from pathlib import Path
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parents[1]
files = {root / "train_mask_set.py", root / "train_direct_semantic_affinity.py"}
for directory in ("models", "utils", "data"):
    files.update((root / directory).glob("*.py"))
for suffix in ("*.yaml", "*.json", "*.txt"):
    files.update((root / "config").rglob(suffix))
files.update((root / "tests").glob("test_mask_set*.py"))
files.update((root / "tools").glob("evaluate_mask_set.py"))
files.update(root / name for name in ("AGENTS.md", "requirements.txt", "COMPETITION_RULES.md"))
destination = root / "tmp" / "mask_set_source.tar.gz"
manifest = {}
with tarfile.open(destination, "w:gz") as archive:
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        archive.add(path, arcname=relative, recursive=False)
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "tmp" / "mask_set_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"archive": str(destination), "files": len(files), "bytes": destination.stat().st_size}))
