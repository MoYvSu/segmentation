# -*- coding: utf-8 -*-
"""阻止赛题产物进入暂存提交或待推送历史；安装副本跨本地分支生效。"""
import os
import re
import subprocess
import sys
from pathlib import Path

MARKER = "segmentationv2-confidential-artifacts-v1"
PRIVATE_DIRS = (
    "output", "outputs", "weights", "downloads", "logs",
    "data/raw", "data/test", "data/unlabeled", "data/purified_gt",
    "data/purified_gt_uncovered", "data/sam2_geometry", "data/sam2_geometry_g2",
    "tmp/musam_oracle_smoke",
)
PRIVATE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
    ".npz", ".npy", ".pth", ".pt", ".ckpt", ".zip", ".7z", ".tar", ".gz",
    "_class.json", "_class_confidence.json",
)
EMBEDDED_IMAGE = re.compile(
    rb'data:image/[a-z0-9.+-]+;base64,[a-z0-9+/]{32,}|"imageData"\s*:\s*"[a-z0-9+/]{32,}',
    re.IGNORECASE,
)


def git(*args, input=None):
    return subprocess.check_output(["git", *args], input=input)


def private_path(path):
    path = path.replace("\\", "/").lower()
    return (
        any(path == d or path.startswith(d + "/") for d in PRIVATE_DIRS)
        or any(part in ("output", "outputs") for part in path.split("/"))
        or path.endswith(PRIVATE_SUFFIXES)
        or (path.startswith(".claude/settings") and path.endswith(".json"))
    )


def fail(items):
    print("保密检查拒绝：发现数据/产物或禁止推送的内部引用。不要绕过检查。", file=sys.stderr)
    for item in sorted(set(items))[:15]:
        print("  " + item, file=sys.stderr)
    return 1


def check_objects(objects):
    blocked = [path for _, path in objects if private_path(path)]
    if blocked:
        return fail(blocked)
    # 同时检查内容，拦截改成普通文本扩展名的图片/数组/压缩包及内嵌原图。
    identities = list(dict.fromkeys(oid for oid, _ in objects))
    if not identities:
        return 0
    metadata = git("cat-file", "--batch-check=%(objectname) %(objecttype)",
                   input=("\n".join(identities) + "\n").encode()).splitlines()
    blobs = [line.split()[0].decode() for line in metadata if line.endswith(b" blob")]
    if not blobs:
        return 0
    names = {oid: path for oid, path in objects}
    stream = git("cat-file", "--batch", input=("\n".join(blobs) + "\n").encode())
    offset = 0
    for oid in blobs:
        end = stream.index(b"\n", offset)
        size = int(stream[offset:end].split()[2])
        data = stream[end + 1:end + 1 + size]
        offset = end + 2 + size
        magic = data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a",
                                 b"II*\0", b"MM\0*", b"PK\x03\x04", b"\x93NUMPY"))
        webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
        if magic or webp or EMBEDDED_IMAGE.search(data):
            blocked.append(names[oid])
    return fail(blocked) if blocked else 0


def staged():
    objects = []
    for row in git("ls-files", "--stage", "-z").split(b"\0"):
        if row:
            metadata, path = row.split(b"\t", 1)
            objects.append((metadata.split()[1].decode(), os.fsdecode(path)))
    return check_objects(objects)


def pre_push():
    tips = []
    for line in sys.stdin:
        local_ref, local_oid, remote_ref, _ = line.split()
        if set(local_oid) == {"0"}:
            continue
        if not remote_ref.startswith(("refs/heads/", "refs/tags/")):
            return fail([remote_ref])
        if not local_ref.startswith(("refs/heads/", "refs/tags/")):
            return fail([local_ref])
        tips.append(local_oid)
    if not tips:
        return 0
    objects = []
    # 必须查完整待推送祖先，不能只查最新树或与远程的差集，否则会重新带回旧产物。
    for row in git("rev-list", "--objects", *dict.fromkeys(tips)).splitlines():
        if b" " in row:
            oid, path = row.split(b" ", 1)
            objects.append((oid.decode(), os.fsdecode(path)))
    return check_objects(objects)


def install():
    custom = subprocess.run(["git", "config", "--get", "core.hooksPath"], capture_output=True)
    if custom.returncode == 0:
        raise RuntimeError("已有core.hooksPath，需显式整合现有hooks，不能覆盖。")
    common = Path(os.fsdecode(git("rev-parse", "--git-common-dir")).strip()).resolve()
    hooks = common / "hooks"
    for name in ("pre-commit", "pre-push"):
        target = hooks / name
        if target.exists() and MARKER not in target.read_text(encoding="utf-8"):
            raise RuntimeError("已有hook，不能覆盖：" + str(target))
    target_dir = common / "confidential-artifacts"
    target_dir.mkdir(exist_ok=True)
    (target_dir / "guard.py").write_bytes(Path(__file__).read_bytes())
    hooks.mkdir(exist_ok=True)
    for name, mode in (("pre-commit", "--staged"), ("pre-push", "--pre-push")):
        target = hooks / name
        target.write_text(
            '#!/bin/sh\n# ' + MARKER + '\n'
            'exec python "$(git rev-parse --git-common-dir)/confidential-artifacts/guard.py" '
            + mode + '\n', encoding="utf-8", newline="\n",
        )
        target.chmod(target.stat().st_mode | 0o111)
    exclude = common / "info/exclude"
    exclude.parent.mkdir(exist_ok=True)
    text = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if MARKER not in text:
        text += "\n# " + MARKER + "\noutput/\noutputs/\n*.png\n*.jpg\n*.jpeg\n*.npz\n*.npy\n*.zip\n"
        exclude.write_text(text, encoding="utf-8", newline="\n")
    print("已安装跨分支Git保密检查：" + str(hooks))
    return 0


if __name__ == "__main__":
    actions = {"--staged": staged, "--pre-push": pre_push, "--install": install}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit("用法：python tools/confidential_artifacts_guard.py --install|--staged|--pre-push")
    raise SystemExit(actions[sys.argv[1]]())
