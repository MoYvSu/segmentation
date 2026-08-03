# -*- coding: utf-8 -*-
"""训练运行记录器。

每次训练启动时创建一个运行目录
``outputs/runs/<时间戳>_<phase>_<tag>/``，并落盘：

- ``run_info.json``：命令、git 提交/分支/工作区脏文件、Python/Torch/CUDA 版本；
- ``config_snapshot.yaml``：本次训练生效的完整配置快照；
- ``metrics.csv``：逐 epoch 训练/验证指标；
- ``best_model.pth``：最优权重副本（与上面三者同目录，可直接复现）。

用法（在 train_stage2.py 内部自动调用）：

    recorder = RunRecorder(project_root, phase="boundary", tag="exp01")
    recorder.save_config(config)
    recorder.save_manifest()
    recorder.append_metrics({...})
    recorder.copy_checkpoint(best_path)
"""

import csv
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys


def _git_info(project_root):
    """读取 git 提交信息；仓库不可用或非 git 时返回 None。"""
    info = {}
    try:
        def _run(args):
            return subprocess.run(
                args, cwd=project_root, capture_output=True,
                text=True, timeout=10,
            )
        out = _run(["git", "rev-parse", "HEAD"])
        if out.returncode == 0:
            info["commit"] = out.stdout.strip()
        out = _run(["git", "branch", "--show-current"])
        if out.returncode == 0:
            info["branch"] = out.stdout.strip()
        out = _run(["git", "status", "--porcelain"])
        if out.returncode == 0:
            dirty = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            info["dirty_files"] = dirty
    except Exception:
        return None
    return info if info else None


def _env_info():
    """记录 Python / PyTorch / CUDA 环境信息。"""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


class RunRecorder:
    """训练运行记录器（见模块 docstring）。"""

    def __init__(self, project_root, output_base="outputs/runs",
                 phase="", tag="", command=None):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [ts]
        if phase:
            parts.append(phase)
        if tag:
            parts.append(tag)
        self.name = "_".join(parts)
        self.run_dir = os.path.join(project_root, output_base, self.name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.manifest = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "phase": phase,
            "tag": tag,
            "command": command or " ".join(sys.argv),
            "git": _git_info(project_root),
            "env": _env_info(),
        }
        self._metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self._metrics_written = os.path.exists(self._metrics_path)

    def save_config(self, config_dict):
        """保存生效配置快照，返回路径。"""
        import yaml
        path = os.path.join(self.run_dir, "config_snapshot.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)
        return path

    def save_manifest(self):
        """保存 run_info.json，返回路径。"""
        path = os.path.join(self.run_dir, "run_info.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        return path

    def append_metrics(self, row):
        """逐 epoch 追加一行指标到 metrics.csv（首次自动写表头）。"""
        write_header = not self._metrics_written
        with open(self._metrics_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._metrics_written = True
            writer.writerow(row)

    def copy_checkpoint(self, src_path, name="best_model.pth"):
        """把最优权重复制到运行目录，返回目标路径（源不存在时返回 None）。"""
        if src_path and os.path.exists(src_path):
            dst = os.path.join(self.run_dir, name)
            shutil.copy2(src_path, dst)
            return dst
        return None
