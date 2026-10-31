# -*- coding: utf-8 -*-
"""沿各自冻结源码接续A/C至累计e10；原e5目录只读，默认仅预检。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


PAIR_FIELDS = ("update", "epoch", "data_epoch", "sources", "pair_sha256",
               "noise_rng_before", "noise_sha256", "profiles", "mask_mean",
               "learning_rate", "samples", "micro_batch", "terminal_step")
MONITORS = {"control": "tools.restore_consistency_monitor", "zero": "tools.restore_zero_monitor"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def validate_logs(directory, epoch):
    directory = Path(directory)
    metrics, epochs = rows(directory / "metrics.jsonl"), rows(directory / "epochs.jsonl")
    if len(metrics) != epoch * 250 or len(epochs) != epoch:
        raise ValueError("Partial or unexpected epoch logs")
    if [row["update"] for row in metrics] != list(range(1, epoch * 250 + 1)):
        raise ValueError("Update order is incomplete")
    for index in range(epoch):
        batch = metrics[index * 250:(index + 1) * 250]
        sources = [source for row in batch for source in row["sources"]]
        if len(sources) != 1000 or len(set(sources)) != 1000:
            raise ValueError("Each epoch must use 1000 unique originals once")
        if any(row["epoch"] != index + 1 or row["data_epoch"] != 60 + index for row in batch):
            raise ValueError("Wrong epoch or data epoch")
        if epochs[index]["epoch"] != index + 1 or epochs[index]["updates"] != (index + 1) * 250:
            raise ValueError("Epoch log is incomplete")
    return metrics


def check_pairs(control, zero, epoch):
    left, right = validate_logs(control, epoch), validate_logs(zero, epoch)
    for a, c in zip(left, right):
        if any(a[key] != c[key] for key in PAIR_FIELDS):
            raise ValueError(f"A/C pair mismatch at update {c['update']}")
        if (a.get("consistency_weight") != 0 or c.get("consistency_weight") != 0
                or c.get("training_noise_mode") != "zero" or c.get("applied_noise_max_abs") != 0
                or c["loss"] != c["supervised_loss"]):
            raise ValueError("Unexpected zero-noise or supervised objective")
    return dict(matched_updates=len(left), pair_fields=list(PAIR_FIELDS),
                every_epoch_1000_unique_sources=True, every_update_applied_c_noise_zero=True)


def continuation_config(config):
    result = copy.deepcopy(config)
    result["restore_consistency"]["stage_epochs"] = 10
    result["restore_consistency"]["monitor_epochs"] = [0, 1, 3, 5, 7, 10]
    return result


# 在各自原冻结目录导入训练入口，不把最新C入口混进A的source_code签名。
CHECK_STATE = r'''
import hashlib, inspect, json, sys, tempfile
from pathlib import Path
import torch
import yaml
sys.path.insert(0, sys.argv[1])
import train_restore_consistency as t
directory, arm, epoch = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
config = json.loads((directory/'run_config.json').read_text())
saved = torch.load(directory/'last.pt', map_location='cpu', weights_only=True)
slim = torch.load(directory/f'epoch_{epoch:03d}.pt', map_location='cpu', weights_only=True)
parameters = inspect.signature(t.source_code_identity).parameters
kwargs = {}
if 'monitor_module' in parameters: kwargs['monitor_module']=sys.argv[5]
if 'arm' in parameters: kwargs['arm']=arm
sources, code_sha = t.source_code_identity(**kwargs)
assert saved['epoch']==epoch and saved['updates']==epoch*250 and saved['arm']==arm
assert saved['source_code_sha256']==code_sha
assert saved['recipe_signature']==t.recipe_signature(config)
assert saved['training_format']==t.TRAINING_FORMAT and not saved['smoke']
assert t.state_sha(saved['model'])==t.state_sha(slim['model'])
assert saved['scheduler']['T_max']==60 and saved['scheduler']['last_epoch']==epoch
assert {int(v['step'].item()) for v in saved['optimizer']['state'].values()}=={epoch*250}
runtime=json.loads((directory/'status.json').read_text())['runtime']
assert str(torch.__version__)==runtime['torch_version']
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090'
assert runtime['training_dtype']=='bfloat16'
assert bool(torch.backends.cuda.matmul.allow_tf32)==runtime['matmul_allow_tf32']
assert bool(torch.backends.cudnn.allow_tf32)==runtime['cudnn_allow_tf32']
cfg=json.loads(json.dumps(config))
cfg['restore_consistency']['stage_epochs']=10
cfg['restore_consistency']['monitor_epochs']=[0,1,3,5,7,10]
assert t.recipe_signature(cfg)==saved['recipe_signature']
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'resume.yaml'
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    decoded=t.load_config(str(path))
    assert decoded==cfg and t.recipe_signature(decoded)==saved['recipe_signature']
    assert isinstance(decoded['rgb_restoration']['train']['learning_rate'], float)
def rng_sha(name):
    return hashlib.sha256(saved['rng'][name].cpu().numpy().tobytes()).hexdigest()
print(json.dumps(dict(epoch=epoch,updates=saved['updates'],source_code_sha256=code_sha,
    recipe_signature=saved['recipe_signature'],model_state_sha256=t.state_sha(saved['model']),
    noise_rng=rng_sha('noise_generator'),loader_rng=rng_sha('loader_generator'),
    next_lr=saved['optimizer']['param_groups'][0]['lr'],device=torch.cuda.get_device_name(),
    runtime=runtime,optimizer_slots=len(saved['optimizer']['state']),typed_config_roundtrip=True)))
'''


def state_check(code, directory, arm, epoch):
    result = subprocess.run([sys.executable, "-B", "-c", CHECK_STATE, str(code), str(directory),
                             arm, str(epoch), MONITORS[arm]], cwd=code,
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout.splitlines()[-1])


def parent_files(directory):
    """核对原臂全部文件及根固定监控；不包含其他臂和分析缓存。"""
    directory = Path(directory)
    files = [path for path in directory.rglob("*") if path.is_file()]
    files += [directory.parent / name for name in
              ("source_pool.json", "monitor_manifest.json", "monitor_fixed.pt")]
    return {str(path.resolve()): sha(path) for path in files}


def verify_hashes(hashes):
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("Original e5 artifacts changed")


def frozen_sources(code):
    code = Path(code)
    manifest = next((path for path in (code.parent / "source_manifest.json", code / "source_manifest.json")
                     if path.is_file()), None)
    if manifest is None:
        raise FileNotFoundError("Frozen source manifest is required")
    hashes = {str((code / name).resolve()): expected for name, expected in read(manifest).items()}
    hashes[str(manifest.resolve())] = sha(manifest)
    verify_hashes(hashes)
    return hashes


def validate_destination(root, parents, codes):
    root = Path(root).resolve()
    protected = [Path(path).resolve().parent for path in parents.values()]
    protected += [Path(path).resolve() for path in codes.values()]
    if root.exists() or any(root.is_relative_to(path) for path in protected):
        raise FileExistsError("Continuation must use a fresh directory outside original experiments")


def prepare(root, parents):
    root = Path(root)
    if root.exists():
        raise FileExistsError("Use a fresh continuation directory")
    root.mkdir(parents=True)
    for arm, source in parents.items():
        # 真实复制，不共享可追加日志或会被原子替换的last.pt。
        shutil.copytree(source, root / arm)
        config = continuation_config(read(Path(source) / "run_config.json"))
        path = root / f"{arm}_config.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if yaml.safe_load(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Continuation YAML must preserve the typed original configuration")
    shutil.copy2(Path(parents["control"]).parent / "source_pool.json", root / "source_pool.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-code", type=Path, required=True)
    parser.add_argument("--zero-code", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    parents = {"control": args.control.resolve(), "zero": args.zero.resolve()}
    codes = {"control": args.control_code.resolve(), "zero": args.zero_code.resolve()}
    root = args.root.resolve()
    validate_destination(root, parents, codes)
    checks, hashes = {}, {}
    for arm in parents:
        status = read(parents[arm] / "status.json")
        if (status.get("status") != "stage_complete" or status.get("epoch") != 5
                or status.get("updates") != 1250 or status.get("arm") != arm
                or not status.get("strict_reload_equal") or status.get("no_backend_loaded") is not True):
            raise ValueError("Parent must be the completed formal matched e5")
        if sha(parents[arm] / "epoch_005.pt") != status["final_checkpoint_sha256"]:
            raise ValueError("Parent e5 checkpoint differs from receipt")
        checks[arm] = state_check(codes[arm], parents[arm], arm, 5)
        hashes.update(parent_files(parents[arm]))
        hashes.update(frozen_sources(codes[arm]))
    for key in ("noise_rng", "loader_rng", "next_lr"):
        if checks["control"][key] != checks["zero"][key]:
            raise ValueError("Saved A/C random stream or learning rate differs")
    # 两模式首步复核的噪声协议有意不同，运行精度单独核对。
    for key in ("torch_version", "training_dtype", "matmul_allow_tf32", "cudnn_allow_tf32"):
        if checks["control"]["runtime"][key] != checks["zero"]["runtime"][key]:
            raise ValueError("Parent A/C runtime differs")
    paired = check_pairs(parents["control"], parents["zero"], 5)
    if read(parents["control"].parent / "monitor_manifest.json") != read(parents["zero"].parent / "monitor_manifest.json"):
        raise ValueError("Original fixed monitor pairs differ")
    state = dict(status="preflight_passed", from_epoch=5, stop_epoch=10,
                 old_epochs_rerun=False, automatic_inference=False, backend_loaded=False,
                 parents={arm: str(path) for arm, path in parents.items()},
                 frozen_code={arm: str(path) for arm, path in codes.items()},
                 checks=checks, parent_hashes=hashes, previous_pairing=paired, arms={})
    if not args.run:
        print(json.dumps({key: value for key, value in state.items() if key != "parent_hashes"}))
        return
    prepare(root, parents)
    receipt = root / "queue.json"
    state["status"] = "running"
    write(receipt, state)
    try:
        for arm in ("control", "zero"):
            command = [sys.executable, "-u", "-B", str(codes[arm] / "train_restore_consistency.py"),
                       "--config", str(root / f"{arm}_config.yaml"), "--root", str(root),
                       "--arm", arm, "--resume", "--stop", "10", "--monitor-module", MONITORS[arm]]
            state["current_arm"] = arm
            state["arms"][arm] = dict(status="running", command=command)
            write(receipt, state)
            with (root / f"{arm}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=codes[arm], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            status = read(root / arm / "status.json")
            if (status["status"] != "stage_complete" or status["epoch"] != 10
                    or status["updates"] != 2500 or not status["strict_reload_equal"]
                    or not status["first_inference_reload_equal"]):
                raise RuntimeError("Continuation did not pass the fixed endpoint/reload gate")
            for name in ("metrics.jsonl", "epochs.jsonl"):
                if not (root / arm / name).read_bytes().startswith((parents[arm] / name).read_bytes()):
                    raise RuntimeError("Old log prefix changed")
            if sha(root / arm / "epoch_005.pt") != sha(parents[arm] / "epoch_005.pt"):
                raise RuntimeError("Copied e5 checkpoint changed")
            if read(root / "monitor_manifest.json") != read(parents[arm].parent / "monitor_manifest.json"):
                raise RuntimeError("Fixed monitor pairs changed during continuation")
            final = state_check(codes[arm], root / arm, arm, 10)
            state["arms"][arm].update(status="stage_complete", final=final,
                                     final_checkpoint_sha256=status["final_checkpoint_sha256"])
            verify_hashes(hashes)
            write(receipt, state)
        state.update(status="stage_complete", **check_pairs(root / "control", root / "zero", 10),
                     parent_artifacts_unchanged=True, previous_log_prefixes_unchanged=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write(receipt, state)


if __name__ == "__main__":
    main()
