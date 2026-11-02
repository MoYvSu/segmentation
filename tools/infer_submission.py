# -*- coding: utf-8 -*-
"""显式三模型身份的提交入口；复用C全零算术，保留旧入口缺省锁。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path, PureWindowsPath
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import infer_compact as compact

entry, zero, flat, naf = compact.entry, compact.zero, compact.flat, compact.naf
FORMAT = "compact_C_submission_v1"
IDENTITY_FORMAT = "compact_C_model_identity_v1"
ALGORITHM = "C_zero_native1024_v1"
DEFAULT_CONFIG = ROOT / "config/inference/submission_current.json"
DEFAULT_MANIFEST = ROOT / "model_identity.json"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _portable(value):
    if isinstance(value, dict):
        if "_base" in value:
            raise ValueError("Submission config must be standalone JSON without _base")
        for item in value.values():
            _portable(item)
    elif isinstance(value, list):
        for item in value:
            _portable(item)
    elif isinstance(value, str):
        if (value.startswith(("/", "\\", "~")) or PureWindowsPath(value).drive
                or ".." in value.replace("\\", "/").split("/") or "$" in value):
            raise ValueError("Submission config must not contain absolute/traversal/environment paths")


def validate_inputs(config_path, identity_path, model_dir):
    """只核文件/配置，不构建模型；三模型均须位于本bundle/model。"""
    config_path, identity_path, model_dir = map(lambda p: Path(p).resolve(),
                                               (config_path, identity_path, model_dir))
    root = ROOT.resolve()
    if model_dir != root / "model" or not config_path.is_relative_to(root) or not identity_path.is_relative_to(root):
        raise ValueError("Config/identity must be inside this bundle; models must use bundle/model")
    identity, config = _json(identity_path), _json(config_path)
    if (identity.get("format") != IDENTITY_FORMAT or identity.get("algorithm") != ALGORITHM
            or not isinstance(identity.get("identity"), str) or not identity["identity"].strip()
            or not _digest(identity.get("config_sha256"))
            or entry.sha(config_path) != identity["config_sha256"]):
        raise ValueError("Model identity version/algorithm/config SHA differs")
    _portable(config)
    allowed = {"format", "paths", "sam2", "inference", "affinity_native", "affinity_deployment", "submission"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Submission config contains training or unsupported sections")
    if (config.get("paths", {}).get("project_root") != "auto"
            or config.get("sam2", {}).get("sam2_repo_path") != "segment-anything-2"):
        raise ValueError("Use portable auto root and the bundled segment-anything-2 source")
    submission = config.get("submission", {})
    if (submission.get("dataset") != "semifinal_150" or submission.get("restoration_noise") != zero.ZERO_MODE
            or submission.get("semantic_head_delta") is not None):
        raise ValueError("Submission requires latest150, explicit zero noise and the complete backend semantic head")
    models = identity.get("models")
    if not isinstance(models, dict) or set(models) != {"backend", "restorer", "boundary_naf"}:
        raise ValueError("Identity must bind exactly backend/restorer/boundary_naf")
    files = {}
    for name, item in models.items():
        if (not isinstance(item, dict) or not isinstance(item.get("file"), str)
                or not _digest(item.get("sha256"))):
            raise ValueError("Missing model file/SHA: " + name)
        file = (model_dir / item["file"]).resolve()
        if not file.is_relative_to(model_dir) or not file.is_file() or entry.sha(file) != item["sha256"]:
            raise ValueError("Model file/SHA differs: " + name)
        files[name] = file
    if len(set(files.values())) != 3 or not _digest(models["boundary_naf"].get("state_sha256")):
        raise ValueError("Require three distinct model files and the complete NAF state SHA")
    configured = submission.get("models", {})
    if set(configured) - {"backend", "restorer", "terminal_semantic", "local_restorer"}:
        raise ValueError("Unexpected upstream model; NAF is loaded separately exactly once")
    for name in ("backend", "restorer"):
        if configured.get(name) != {"file": models[name]["file"], "sha256": models[name]["sha256"]}:
            raise ValueError("Config/model identity mismatch: " + name)
    if configured.get("terminal_semantic") is not None or configured.get("local_restorer") is not None:
        raise ValueError("C submission uses one complete backend and one shared restorer")
    return config, identity, files


def source_closure(config_path, identity_path, files):
    proof = entry.verify_source_manifest(ROOT.resolve())
    if not proof.get("manifest_present"):
        raise ValueError("Submission bundle requires its source manifest.json")
    declared = {row["path"] for row in _json(ROOT / "manifest.json")["files"]}
    required = {"tools/infer_submission.py", "tools/infer_compact.py", "tools/infer_zero.py",
                "tools/infer_flat_semantic.py", "tools/infer_boundary_rgb.py", "tools/infer_semifinal.py",
                "utils/compact_instance_readout.py"}
    required |= {Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
                 for p in (config_path, identity_path, *files.values())}
    if not required <= declared:
        raise ValueError("Source closure omits actual config/models/entry dependencies: " + str(sorted(required-declared)))
    return proof, declared


def zero_validate(original, config):
    if config.get("submission", {}).get("restoration_noise") != zero.ZERO_MODE:
        raise ValueError("Public submission config must explicitly declare zero prediction noise")
    checked = deepcopy(config)
    # 仅适配旧校验器的协议标签；其余C几何与读出检查原样执行。
    checked["submission"]["restoration_noise"] = entry.NOISE_MODE
    return original(checked)


def zero_configure(restorer, mode):
    if mode != zero.ZERO_MODE:
        raise ValueError("Submission forbids any nonzero restoration noise protocol")
    restorer.inference_noise_mode = zero.ZERO_MODE


def zero_runtime(original, *, restoration_noise_mode=zero.ZERO_MODE):
    if restoration_noise_mode != zero.ZERO_MODE:
        raise ValueError("Runtime must report explicit zero noise")
    result = original(restoration_noise_mode=entry.NOISE_MODE)
    result["restoration_noise"] = dict(name=zero.ZERO_MODE, global_noise="zero", local_noise="zero",
        random_noise_generated=False, prediction_noise_generators_forbidden=True)
    return result


def audit_document(document, identity, identity_sha, config_sha, source_proof):
    value = deepcopy(document)
    value.setdefault("underlying_format", value.get("format"))
    value.update(format=FORMAT, algorithm=ALGORITHM, training=False,
        model_identity=deepcopy(identity), model_identity_sha256=identity_sha,
        portable_config_sha256=config_sha, source_closure=deepcopy(source_proof),
        inference_entry="tools/infer_submission.py", original_model_identity_claimed=False)
    if "semantic" in value:
        value["semantic"] = "raw RGB semantic probability from the explicitly identified complete backend"
    if "geometry" in value:
        value["geometry"] = "C restored whole semantic + local zero-D5a affinity + NAF/S0 + raw RGB vote + final200"
    def annotate_contract(container):
        if isinstance(container, dict) and isinstance(container.get("compact_readout"), dict):
            contract = container["compact_readout"]
            contract.pop("original_backend_sha256", None)
            contract.update(backend_file_sha256=identity["models"]["backend"]["sha256"],
                restorer_file_sha256=identity["models"]["restorer"]["sha256"],
                NAF_model_sha256=identity["models"]["boundary_naf"]["state_sha256"],
                model_identity=identity["identity"])
    annotate_contract(value)
    for name in ("config", "baseline_config"):
        annotate_contract(value.get(name))
    for name in ("runtime", "baseline_runtime"):
        if isinstance(value.get(name), dict):
            value[name]["restoration_noise"] = dict(name=zero.ZERO_MODE, global_noise="zero", local_noise="zero",
                random_noise_generated=False, prediction_noise_generators_forbidden=True)
    if "config_role" in value:
        value["config_role"] = "effective C zero execution; the separately hashed portable config declares zero noise"
    return value


def run(args, *, device="cuda"):
    args = argparse.Namespace(**vars(args))
    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG).resolve()
    identity_path = Path(getattr(args, "manifest", None) or DEFAULT_MANIFEST).resolve()
    model_dir = Path(getattr(args, "model_dir", None) or ROOT / "model").resolve()
    config, identity, files = validate_inputs(config_path, identity_path, model_dir)
    proof, declared = source_closure(config_path, identity_path, files)
    identity_sha, config_sha = entry.sha(identity_path), entry.sha(config_path)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh submission output directory")
    original_validate, original_load = entry.validate_config, entry.load_config
    original_runtime, original_write = entry.runtime_record, entry.write

    def load(path):
        if Path(path).resolve() != config_path or entry.sha(path) != config_sha:
            raise ValueError("Execution attempted an unsealed config")
        value = original_load(str(path))
        value["paths"]["project_root"] = str(ROOT.resolve())
        value["sam2"]["sam2_repo_path"] = str((ROOT / "segment-anything-2").resolve())
        return value

    def write(path, value):
        if Path(path).parent == output / "audit":
            value = audit_document(value, identity, identity_sha, config_sha, proof)
        return original_write(path, value)

    args.config, args.model_dir = str(config_path), str(model_dir)
    args.dataset, args.geometry_mode = "semifinal_150", "C"
    args.naf_checkpoint = str(files["boundary_naf"])
    args.expected_model_sha = identity["models"]["boundary_naf"]["state_sha256"]
    args.global_noise, args.seed_support, args.capture_names = "zero", "original-zero", []
    try:
        with ExitStack() as stack:
            for target, name, replacement in (
                (naf, "EXPECTED_MODEL_SHA", args.expected_model_sha),
                (flat, "BACKEND_SHA", identity["models"]["backend"]["sha256"]),
                (flat, "RESTORER_SHA", identity["models"]["restorer"]["sha256"]),
                (entry, "load_config", load),
                (entry, "validate_config", lambda value: zero_validate(original_validate, value)),
                (entry, "configure_restoration_noise", zero_configure),
                (entry, "runtime_record", lambda **kw: zero_runtime(original_runtime, **kw)),
                (entry, "write", write),
            ):
                stack.enter_context(patch.object(target, name, new=replacement))
            # 全零predict的禁RNG/真实q_sample审计、compact读出与final200均复用原实现。
            result = compact.run(args, device=device)
        validate_inputs(config_path, identity_path, model_dir)
        final_proof, _ = source_closure(config_path, identity_path, files)
        if entry.sha(identity_path) != identity_sha or entry.sha(config_path) != config_sha or final_proof != proof:
            raise RuntimeError("Submission identity/config/source changed during inference")
        report_path = output / "audit/report.json"
        if report_path.is_file():
            report = _json(report_path)
            if set(report.get("imported_sources", {})) - declared:
                raise RuntimeError("Actual imported source is absent from the sealed bundle closure")
        result = audit_document(result, identity, identity_sha, config_sha, proof)
        original_write(output / "audit/submission_report.json", result)
        return result
    except Exception as exc:
        # 内层完成不代表便携封装已验收；收尾校验失败时也撤销所有成功回执。
        audit = output / "audit"
        if audit.is_dir():
            for path in sorted(audit.glob("*.json")):
                document = _json(path)
                document.update(complete=False, submission_error=repr(exc))
                if path.name == "status.json":
                    document["status"] = "failed"
                original_write(path, audit_document(document, identity, identity_sha, config_sha, proof))
            failure = dict(complete=False, status="failed", error=repr(exc))
            original_write(audit / "submission_failed.json",
                           audit_document(failure, identity, identity_sha, config_sha, proof))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-dir", default=str(ROOT / "model"))
    parser.add_argument("--only", nargs="+", help="仅用于固定少量源的smoke；正式运行不传")
    result = run(parser.parse_args(argv))
    print(json.dumps(dict(complete=bool(result.get("complete")), format=FORMAT,
        identity=result["model_identity"]["identity"], algorithm=ALGORITHM,
        images=len(result.get("images", []))), ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()
