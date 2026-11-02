# -*- coding: utf-8 -*-
"""提交入口的身份/全零标签适配/旧锁恢复合同；不加载模型或GPU。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import infer_submission as submission
from utils.config import load_config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    original_root = submission.ROOT
    base = load_config(str(original_root / "config/inference/area_floor_200.yaml"))
    config = {key: deepcopy(base[key]) for key in
              ("paths", "sam2", "inference", "affinity_native", "affinity_deployment", "submission")}
    config["paths"] = {"project_root":"auto"}
    config["sam2"] = {"input_normalization":"legacy_none", "sam2_repo_path":"segment-anything-2"}
    config["submission"].update(dataset="semifinal_150", restoration_noise=submission.zero.ZERO_MODE,
        local_restoration={"mode":"zero", "model":"restorer"}, native_sizes=[1024])
    models = {}
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name, filename in (("backend","backend.pt"),("restorer","restorer.pt"),("boundary_naf","naf.pt")):
        path = model_dir / filename
        path.write_bytes(("fake " + name).encode())
        models[name] = {"file":filename,"sha256":submission.entry.sha(path)}
    models["boundary_naf"]["state_sha256"] = "a" * 64
    config["submission"]["models"] = {"backend":deepcopy(models["backend"]),
        "restorer":deepcopy(models["restorer"]), "terminal_semantic":None,"local_restorer":None}
    config_path = tmp_path / "config/inference/submission_current.json"
    write_json(config_path, config)
    identity = {"format":submission.IDENTITY_FORMAT,"algorithm":submission.ALGORITHM,
        "identity":"toy_retrained_three_models","config_sha256":submission.entry.sha(config_path),"models":models}
    identity_path = tmp_path / "model_identity.json"
    write_json(identity_path, identity)
    sources = ["tools/infer_submission.py","tools/infer_compact.py","tools/infer_zero.py",
        "tools/infer_flat_semantic.py","tools/infer_boundary_rgb.py","tools/infer_semifinal.py",
        "utils/compact_instance_readout.py"]
    for name in sources:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic closure\n",encoding="utf-8")
    all_files = [tmp_path/name for name in sources] + [config_path,identity_path] + list(model_dir.glob("*.pt"))
    write_json(tmp_path / "manifest.json", {"files":[{"path":p.relative_to(tmp_path).as_posix(),
        "sha256":submission.entry.sha(p)} for p in all_files]})
    monkeypatch.setattr(submission,"ROOT",tmp_path)
    args = argparse.Namespace(input_dir=str(tmp_path/"images"), output_dir=str(tmp_path/"prediction"),
        config=str(config_path),manifest=str(identity_path),model_dir=str(model_dir),only=["test_169"])
    return SimpleNamespace(root=tmp_path, config=config, identity=identity, config_path=config_path,
        identity_path=identity_path,model_dir=model_dir,args=args)


def test_bundle_binds_exact_three_model_files_and_source_closure(bundle):
    config, identity, files = submission.validate_inputs(bundle.config_path,bundle.identity_path,bundle.model_dir)
    proof, declared = submission.source_closure(bundle.config_path,bundle.identity_path,files)
    assert config["submission"]["restoration_noise"] == "explicit_zero_v1"
    assert identity["identity"] == "toy_retrained_three_models"
    assert set(files) == {"backend","restorer","boundary_naf"}
    assert proof["manifest_present"] and "tools/infer_submission.py" in declared


@pytest.mark.parametrize("defect", ["model_bytes","NAF_state","config_absolute","config_sha"])
def test_identity_rejects_changed_files_or_nonportable_config(bundle, defect):
    if defect == "model_bytes":
        (bundle.model_dir/"backend.pt").write_bytes(b"changed")
    elif defect == "NAF_state":
        bundle.identity["models"]["boundary_naf"]["state_sha256"] = "bad"
        write_json(bundle.identity_path,bundle.identity)
    else:
        bundle.config["paths"]["project_root"] = "/root/old_training" if defect == "config_absolute" else "different"
        write_json(bundle.config_path,bundle.config)
        if defect == "config_absolute":
            bundle.identity["config_sha256"] = submission.entry.sha(bundle.config_path)
            write_json(bundle.identity_path,bundle.identity)
    with pytest.raises(ValueError):
        submission.validate_inputs(bundle.config_path,bundle.identity_path,bundle.model_dir)


def test_zero_adapters_preserve_other_geometry_and_do_not_call_noise_protocol():
    original = {"submission":{"restoration_noise":"explicit_zero_v1"},"inference":{"threshold":.5}}
    observed = []
    submission.zero_validate(lambda cfg: observed.append(deepcopy(cfg)), original)
    assert observed[0]["submission"]["restoration_noise"] == submission.entry.NOISE_MODE
    assert observed[0]["inference"] == original["inference"]
    assert original["submission"]["restoration_noise"] == "explicit_zero_v1"
    restorer = SimpleNamespace()
    submission.zero_configure(restorer,"explicit_zero_v1")
    assert restorer.inference_noise_mode == "explicit_zero_v1"
    runtime = submission.zero_runtime(lambda **kw: {"precision":"FP32","threads":4,"restoration_noise":kw})
    assert runtime["precision"] == "FP32" and runtime["threads"] == 4
    assert runtime["restoration_noise"]["random_noise_generated"] is False
    with pytest.raises(ValueError):
        submission.zero_configure(restorer,"single")


def test_run_uses_C_full_heads_and_restores_old_locks(bundle,monkeypatch):
    old = (submission.naf.EXPECTED_MODEL_SHA,submission.flat.BACKEND_SHA,submission.flat.RESTORER_SHA)
    monkeypatch.setattr(submission.entry,"runtime_record",lambda **kw:{"precision":"FP32","restoration_noise":kw})
    def fake_compact(args, *, device):
        assert args.geometry_mode == "C" and args.global_noise == "zero"
        assert args.seed_support == "original-zero" and args.capture_names == []
        assert args.expected_model_sha == "a"*64
        assert submission.naf._s0_expected_model_sha(args.expected_model_sha) == "a"*64
        assert submission.flat.BACKEND_SHA == bundle.identity["models"]["backend"]["sha256"]
        cfg = submission.entry.load_config(args.config)
        assert cfg["sam2"]["sam2_repo_path"] == str(bundle.root/"segment-anything-2")
        submission.entry.validate_config(cfg)  # 真实旧C几何校验，经局部zero标签适配。
        report = {"complete":True,"images":[{"source":"test_169"}],"imported_sources":{},
            "semantic":"raw original R1","compact_readout":submission.compact.contract("C"),
            "runtime":submission.entry.runtime_record(restoration_noise_mode="explicit_zero_v1")}
        audit = Path(args.output_dir)/"audit"
        audit.mkdir(parents=True)
        submission.entry.write(audit/"report.json",report)
        return report
    monkeypatch.setattr(submission.compact,"run",fake_compact)
    result = submission.run(bundle.args,device="cpu")
    assert (submission.naf.EXPECTED_MODEL_SHA,submission.flat.BACKEND_SHA,submission.flat.RESTORER_SHA) == old
    assert result["model_identity"] == bundle.identity
    assert result["original_model_identity_claimed"] is False
    assert "original_backend_sha256" not in result["compact_readout"]
    actual = json.loads((Path(bundle.args.output_dir)/"audit/report.json").read_text())
    assert actual["format"] == submission.FORMAT
    assert actual["runtime"]["restoration_noise"]["name"] == "explicit_zero_v1"


def test_exception_restores_every_scoped_adapter(bundle,monkeypatch):
    names = [(submission.naf,"EXPECTED_MODEL_SHA"),(submission.flat,"BACKEND_SHA"),
        (submission.flat,"RESTORER_SHA"),(submission.entry,"load_config"),
        (submission.entry,"validate_config"),(submission.entry,"configure_restoration_noise"),
        (submission.entry,"runtime_record"),(submission.entry,"write")]
    before = [getattr(target,name) for target,name in names]
    def fail(*args,**kwargs):
        raise RuntimeError("synthetic inference failure")
    monkeypatch.setattr(submission.compact,"run",fail)
    with pytest.raises(RuntimeError,match="synthetic inference failure"):
        submission.run(bundle.args,device="cpu")
    assert all(getattr(target,name) is value for (target,name),value in zip(names,before))


@pytest.mark.parametrize("failure", ["execution", "model_identity", "source_closure", "imported_sources"])
def test_late_failure_retracts_inner_success_reports_and_restores_adapters(bundle, monkeypatch, failure):
    names = [(submission.naf,"EXPECTED_MODEL_SHA"),(submission.flat,"BACKEND_SHA"),
        (submission.flat,"RESTORER_SHA"),(submission.entry,"load_config"),
        (submission.entry,"validate_config"),(submission.entry,"configure_restoration_noise"),
        (submission.entry,"runtime_record"),(submission.entry,"write")]
    before = [getattr(target,name) for target,name in names]
    original_closure = submission.source_closure
    closure_calls = []

    def closure(*args):
        closure_calls.append(True)
        if failure == "source_closure" and len(closure_calls) == 2:
            raise RuntimeError("synthetic final source closure failure")
        return original_closure(*args)

    def fake_compact(args, *, device):
        audit = Path(args.output_dir)/"audit"
        audit.mkdir(parents=True)
        report = {"complete":True,"status":"complete","images":[{"source":"test_169"}],
            "imported_sources":{"utils/not_sealed.py":"a"*64} if failure == "imported_sources" else {}}
        for name in ("status.json", "report.json", "compact_report.json"):
            submission.entry.write(audit/name,report)
        if failure == "execution":
            raise RuntimeError("synthetic inference failure after reports")
        if failure == "model_identity":
            (bundle.model_dir/"backend.pt").write_bytes(b"changed during inference")
        return report

    monkeypatch.setattr(submission,"source_closure",closure)
    monkeypatch.setattr(submission.compact,"run",fake_compact)
    with pytest.raises((RuntimeError, ValueError)):
        submission.run(bundle.args,device="cpu")
    assert all(getattr(target,name) is value for (target,name),value in zip(names,before))
    audit = Path(bundle.args.output_dir)/"audit"
    for name in ("status.json", "report.json", "compact_report.json", "submission_failed.json"):
        document = json.loads((audit/name).read_text(encoding="utf-8"))
        assert document["complete"] is False
        assert document["model_identity"] == bundle.identity
    assert json.loads((audit/"status.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert not (audit/"submission_report.json").exists()
