# -*- coding: utf-8 -*-
"""独立CPU审查：直接调用桌面包的调度/反序列化分支，不导入CUDA运行时。"""
import builtins
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest


PACKAGE = Path(__file__).resolve().parents[1] / "submission/AIC-2026-59388388-代码与数据"
ROLES = ("trunk", "semantic", "affinity", "naf", "d5a")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_engines(folder, marker):
    folder.mkdir()
    for role in ROLES:
        (folder / (role + ".engine")).write_bytes(marker)
    (folder / "engines.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def scheduler(tmp_path):
    module = load_file("review_infer", PACKAGE / "infer.py")
    root = tmp_path / "engines"
    root.mkdir()
    return module, SimpleNamespace(engine_dir=str(root), engine_manifest=None,
        model_dir=str(tmp_path / "my_models"), input_dir="images", output_dir="predictions")


def test_parent_does_not_import_torch_and_defaults_are_portable():
    original_import = builtins.__import__

    def cpu_only(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "tensorrt", "inference"}, name
        return original_import(name, *args, **kwargs)

    seen = []
    with patch("builtins.__import__", new=cpu_only):
        module = load_file("review_infer_cpu_parent", PACKAGE / "infer.py")
        module.run_with_rebuild = lambda args: seen.append(args) or 0
        assert module.main(["--input-dir", "images", "--output-dir", "predictions"]) == 0
    assert Path(seen[0].engine_dir) == PACKAGE.parent / "AIC-2026-59388388-模型文件/engines"
    assert Path(seen[0].model_dir) == PACKAGE.parent / "AIC-2026-59388388-模型文件/weights"
    assert not seen[0]._worker


@pytest.mark.parametrize("code", [0, 1, 7, 137])
def test_success_or_unrelated_failure_does_not_build(scheduler, code):
    module, args = scheduler
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert "--_worker" in command
        return SimpleNamespace(returncode=code)

    with patch.object(module.subprocess, "run", new=child):
        assert module.run_with_rebuild(args) == code
    assert len(calls) == 1


@pytest.mark.parametrize("old_cache", [False, True])
def test_compatibility_rebuild_atomically_publishes_and_clears_manifest(scheduler, old_cache):
    module, args = scheduler
    root = Path(args.engine_dir)
    manifest = root / "selected.json"
    manifest.write_text("{}", encoding="utf-8")
    args.engine_manifest = str(manifest)
    local = root / "local"
    if old_cache:
        write_engines(local, b"old")
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        if "build_engines.py" in Path(command[2]).name:
            assert kwargs == {"check": True}
            assert command[command.index("--model-dir") + 1] == args.model_dir
            assert Path(command[command.index("--output-dir") + 1]) == local
            assert not local.exists()
            previous = list(root.glob(".engine-rebuild-*/previous"))
            assert bool(previous) == old_cache
            if old_cache:
                assert (previous[0] / "trunk.engine").read_bytes() == b"old"
            staged = local.parent / "staged-complete"
            write_engines(staged, b"new")
            staged.rename(local)
            return SimpleNamespace(returncode=0)
        if len(calls) == 1:
            assert ("--engine-manifest" in command) == (not old_cache)
            return SimpleNamespace(returncode=86)
        assert "--engine-manifest" not in command
        assert Path(command[command.index("--engine-dir") + 1]) == local
        return SimpleNamespace(returncode=0)

    with patch.object(module.subprocess, "run", new=child):
        assert module.run_with_rebuild(args) == 0
    assert len(calls) == 3
    assert (local / "trunk.engine").read_bytes() == b"new"
    assert not list(root.glob(".engine-rebuild-*"))
    assert manifest.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("old_cache", [False, True])
def test_failed_build_leaves_no_partial_cache_and_does_not_retry(scheduler, old_cache):
    module, args = scheduler
    root = Path(args.engine_dir)
    local = root / "local"
    if old_cache:
        write_engines(local, b"old-incompatible")
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        if "--_worker" in command:
            return SimpleNamespace(returncode=86)
        assert not local.exists()
        raise module.subprocess.CalledProcessError(4, command)

    with patch.object(module.subprocess, "run", new=child):
        with pytest.raises(module.subprocess.CalledProcessError):
            module.run_with_rebuild(args)
    assert len(calls) == 2
    assert not local.exists()
    assert not list(root.glob(".engine-rebuild-*"))


def test_rebuilt_engine_failure_is_not_a_rebuild_loop(scheduler):
    module, args = scheduler
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        if "--_worker" in command:
            return SimpleNamespace(returncode=86)
        write_engines(Path(command[command.index("--output-dir") + 1]), b"new")
        return SimpleNamespace(returncode=0)

    with patch.object(module.subprocess, "run", new=child):
        assert module.run_with_rebuild(args) == 86
    assert len(calls) == 3


def fake_runtime(messages, success=False, context_success=True):
    class ILogger:
        ERROR, WARNING = 1, 2
        def __init__(self):
            self.base_initialized = True

    class Logger:
        WARNING = 2
        def __init__(self, severity):
            self.forwarded = []
        def log(self, severity, message):
            self.forwarded.append((severity, message))

    class Engine:
        num_io_tensors = 2
        def create_execution_context(self):
            return object() if context_success else None
        def get_tensor_name(self, index):
            return ("input", "output")[index]
        def get_tensor_mode(self, name):
            return 0 if name == "input" else 1
        def get_tensor_location(self, name):
            return 0
        def is_shape_inference_io(self, name):
            return False
        def get_tensor_format(self, name, profile):
            return 0
        def get_tensor_dtype(self, name):
            return 0
        def get_tensor_shape(self, name):
            return (1,)

    class Runtime:
        def __init__(self, logger):
            assert logger.base_initialized
            self.logger = logger
        def deserialize_cuda_engine(self, payload):
            for severity, message in messages:
                self.logger.log(severity, message)
            return Engine() if success else None

    return SimpleNamespace(ILogger=ILogger, Logger=Logger, Runtime=Runtime,
        float32=0, int64=1, TensorIOMode=SimpleNamespace(INPUT=0, OUTPUT=1),
        TensorLocation=SimpleNamespace(DEVICE=0), TensorFormat=SimpleNamespace(LINEAR=0))


def engine_module():
    fake_torch = SimpleNamespace(device=lambda *args: SimpleNamespace(index=0),
        cuda=SimpleNamespace(current_device=lambda: 0), float32=0, int64=1)
    with patch.dict("sys.modules", {"torch": fake_torch}):
        return load_file("review_engine", PACKAGE / "inference/engine.py")


@pytest.mark.parametrize("message", [
    "Platform specific tag mismatch",
    "Engine only supported on the target runtime platform",
    "The engine plan file is generated on an incompatible device",
    "The engine plan file is generated on an incompatible platform",
    "The engine plan file is not compatible with this version of TensorRT",
    "Version tag does not match",
])
def test_explicit_compatibility_error_is_distinguished(tmp_path, message):
    module = engine_module()
    path = tmp_path / "model.engine"
    path.write_bytes(b"mock")
    runtime = fake_runtime([(1, message)])
    with patch.object(module.importlib, "import_module", return_value=runtime):
        with pytest.raises(module.EngineCompatibilityError, match="Could not deserialize"):
            module.Engine(path)


@pytest.mark.parametrize("message", [
    "GPU memory allocation failed during initialization",
    "Allocation failed during the deserialization of weights",
    "Could not find plugin creator",
    "Magic tag does not match",
])
def test_unrelated_deserialization_failure_is_not_compatibility(tmp_path, message):
    module = engine_module()
    path = tmp_path / "model.engine"
    path.write_bytes(b"mock")
    with patch.object(module.importlib, "import_module", return_value=fake_runtime([(1, message)])):
        with pytest.raises(RuntimeError) as error:
            module.Engine(path)
    assert not isinstance(error.value, module.EngineCompatibilityError)


def test_warning_does_not_trigger_rebuild_and_logger_remains_alive(tmp_path):
    module = engine_module()
    path = tmp_path / "model.engine"
    path.write_bytes(b"mock")
    warning = "Using an engine plan file across different models of devices is not recommended"
    with patch.object(module.importlib, "import_module", return_value=fake_runtime([(2, warning)], success=True)):
        engine = module.Engine(path)
    ref = weakref.ref(engine.logger)
    gc.collect()
    assert ref() is engine.runtime.logger
    assert not engine.logger.errors
    assert engine.logger.forward.forwarded == [(2, warning)]


def test_context_failure_remains_an_unrelated_error(tmp_path):
    module = engine_module()
    path = tmp_path / "model.engine"
    path.write_bytes(b"mock")
    with patch.object(module.importlib, "import_module", return_value=fake_runtime([], success=True, context_success=False)):
        with pytest.raises(RuntimeError) as error:
            module.Engine(path)
    assert not isinstance(error.value, module.EngineCompatibilityError)


@pytest.mark.parametrize("exception_name", ["compatibility", "runtime", "value"])
def test_worker_only_converts_the_typed_compatibility_error(exception_name):
    module = load_file("review_worker", PACKAGE / "infer.py")
    engine = engine_module()
    exception = {"compatibility": engine.EngineCompatibilityError,
                 "runtime": RuntimeError, "value": ValueError}[exception_name]

    def run(args):
        raise exception("simulated failure")

    module.run_inference = run
    with patch.dict("sys.modules", {"inference.engine": engine}):
        arguments = ["--input-dir", "images", "--output-dir", "predictions", "--_worker"]
        if exception_name == "compatibility":
            assert module.main(arguments) == 86
        else:
            with pytest.raises(exception, match="simulated failure"):
                module.main(arguments)
