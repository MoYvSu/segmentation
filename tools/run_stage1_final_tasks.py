# -*- coding: utf-8 -*-
"""串行执行最终任务适配、严格权重交接、affinity及语义精修。"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from inference import build_model
from models.direct_semantic_affinity import load_direct_semantic_affinity_model
from models.offset_geometry import semantic_state_digest, lora_state_dict_digest
from models.stage1_final_tasks import export_final_task_checkpoints
from train_affinity_geometry_g1 import build_system
from utils.config import load_config, project_path
from utils.semantic_challenger import SemanticChallenger


ADAPT = "config/train/stage1_final_tasks_20260919.yaml"
SEMANTIC = "config/train/stage1_final_semantic20_20260919.yaml"
GEOMETRY = "config/train/stage1_final_geometry120_20260919.yaml"


def export_and_verify(config, semantic_config, geometry_config, task_path, export_dir):
    task = torch.load(task_path, map_location="cpu", weights_only=False)
    stage1 = torch.load(project_path(config, config["stage1_final_tasks"]["checkpoint"]),
                        map_location="cpu", weights_only=False)
    reference, geometry = export_final_task_checkpoints(task, stage1, semantic_config, export_dir)
    model = build_model(geometry_config, "cpu", str(export_dir / "reference.pth"))
    # 不依赖宽松读取：新增/遗漏/异形键在进入长程训练前必须失败。
    model.decoder.load_state_dict(reference["decoder_state_dict"], strict=True)
    geometry["semantic_state_digest"] = semantic_state_digest(model)
    if lora_state_dict_digest(task["lora_state_dict"]) != lora_state_dict_digest(reference["lora_state_dict"]):
        raise RuntimeError("导出LoRA不一致")
    torch.save(geometry, export_dir / "affinity.pth")
    del model
    direct, _ = load_direct_semantic_affinity_model(task_path, config, "cuda")
    system, _, _, _ = build_system(geometry_config, geometry_config["affinity_geometry_g1"], "cuda")
    reference_model = system.reference_model
    semantic = SemanticChallenger(reference_model.decoder,
        semantic_residual=reference_model.decoder.semantic_residual,
        semantic_residual_version="highres_v1").cuda().eval()
    direct.eval()
    system.eval()
    torch.manual_seed(42)
    image = torch.rand(1, 3, 1024, 1024, device="cuda")
    with torch.no_grad():
        expected = direct(image)
        features = reference_model.encoder(image)
        actual = {"semantic_logits": semantic(features, image),
                  "affinity_logits": system.geometry_forward_from_features(image, features)["affinity_logits"]}
        deltas = {k: float((expected[k] - actual[k]).abs().max()) for k in actual}
    if any(v > 1e-6 for v in deltas.values()):
        raise RuntimeError("交接改变了前向: " + str(deltas))
    report = {"logit_max_delta": deltas, "semantic_state_digest": geometry["semantic_state_digest"],
              "lora_digest": lora_state_dict_digest(reference["lora_state_dict"]),
              "source_epoch": task["epoch"], "source_phase": task["phase"],
              "selection": task["selection"]}
    (export_dir / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("EXPORT VERIFIED", json.dumps(report), flush=True)
    del system, direct, semantic, expected, actual, reference_model, features
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start-at", choices=("adaptation", "export", "geometry", "semantic"), default="adaptation")
    parser.add_argument("--resume-adaptation")
    args = parser.parse_args()
    config, sem, geo = [load_config(str(ROOT / p)) for p in (ADAPT, SEMANTIC, GEOMETRY)]
    base = Path(project_path(config, config["stage1_final_tasks"]["output_dir"]))
    run_root = base / "smoke" if args.smoke else base
    run_root.mkdir(parents=True, exist_ok=True)
    export = run_root / "export"
    cfg_dir = run_root / "runtime_configs"
    cfg_dir.mkdir(exist_ok=True)
    geo["affinity_geometry_g1"].update(reference_checkpoint=str(export / "reference.pth"),
        geometry_init_checkpoint=str(export / "affinity.pth"), output_dir=str(run_root / "geometry"))
    sem["semi_supervised"].update(base_checkpoint=str(export / "reference.pth"), output_dir=str(run_root / "semantic"))
    sem["semi_supervised"]["monitor"]["output_dir"] = str(run_root / "semantic/monitor")
    if args.smoke:
        geo["affinity_geometry_g1"].update(epochs=1, monitor_interval=99)
        geo["affinity_geometry_g1"]["sam2_geometry"]["samples_per_epoch"] = 4
        sem["semi_supervised"].update(epochs=1, labeled_steps_per_epoch=1, unlabeled_samples_per_epoch=4, checkpoint_interval=99)
        sem["data"]["num_workers"] = 0
    for name, value in (("semantic", sem), ("geometry", geo)):
        (cfg_dir / (name + ".yaml")).write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    status_path = run_root / "pipeline_status.json"
    status = {"status": "running", "smoke": args.smoke, "completed": []}
    def stage(name, operation):
        status["stage"] = name
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print("PIPELINE STAGE", name, flush=True)
        operation()
        status["completed"].append(name)
    def execute(script, *extra):
        subprocess.run([sys.executable, str(ROOT / script), *extra], cwd=ROOT, check=True)
    order = ("adaptation", "export", "geometry", "semantic")
    start = order.index(args.start_at)
    try:
        if start <= 0:
            extra = ["--smoke"] if args.smoke else []
            if args.resume_adaptation:
                extra += ["--resume", args.resume_adaptation]
            stage("adaptation", lambda: execute("train_stage1_final_tasks.py", "--config", ADAPT, *extra))
        if start <= 1:
            task_path = (base / "smoke" if args.smoke else base / "adaptation") / "best_joint_lora.pth"
            stage("export", lambda: export_and_verify(config, sem, geo, task_path, export))
        if start <= 2:
            stage("geometry", lambda: execute("train_affinity_geometry_g1.py", "--config", str(cfg_dir / "geometry.yaml")))
        stage("semantic", lambda: execute("train_stage2.py", "--config", str(cfg_dir / "semantic.yaml")))
        reference = torch.load(export / "reference.pth", map_location="cpu", weights_only=False)
        final_sem = torch.load(run_root / "semantic/best_model_stage2.pth", map_location="cpu", weights_only=False)
        if lora_state_dict_digest(final_sem["lora_state_dict"]) != lora_state_dict_digest(reference["lora_state_dict"]):
            raise RuntimeError("语义精修改变了共享LoRA")
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
