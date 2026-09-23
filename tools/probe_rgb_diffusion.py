# -*- coding: utf-8 -*-
"""固定训练配对的扩散诊断：多个种子、首步/全程误差和训练/采样状态差距。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.rgb_restoration import load_rgb_restorer
from train_rgb_diffusion_restoration import error_report, overfit_gate, save_monitor_image
from train_rgb_restoration import reconstruction_errors, write_json


@torch.inference_mode()
def probe(checkpoint_path, *, device, seeds, output_dir):
    checkpoint_path, output_dir = Path(checkpoint_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = load_rgb_restorer(str(checkpoint_path), device=device).eval().requires_grad_(False)
    fixed = torch.load(checkpoint_path.parent / "fixed_samples.pt", map_location="cpu", weights_only=True)
    result = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "updates": payload["updates"], "precision": "FP32", "model_config": model.model_config,
              "scope": "seen_training_pairs_only_not_generalization_or_segmentation_score", "seeds": []}
    for seed in seeds:
        first_predictions, final_predictions, gap_rows = [], [], []
        for index, source in enumerate(fixed["source"]):
            y, target, valid = [fixed[key][index:index + 1].to(device) for key in ("input", "target", "valid")]
            rng = torch.Generator(device=device).manual_seed(seed + index)
            noise = torch.randn(y.shape, device=device, generator=rng)
            state = y + model.kappa * noise
            first = None
            for step in range(model.num_steps, 0, -1):
                t = torch.full((1,), step, device=device, dtype=torch.long)
                prediction = model.denoise(state, y, t)
                if first is None:
                    first = prediction.clamp(0, 1).cpu()
                if seed == seeds[0] and step in {model.num_steps, model.num_steps // 2, 4, 1}:
                    forward_prediction = model.denoise(model.q_sample(target, y, t, noise=noise), y, t)
                    row = {"source": source, "t": step, "eta": float(model.eta[step])}
                    for name, value in (("forward_state_x0_error", forward_prediction), ("free_sampling_x0_error", prediction)):
                        row[name] = {key: float(error.mean()) for key, error in reconstruction_errors(value, target, valid).items()}
                    gap_rows.append(row)
                state, variance = model.posterior_mean_variance(state, prediction, t)
                if step > 1:
                    state += variance.sqrt() * torch.randn(y.shape, device=device, generator=rng)
            final = (y + (state - y)).clamp(0, 1).cpu()
            first_predictions.append(first)
            final_predictions.append(final)
            if seed == seeds[0]:
                for name, prediction in (("first", first), ("full", final)):
                    save_monitor_image(output_dir / f"{index:02d}_{Path(source).stem}_{name}.png", fixed["input"][index],
                                       prediction[0], fixed["target"][index],
                                       title=f"Update {payload['updates']} | {name} | seed {seed + index}")
                # 诊断重放必须等于实际部署forward，避免手写循环偏离模型。
                deployed = model(y, generator=torch.Generator(device=device).manual_seed(seed + index)).cpu()
                if not torch.equal(final, deployed):
                    raise AssertionError("probe sampling differs from deployed forward")
        reports = {name: error_report(fixed["input"], torch.cat(predictions), fixed["target"], fixed["valid"], fixed["source"])
                   for name, predictions in (("first", first_predictions), ("full", final_predictions))}
        reports["full_gate"] = overfit_gate(reports["full"], payload["experiment_config"]["overfit"]["gate"])
        result["seeds"].append({"seed": seed, **reports})
        if gap_rows:
            result["sampling_gap"] = gap_rows
    write_json(output_dir / "probe.json", result)
    print(json.dumps({"updates": result["updates"], "checkpoint_sha256": result["checkpoint_sha256"],
                      "seeds": [{"seed": row["seed"], "first_ratios": row["first"]["mean"]["ratios"],
                                 "full_ratios": row["full"]["mean"]["ratios"], "full_gate": row["full_gate"]["passed"]}
                                for row in result["seeds"]]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[314159, 271828, 161803])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    probe(args.checkpoint, device=torch.device(args.device), seeds=args.seeds, output_dir=args.output_dir)
