# -*- coding: utf-8 -*-
"""Build one self-contained E10a semantic + G4b affinity checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fused_deployment import (
    FUSED_DEPLOYMENT_FORMAT,
    FusedPhaseAffinityModel,
    load_fused_deployment_model,
)
from train_affinity_geometry_g1 import build_system, load_geometry_checkpoint_state
from train_offset_geometry import file_sha256
from utils.affinity_deployment import prepare_image
from utils.config import load_config, project_path
from utils.semantic_challenger import build_semantic_challenger


def _lora_architecture(checkpoint):
    state = checkpoint.get("lora_state_dict", {})
    rank = next(
        (int(value.shape[0]) for key, value in state.items() if "lora_A" in key),
        None,
    )
    if rank is None:
        raise ValueError("semantic checkpoint contains no LoRA rank metadata")
    cfg = checkpoint.get("config", {}).get("lora", {})
    return {
        "rank": rank,
        "alpha": float(cfg.get("alpha", 32.0)),
        "target_layers": cfg.get("target_layers"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml",
    )
    parser.add_argument(
        "--output",
        default="outputs/deployment/e10a_g4b_fused.pth",
    )
    parser.add_argument("--verify-image", default="data/test/test_009.jpg")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["affinity_geometry_g1"]
    deployment_cfg = config["affinity_deployment"]
    device = torch.device(
        "cuda"
        if config["sam2"].get("device") == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    system, reference_path, _, _ = build_system(config, cfg, device)
    geometry_path = Path(project_path(config, deployment_cfg["checkpoint"]))
    geometry_checkpoint = torch.load(
        geometry_path, map_location="cpu", weights_only=False
    )
    load_geometry_checkpoint_state(system, geometry_checkpoint)
    semantic_decoder, semantic_metadata = build_semantic_challenger(
        config, system, reference_path, device
    )
    if semantic_decoder is None:
        raise RuntimeError("fused deployment requires an enabled semantic challenger")
    semantic_path = Path(semantic_metadata["checkpoint"])
    semantic_checkpoint = torch.load(
        semantic_path, map_location="cpu", weights_only=False
    )

    fused = FusedPhaseAffinityModel(
        system.reference_model.encoder,
        semantic_decoder,
        system.geometry_decoder,
        geometry_feature_adapter=system.geometry_feature_adapter,
    ).to(device).eval()
    verify_path = Path(project_path(config, args.verify_image))
    image, tensor, _, _ = prepare_image(
        verify_path, int(cfg.get("input_size", 1024)), device
    )
    with torch.no_grad():
        reference_output = {
            key: value.detach().cpu()
            for key, value in fused(tensor).items()
        }

    decoder_cfg = dict(semantic_checkpoint.get("config", {}).get("decoder", {}))
    decoder_cfg.setdefault("fpn_channels", int(config["decoder"]["fpn_channels"]))
    decoder_cfg.setdefault("num_classes", int(config["decoder"]["num_classes"]))
    decoder_cfg.setdefault("dropout", float(config["decoder"]["dropout"]))
    decoder_cfg.setdefault("use_bn", bool(config["decoder"]["use_bn"]))
    adapter_cfg = cfg.get("feature_adapter", {})
    architecture = {
        "sam2": {
            "config_file": str(config["sam2"]["config_file"]),
            "sam2_repo_path": str(config["sam2"]["sam2_repo_path"]),
        },
        "lora": _lora_architecture(semantic_checkpoint),
        "semantic_decoder": decoder_cfg,
        "affinity_decoder": {
            "affinity_channels": int(system.geometry_decoder.affinity_channels),
            "fpn_channels": int(cfg.get("fpn_channels", 256)),
            "up_channels": int(cfg.get("up_channels", 128)),
            "output_grid": int(cfg.get("output_grid", 512)),
        },
        "geometry_feature_adapter": (
            {
                "bottleneck_ratio": int(adapter_cfg.get("bottleneck_ratio", 8)),
                "gate_mode": str(adapter_cfg.get("gate_mode", "scalar")),
                "active_scales": adapter_cfg.get("active_scales"),
            }
            if system.geometry_feature_adapter is not None else None
        ),
    }
    bundle = {
        "format": FUSED_DEPLOYMENT_FORMAT,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": architecture,
        "model_state_dict": {
            key: value.detach().cpu().contiguous()
            for key, value in fused.state_dict().items()
        },
        "parameter_summary": fused.parameter_summary(),
        "deployment": {
            "fusion": dict(deployment_cfg),
            "inference": dict(config["inference"]),
        },
        "sources": {
            "reference": {
                "path": str(Path(reference_path).resolve()),
                "sha256": file_sha256(reference_path),
            },
            "semantic": {
                **semantic_metadata,
                "sha256": file_sha256(semantic_path),
            },
            "affinity": {
                "path": str(geometry_path.resolve()),
                "sha256": file_sha256(geometry_path),
                "epoch": int(geometry_checkpoint.get("epoch", -1)),
            },
        },
    }
    output_path = Path(project_path(config, args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)

    del fused, system, semantic_decoder, geometry_checkpoint, semantic_checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded, reloaded_bundle = load_fused_deployment_model(
        output_path, config, device
    )
    with torch.no_grad():
        reloaded_output = reloaded(tensor)
    maximum_error = {
        key: float((reloaded_output[key].cpu() - value).abs().max())
        for key, value in reference_output.items()
    }
    if any(value != 0.0 for value in maximum_error.values()):
        raise RuntimeError(f"fused reload is not bit-exact: {maximum_error}")
    report = {
        "checkpoint": str(output_path.resolve()),
        "sha256": file_sha256(output_path),
        "bytes": int(output_path.stat().st_size),
        "format": reloaded_bundle["format"],
        "parameter_summary": reloaded_bundle["parameter_summary"],
        "verify_image": str(verify_path.resolve()),
        "reload_max_abs_error": maximum_error,
        "input_shape": list(image.shape),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
