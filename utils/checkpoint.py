# -*- coding: utf-8 -*-
"""Checkpoint metadata, compatibility validation, and payload creation."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Iterable, Optional


FORMAT_VERSION = 6


def architecture_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    decoder = config.get("decoder", {})
    lora = config.get("lora", {})
    gda = config.get("gda", {})
    edge_prior = config.get("edge_prior", {})
    sam2 = config.get("sam2", {})
    boundary_refine = bool(decoder.get("boundary_refine", False))
    return {
        "model": "SegmentationModel",
        "encoder": sam2.get("model_version", "sam2_hiera_base_plus"),
        "decoder": "FPNDecoder",
        "fpn_channels": int(decoder.get("fpn_channels", 256)),
        "num_classes": int(decoder.get("num_classes", 2)),
        "boundary_refine": boundary_refine,
        "boundary_refine_version": (
            decoder.get("boundary_refine_version", "legacy_lowres")
            if boundary_refine else "none"
        ),
        "center_head": bool(decoder.get("center_head", False)),
        "semantic_residual": bool(decoder.get("semantic_residual", False)),
        "semantic_residual_version": (
            str(decoder.get("semantic_residual_version", "lowres_v1"))
            if decoder.get("semantic_residual", False) else "none"
        ),
        "semantic_residual_hidden": (
            int(decoder.get("semantic_residual_hidden", 64))
            if decoder.get("semantic_residual", False) else 0
        ),
        "semantic_residual_color_channels": (
            int(decoder.get("semantic_residual_color_channels", 16))
            if decoder.get("semantic_residual", False) else 0
        ),
        "semantic_residual_use_photometric": (
            bool(decoder.get("semantic_residual_use_photometric", True))
            if decoder.get("semantic_residual", False) else False
        ),
        "semantic_residual_half_channels": (
            int(decoder.get("semantic_residual_half_channels", 48))
            if decoder.get("semantic_residual", False)
            and decoder.get("semantic_residual_version", "lowres_v1") == "highres_v1"
            else 0
        ),
        "semantic_residual_full_channels": (
            int(decoder.get("semantic_residual_full_channels", 24))
            if decoder.get("semantic_residual", False)
            and decoder.get("semantic_residual_version", "lowres_v1") == "highres_v1"
            else 0
        ),
        "lora_enabled": bool(lora.get("enabled", False)),
        "lora_rank": int(lora.get("rank", 0)) if lora.get("enabled", False) else 0,
        "gda_enabled": bool(gda.get("enabled", False)),
        "gda_bottleneck_ratio": (
            int(gda.get("bottleneck_ratio", 8))
            if gda.get("enabled", False) else 0
        ),
        "gda_gate_mode": (
            gda.get("gate_mode", "scalar") if gda.get("enabled", False) else "none"
        ),
        "gda_active_scales": (
            tuple(gda.get("active_scales", (0, 1, 2, 3)))
            if gda.get("enabled", False) else ()
        ),
        "edge_prior_enabled": bool(edge_prior.get("enabled", False)),
        "edge_prior_hidden_channels": (
            int(edge_prior.get("hidden_channels", 64))
            if edge_prior.get("enabled", False) else 0
        ),
        "edge_prior_fusion": bool(decoder.get("edge_prior_fusion", False)),
        "edge_prior_fusion_hidden": (
            int(decoder.get("edge_prior_fusion_hidden", 32))
            if decoder.get("edge_prior_fusion", False) else 0
        ),
    }


def architecture_from_state_dict(
    state_dict: Dict[str, Any], lora_state_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Infer the architecture of legacy checkpoints from parameter names."""
    keys = tuple(state_dict.keys())
    fpn_channels = None
    probe = state_dict.get("seg_fpn.lateral_convs.0.weight")
    if probe is not None and hasattr(probe, "shape"):
        fpn_channels = int(probe.shape[0])
    lora_rank = None
    for name, tensor in (lora_state_dict or {}).items():
        if "lora_A" in name and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            lora_rank = int(tensor.shape[0])
            break
    semantic_highres = any(
        k.startswith("semantic_residual.up_half.") for k in keys
    )
    return {
        "model": "SegmentationModel",
        "decoder": "FPNDecoder",
        "fpn_channels": fpn_channels,
        "num_classes": 2,
        "boundary_refine": any(k.startswith("boundary_refine_head.") for k in keys),
        "center_head": any(k.startswith("center_branch.") for k in keys),
        "semantic_residual": any(k.startswith("semantic_residual.") for k in keys),
        "semantic_residual_version": (
            "highres_v1" if semantic_highres else "lowres_v1"
        ) if any(k.startswith("semantic_residual.") for k in keys) else "none",
        "semantic_residual_hidden": (
            int(state_dict["semantic_residual.feature_path.0.weight"].shape[0])
            if "semantic_residual.feature_path.0.weight" in state_dict else 0
        ),
        "semantic_residual_color_channels": (
            int(state_dict["semantic_residual.photometric_path.0.weight"].shape[0])
            if "semantic_residual.photometric_path.0.weight" in state_dict
            else int(state_dict["semantic_residual.half_image_path.0.weight"].shape[0])
            if "semantic_residual.half_image_path.0.weight" in state_dict
            else 0
        ),
        "semantic_residual_use_photometric": any(
            k.startswith("semantic_residual.photometric_path.") for k in keys
        ) or (
            semantic_highres
            and state_dict.get("semantic_residual.half_image_path.0.weight") is not None
            and int(state_dict["semantic_residual.half_image_path.0.weight"].shape[1]) == 7
        ),
        "semantic_residual_half_channels": (
            int(state_dict["semantic_residual.up_half.0.weight"].shape[0])
            if "semantic_residual.up_half.0.weight" in state_dict else 0
        ),
        "semantic_residual_full_channels": (
            int(state_dict["semantic_residual.up_full.0.weight"].shape[0])
            if "semantic_residual.up_full.0.weight" in state_dict else 0
        ),
        "lora_enabled": bool(lora_state_dict),
        "lora_rank": lora_rank,
        "gda_enabled": False,
        "gda_bottleneck_ratio": 0,
        "gda_gate_mode": "none",
        "gda_active_scales": (),
        "edge_prior_enabled": False,
        "edge_prior_hidden_channels": 0,
        "edge_prior_fusion": any(
            k.startswith("edge_prior_fusion.") for k in keys
        ),
        "edge_prior_fusion_hidden": None,
    }


def checkpoint_architecture(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Return explicit metadata, falling back to legacy state-dict inference."""
    explicit = checkpoint.get("architecture")
    if explicit:
        return dict(explicit)
    state = checkpoint.get("decoder_state_dict", {})
    inferred = architecture_from_state_dict(state, checkpoint.get("lora_state_dict"))
    embedded = checkpoint.get("config")
    if embedded:
        configured = architecture_from_config(embedded)
        # Parameter keys are authoritative for optional decoder branches.  The
        # embedded config supplies fields that cannot be inferred reliably.
        configured.update({k: v for k, v in inferred.items() if v is not None})
        return configured
    return inferred


def architecture_mismatches(
    expected: Dict[str, Any], actual: Dict[str, Any],
    fields: Iterable[str] = (
        "encoder", "decoder", "fpn_channels", "num_classes", "boundary_refine",
        "boundary_refine_version", "center_head", "lora_enabled", "lora_rank",
        "semantic_residual", "semantic_residual_hidden",
        "semantic_residual_version",
        "semantic_residual_color_channels", "semantic_residual_use_photometric",
        "semantic_residual_half_channels", "semantic_residual_full_channels",
        "gda_enabled", "gda_bottleneck_ratio",
        "gda_gate_mode", "gda_active_scales",
        "edge_prior_enabled", "edge_prior_hidden_channels",
        "edge_prior_fusion", "edge_prior_fusion_hidden",
    ),
) -> Dict[str, tuple[Any, Any]]:
    mismatches = {}
    for field in fields:
        if field in expected and field in actual and actual[field] is not None:
            if expected[field] != actual[field]:
                mismatches[field] = (expected[field], actual[field])
    return mismatches


def validate_checkpoint_architecture(
    checkpoint: Dict[str, Any], config: Dict[str, Any], *, allow_mismatch: bool = False
) -> Dict[str, Any]:
    """Validate that inference config and checkpoint instantiate the same model."""
    expected = architecture_from_config(config)
    actual = checkpoint_architecture(checkpoint)
    mismatches = architecture_mismatches(expected, actual)
    if mismatches and not allow_mismatch:
        detail = ", ".join(
            f"{name}: config={left!r}, checkpoint={right!r}"
            for name, (left, right) in mismatches.items()
        )
        raise RuntimeError(
            "Checkpoint architecture does not match the active config ("
            + detail
            + "). Use the checkpoint's experiment config, or pass "
              "--allow-architecture-mismatch only for an intentional ablation."
        )
    return {"expected": expected, "actual": actual, "mismatches": mismatches}


def _git_commit(project_root: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def build_checkpoint(
    *, model: Any, config: Dict[str, Any], epoch: int,
    lora_state_dict: Dict[str, Any], best_composite_score: float,
    optimizer: Any = None, scheduler: Any = None,
) -> Dict[str, Any]:
    """Create the common checkpoint payload used by Stage 1 and Stage 2."""
    root = config.get("paths", {}).get("project_root", os.getcwd())
    payload = {
        "format_version": FORMAT_VERSION,
        "architecture": architecture_from_config(config),
        "provenance": {"git_commit": _git_commit(root)},
        "epoch": int(epoch),
        "decoder_state_dict": model.decoder.state_dict(),
        "lora_state_dict": lora_state_dict,
        "best_composite_score": float(best_composite_score),
        "config": config,
    }
    boundary_adapter = getattr(model, "boundary_adapter", None)
    if boundary_adapter is not None:
        payload["gda_state_dict"] = boundary_adapter.state_dict()
    edge_prior = getattr(model, "edge_prior", None)
    if edge_prior is not None:
        payload["edge_prior_state_dict"] = edge_prior.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload
