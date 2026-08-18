# -*- coding: utf-8 -*-
"""Checkpoint metadata, compatibility validation, and payload creation."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Iterable, Optional


FORMAT_VERSION = 2


def architecture_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    decoder = config.get("decoder", {})
    lora = config.get("lora", {})
    sam2 = config.get("sam2", {})
    return {
        "model": "SegmentationModel",
        "encoder": sam2.get("model_version", "sam2_hiera_base_plus"),
        "decoder": "FPNDecoder",
        "fpn_channels": int(decoder.get("fpn_channels", 256)),
        "num_classes": int(decoder.get("num_classes", 2)),
        "boundary_refine": bool(decoder.get("boundary_refine", False)),
        "center_head": bool(decoder.get("center_head", False)),
        "lora_enabled": bool(lora.get("enabled", False)),
        "lora_rank": int(lora.get("rank", 0)) if lora.get("enabled", False) else 0,
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
    return {
        "model": "SegmentationModel",
        "decoder": "FPNDecoder",
        "fpn_channels": fpn_channels,
        "num_classes": 2,
        "boundary_refine": any(k.startswith("boundary_refine_head.") for k in keys),
        "center_head": any(k.startswith("center_branch.") for k in keys),
        "lora_enabled": bool(lora_state_dict),
        "lora_rank": lora_rank,
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
        "center_head", "lora_enabled", "lora_rank",
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
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload
