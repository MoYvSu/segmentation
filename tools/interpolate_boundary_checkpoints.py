"""Interpolate only the boundary decoder weights between two checkpoints.

This creates a single ordinary checkpoint. Semantic decoder weights and LoRA
state are copied from the base checkpoint unchanged, so the experiment isolates
how much of a boundary-only fine-tuning direction should be retained.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Iterable, Mapping

import torch


DEFAULT_PREFIXES = ("boundary_fpn.", "boundary_branch.")


def interpolate_decoder_state(
    base_state: Mapping[str, torch.Tensor],
    tuned_state: Mapping[str, torch.Tensor],
    alpha: float,
    prefixes: Iterable[str] = DEFAULT_PREFIXES,
):
    """Return base decoder state plus ``alpha`` of the boundary-only delta."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if set(base_state) != set(tuned_state):
        missing = sorted(set(base_state) - set(tuned_state))
        extra = sorted(set(tuned_state) - set(base_state))
        raise ValueError(f"decoder key mismatch: missing={missing}, extra={extra}")

    prefixes = tuple(prefixes)
    output = {}
    changed = []
    for key, base_value in base_state.items():
        tuned_value = tuned_state[key]
        if base_value.shape != tuned_value.shape:
            raise ValueError(
                f"shape mismatch for {key}: {base_value.shape} != {tuned_value.shape}"
            )
        if key.startswith(prefixes):
            if not (torch.is_floating_point(base_value) and torch.is_floating_point(tuned_value)):
                raise TypeError(f"boundary tensor must be floating point: {key}")
            mixed = base_value.float().lerp(tuned_value.float(), alpha)
            output[key] = mixed.to(dtype=base_value.dtype)
            changed.append(key)
        else:
            output[key] = base_value.clone()
    if not changed:
        raise ValueError(f"no decoder keys matched prefixes {prefixes}")
    return output, changed


def create_interpolated_checkpoint(base_checkpoint, tuned_checkpoint, alpha):
    if "decoder_state_dict" not in base_checkpoint:
        raise KeyError("base checkpoint has no decoder_state_dict")
    if "decoder_state_dict" not in tuned_checkpoint:
        raise KeyError("tuned checkpoint has no decoder_state_dict")

    output = copy.deepcopy(base_checkpoint)
    decoder_state, changed = interpolate_decoder_state(
        base_checkpoint["decoder_state_dict"],
        tuned_checkpoint["decoder_state_dict"],
        alpha,
    )
    output["decoder_state_dict"] = decoder_state
    output["interpolation"] = {
        "alpha": float(alpha),
        "prefixes": list(DEFAULT_PREFIXES),
        "changed_tensor_count": len(changed),
    }
    return output, changed


def alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 100)):03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate V6->N1 boundary weights without changing semantic/LoRA"
    )
    parser.add_argument("--base", required=True, help="Base checkpoint (V6)")
    parser.add_argument("--tuned", required=True, help="Boundary-tuned checkpoint (N1)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=False)
    os.makedirs(args.output_dir, exist_ok=True)

    for alpha in args.alphas:
        output, changed = create_interpolated_checkpoint(base, tuned, alpha)
        output["interpolation"].update(
            {
                "base_checkpoint": str(Path(args.base)),
                "tuned_checkpoint": str(Path(args.tuned)),
            }
        )
        output_path = os.path.join(
            args.output_dir, f"boundary_interpolation_{alpha_tag(alpha)}.pth"
        )
        torch.save(output, output_path)
        print(
            f"saved {output_path}: alpha={alpha:.3f}, "
            f"boundary_tensors={len(changed)}"
        )


if __name__ == "__main__":
    main()
