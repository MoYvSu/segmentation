# -*- coding: utf-8 -*-
"""Shared affinity deployment inference used by training validation and audits."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from data.dataset import letterbox
from utils.affinity_fusion import affinity_boundary_probability
from utils.post_process import post_process_prediction_boundary


def affinity_mean_boundary_probability(affinity_logits: torch.Tensor) -> torch.Tensor:
    return affinity_boundary_probability(affinity_logits, mode="mean")


def probability_to_logit(probability: torch.Tensor, eps: float = 1.0e-5):
    value = probability.clamp(float(eps), 1.0 - float(eps))
    return torch.log(value) - torch.log1p(-value)


def crop_letterbox_output(
    output: torch.Tensor,
    image_size: int,
    pad_h: int,
    pad_w: int,
    original_size,
):
    out_h, out_w = output.shape[-2:]
    content_h = max(1, int(round((image_size - pad_h) * out_h / image_size)))
    content_w = max(1, int(round((image_size - pad_w) * out_w / image_size)))
    output = output[:, :, :content_h, :content_w]
    return F.interpolate(output, size=original_size, mode="bilinear", align_corners=True)


def prepare_image(image_path: str | Path, image_size: int, device):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_lb, _, pad_h, pad_w = letterbox(rgb, image_size)
    tensor = (
        torch.from_numpy(image_lb).permute(2, 0, 1).float().unsqueeze(0).to(device)
        / 255.0
    )
    return rgb, tensor, int(pad_h), int(pad_w)


@torch.no_grad()
def predict_maps(
    system,
    image_path,
    image_size,
    device,
    fusion_mode="mean",
    fusion_kwargs=None,
):
    image, tensor, pad_h, pad_w = prepare_image(image_path, image_size, device)
    original_size = image.shape[:2]
    reference_output = system.reference_model(tensor)
    affinity_output = system.geometry_forward(tensor)["affinity_logits"]
    reference_native = crop_letterbox_output(
        reference_output, image_size, pad_h, pad_w, original_size
    ).cpu()
    affinity_boundary = affinity_boundary_probability(
        affinity_output,
        mode=fusion_mode,
        **(fusion_kwargs or {}),
    )
    affinity_boundary_native = crop_letterbox_output(
        affinity_boundary, image_size, pad_h, pad_w, original_size
    ).cpu()
    affinity_logits_native = probability_to_logit(affinity_boundary_native)
    affinity_watershed_output = torch.cat(
        [reference_native[:, :1], affinity_logits_native], dim=1
    )
    return image, reference_native, affinity_watershed_output, affinity_boundary_native


def postprocess(
    output,
    original_size,
    output_dir,
    basename,
    infer_cfg,
    boundary_threshold,
    save_visualization,
    image_rgb=None,
):
    return post_process_prediction_boundary(
        output=output,
        original_size=original_size,
        output_dir=str(output_dir),
        image_basename=basename,
        min_instance_area=int(infer_cfg.get("min_instance_area", 50)),
        max_instance_id=int(infer_cfg.get("max_instance_id", 255)),
        threshold=float(infer_cfg.get("threshold", 0.5)),
        boundary_threshold=float(boundary_threshold),
        boundary_logit_scale=1.0,
        sem_edge_boost_alpha=0.0,
        sem_edge_merge_weight=0.0,
        watershed_dilate_width=int(infer_cfg.get("watershed_dilate_width", 1)),
        bridge_width=int(infer_cfg.get("bridge_width", 1)),
        marker_border_seal_width=int(
            infer_cfg.get("marker_border_seal_width", 0)
        ),
        marker_boundary_low_threshold=infer_cfg.get(
            "marker_boundary_low_threshold"
        ),
        marker_boundary_reconstruction_steps=int(
            infer_cfg.get("marker_boundary_reconstruction_steps", 0)
        ),
        semantic_vote_mode=str(
            infer_cfg.get("semantic_vote_mode", "hard_majority")
        ),
        semantic_vote_erode_width=int(
            infer_cfg.get("semantic_vote_erode_width", 0)
        ),
        semantic_vote_threshold=float(
            infer_cfg.get("semantic_vote_threshold", 0.5)
        ),
        semantic_vote_options={
            "core_fraction": float(
                infer_cfg.get("semantic_vote_core_fraction", 0.40)
            ),
            "core_min_pixels": int(
                infer_cfg.get("semantic_vote_core_min_pixels", 8)
            ),
            "core_distance_power": float(
                infer_cfg.get("semantic_vote_core_distance_power", 2.0)
            ),
            "color_uncertain_low": float(
                infer_cfg.get("semantic_vote_color_uncertain_low", 0.35)
            ),
            "color_uncertain_high": float(
                infer_cfg.get("semantic_vote_color_uncertain_high", 0.65)
            ),
            "color_weight": float(
                infer_cfg.get("semantic_vote_color_weight", 0.25)
            ),
            "color_min_separation": float(
                infer_cfg.get("semantic_vote_color_min_separation", 1.0)
            ),
        },
        original_image_rgb=image_rgb,
        use_center_seeds=False,
        save_visualization=save_visualization,
    )
