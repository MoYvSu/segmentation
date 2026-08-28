# -*- coding: utf-8 -*-
"""
推理入口（边界预测版本）
========================
加载训练好的 FPN decoder 权重，对测试图像执行前向推理，
执行边界骨架化 + 受阻分水岭 + 语义投票实现实例分割。

所有图像通过 Letterbox 处理到 1024x1024（与训练一致）。
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import letterbox
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.edge_prior import load_pretrained_edge_prior
from models.gda_mim import load_pretrained_gda
from models.sam2_encoder import SAM2Encoder
from utils.checkpoint import (
    checkpoint_architecture,
    validate_checkpoint_architecture,
)
from utils.config import load_config, project_path
from utils.post_process import post_process_prediction_boundary
from utils.scale_policy import resolution_scaled_min_area
from utils.quality_aware import (
    assess_image_quality,
    classify_quality,
    effective_boundary_threshold,
    enhance_weak_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_model(
    config, device, checkpoint_path=None, *, allow_architecture_mismatch=False
):
    """Build model and load trained decoder weights."""
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]

    ckpt_path = project_path(config, paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    if not os.path.exists(ckpt_path):
        logger.warning(f"SAM 2 checkpoint not found: {ckpt_path}")

    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
        device=device,
        freeze=True,
        sam2_repo_path=os.path.join(paths_cfg["project_root"], sam2_cfg["sam2_repo_path"]),
    )

    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=decoder_cfg["fpn_channels"],
        num_classes=decoder_cfg["num_classes"],
        dropout=decoder_cfg["dropout"],
        use_bn=decoder_cfg["use_bn"],
        boundary_refine=decoder_cfg.get("boundary_refine", False),
        boundary_refine_version=decoder_cfg.get(
            "boundary_refine_version", "legacy_lowres"
        ),
        center_head=decoder_cfg.get("center_head", False),
        edge_prior_fusion=decoder_cfg.get("edge_prior_fusion", False),
        edge_prior_fusion_hidden=decoder_cfg.get(
            "edge_prior_fusion_hidden", 32
        ),
        edge_prior_max_logit_delta=decoder_cfg.get(
            "edge_prior_max_logit_delta", 1.0
        ),
        semantic_residual=decoder_cfg.get("semantic_residual", False),
        semantic_residual_hidden=decoder_cfg.get("semantic_residual_hidden", 64),
        semantic_residual_color_channels=decoder_cfg.get(
            "semantic_residual_color_channels", 16
        ),
        semantic_residual_use_photometric=decoder_cfg.get(
            "semantic_residual_use_photometric", True
        ),
        semantic_residual_max_logit_delta=decoder_cfg.get(
            "semantic_residual_max_logit_delta", 2.0
        ),
    )

    boundary_adapter = None
    gda_cfg = config.get("gda", {})
    if gda_cfg.get("enabled", False):
        gda_checkpoint = project_path(config, gda_cfg.get("checkpoint", ""))
        if not gda_checkpoint or not os.path.exists(gda_checkpoint):
            raise FileNotFoundError(
                f"GDA checkpoint not found: {gda_checkpoint or '<empty>'}"
            )
        boundary_adapter = load_pretrained_gda(
            gda_checkpoint,
            channels=encoder.get_stage_channels(),
            bottleneck_ratio=int(gda_cfg.get("bottleneck_ratio", 8)),
            gate_mode=gda_cfg.get("gate_mode", "scalar"),
            active_scales=gda_cfg.get("active_scales"),
            map_location=device,
            freeze_adapters=True,
            train_gates=False,
        )
        logger.info("Boundary GDA enabled: %s", gda_checkpoint)

    edge_prior = None
    edge_prior_cfg = config.get("edge_prior", {})
    if edge_prior_cfg.get("enabled", False):
        prior_checkpoint = project_path(
            config, edge_prior_cfg.get("checkpoint", "")
        )
        if not prior_checkpoint or not os.path.exists(prior_checkpoint):
            raise FileNotFoundError(
                f"Edge-prior checkpoint not found: {prior_checkpoint or '<empty>'}"
            )
        edge_prior = load_pretrained_edge_prior(
            prior_checkpoint,
            in_channels=encoder.get_stage_channels()[:2],
            hidden_channels=int(edge_prior_cfg.get("hidden_channels", 64)),
            map_location=device,
        )
        logger.info("Retained G0b edge prior enabled: %s", prior_checkpoint)
    if decoder_cfg.get("edge_prior_fusion", False) and edge_prior is None:
        raise ValueError(
            "decoder.edge_prior_fusion=True requires edge_prior.enabled=True"
        )

    model = SegmentationModel(
        encoder, decoder, boundary_adapter=boundary_adapter,
        edge_prior=edge_prior,
    )
    model = model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading decoder weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        validation = validate_checkpoint_architecture(
            checkpoint, config, allow_mismatch=allow_architecture_mismatch
        )
        if validation["mismatches"]:
            logger.warning("Intentional architecture mismatch: %s", validation["mismatches"])
        from models.fpn_decoder import load_decoder_state
        load_decoder_state(model.decoder, checkpoint["decoder_state_dict"])
        if model.edge_prior is not None and checkpoint.get("edge_prior_state_dict"):
            model.edge_prior.load_state_dict(
                checkpoint["edge_prior_state_dict"], strict=True
            )
            model.edge_prior.eval()
            logger.info("Embedded retained edge-prior state loaded")
        if model.boundary_adapter is not None:
            gda_state = checkpoint.get("gda_state_dict")
            if not gda_state:
                raise KeyError(
                    "GDA inference config requires checkpoint.gda_state_dict"
                )
            model.boundary_adapter.load_state_dict(gda_state, strict=True)
            logger.info(
                "GDA state loaded; gate_stats=%s",
                model.boundary_adapter.gate_statistics(),
            )
        # 含 lora_state_dict 的检查点：注入并加载 LoRA（trunk 域适配）
        from models.lora import load_lora_from_checkpoint
        load_lora_from_checkpoint(model, checkpoint)
        model.to(device)   # 注入发生在 .to() 之后，需再移动一次 LoRA 参数
        # 兼容新旧 checkpoint key
        best_score = checkpoint.get(
            "best_composite_score", checkpoint.get("best_val_iou", "?")
        )
        logger.info(
            f"Weights loaded (epoch={checkpoint.get('epoch', '?')}, "
            f"best_score={best_score})"
        )
    else:
        logger.warning("No decoder weights loaded, using random init!")

    model.eval()
    return model


@torch.no_grad()
def _predict_with_tta(model, image_tensor, use_tta=False):
    """固定几何视角平均；不更新模型或归一化统计。"""
    if not use_tta:
        return model(image_tensor)
    views = torch.cat(
        [
            image_tensor,
            torch.flip(image_tensor, dims=[3]),
            torch.flip(image_tensor, dims=[2]),
            torch.rot90(image_tensor, 2, dims=[2, 3]),
        ],
        dim=0,
    )
    outs = model(views)
    return (
        outs[0:1]
        + torch.flip(outs[1:2], dims=[3])
        + torch.flip(outs[2:3], dims=[2])
        + torch.rot90(outs[3:4], 2, dims=[2, 3])
    ) / 4.0


def predict_single_image(
    model, image_path, device,
    image_size=1024,
    min_instance_area=50, max_instance_id=255,
    threshold=0.5, boundary_threshold=0.5,
    boundary_logit_scale=1.0,
    sem_edge_boost_alpha=0.0,
    sem_edge_merge_weight=0.0,
    sem_edge_smooth=1.0,
    use_tta=False,
    bridge_width=1,
    watershed_dilate_width=2,
    marker_border_seal_width=0,
    marker_boundary_low_threshold=None,
    marker_boundary_reconstruction_steps=0,
    semantic_vote_mode="hard_majority",
    semantic_vote_erode_width=0,
    semantic_vote_threshold=0.5,
    use_center_seeds=True, center_threshold=0.25, center_nms_kernel=9,
    min_instance_area_policy=None,
    quality_aware_config=None,
    output_dir=None, save_visualization=True,
):
    """Letterbox inference + boundary watershed post-processing."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]
    effective_min_instance_area = resolution_scaled_min_area(
        min_instance_area, (h_orig, w_orig), min_instance_area_policy
    )
    quality_cfg = quality_aware_config or {}
    quality_enabled = bool(quality_cfg.get("enabled", False))
    quality_metrics = assess_image_quality(image_rgb) if quality_enabled else {}
    if quality_enabled:
        quality_profile, quality_reasons = classify_quality(
            quality_metrics, quality_cfg
        )
    else:
        quality_profile, quality_reasons = "disabled", []

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Letterbox
    image_lb, scale, pad_h, pad_w = letterbox(image_rgb, image_size)
    image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    output = _predict_with_tta(model, image_tensor, use_tta=use_tta)
    enhanced_applied = []
    if quality_enabled and quality_profile == "weak":
        enhanced_rgb, enhanced_applied = enhance_weak_image(
            image_rgb, quality_metrics, quality_cfg
        )
        enhanced_lb, _, _, _ = letterbox(enhanced_rgb, image_size)
        enhanced_tensor = (
            torch.from_numpy(enhanced_lb)
            .float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        )
        enhanced_output = _predict_with_tta(
            model,
            enhanced_tensor,
            use_tta=bool(quality_cfg.get("enhanced_tta", False)),
        )
        enhanced_weight = float(quality_cfg.get("enhanced_weight", 0.35))
        enhanced_weight = float(np.clip(enhanced_weight, 0.0, 1.0))
        output = output * (1.0 - enhanced_weight) + enhanced_output * enhanced_weight

    selected_boundary_threshold = effective_boundary_threshold(
        boundary_threshold, quality_profile, quality_cfg
    ) if quality_enabled else float(boundary_threshold)

    # Inverse Letterbox: crop the bottom/right padding in the model's actual
    # output resolution.  The legacy decoder emitted at stride 4 (256 for a
    # 1024 input), while boundary_refine emits at full resolution.  A fixed
    # ``/4`` therefore crops a full-resolution prediction down to its top-left
    # quarter and causes severe under-segmentation after it is stretched back.
    out_h, out_w = output.shape[-2:]
    resized_h = image_size - pad_h
    resized_w = image_size - pad_w
    content_h = int(round(resized_h * out_h / image_size))
    content_w = int(round(resized_w * out_w / image_size))
    content_h = max(1, min(content_h, out_h))
    content_w = max(1, min(content_w, out_w))
    output = output[:, :, :content_h, :content_w]
    output = F.interpolate(output, size=(h_orig, w_orig), mode="bilinear", align_corners=True)

    if output_dir is not None:
        output_paths, inst_map, class_map = post_process_prediction_boundary(
            output=output,
            original_size=(h_orig, w_orig),
            output_dir=output_dir,
            image_basename=basename,
            min_instance_area=effective_min_instance_area,
            max_instance_id=max_instance_id,
            threshold=threshold,
            boundary_threshold=selected_boundary_threshold,
            boundary_logit_scale=boundary_logit_scale,
            sem_edge_boost_alpha=sem_edge_boost_alpha,
            sem_edge_merge_weight=sem_edge_merge_weight,
            sem_edge_smooth=sem_edge_smooth,
            watershed_dilate_width=watershed_dilate_width,
            bridge_width=bridge_width,
            marker_border_seal_width=marker_border_seal_width,
            marker_boundary_low_threshold=marker_boundary_low_threshold,
            marker_boundary_reconstruction_steps=marker_boundary_reconstruction_steps,
            semantic_vote_mode=semantic_vote_mode,
            semantic_vote_erode_width=semantic_vote_erode_width,
            semantic_vote_threshold=semantic_vote_threshold,
            use_center_seeds=use_center_seeds,
            center_threshold=center_threshold,
            center_nms_kernel=center_nms_kernel,
            save_visualization=save_visualization,
        )
    else:
        output_paths = {}
        inst_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
        class_map = {}

    n_ferrite = sum(1 for v in class_map.values() if v == 1)
    n_pearlite = sum(1 for v in class_map.values() if v == 0)
    if len(class_map) > min(int(max_instance_id), 255):
        raise RuntimeError(
            f"Instance cap violated: {len(class_map)} > {min(int(max_instance_id), 255)}"
        )

    return {
        "image_path": image_path,
        "original_size": (h_orig, w_orig),
        "num_instances": len(class_map),
        "num_ferrite": n_ferrite,
        "num_pearlite": n_pearlite,
        "output_paths": output_paths,
        "effective_min_instance_area": effective_min_instance_area,
        "quality_profile": quality_profile,
        "quality_reasons": quality_reasons,
        "quality_metrics": quality_metrics,
        "enhanced_applied": enhanced_applied,
        "effective_boundary_threshold": selected_boundary_threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Metallographic segmentation inference (boundary prediction version)",
    )
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tta", action="store_true",
                        help="推理 TTA（hflip/vflip/rot180 平均）")
    parser.add_argument(
        "--quality-aware", action=argparse.BooleanOptionalAction, default=None,
        help="启用/关闭单图质量感知增强与小幅边界阈值档位选择",
    )
    parser.add_argument("--boundary-threshold", type=float, default=None)
    parser.add_argument("--min-instance-area", type=int, default=None)
    parser.add_argument("--center-threshold", type=float, default=None)
    parser.add_argument("--center-seeds", action=argparse.BooleanOptionalAction,
                        default=None, help="启用/关闭中心热图种子")
    parser.add_argument("--save-visualization", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--allow-architecture-mismatch", action="store_true",
                        help="仅用于有意消融；允许配置与 checkpoint 架构不一致")
    args = parser.parse_args()

    config = load_config(args.config)
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]
    infer_cfg = config["inference"]
    post_cfg = config["post_process"]
    data_cfg = config["data"]

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    test_dir = project_path(config, args.test_dir or infer_cfg["test_dir"])
    output_dir = project_path(config, args.output_dir or infer_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Checkpoint path
    if args.checkpoint:
        checkpoint_path = project_path(config, args.checkpoint)
        logger.info(f"Using checkpoint from CLI: {checkpoint_path}")
    else:
        checkpoint_stage = infer_cfg.get("checkpoint_stage", "stage1")
        if checkpoint_stage == "stage2":
            checkpoint_path = project_path(
                config, infer_cfg.get("stage2_checkpoint", "outputs/stage2/best_model_stage2.pth")
            )
        else:
            checkpoint_path = project_path(
                config, infer_cfg.get("stage1_checkpoint", "outputs/stage1/best_model.pth")
            )
        logger.info(
            f"Using {checkpoint_stage} checkpoint from config: {checkpoint_path}"
        )

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise SystemExit(2)

    logger.info(f"Test dir: {test_dir}")
    logger.info(f"Output dir: {output_dir}")

    model = build_model(
        config, device, checkpoint_path,
        allow_architecture_mismatch=args.allow_architecture_mismatch,
    )

    quality_cfg = dict(infer_cfg.get("quality_aware", {}))
    if args.quality_aware is not None:
        quality_cfg["enabled"] = bool(args.quality_aware)

    effective = {
        "boundary_threshold": (
            args.boundary_threshold if args.boundary_threshold is not None
            else infer_cfg.get("boundary_threshold", 0.5)
        ),
        "min_instance_area": (
            args.min_instance_area if args.min_instance_area is not None
            else infer_cfg.get("min_instance_area", 50)
        ),
        "center_seeds": (
            args.center_seeds if args.center_seeds is not None
            else infer_cfg.get("center_seeds", False)
        ),
        "center_threshold": (
            args.center_threshold if args.center_threshold is not None
            else infer_cfg.get("center_threshold", 0.25)
        ),
        "tta": bool(infer_cfg.get("tta", False) or args.tta),
        "quality_aware": bool(quality_cfg.get("enabled", False)),
        "save_visualization": (
            args.save_visualization if args.save_visualization is not None
            else post_cfg.get("save_visualization", False)
        ),
    }
    logger.info("Effective inference settings: %s", effective)

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        logger.error(f"No images found in test dir: {test_dir}")
        return

    logger.info(f"Found {len(image_paths)} test images")

    all_results = []
    total_time = 0.0

    for img_path in image_paths:
        start_time = time.time()
        logger.info(f"Processing: {os.path.basename(img_path)}")

        result = predict_single_image(
            model, img_path, device,
            image_size=data_cfg["image_size"],
            min_instance_area=effective["min_instance_area"],
            max_instance_id=infer_cfg["max_instance_id"],
            threshold=infer_cfg.get("threshold", 0.5),
            boundary_threshold=effective["boundary_threshold"],
            boundary_logit_scale=infer_cfg.get("boundary_logit_scale", 1.0),
            sem_edge_boost_alpha=infer_cfg.get("sem_edge_boost_alpha", 0.0),
            sem_edge_merge_weight=infer_cfg.get("sem_edge_merge_weight", 0.0),
            sem_edge_smooth=infer_cfg.get("sem_edge_smooth", 1.0),
            use_tta=effective["tta"],
            watershed_dilate_width=infer_cfg.get("watershed_dilate_width", 2),
            bridge_width=infer_cfg.get("bridge_width", 1),
            marker_border_seal_width=infer_cfg.get("marker_border_seal_width", 0),
            marker_boundary_low_threshold=infer_cfg.get(
                "marker_boundary_low_threshold"
            ),
            marker_boundary_reconstruction_steps=infer_cfg.get(
                "marker_boundary_reconstruction_steps", 0
            ),
            semantic_vote_mode=infer_cfg.get(
                "semantic_vote_mode", "hard_majority"
            ),
            semantic_vote_erode_width=infer_cfg.get(
                "semantic_vote_erode_width", 0
            ),
            semantic_vote_threshold=infer_cfg.get(
                "semantic_vote_threshold", 0.5
            ),
            use_center_seeds=effective["center_seeds"],
            center_threshold=effective["center_threshold"],
            center_nms_kernel=infer_cfg.get("center_nms_kernel", 9),
            min_instance_area_policy=infer_cfg.get("resolution_aware_min_area", {}),
            quality_aware_config=quality_cfg,
            output_dir=output_dir,
            save_visualization=effective["save_visualization"],
        )

        elapsed = time.time() - start_time
        total_time += elapsed
        all_results.append(result)

        logger.info(
            f"  Done ({elapsed:.2f}s): "
            f"instances={result['num_instances']} "
            f"(ferrite={result['num_ferrite']}, pearlite={result['num_pearlite']}) "
            f"quality={result['quality_profile']} "
            f"bnd_th={result['effective_boundary_threshold']:.3f}"
        )

    logger.info("=" * 60)
    logger.info("Inference complete:")
    logger.info(f"  Total images: {len(all_results)}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time / len(all_results):.2f}s/img)")
    total_instances = sum(r["num_instances"] for r in all_results)
    total_ferrite = sum(r["num_ferrite"] for r in all_results)
    total_pearlite = sum(r["num_pearlite"] for r in all_results)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_architecture": checkpoint_architecture(checkpoint),
        "effective_inference": effective,
        "test_dir": os.path.abspath(test_dir),
        "images": len(all_results),
        "instances": {
            "total": total_instances,
            "ferrite": total_ferrite,
            "pearlite": total_pearlite,
        },
        "quality_aware_results": [
            {
                "image": os.path.basename(result["image_path"]),
                "profile": result["quality_profile"],
                "reasons": result["quality_reasons"],
                "metrics": result["quality_metrics"],
                "enhanced_applied": result["enhanced_applied"],
                "boundary_threshold": result["effective_boundary_threshold"],
                "instances": result["num_instances"],
            }
            for result in all_results
        ],
    }
    with open(os.path.join(output_dir, "inference_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"  Total instances: {total_instances} (ferrite={total_ferrite}, pearlite={total_pearlite})")
    logger.info(f"  Output saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
