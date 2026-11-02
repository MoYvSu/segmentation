# -*- coding: utf-8 -*-
"""V6 carrier 删除的合成 CPU 桥接；不读取赛题数据或真实 checkpoint。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.lora import LoRALinear
from train_stage2 import build_teacher_model, set_student_train_modes, validate
from utils.config import load_config
from utils.loss import BoundaryLoss
from utils.loss_semi import compute_unsupervised_loss


SEMANTIC = ("decoder.seg_fpn.", "decoder.seg_branch.", "decoder.semantic_residual.")


def digest(state):
    out = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        out.update(name.encode())
        out.update(str((value.dtype, tuple(value.shape))).encode())
        out.update(value.numpy().tobytes())
    return out.hexdigest()


def select_state(model, prefixes):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(prefixes)}


class TinySAM2(nn.Module):
    """仅编码器缩小；LoRA、解码器、GroupNorm 和训练损失均使用项目实现。"""
    def __init__(self):
        super().__init__()
        self.trunk = LoRALinear(nn.Linear(3, 8), rank=2, alpha=4)
        self.projections = nn.ModuleList(nn.Conv2d(8, n, 1) for n in (8, 16, 32, 64))
        with torch.no_grad():
            self.trunk.lora_B.copy_(torch.linspace(-.05, .05, 16).reshape(8, 2))
        self.trainable_lora = False
        self.requires_grad_(False)

    def forward(self, image):
        x = F.avg_pool2d(image, 4).permute(0, 2, 3, 1)
        x = self.trunk(x).permute(0, 3, 1, 2)
        return [layer(F.avg_pool2d(x, 2 ** i)) for i, layer in enumerate(self.projections)]


def recipe():
    cfg = load_config(str(ROOT / "config/train/stage2_semantic_gt_aligned20.yaml"))
    semi = cfg["semi_supervised"]
    assert semi["freeze"] == {"seg_branch": False, "boundary_branch": True}
    assert semi["selection_metric"] == "val_loss"
    assert semi.get("sem_boundary_align_weight", 0) == 0
    assert cfg["decoder"]["semantic_residual_version"] == "highres_v1"
    return cfg


def criterion_from_recipe(cfg, freeze_boundary=True):
    train = cfg["train"]
    # 与 train_stage2 的当前 S-align 参数一致；未启用的边界增强项不参与冻结语义损失。
    names = ("seg_dice_weight", "semantic_instance_weight", "semantic_core_radius",
             "semantic_core_min_pixels", "semantic_core_boundary_threshold",
             "semantic_instance_class_balance", "semantic_instance_ferrite_weight",
             "semantic_instance_hard_gamma", "semantic_instance_hard_floor",
             "semantic_instance_pool_weight", "semantic_thin_instance_weight",
             "semantic_tversky_weight", "semantic_tversky_alpha", "semantic_tversky_beta")
    kwargs = {k: train[k] for k in names if k in train}
    return BoundaryLoss(gamma=train.get("focal_gamma", 2),
                        alpha_boundary=train.get("boundary_alpha", 1),
                        alpha_focal=train.get("focal_alpha", .75),
                        freeze_seg=False, freeze_boundary=freeze_boundary, **kwargs)


def synthetic_batch():
    generator = torch.Generator().manual_seed(667)
    image = torch.rand((2, 3, 64, 64), generator=generator)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    ids = 1 + (xx >= 32).long() + 2 * (yy >= 32).long()
    semantic = (xx >= 32).float()
    boundary = ((xx == 31) | (xx == 32) | (yy == 31) | (yy == 32)).float()
    targets = torch.stack((semantic, boundary))[None].repeat(2, 1, 1, 1)
    targets[:, :, :3, :3] = -1
    ids = ids[None].repeat(2, 1, 1)
    ids[:, :3, :3] = 0
    return dict(image=image, target=targets, weight=torch.ones((2, 1, 64, 64)), instance_map=ids)


def run_worker(variant, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = recipe()
    decoder = FPNDecoder(in_channels=[8, 16, 32, 64], fpn_channels=16, dropout=.1,
        use_bn=True, semantic_residual=True, semantic_residual_version="highres_v1",
        semantic_residual_hidden=64, semantic_residual_color_channels=16,
        semantic_residual_half_channels=48, semantic_residual_full_channels=24,
        semantic_residual_max_logit_delta=2.)
    model = SegmentationModel(TinySAM2(), decoder)
    # 使无标签蒸馏确有高置信有效像素，不让零 loss 掩盖依赖差异。
    with torch.no_grad():
        model.decoder.seg_branch[-1].bias.add_(3.)
        if variant == "v6":
            for name, parameter in model.decoder.named_parameters():
                if name.startswith(("boundary_fpn.", "boundary_branch.")):
                    parameter.mul_(1.05)
            model.decoder.boundary_branch[-1].bias.add_(4.)
    initial_encoder = select_state(model, ("encoder.",))
    teacher = build_teacher_model(model)
    batch = synthetic_batch()
    with torch.no_grad():
        teacher_output = teacher(batch["image"], output_size=(64, 64))
        teacher_probability = teacher_output[:, 0].sigmoid()
        features = teacher.encoder(batch["image"])
    model.decoder.reset_semantic_branch()
    model.decoder.set_semantic_cold_start_only()
    model.encoder.requires_grad_(False)
    initial_semantic = select_state(model, SEMANTIC)
    boundary_before = select_state(model, ("decoder.boundary_fpn.", "decoder.boundary_branch."))
    criterion = criterion_from_recipe(cfg)
    semi = cfg["semi_supervised"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=float(semi["learning_rate"]) * semi["seg_lr_ratio"],
        weight_decay=float(cfg["train"]["weight_decay"]))
    vals, rows, gradients, states = [], [], [], []
    vals.append(validate(model, [batch], criterion, "cpu")["loss"])
    for step in range(3):
        set_student_train_modes(model)
        optimizer.zero_grad()
        prediction = model(batch["image"], output_size=(64, 64))
        supervised, seg_value, boundary_value = criterion(prediction, batch["target"],
            batch["weight"], instance_map=batch["instance_map"])
        unsup, unsup_boundary, stats = compute_unsupervised_loss(model,
            batch["image"], batch["image"] * .95, torch.zeros((2, 1, 64, 64)),
            output_size=(64, 64), teacher_model=teacher, freeze_boundary=True,
            seg_mask_region_weight=semi.get("seg_mask_region_weight", 1.),
            seg_sharpen_temperature=semi.get("unsup_seg_sharpen_temperature", 1.),
            seg_confidence_threshold=semi["unsup_seg_confidence_threshold"],
            sem_boundary_align_weight=semi.get("sem_boundary_align_weight", 0.))
        loss = supervised + .12 * unsup
        loss.backward()
        gradients.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                          if p.requires_grad and p.grad is not None})
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        states.append(select_state(model, SEMANTIC))
        rows.append(dict(step=step + 1, total=float(loss.detach()), supervised=seg_value,
                         boundary=boundary_value, unsup=float(unsup.detach()),
                         unsup_boundary=float(unsup_boundary), stats=stats))
        vals.append(validate(model, [batch], criterion, "cpu")["loss"])
    # 真正的反例门：错误地打开边界 loss 后，两个 carrier 的验证值应不同。
    wrong_val = validate(model, [batch], criterion_from_recipe(cfg, False), "cpu")["loss"]
    payload = dict(variant=variant, teacher_probability=teacher_probability,
        teacher_boundary=teacher_output[:, 1], features=features,
        encoder=initial_encoder, encoder_after=select_state(model, ("encoder.",)),
        lora_sha256=digest({k: v for k, v in initial_encoder.items() if "lora_" in k}),
        initial_semantic=initial_semantic, boundary_before=boundary_before,
        boundary_after=select_state(model, ("decoder.boundary_fpn.", "decoder.boundary_branch.")),
        gradient_steps=gradients, semantic_steps=states, optimizer=optimizer.state_dict(),
        rows=rows, val_loss=vals, best_epoch_1based=min(range(1, 4), key=lambda i: vals[i]),
        wrong_unfrozen_boundary_val=wrong_val, rng=torch.get_rng_state(),
        group_norm_count=sum(isinstance(m, nn.GroupNorm) for m in decoder.modules()),
        highres_version=decoder.semantic_residual_version)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(x, y) for x, y in zip(left, right))
    return left == right


def rounding_example():
    a = torch.tensor(1., dtype=torch.float32)
    b = torch.nextafter(a, torch.tensor(0.))
    offset = torch.tensor(8.)
    scores = [float(a), float(b)]
    with_offset = [float(a + offset), float(b + offset)]
    return dict(semantic_scores=scores, counterfactual_scores=with_offset,
                semantic_best=1 + scores.index(min(scores)),
                counterfactual_best=1 + with_offset.index(min(with_offset)),
                applies_to_current_salign=False,
                reason="BoundaryLoss(freeze_boundary=True) skips boundary loss before addition")


def compare(left, right):
    checks = {key: exact(left[key], right[key]) for key in (
        "teacher_probability", "features", "encoder", "lora_sha256", "initial_semantic",
        "gradient_steps", "semantic_steps", "optimizer", "rows", "val_loss",
        "best_epoch_1based", "rng")}
    checks.update(boundary_carriers_differ=not exact(left["boundary_before"], right["boundary_before"]),
                  boundary_outputs_differ=not exact(left["teacher_boundary"], right["teacher_boundary"]),
                  wrong_unfrozen_validation_differ=left["wrong_unfrozen_boundary_val"] != right["wrong_unfrozen_boundary_val"],
                  encoder_unchanged=all(exact(x["encoder"], x["encoder_after"]) for x in (left, right)),
                  frozen_boundary_unchanged=all(exact(x["boundary_before"], x["boundary_after"]) for x in (left, right)),
                  boundary_loss_zero=all(r["boundary"] == r["unsup_boundary"] == 0 for x in (left, right) for r in x["rows"]),
                  meaningful_distillation=all(r["unsup"] > 0 for x in (left, right) for r in x["rows"]),
                  semantic_updated=not exact(left["initial_semantic"], left["semantic_steps"][-1]),
                  actual_group_norm=left["group_norm_count"] > 0,
                  highres_v1=left["highres_version"] == right["highres_version"] == "highres_v1")
    rounding = rounding_example()
    checks["rounding_counterexample_detected"] = rounding["semantic_best"] != rounding["counterfactual_best"]
    return dict(format="v6_carrier_elision_synthetic_cpu_v1", complete=all(checks.values()),
        checks=checks, val_loss=left["val_loss"], best_epoch_1based=left["best_epoch_1based"],
        lora_sha256=left["lora_sha256"], initial_semantic_sha256=digest(left["initial_semantic"]),
        final_semantic_sha256=digest(left["semantic_steps"][-1]),
        wrong_unfrozen_boundary_val={"joint": left["wrong_unfrozen_boundary_val"], "v6": right["wrong_unfrozen_boundary_val"]},
        rounding_counterexample=rounding,
        limits=["synthetic inputs; real decoder/loss/LoRA implementation; tiny encoder and FPN widths",
                "CPU FP32, one thread, three updates; not CUDA AMP equivalence or actual-run checkpoint evidence",
                "boundary predictions/metrics intentionally differ and may remain visible in diagnostic monitors",
                "requires frozen shared encoder/semantic carrier, boundary loss skipped, semantic-boundary alignment disabled"])


def run_bridge(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("joint", "v6"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", variant,
                        "--output", str(output_dir / f"{variant}.pt")], check=True, cwd=ROOT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=120)
    pair = [torch.load(output_dir / f"{v}.pt", map_location="cpu", weights_only=False) for v in ("joint", "v6")]
    result = compare(*pair)
    result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "models/fpn_decoder.py", ROOT / "models/lora.py", ROOT / "utils/loss.py",
                  ROOT / "utils/loss_semi.py", ROOT / "train_stage2.py", Path(__file__).resolve())}
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["complete"]:
        raise RuntimeError(json.dumps(result["checks"]))
    return result



@pytest.fixture(scope="module")
def bridge():
    return sys.modules[__name__]


@pytest.fixture(scope="module")
def result(bridge, tmp_path_factory):
    return bridge.run_bridge(tmp_path_factory.mktemp("v6_carrier"))


def test_current_salign_configuration_skips_frozen_boundary_loss(bridge):
    cfg = bridge.recipe()
    assert bridge.criterion_from_recipe(cfg).freeze_boundary
    assert not bridge.criterion_from_recipe(cfg, False).freeze_boundary


def test_separate_process_teacher_initialization_and_lora_are_exact(result):
    assert result["complete"]
    for name in ("teacher_probability", "features", "encoder", "lora_sha256", "initial_semantic", "rng"):
        assert result["checks"][name]
    assert result["checks"]["actual_group_norm"] and result["checks"]["highres_v1"]


def test_same_batch_semantic_gradients_adam_updates_and_freezes_are_exact(result):
    for name in ("gradient_steps", "semantic_steps", "optimizer", "semantic_updated",
                 "meaningful_distillation", "encoder_unchanged", "frozen_boundary_unchanged"):
        assert result["checks"][name]


def test_actual_validation_loss_and_best_epoch_are_exact_despite_different_boundary(result):
    for name in ("val_loss", "best_epoch_1based", "boundary_loss_zero", "boundary_outputs_differ",
                 "boundary_carriers_differ", "wrong_unfrozen_validation_differ"):
        assert result["checks"][name]


def test_counterfactual_constant_can_hide_a_one_ulp_improvement(bridge):
    result = bridge.rounding_example()
    assert result["semantic_scores"][1] < result["semantic_scores"][0]
    assert result["counterfactual_scores"][0] == result["counterfactual_scores"][1]
    assert result["semantic_best"] == 2 and result["counterfactual_best"] == 1
    assert result["applies_to_current_salign"] is False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("joint", "v6"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/v6_skip_20261101/cpu")
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker, args.output)
    else:
        print(json.dumps(run_bridge(args.output_dir), ensure_ascii=False, indent=2))
