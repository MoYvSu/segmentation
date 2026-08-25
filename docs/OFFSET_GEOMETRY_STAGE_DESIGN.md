# Global center-offset geometry stage design

## O1 decision

O1 reproduces the project's actual 1024 letterbox arithmetic. Geometry padding
is zero and excluded from loss even though the RGB image keeps reflection
padding. The fixed seven-image validation split contains 1114 instances.

| Geometry grid | Pred / GT | Valid matches | Valid instance mIoU | Ferrite area error | Proxy total |
|---:|---:|---:|---:|---:|---:|
| 256 | 1114 / 1114 | 1109 | 0.82733 | 0.00145 | 91.294 |
| 512 | 1114 / 1114 | 1114 | 0.89973 | 0.00030 | 94.972 |
| 1024 | 1114 / 1114 | 1114 | 0.93882 | 0.00015 | 96.933 |

At the selected 512 grid, endpoint noise of one grid pixel has no measurable
effect. Noise sigma 2 yields 1130 predictions, 1107 valid matches, 0.89811
valid mIoU, 0.00748 ferrite area error and 94.531 proxy score.

Use a 512 geometry grid for the first learned model. It preserves every GT
instance in the oracle, retains useful regression-error tolerance and avoids
the parameter/data cost of a full-resolution image-guided head. A 1024 output
is a later ceiling experiment, not the first implementation.

## Architecture G0

Keep the verified V6 semantic path and add one independent geometry decoder:

```text
SAM2 encoder + retained V6 LoRA
├── frozen V6 semantic FPN -> semantic logits (unchanged)
└── geometry FPN -> 256 feature grid
      └── learned 2x upsample block -> 512 feature grid
            ├── center heatmap (1 channel)
            └── normalized center offset dy/dx (2 channels)
```

Center and offset are one coupled instance representation, not two unrelated
tasks. The center heatmap only identifies global attractors. Dense offsets own
pixel-to-instance assignment; the center heatmap is not passed to the old
boundary watershed.

The first ablation compares geometry-FPN initialization from the V6 boundary
FPN against random initialization. The output heads and 512 upsample block are
always newly initialized. No boundary output participates in inference.

## Supervision

- One interior center per LabelMe instance: the maximum of its EDT.
- Adaptive Gaussian center target, with sigma clamped to a small grid-relative
  range instead of one fixed sigma for every grain size.
- Offsets are divided by 512 for bounded regression and multiplied back before
  endpoint grouping.
- Smooth-L1 offset loss is evaluated only on covered instance pixels and valid
  letterbox content. A cosine direction term is optional after the baseline.
- Center focal loss is evaluated only on valid letterbox content.
- Uncovered polygon gaps and letterbox padding do not receive offset targets.
- Checkpoint selection uses class-aware instance metrics and ferrite mean-area
  error, not boundary IoU.

## Semantic reproducibility contract

Adding geometry is not allowed to silently change the successful semantic
model. Before any geometry training:

1. Load the exact verified V6 decoder and LoRA states.
2. Hash every `seg_fpn.*`, `seg_branch.*` and LoRA tensor before and after model
   construction/checkpoint save-load.
3. Set semantic modules to `requires_grad=False` and keep frozen dropout/norm
   modules in eval mode during geometry training.
4. On the fixed seven images, compare V6 semantic logits before and after the
   architecture change. Require identical tensor states and numerical output
   agreement within a fixed tolerance (target max absolute difference <=1e-6).
5. Save geometry state under separate checkpoint keys so legacy V6 loading and
   inference remain reproducible.

The first geometry experiment must stop if this contract fails, regardless of
center/offset training metrics.

## Training stages

### G0: implementation and overfit audit

- Two labeled images, fixed augmentations, 300-500 optimizer steps.
- Freeze encoder, LoRA and complete semantic path.
- Train only geometry FPN, 512 upsampler, center and offset heads.
- Success means endpoint clusters reproduce the labeled instance count and the
  loss/visualization proves both heads move away from initialization.

### G1: labeled geometry baseline

- Fixed labeled train split, 15-20 epochs.
- Global 1024 letterbox samples only; no native instance crops.
- Keep the semantic path frozen.
- Evaluate global instance reconstruction at 512 and compare boundary-FPN vs
  random geometry initialization.
- Do not add pseudo labels until a learned supervised head beats the V6
  instance baseline on the fixed validation proxy.

### G2: allowed unlabeled geometry expansion

SAM2 automatic masks from the competition's 1000 unlabeled images are treated
as class-agnostic proposals, never as ground truth or external labeled data.
They may supervise geometry only after filtering:

1. predicted-IoU and stability thresholds;
2. mask-IoU NMS for duplicate proposals;
3. containment suppression for nested near-duplicates and thin rings;
4. deterministic overlap arbitration to produce non-overlapping instances;
5. minimum area and shape sanity checks;
6. confidence-weighted geometry loss.

SAM2 proposals do not update the V6 semantic branch in G2. If phase classes are
needed, frozen V6 semantic probabilities may assign a class for diagnostics or
inference, but low-confidence class assignments are ignored rather than used as
semantic training targets. Human-reviewed proposals remain subject to the
competition's <=500 self-labeled-image limit.

## Inference contract

1. Run one global 1024 letterbox geometry pass.
2. Detect at most 255 center peaks after NMS.
3. Add predicted offsets to every valid geometry pixel and assign endpoints to
   detected global centers.
4. Inverse-letterbox the coarse instance map to native resolution.
5. Use the retained V6 native semantic output to classify each instance by
   robust probability voting.
6. Native evidence may adjust a boundary locally only if it preserves the two
   neighboring global instance ids; it cannot create an independent instance.

This hierarchy prevents the native-tile duplicate-center failure measured in
O0 while preserving the known-good semantic branch.

## Affinity watershed postprocess A/B (2026-08-26)

`utils/affinity_fusion.py` adds a short-range-dominant, gated long-range fusion
for the eight-channel affinity head:

```text
short = boundary probability from four unit-offset channels
support(d) = soft dilation of short to radius d
fused = normalized(short + 0.50 * support(2) * distance2 + 0.25 * support(4) * distance4)
```

On the fixed six-image labeled split and the G2 latest checkpoint:

| Boundary threshold | Mean fusion proxy | Gated fusion proxy | Mean/Gated predictions |
|---:|---:|---:|---:|
| 0.45 | 84.53 | 85.81 | 1240 / 1228 |
| 0.55 | 88.94 | 89.40 | 1182 / 1177 |

The gain is small and comes mainly from the ferrite mean-area term; mIoU changes
only slightly. On the same 12 label-free monitor images, gated fusion reduces
mean boundary probability from about 0.271 to 0.266 while leaving the mean final
instance count nearly unchanged. Raw components above 255 are diagnostic only;
the exported map must still enforce `max_instance_id=255`.
