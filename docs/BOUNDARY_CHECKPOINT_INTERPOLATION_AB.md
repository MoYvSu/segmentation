# V6/N1 boundary checkpoint interpolation A/B

## Purpose

Test whether a small amount of the cleaner N1 boundary branch can improve the
reproducible V6 baseline without inheriting N1's ferrite fragmentation. Only
the 36 tensors under `boundary_fpn.*` and `boundary_branch.*` are interpolated:

```text
W(alpha) = (1 - alpha) * W(V6) + alpha * W(N1)
```

The semantic decoder and encoder LoRA are copied byte-for-byte from V6. The
checkpoint generator is `tools/interpolate_boundary_checkpoints.py`.

## Fixed evaluation conditions

- Validation set: 7 labeled native-resolution images.
- Inference: native 1024 tiles, overlap 256, no TTA.
- Postprocess: boundary threshold 0.45, minimum instance area 50.
- V6 source: `outputs/stage2_v6/best_model_stage2.pth`.
- N1 source: `outputs/stage2_boundary_v6_n1_native_multiscale15/best_model_stage2.pth`.
- Reports: `outputs/experiments/v6_n1_interpolation/*/ab_summary.json`.

## Results

| alpha | predicted instances | valid matches | valid instance mIoU | GT-penalized mIoU | ferrite predicted | ferrite area error | proxy total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (V6) | 1495 | 952 | 0.88447 | 0.75585 | 880 | 0.10913 | 88.767 |
| 0.10 | 1513 | 975 | 0.88594 | 0.77540 | 907 | 0.13938 | 87.328 |
| 0.25 | 1573 | 996 | 0.88838 | 0.79428 | 960 | 0.18490 | 85.174 |
| 0.50 | 1626 | 1011 | 0.89175 | 0.80930 | 1015 | 0.23080 | 83.048 |
| 0.75 | 1692 | 1026 | 0.89303 | 0.82248 | 1093 | 0.28693 | 80.305 |

## Decision

Interpolation produces a consistent boundary/recall improvement, but the same
change monotonically fragments ferrite: instance count rises, mean ferrite
area falls and the area score deteriorates faster than the mIoU component
improves. Even alpha 0.10 is below V6 by 1.439 proxy points.

Stop this route. Do not continue N1 training, use larger interpolation weights,
or hide the model-side fragmentation with an increasingly complex adaptive
postprocessor. Keep V6 as the production checkpoint. A later training attempt
should explicitly distinguish useful missing boundaries from internal texture,
instead of globally increasing boundary response.

## Reproduction

```bash
python tools/interpolate_boundary_checkpoints.py \
  --base outputs/stage2_v6/best_model_stage2.pth \
  --tuned outputs/stage2_boundary_v6_n1_native_multiscale15/best_model_stage2.pth \
  --output-dir outputs/checkpoint_interpolation/v6_n1 \
  --alphas 0.10 0.25 0.50 0.75

python tools/run_instance_ab.py \
  --config config/experiments/native_tiles_v6_bt045.yaml \
  --checkpoint outputs/checkpoint_interpolation/v6_n1/boundary_interpolation_a010.pth \
  --output-dir outputs/experiments/v6_n1_interpolation/a010_min50 \
  --subset val --modes native_tiles
```
