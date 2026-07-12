# Agent.md: Low-carbon Steel Metallographic Image Phase Segmentation

python env: sam2_env via conda activate sam2_env or D:\\Anaconda\\envs\\sam2_env\\python.exe

## 1. Core Objective

Frozen SAM 2 Image Encoder + custom binary FPN decoder (dual-task: binary mask + distance field regression) + online Letterbox pipeline + multi-resolution adaptive inference + watershed topological separation post-processing.

## 2. Hard Constraints

1. Parameter limit: Total < 500M.
2. Zero pretrained decoder: No SAM 2 Mask Decoder weights. Fully random init.
3. Local isolation: All weights in weights/ dir, no ~/.cache/.
4. Aspect ratio preservation: No forced squeeze/distortion.

## 3. Project Tree

project/
+-- config/default_config.yaml
+-- data/raw/  (images + labelme .json)
+-- data/smoketest/  (test images)
+-- data/dataset.py  (online pipeline)
+-- data/active_learning.py  (pseudo-labels)
+-- models/sam2_encoder.py  (frozen encoder)
+-- models/fpn_decoder.py  (dual-task FPN decoder)
+-- utils/metrics.py
+-- utils/loss.py  (FocalDistanceFieldLoss)
+-- utils/post_process.py  (compensation + watershed + instance ID)
+-- weights/
+-- train.py
+-- inference.py  (multi-resolution adaptive)
+-- debug_iou.py  (zero-epoch IoU audit)
+-- debug_pipeline.py  (pipeline diagnostic)

## 4. Module Specs

### 4.1 Model (models/)
- sam2_encoder.py: sam2_hiera_base_plus, frozen, Stage 1-4 features.
+ fpn_decoder.py: FPN 256ch, ResidualBlocks, semantic head, dual-branch:
  - cls_branch: Conv(256->128)+BN+ReLU+Conv(128->1), raw logits
  - reg_branch: Conv(256->128)+BN+ReLU+Conv(128->1)+Sigmoid, [0,1] dist field

### 4.2 Data Pipeline (data/dataset.py)
- Online Letterbox: long edge to 1024, pad short edge with zeros.
- JSON labels: 1=ferrite, 0=pearlite. Also text labels.
+ Dual-task: ch0=binary mask (ferrite=1), ch1=distance field (dist/(dist+10)).

### 4.3 Multi-Resolution Adaptive Inference (inference.py)
Test data: 2448x2048 and 1224x1024 (both 1.2:1).
+ Case A (max dim <= 1224): Letterbox to 1024x1024, single forward.
+ Case B (max dim > 1224): Tiled inference:
  - Window 1024x1024, stride 512 (50% overlap)
- Gaussian weight blending for seamless stitching.
+ Distance field spatial compensation:
  spatial_scale = train_max_dim / image_size = 2584/1024 = 2.5234

### 4.4 Post-Processing (utils/post_process.py)
- Distance compensation (non-linear inverse transform):
  1. De-normalize: dist_raw = dist_norm * 10 / (1 - dist_norm)
- 2. Scale: dist_raw_corrected = dist_raw * 2.5234
  3. Re-normalize: dist_compensated = dist_raw_corrected / (dist_raw_corrected + 10)
- Watershed separation:
  1. Dilate + erode distance field for local maxima
  2. Fallback: 75th percentile threshold if no seeds
  3. cv2.watershed() to split touching grains
  4. Dynamic kernel: base_kernel * sqrt(area / 1024^2)
- Instance IDs 1-255 by descending area.
  Output: _inst.png (uint8) + _class.json ({id: class})

## 5. Loss & Training

- Loss: FocalDistanceFieldLoss = Focal(gamma=2.0, alpha=0.95) + 10.0 * MSE
- Optimizer: AdamW(lr=1.5e-4, weight_decay=1e-4, eps=1e-4)
- Scheduler: Warmup(25) -> CosineAnnealing(475)
- Grad clip: 1.0, FP32, batch=4, epochs=500

## 6. Execution Steps

1. Build structure and scripts.
2. Complete config/default_config.yaml and data/dataset.py.
3. Place raw images and annotations in data/raw/.
4. Run debug_pipeline.py to verify pipeline.
5. Run debug_iou.py for zero-epoch IoU audit.
6. Train: python train.py --config config/default_config.yaml
7. Infer: python inference.py --config config/default_config.yaml --checkpoint outputs/best_model.pth

## 7. Stage 2: Semi-Supervised Fine-tuning

### 7.1 Technical Specification

- **Freeze**: SAM 2 Encoder + FPN Decoder main body (lateral_convs, residual_blocks, semantic_head).
- **Trainable**: Only `decoder.cls_branch` + `decoder.reg_branch` (two decoupled heads).
- **Dual-stream**: Labeled data (Focal+MSE) + Unlabeled data (consistency loss).
- **Labeled stream**: `LabeledDataset` -> FocalDistanceFieldLoss (same as Stage 1).
- **Unlabeled stream**: `UnlabeledDataset` -> 3-way fork augmentation:
  - `img_weak`: letterbox only (teacher prediction source).
  - `img_strong_appearance`: Gaussian blur + brightness/contrast jitter (for cls consistency).
  - `img_strong_geometric`: random 90-degree rotation or flip + metadata T (for reg consistency).
- **Unsupervised loss** (`compute_stage2_unsupervised_loss`):
  - Classification consistency: pseudo-labels from `img_weak` (confidence > 0.90) vs `img_strong_appearance` prediction (regional BCE).
  - Regression geometric consistency: `img_weak` distance field transformed by T vs `img_strong_geometric` prediction (pixel-level MSE).
  - Total: `UNSUPERVISED_WEIGHT * (loss_cls + DIST_WEIGHT * loss_reg)`.
- **Mixed batch**: `itertools.cycle(labeled_loader)` wraps labeled stream; unlabeled stream drives iteration count.
- **Optimizer**: AdamW(lr=5.0e-5, weight_decay=1e-4, eps=1e-4), only cls_branch + reg_branch params.
- **Scheduler**: Warmup(5) -> CosineAnnealing(95).
- **Grad clip**: 1.0, FP32.

### 7.2 File Listing

| File | Description |
|------|-------------|
| `config/stage2_config.yaml` | Stage-2 config (checkpoint path, batch sizes, unsup weight) |
| `data/dataset_semi.py` | `LabeledDataset`, `UnlabeledDataset`, collate fns, transform utils |
| `utils/loss_semi.py` | `compute_stage2_unsupervised_loss` function |
| `train_stage2.py` | Main entry: load Stage-1, freeze, dual-stream training loop |
| `data/unlabeled/` | Unlabeled images directory (semi-supervised training only) |

### 7.3 Commands

```bash
conda activate sam2_env

# Stage-2 semi-supervised fine-tuning
python train_stage2.py --config config/stage2_config.yaml

# Resume from checkpoint
python train_stage2.py --config config/stage2_config.yaml --resume outputs/stage2/best_model_stage2.pth
```

### 7.4 Checkpoint Format

Stage-2 checkpoints save only `cls_branch_state_dict` + `reg_branch_state_dict` (not full decoder):
- `best_model_stage2.pth`: best validation mIoU.
- `stage2_epoch{N}.pth`: periodic checkpoints (every 10 epochs).
- `final_model_stage2.pth`: final model.
