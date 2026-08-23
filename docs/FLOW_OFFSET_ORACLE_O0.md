# O0: offset/flow instance representation oracle

## Question

Before training another decoder, determine whether dense offsets can reconstruct
the labeled instances under the competition metrics, output resolution and
native-tile constraints. Reconstruction is not allowed to read GT instance ids
or a GT instance count after the vector target has been generated.

The experiment reuses `utils.instance_metrics` and the fixed seven-image
validation split (1114 GT instances, including 702 ferrite instances).

## Representations

1. `center`: each labeled pixel regresses `(dy, dx)` to one interior center,
   selected as the maximum of the instance EDT.
2. `edt`: each labeled pixel predicts the normalized local gradient of its
   instance EDT and reaches an attractor through Euler integration.
3. `center_tile_local`: pessimistic native-tile simulation. Each 1024 tile can
   only predict the center of the visible clipped fragment; overlapping offset
   fields are blended before global endpoint clustering.

Endpoint components are discovered without a target instance count. Components
below the fixed area floor are removed and the 255-instance contract is enforced.

## Direct-center results

### Output resolution, zero regression noise

| Original-image stride | Pred / GT | Valid matches | Valid instance mIoU | Ferrite area error | Proxy total |
|---:|---:|---:|---:|---:|---:|
| 2 | 1114 / 1114 | 1114 | 0.98072 | 0.00112 | 98.980 |
| 4 | 1114 / 1114 | 1114 | 0.94527 | 0.00097 | 97.215 |
| 8 | 1114 / 1114 | 1114 | 0.88105 | 0.00047 | 94.029 |

Stride 4 is sufficient for a first learned head. A stride-2 refinement has a
measurable ceiling advantage but is not required to test learnability.

### Stride-4 endpoint regression noise

Noise is measured in decoder-grid pixels; one grid pixel corresponds to about
four original pixels.

| Endpoint noise sigma | Pred / GT | Valid matches | Valid instance mIoU | Ferrite area error | Proxy total |
|---:|---:|---:|---:|---:|---:|
| 0 | 1114 / 1114 | 1114 | 0.94527 | 0.00097 | 97.215 |
| 1 | 1114 / 1114 | 1114 | 0.94527 | 0.00097 | 97.215 |
| 2 | 1113 / 1114 | 1113 | 0.94414 | 0.00100 | 97.157 |
| 4 | 1108 / 1114 | 1021 | 0.92254 | 0.00111 | 96.071 |

The endpoint representation is robust to realistic small regression error.

## Rejected branches

### Naive EDT-gradient flow

On the first validation image, 150 GT instances produce 282 raw attractor
components. After the 255 cap it obtains 135 valid matches, 0.8521 valid mIoU,
0.4263 ferrite area error and 71.29 proxy score. Endpoint closing radii 1, 2,
3 and 4 do not change the result. The problem is multiple separated stationary
regions inside one instance, not a clustering-radius setting.

This does not reject a full Omnipose implementation, but it rejects adding a
naive normalized EDT gradient to this project as the next experiment.

### Tile-local center offsets

| Mode | Pred / GT | Valid matches | Valid instance mIoU | GT-penalized mIoU | Ferrite area error | Proxy total |
|---|---:|---:|---:|---:|---:|---:|
| Global center, stride 4 | 1114 / 1114 | 1114 | 0.94527 | 0.94527 | 0.00097 | 97.215 |
| 1024 tile-local center, overlap 256 | 1542 / 1114 | 1096 | 0.90464 | 0.89002 | 0.31042 | 79.711 |

Recomputing a center from each visible tile fragment reproduces the exact
failure mode that the project must avoid: cross-tile instances split into
multiple basins, instance count rises and ferrite mean area collapses.

## Decision

Proceed with direct center offsets, but do not attach them to the current
native-tile-only instance path. The next architecture should be hierarchical:

1. A global resized-image geometry pass predicts a coupled center heatmap and
   dense `(dy, dx)` offsets. This pass owns instance identity and instance count.
2. The retained V6 native-tile semantic pass supplies phase probabilities and
   high-resolution support. It must not independently create new instances.
3. Each native pixel is assigned to a global center; high-resolution evidence
   may adjust boundaries locally without creating another center.

Before neural training, O1 must test exact 1024 letterbox reconstruction at
geometry output grids 256, 512 and 1024. This determines whether the geometry
head needs a learned stride-2/full-resolution upsampler.

## Reproduction

```bash
python tools/run_flow_oracle.py --subset val --methods center --strides 4 \
  --center-noise-px 0 1 2 4 \
  --output-dir outputs/experiments/flow_oracle_o0_center_val

python tools/run_flow_oracle.py --subset val --methods center --strides 2 8 \
  --center-noise-px 0 \
  --output-dir outputs/experiments/flow_oracle_o0_center_stride

python tools/run_flow_oracle.py --subset val --methods center_tile_local \
  --strides 4 --center-noise-px 0 --tile-size 1024 --tile-overlap 256 \
  --output-dir outputs/experiments/flow_oracle_o0_tile_val
```
