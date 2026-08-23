# N1: native/multi-scale source-space crop adaptation

N1 addresses the measured mismatch between training and native-tile inference.
The legacy labeled path first resized a 2584x1936 image to 1024 and then took a
512 crop. Native-tile inference instead presents approximately 1024 original
pixels to the model. The two paths therefore differ in physical pixel scale and
field of view.

N1 changes only labeled sampling geometry:

- 50%: crop 1024x1024 in original coordinates and keep it at 1024.
- 50%: request a 2048x2048 source crop, clipped to the 1936-pixel short side,
  then resize it to 1024. This approximates the lower-resolution test domain
  without creating padded pseudo labels.
- Image, semantic mask, soft boundary target and hard boundary core share the
  exact crop. EDT weights are recomputed after resizing.
- The semantic branch, encoder/LoRA and center task remain frozen/disabled.
- No unlabeled loss, new loss term or decoder architecture is introduced.

N1 initializes from the reproducible V6 checkpoint because the migrated server
contains V6 but not the later E1/B2 checkpoint. This deliberately makes N1 a
clean scale-adaptation ablation; B2/affinity work should follow only if N1 is
useful.

Inference additionally supports a deterministic resolution-aware tiny-instance
floor: `50 * image_area / (1044 * 1244)`, clamped to `[50, 200]`. This preserves
the low-resolution setting and yields 193 pixels for 2448x2048 images.
