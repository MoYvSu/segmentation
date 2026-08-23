"""Resolution-matched source-space crops for boundary training.

The legacy pipeline letterboxes a full metallography image to 1024 and then
crops that resized image.  Native-tile inference instead sees roughly 1024
source pixels at a time.  This module crops all tensors in original-image
coordinates first, then resizes the crop to the fixed model input size.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


def _normalise_probabilities(probabilities: Sequence[float], count: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (count,) or np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("source_crop_probabilities must be non-negative and match crop sizes")
    return probs / probs.sum()


def _center_heatmap(
    centers: Iterable[Tuple[float, float]],
    left: int,
    top: int,
    side: int,
    output_size: int,
    sigma: float,
) -> np.ndarray:
    heatmap = np.zeros((output_size, output_size), dtype=np.float32)
    scale = output_size / float(side)
    sigma_out = max(0.5, float(sigma) * scale)
    radius = max(1, int(round(3.0 * sigma_out)))
    for x, y in centers:
        if not (left <= x < left + side and top <= y < top + side):
            continue
        xx = int(round((x - left) * scale))
        yy = int(round((y - top) * scale))
        if not (0 <= xx < output_size and 0 <= yy < output_size):
            continue
        x0, x1 = max(0, xx - radius), min(output_size, xx + radius + 1)
        y0, y1 = max(0, yy - radius), min(output_size, yy + radius + 1)
        xs = np.arange(x0, x1, dtype=np.float32) - xx
        ys = np.arange(y0, y1, dtype=np.float32) - yy
        peak = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2.0 * sigma_out ** 2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], peak)
    return heatmap


def native_multiscale_crop(
    image: np.ndarray,
    semantic: np.ndarray,
    boundary: np.ndarray,
    boundary_core: np.ndarray,
    centers: Iterable[Tuple[float, float]],
    output_size: int,
    source_crop_sizes: Sequence[int],
    source_crop_probabilities: Sequence[float],
    center_sigma: float = 4.0,
):
    """Randomly crop one configured source scale and resize all targets together.

    A requested crop larger than the image's short side is clipped to that short
    side.  This avoids reflected/zero-padded pseudo labels; for the 2584x1936
    labeled images, the nominal 2048 mode therefore becomes a 1936-pixel crop.
    """
    if len(source_crop_sizes) == 0:
        raise ValueError("source_crop_sizes must not be empty")
    probs = _normalise_probabilities(source_crop_probabilities, len(source_crop_sizes))
    requested_side = int(np.random.choice(source_crop_sizes, p=probs))
    h, w = image.shape[:2]
    side = max(1, min(requested_side, h, w))
    top = int(np.random.randint(0, h - side + 1))
    left = int(np.random.randint(0, w - side + 1))
    ys = slice(top, top + side)
    xs = slice(left, left + side)

    image_out = cv2.resize(image[ys, xs], (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    semantic_out = cv2.resize(
        semantic[ys, xs], (output_size, output_size), interpolation=cv2.INTER_NEAREST
    )
    boundary_out = cv2.resize(
        boundary[ys, xs], (output_size, output_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    core_out = cv2.resize(
        boundary_core[ys, xs], (output_size, output_size), interpolation=cv2.INTER_NEAREST
    )
    center_out = _center_heatmap(
        centers, left, top, side, output_size, center_sigma
    )
    metadata = {
        "requested_source_crop_size": requested_side,
        "source_crop_size": side,
        "crop_top": top,
        "crop_left": left,
        "scale": output_size / float(side),
    }
    return image_out, semantic_out, boundary_out, core_out, center_out, metadata
