# -*- coding: utf-8 -*-
"""Geometry-safe letterbox transforms for center/offset instance heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryLetterbox:
    original_height: int
    original_width: int
    input_size: int
    output_grid: int
    resized_height: int
    resized_width: int
    content_height: int
    content_width: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def geometry_letterbox_metadata(
    original_shape: Tuple[int, int],
    input_size: int,
    output_grid: int,
) -> GeometryLetterbox:
    """Match the project's image letterbox and decoder-grid crop arithmetic."""
    height, width = map(int, original_shape)
    if min(height, width, input_size, output_grid) <= 0:
        raise ValueError(
            f"invalid geometry: shape={original_shape}, input={input_size}, "
            f"grid={output_grid}"
        )
    scale = float(input_size) / max(height, width)
    resized_height = int(round(height * scale))
    resized_width = int(round(width * scale))
    content_height = int(round(resized_height * output_grid / input_size))
    content_width = int(round(resized_width * output_grid / input_size))
    content_height = max(1, min(content_height, int(output_grid)))
    content_width = max(1, min(content_width, int(output_grid)))
    return GeometryLetterbox(
        original_height=height,
        original_width=width,
        input_size=int(input_size),
        output_grid=int(output_grid),
        resized_height=resized_height,
        resized_width=resized_width,
        content_height=content_height,
        content_width=content_width,
    )


def letterbox_instance_geometry(
    instance_map: np.ndarray,
    input_size: int = 1024,
    output_grid: int = 256,
):
    """Resize instances to a decoder grid and return a valid-content mask.

    Unlike the RGB image letterbox, geometry padding is constant zero. Mirrored
    instance ids in padding would create fake offset targets with no valid
    center and must therefore be excluded from geometry losses.
    """
    if instance_map.ndim != 2:
        raise ValueError(f"instance_map must be 2-D, got {instance_map.shape}")
    metadata = geometry_letterbox_metadata(
        instance_map.shape, input_size=input_size, output_grid=output_grid
    )
    resized = cv2.resize(
        instance_map.astype(np.int32, copy=False),
        (metadata.content_width, metadata.content_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32, copy=False)
    grid = np.zeros((output_grid, output_grid), dtype=np.int32)
    grid[: metadata.content_height, : metadata.content_width] = resized
    valid = np.zeros((output_grid, output_grid), dtype=bool)
    valid[: metadata.content_height, : metadata.content_width] = True
    return grid, valid, metadata


def inverse_letterbox_instances(
    grid_instances: np.ndarray,
    metadata: GeometryLetterbox,
) -> np.ndarray:
    """Crop decoder padding and restore an integer instance map."""
    expected = (metadata.output_grid, metadata.output_grid)
    if grid_instances.shape != expected:
        raise ValueError(
            f"grid shape mismatch: got {grid_instances.shape}, expected {expected}"
        )
    content = grid_instances[
        : metadata.content_height, : metadata.content_width
    ]
    restored = cv2.resize(
        content.astype(np.int32, copy=False),
        (metadata.original_width, metadata.original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return restored.astype(np.int32, copy=False)
