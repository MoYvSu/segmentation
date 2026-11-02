"""Small deterministic policies shared by inference entry points."""

from __future__ import annotations


def resolution_scaled_min_area(base_area, image_shape, policy=None):
    """Scale the instance-area floor by image area and clamp it.

    The default reference is the larger low-resolution test shape (1044x1244).
    The rule uses only image dimensions, never predictions or test labels.
    """
    cfg = policy or {}
    base = max(1, int(base_area))
    if not cfg.get("enabled", False):
        return base
    height, width = int(image_shape[0]), int(image_shape[1])
    ref_shape = cfg.get("reference_shape", [1044, 1244])
    ref_height, ref_width = max(1, int(ref_shape[0])), max(1, int(ref_shape[1]))
    minimum = int(cfg.get("min_area", base))
    maximum = int(cfg.get("max_area", max(base, 200)))
    scaled = round(base * height * width / float(ref_height * ref_width))
    return max(minimum, min(maximum, int(scaled)))
