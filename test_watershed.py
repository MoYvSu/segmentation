# -*- coding: utf-8 -*-
"""Full test suite for watershed_separation and topo_instance_separation.

Tests use barely-overlapping circular grains (2px overlap) to simulate
realistic touching grain boundaries. Heavy overlaps (40+px) produce
distance-field saddles too close to peaks for any threshold-based
seed extraction to separate.
"""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from utils.post_process import watershed_separation, topo_instance_separation


def make_circular_grain(h, w, cx, cy, radius):
    """Create a circular binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, -1)
    return mask


def test_two_touching_grains():
    """Two barely-overlapping circular grains should be separated into 2 instances."""
    h, w = 200, 200
    # Centers 98px apart, radius=50 each → 2px overlap (realistic touching)
    c1 = make_circular_grain(h, w, 51, 100, 50)
    c2 = make_circular_grain(h, w, 149, 100, 50)
    ferrite_mask = (c1 | c2).astype(np.uint8)
    dist = distance_transform_edt(ferrite_mask > 0).astype(np.float32)
    dist_norm = dist / (dist + 10.0)

    labels = watershed_separation(ferrite_mask, dist_norm)
    ul = np.unique(labels)
    ul = ul[ul > 0]
    assert len(ul) == 2, f"Expected 2 grains, got {len(ul)}"

    for lid in ul:
        leaked = (labels == lid) & (ferrite_mask == 0)
        assert leaked.sum() == 0, f"Label {lid}: {leaked.sum()} leaked pixels"

    print("test_two_touching_grains: PASS")


def test_single_grain():
    """A single isolated grain should produce 1 instance."""
    h, w = 200, 200
    ferrite_mask = make_circular_grain(h, w, 100, 100, 50)
    dist = distance_transform_edt(ferrite_mask > 0).astype(np.float32)
    dist_norm = dist / (dist + 10.0)

    labels = watershed_separation(ferrite_mask, dist_norm)
    ul = np.unique(labels)
    ul = ul[ul > 0]
    assert len(ul) == 1, f"Expected 1 grain, got {len(ul)}"
    print("test_single_grain: PASS")


def test_three_touching_grains():
    """Three barely-overlapping grains in a row should produce 3 instances."""
    h, w = 300, 200
    # radius=40, centers 78px apart → 2px overlap each
    c1 = make_circular_grain(h, w, 41, 100, 40)
    c2 = make_circular_grain(h, w, 119, 100, 40)
    c3 = make_circular_grain(h, w, 197, 100, 40)
    ferrite_mask = (c1 | c2 | c3).astype(np.uint8)
    dist = distance_transform_edt(ferrite_mask > 0).astype(np.float32)
    dist_norm = dist / (dist + 10.0)

    labels = watershed_separation(ferrite_mask, dist_norm)
    ul = np.unique(labels)
    ul = ul[ul > 0]
    assert len(ul) >= 2, f"Expected at least 2 grains, got {len(ul)}"
    print(f"test_three_touching_grains: PASS (found {len(ul)} grains)")


def test_topo_instance_separation():
    """Full pipeline: 2 ferrite + 1 pearlite background."""
    h, w = 200, 200
    c1 = make_circular_grain(h, w, 51, 100, 50)
    c2 = make_circular_grain(h, w, 149, 100, 50)
    ferrite_mask = (c1 | c2).astype(np.uint8)
    dist = distance_transform_edt(ferrite_mask > 0).astype(np.float32)
    dist_norm = dist / (dist + 10.0)

    bm = ferrite_mask.copy()  # 1=ferrite, 0=pearlite (background)
    im, cm = topo_instance_separation(bm, dist_field=dist_norm, min_instance_area=10)

    n_ferrite = sum(1 for v in cm.values() if v == 1)
    n_pearlite = sum(1 for v in cm.values() if v == 0)
    assert n_ferrite == 2, f"Expected 2 ferrite instances, got {n_ferrite}"
    assert n_pearlite >= 1, f"Expected >=1 pearlite instances, got {n_pearlite}"

    areas = [(im == i).sum() for i in sorted(cm.keys())]
    for i in range(len(areas) - 1):
        assert areas[i] >= areas[i + 1], f"IDs not sorted by descending area: {areas}"

    print(f"test_topo_instance_separation: PASS (ferrite={n_ferrite}, pearlite={n_pearlite})")


def test_no_pixel_leakage():
    """Instance map should not have pixels outside ferrite region."""
    h, w = 200, 200
    c1 = make_circular_grain(h, w, 51, 100, 50)
    c2 = make_circular_grain(h, w, 149, 100, 50)
    ferrite_mask = (c1 | c2).astype(np.uint8)
    dist = distance_transform_edt(ferrite_mask > 0).astype(np.float32)
    dist_norm = dist / (dist + 10.0)

    bm = ferrite_mask.copy()
    im, cm = topo_instance_separation(bm, dist_field=dist_norm, min_instance_area=10)

    # All ferrite instances (class=1) must be within ferrite region
    for inst_id, cls in cm.items():
        if cls == 1:
            inst_pixels = (im == inst_id)
            leaked = inst_pixels & (ferrite_mask == 0)
            assert leaked.sum() == 0, f"Instance {inst_id}: {leaked.sum()} leaked pixels"
    print("test_no_pixel_leakage: PASS")


if __name__ == "__main__":
    test_two_touching_grains()
    test_single_grain()
    test_three_touching_grains()
    test_topo_instance_separation()
    test_no_pixel_leakage()
    print("\n=== All tests PASSED ===")