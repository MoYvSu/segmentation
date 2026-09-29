# -*- coding: utf-8 -*-
"""固定实例诊断的解析合成场景，覆盖裁剪边缘、薄实例和混合区域。"""
import json

import numpy as np
import pytest

from tools.semantic_instance_metrics import analyze_instance, instance_bands


def test_uniform_confident_instance_is_not_heterogeneous_or_uncertain():
    mask = np.ones((40, 60), dtype=bool)
    result = analyze_instance(mask, np.full(mask.shape, .9, dtype=np.float32))
    assert result["area"] == 2400
    assert result["mean_probability"] == pytest.approx(.9)
    assert result["mean_class"] == result["hard_class"] == 1
    assert result["strong_f"]["largest_component"] == 2400
    assert result["strong_p"]["largest_component"] == 0
    assert result["uncertain_fraction"] == 0
    assert result["meaningful_core"] and not result["whole_vs_core_flip"]
    assert not result["heterogeneous"]
    assert sum(layer["area"] for layer in result["layers"]) == 2400
    json.dumps(result, allow_nan=False)


def test_uniform_uncertainty_is_distinguished_from_two_confident_regions():
    mask = np.ones((40, 60), dtype=bool)
    uncertain = analyze_instance(mask, np.full(mask.shape, .5, dtype=np.float32))
    mixed_probability = np.full(mask.shape, .1, dtype=np.float32)
    mixed_probability[:, :30] = .9
    mixed = analyze_instance(mask, mixed_probability)
    assert uncertain["mean_probability"] == pytest.approx(mixed["mean_probability"])
    assert uncertain["uncertain_fraction"] == 1
    assert uncertain["mean_class"] == uncertain["hard_class"] == 0
    assert not uncertain["heterogeneous"]
    assert mixed["uncertain_fraction"] == 0 and mixed["heterogeneous"]
    assert mixed["strong_f"]["largest_fraction"] == .5
    assert mixed["strong_p"]["largest_fraction"] == .5
    assert mixed["interior_heterogeneous"]
    assert mixed["interior_strong_f"]["largest_fraction"] == .5


def test_dark_rim_can_reverse_whole_vote_while_core_stays_bright():
    mask = np.ones((50, 50), dtype=bool)
    probability = np.full(mask.shape, .05, dtype=np.float32)
    probability[10:40, 10:40] = .95
    result = analyze_instance(mask, probability)
    assert result["meaningful_core"] and result["whole_vs_core_flip"]
    assert result["mean_probability"] < .5 < result["core"]["mean_probability"]
    assert result["core_minus_outer"] > .7
    assert result["layers"][0]["mean_probability"] < result["layers"][-1]["mean_probability"]


def test_outer_dark_band_is_removed_from_interior_mixture_diagnostic():
    mask = np.ones((50, 50), dtype=bool)
    probability = np.full(mask.shape, .1, dtype=np.float32)
    probability[3:-3, 3:-3] = .9
    result = analyze_instance(mask, probability)
    assert result["heterogeneous"]
    assert not result["interior_heterogeneous"]
    assert result["interior_strong_p"]["area"] == 0
    assert result["interior"]["mean_probability"] == pytest.approx(.9)


@pytest.mark.parametrize("shape", [(2, 200), (3, 3), (1, 1)])
def test_thin_or_small_instance_does_not_claim_meaningful_core(shape):
    mask = np.ones(shape, dtype=bool)
    result = analyze_instance(mask, np.full(shape, .9, dtype=np.float32))
    assert not result["meaningful_core"] and result["core_limitations"]
    assert result["whole_vs_core_flip"] is None
    assert result["distance_max"] <= 2
    assert result["interior"]["area"] == 0
    assert not result["interior_heterogeneous"]
    json.dumps(result, allow_nan=False)


def test_tight_crop_matches_explicit_background_and_full_image_edge_is_finite():
    crop = np.ones((40, 60), dtype=bool)
    probability = np.linspace(0, 1, crop.size).reshape(crop.shape)
    a = analyze_instance(crop, probability)
    full = np.pad(crop, ((3, 5), (4, 7)))
    full_probability = np.pad(probability, ((3, 5), (4, 7)), constant_values=np.nan)
    b = analyze_instance(full, full_probability)
    assert a == b
    assert a["distance_max"] == 20
    bands = instance_bands(crop)
    assert bands["distance"][0, 0] == 1
    assert bands["distance"][20, 30] == 20
    assert np.isfinite(bands["distance"]).all()


def test_disconnected_noise_is_not_a_large_coherent_opposite_region():
    mask = np.ones((40, 60), dtype=bool)
    probability = np.full(mask.shape, .9, dtype=np.float32)
    probability[::3, ::3] = .1
    result = analyze_instance(mask, probability)
    assert result["strong_p"]["fraction"] > .1
    assert result["strong_p"]["largest_component"] == 1
    assert not result["heterogeneous"]


def test_mask_holes_exclude_probability_and_affect_boundary_distance():
    mask = np.ones((50, 50), dtype=bool)
    mask[20:30, 20:30] = False
    probability = np.full(mask.shape, .9, dtype=np.float32)
    probability[~mask] = np.nan
    result = analyze_instance(mask, probability)
    bands = instance_bands(mask)
    assert result["area"] == 2400 and result["mean_probability"] == pytest.approx(.9)
    assert bands["distance"][19, 25] == 1
    assert (bands["layers"][~mask] == 0).all()


@pytest.mark.parametrize("bad_probability", [np.nan, np.inf, -.01, 1.01])
def test_invalid_probability_inside_instance_is_rejected(bad_probability):
    with pytest.raises(ValueError, match="finite and in"):
        analyze_instance(np.ones((4, 4), bool), np.full((4, 4), bad_probability))


def test_empty_mask_wrong_shape_or_integer_instance_ids_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        analyze_instance(np.zeros((4, 4), bool), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="shape"):
        analyze_instance(np.ones((4, 4), bool), np.zeros((3, 4)))
    with pytest.raises(ValueError, match="bool"):
        analyze_instance(np.ones((4, 4), np.uint16), np.zeros((4, 4)))


def test_models_share_exact_same_bands_without_recomputing_or_mutating(monkeypatch):
    mask = np.ones((43, 67), dtype=bool)
    mask[12:17, 20:24] = False
    bands = instance_bands(mask)
    snapshots = {key: value.copy() for key, value in bands.items() if isinstance(value, np.ndarray)}

    def forbidden(*args, **kwargs):
        raise AssertionError("shared bands must not recompute the distance transform")

    monkeypatch.setattr("tools.semantic_instance_metrics.cv2.distanceTransform", forbidden)
    first = analyze_instance(mask, np.full(mask.shape, .1, np.float32), bands=bands)
    second = analyze_instance(mask, np.full(mask.shape, .9, np.float32), bands=bands)
    assert [r["area"] for r in first["layers"]] == [r["area"] for r in second["layers"]]
    for region in ("core", "outer", "interior"):
        assert first[region]["area"] == second[region]["area"]
    assert all(np.array_equal(value, bands[key]) for key, value in snapshots.items())


def test_precomputed_bands_reject_different_shape_or_instance():
    mask = np.ones((40, 60), dtype=bool)
    with pytest.raises(ValueError, match="bands shape"):
        analyze_instance(mask, np.zeros(mask.shape), bands=instance_bands(np.ones((41, 60), bool)))
    different_mask = mask.copy()
    different_mask[20, 20] = False
    with pytest.raises(ValueError, match="same instance mask"):
        analyze_instance(mask, np.zeros(mask.shape), bands=instance_bands(different_mask))
