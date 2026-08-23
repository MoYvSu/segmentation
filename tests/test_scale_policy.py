from utils.scale_policy import resolution_scaled_min_area


POLICY = {
    "enabled": True,
    "reference_shape": [1044, 1244],
    "min_area": 50,
    "max_area": 200,
}


def test_disabled_policy_preserves_base_area():
    assert resolution_scaled_min_area(50, (2048, 2448), {}) == 50


def test_reference_resolution_preserves_base_area():
    assert resolution_scaled_min_area(50, (1044, 1244), POLICY) == 50


def test_high_resolution_scales_and_clamps():
    assert resolution_scaled_min_area(50, (2048, 2448), POLICY) == 193
    assert resolution_scaled_min_area(50, (10000, 10000), POLICY) == 200
