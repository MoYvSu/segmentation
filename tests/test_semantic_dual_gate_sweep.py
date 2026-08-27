from tools.sweep_semantic_dual_gate import GateArm, classify_cached_instance


STRICT = GateArm("strict", 0.35, 0.85, 0.08, 0.65, 0.15)


def test_cached_gate_reproduces_strict_pearlite_to_ferrite_override():
    cls, reason = classify_cached_instance(
        {
            "hard_ratio": 0.40,
            "candidate_core_score": 0.90,
            "dual_core_gain": 0.10,
        },
        STRICT,
    )
    assert cls == 1
    assert reason == "pearlite_to_ferrite_black_rim"


def test_cached_gate_has_no_instance_area_condition():
    common = {
        "hard_ratio": 0.60,
        "candidate_core_score": 0.10,
        "dual_core_gain": 0.0,
    }
    small = {**common, "area": 10}
    large = {**common, "area": 1_000_000}
    assert classify_cached_instance(small, STRICT) == classify_cached_instance(
        large, STRICT
    )
    assert classify_cached_instance(large, STRICT)[0] == 0


def test_cached_gate_rejects_candidate_when_threshold_is_not_met():
    cls, reason = classify_cached_instance(
        {
            "hard_ratio": 0.34,
            "candidate_core_score": 0.90,
            "dual_core_gain": 0.10,
        },
        STRICT,
    )
    assert cls == 0
    assert reason == "gate_rejected"
