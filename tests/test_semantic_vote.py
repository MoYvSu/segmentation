import numpy as np

from utils import affinity_deployment
from utils.semantic_vote import (
    adaptive_instance_core,
    build_adaptive_lab_prior,
    instance_semantic_vote,
)


def test_adaptive_core_ignores_a_dominant_dark_rim():
    instance = np.ones((21, 21), dtype=bool)
    semantic = np.zeros((21, 21), dtype=np.uint8)
    probability = np.full((21, 21), 0.10, dtype=np.float32)
    probability[5:16, 5:16] = 0.90
    semantic[probability > 0.5] = 1

    hard_class, _ = instance_semantic_vote(instance, semantic)
    core_class, score, details = instance_semantic_vote(
        instance,
        semantic,
        semantic_probability=probability,
        mode="adaptive_core",
        core_fraction=0.30,
        return_details=True,
    )

    assert hard_class == 0
    assert core_class == 1
    assert score > 0.75
    assert details["core_pixels"] < int(instance.sum())


def test_adaptive_core_remains_defined_for_a_thin_instance():
    instance = np.zeros((20, 20), dtype=bool)
    instance[3:17, 9:11] = True
    core, weights = adaptive_instance_core(instance, fraction=0.40, min_pixels=8)

    assert 8 <= int(core.sum()) <= int(instance.sum())
    assert np.all(weights[core] > 0)
    assert not np.any(core & ~instance)


def test_lab_prior_can_flip_only_an_uncertain_bright_core():
    image = np.full((32, 32, 3), 30, dtype=np.uint8)
    image[:, 16:] = 220
    prior = build_adaptive_lab_prior(image)
    instance = np.zeros((32, 32), dtype=bool)
    instance[4:28, 20:28] = True
    semantic = np.zeros((32, 32), dtype=np.uint8)
    probability = np.full((32, 32), 0.48, dtype=np.float32)

    cls, score, details = instance_semantic_vote(
        instance,
        semantic,
        semantic_probability=probability,
        mode="adaptive_core_lab",
        lab_prior=prior,
        color_weight=0.30,
        color_min_separation=1.0,
        return_details=True,
    )

    assert prior["separation"] > 1.0
    assert details["color_used"] is True
    assert details["color_score"] > 0.8
    assert cls == 1 and score > 0.5


def test_lab_prior_does_not_touch_a_confident_semantic_vote():
    image = np.full((24, 24, 3), 220, dtype=np.uint8)
    image[:, :12] = 30
    prior = build_adaptive_lab_prior(image)
    instance = np.zeros((24, 24), dtype=bool)
    instance[3:21, 3:10] = True
    semantic = np.ones((24, 24), dtype=np.uint8)
    probability = np.full((24, 24), 0.90, dtype=np.float32)

    cls, score, details = instance_semantic_vote(
        instance,
        semantic,
        semantic_probability=probability,
        mode="adaptive_core_lab",
        lab_prior=prior,
        return_details=True,
    )

    assert cls == 1 and score > 0.85
    assert details["color_used"] is False


def test_conservative_dual_repairs_an_uncertain_black_rim_instance():
    instance = np.ones((21, 21), dtype=bool)
    base_probability = np.full((21, 21), 0.20, dtype=np.float32)
    base_probability[4:17, 4:17] = 0.60
    base_semantic = (base_probability > 0.5).astype(np.uint8)
    candidate = np.full((21, 21), 0.55, dtype=np.float32)
    candidate[4:17, 4:17] = 0.95

    cls, score, details = instance_semantic_vote(
        instance,
        base_semantic,
        semantic_probability=base_probability,
        candidate_semantic_probability=candidate,
        mode="conservative_dual",
        core_fraction=0.30,
        return_details=True,
    )

    assert cls == 1 and score > 0.85
    assert details["dual_override"] is True
    assert details["dual_reason"] == "pearlite_to_ferrite_black_rim"


def test_conservative_dual_rejects_uniform_ferrite_bias():
    instance = np.ones((21, 21), dtype=bool)
    base_probability = np.full((21, 21), 0.40, dtype=np.float32)
    base_semantic = np.zeros((21, 21), dtype=np.uint8)
    base_semantic.flat[::3] = 1
    candidate = np.full((21, 21), 0.95, dtype=np.float32)

    cls, score, details = instance_semantic_vote(
        instance,
        base_semantic,
        semantic_probability=base_probability,
        candidate_semantic_probability=candidate,
        mode="conservative_dual",
        return_details=True,
    )

    assert cls == 0 and score < 0.5
    assert details["dual_override"] is False
    assert details["dual_reason"] == "gate_rejected"


def test_probability_mean_uses_classifier_challenger_when_present():
    instance = np.ones((12, 12), dtype=bool)
    base_probability = np.full((12, 12), 0.80, dtype=np.float32)
    base_semantic = np.ones((12, 12), dtype=np.uint8)
    challenger_probability = np.full((12, 12), 0.20, dtype=np.float32)

    cls, score, details = instance_semantic_vote(
        instance,
        base_semantic,
        semantic_probability=base_probability,
        candidate_semantic_probability=challenger_probability,
        mode="probability_mean",
        return_details=True,
    )

    assert cls == 0
    assert np.isclose(score, 0.20)
    assert np.isclose(details["semantic_score"], 0.20)


def test_affinity_deployment_forwards_semantic_vote_options(monkeypatch, tmp_path):
    captured = {}

    def fake_postprocess(**kwargs):
        captured.update(kwargs)
        return {}, None, {}

    monkeypatch.setattr(
        affinity_deployment, "post_process_prediction_boundary", fake_postprocess
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    affinity_deployment.postprocess(
        np.zeros((1, 2, 4, 4), dtype=np.float32),
        (4, 4),
        tmp_path,
        "sample",
        {
            "semantic_vote_mode": "adaptive_core_lab",
            "semantic_vote_core_fraction": 0.35,
            "semantic_vote_color_weight": 0.20,
        },
        0.65,
        False,
        image_rgb=image,
    )

    assert captured["original_image_rgb"] is image
    assert captured["semantic_vote_options"]["core_fraction"] == 0.35
    assert captured["semantic_vote_options"]["color_weight"] == 0.20
