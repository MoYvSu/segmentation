# -*- coding: utf-8 -*-
"""复用已有控制时必须拒绝错配输入和未真正归零的噪声。"""
import copy
import json

import pytest

from tools.run_restore_zero import PAIR_FIELDS, check_updates, comparable_recipe


def test_comparable_recipe_ignores_only_declared_noise_and_location():
    base = {"paths": {"project_root": "old"},
            "rgb_restoration": {"output_dir": "old", "model": {"kappa": .03}},
            "restore_consistency": {"consistency_weight": .1, "stage_epochs": 5}}
    zero = copy.deepcopy(base)
    zero["paths"]["project_root"] = "new"
    zero["rgb_restoration"]["output_dir"] = "new"
    zero["restore_consistency"]["noise_mode"] = "zero"
    zero["restore_consistency"]["consistency_weight"] = 0.
    assert comparable_recipe(base) == comparable_recipe(zero)
    zero["rgb_restoration"]["model"]["kappa"] = .01
    assert comparable_recipe(base) != comparable_recipe(zero)


def rows(tmp_path, change=None):
    old = {**{key: 1 for key in PAIR_FIELDS}, "consistency_weight": 0.}
    new = {**old, "training_noise_mode": "zero", "consistency_weight": 0,
           "applied_noise_max_abs": 0., "loss": .2, "supervised_loss": .2}
    if change:
        new.update(change)
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for path, row in ((reference, old), (candidate, new)):
        (path / "metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return reference, candidate


def test_complete_pairing_retains_draw_stream(tmp_path):
    reference, candidate = rows(tmp_path)
    result = check_updates(reference, candidate, expected=1)
    assert result["matched_updates"] == 1
    assert result["every_update_inputs_order_draws_lr_equal"]


@pytest.mark.parametrize("change", [
    {"pair_sha256": "changed"}, {"noise_rng_before": "changed"},
    {"learning_rate": .5}, {"applied_noise_max_abs": .03},
    {"training_noise_mode": "gaussian"}, {"consistency_weight": .1},
    {"loss": .3},
])
def test_mismatch_cannot_silently_reuse_control(tmp_path, change):
    reference, candidate = rows(tmp_path, change)
    with pytest.raises(ValueError):
        check_updates(reference, candidate, expected=1)


def test_missing_or_extra_updates_rejected(tmp_path):
    reference, candidate = rows(tmp_path)
    with pytest.raises(ValueError, match="update count"):
        check_updates(reference, candidate, expected=2)
