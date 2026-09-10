# -*- coding: utf-8 -*-
"""实例集合监督的 ignore、容量、小实例覆盖和辅助层合同。"""

import pytest
import torch

from utils.mask_set_loss import MaskSetCriterion, targets_from_direct_batch


def seeded(seed=71):
    return torch.Generator().manual_seed(seed)


def target_from_map(instance_map, labels):
    valid = instance_map > 0
    ids = torch.unique(instance_map[valid], sorted=True)
    return {"labels": torch.tensor(labels, dtype=torch.long),
            "masks": instance_map[None] == ids[:, None, None], "valid": valid}


def test_adapter_keeps_unknown_ignored_and_uses_explicit_classes():
    instance_map = torch.tensor([[[3, 3, 0], [0, 17, 17]]])
    batch = {"affinity_instance_map": instance_map,
             "affinity_valid_content": (instance_map > 0)[:, None],
             "semantic_target": torch.zeros(1, 1, 4, 6),
             "instance_classes": [{"3": 1, "17": 0, "22": 1}]}
    target = targets_from_direct_batch(batch, "cpu")[0]
    assert target["labels"].tolist() == [1, 0]
    assert target["instance_ids"].tolist() == [3, 17]
    assert target["masks"].dtype == torch.bool
    assert target["masks"].shape == (2, 2, 3)
    assert not target["masks"][:, ~target["valid"]].any()
    assert "content_valid" not in target
    batch["instance_classes"] = [{3: 1}]
    with pytest.raises(ValueError, match="missing class"):
        targets_from_direct_batch(batch, "cpu")
    batch["instance_classes"] = [{3: 1, 17: 2}]
    with pytest.raises(ValueError, match="classes 0/1"):
        targets_from_direct_batch(batch, "cpu")


def test_adapter_semantic_fallback_uses_same_ids_and_rejects_mixed_class():
    coarse = torch.tensor([[[3, 0], [17, 17]]])
    fine = torch.tensor([[[3, 3, 0, 0], [3, 0, 0, 0],
                          [17, 17, 17, 17], [17, 17, 17, 17]]])
    semantic = (fine == 3).float()
    batch = {"affinity_instance_map": coarse,
             "affinity_valid_content": coarse > 0,
             "semantic_instance_map": fine,
             "semantic_target": semantic[:, None]}
    assert targets_from_direct_batch(batch, "cpu")[0]["labels"].tolist() == [1, 0]
    semantic[0, 0, 1] = 0
    with pytest.raises(ValueError, match="mixed semantic"):
        targets_from_direct_batch(batch, "cpu")


def test_sampling_covers_single_pixel_instance_and_is_seeded():
    instance_map = torch.ones(32, 32, dtype=torch.long)
    instance_map[16, 16] = 2
    instance_map[:, -4:] = 0
    target = target_from_map(instance_map, [0, 1])
    criterion = MaskSetCriterion(num_points=9, mask_points=7)
    first = criterion._sample_points(target, 9, seeded())
    second = criterion._sample_points(target, 9, seeded())
    assert torch.equal(first, second)
    assert target["valid"].flatten()[first].all()
    assert (first == 16 * 32 + 16).any()
    outputs = {"pred_logits": torch.randn(1, 3, 3, generator=seeded(9)),
               "pred_masks": torch.randn(1, 3, 32, 32, generator=seeded(8))}
    one = criterion(outputs, [target], seeded())
    two = criterion(outputs, [target], seeded())
    for key in one:
        torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)


def test_unknown_values_do_not_change_matching_or_mask_loss_and_have_zero_gradient():
    instance_map = torch.tensor([[1, 1, 2, 0, 0], [1, 1, 2, 0, 0],
                                 [1, 1, 2, 0, 0], [1, 1, 2, 0, 0]])
    target = target_from_map(instance_map, [0, 1])
    predictions = torch.stack([target["masks"][1], target["masks"][0]]).float() * 8 - 4
    predictions[:, ~target["valid"]] = -20
    outputs = {"pred_logits": torch.tensor([[[0., 7., -3.], [7., 0., -3.]]], requires_grad=True),
               "pred_masks": predictions[None].requires_grad_()}
    alternate = {"pred_logits": outputs["pred_logits"].detach().clone(),
                 "pred_masks": predictions[None].detach().clone()}
    alternate["pred_masks"][:, :, ~target["valid"]] = 40
    criterion = MaskSetCriterion(num_points=7, mask_points=5)
    rows, columns = criterion.match(outputs, [target], seeded())[0]
    assert rows.tolist() == [0, 1] and columns.tolist() == [1, 0]
    changed_match = criterion.match(alternate, [target], seeded())[0]
    assert all(torch.equal(a, b) for a, b in zip((rows, columns), changed_match))
    original = criterion(outputs, [target], seeded())
    changed = criterion(alternate, [target], seeded())
    torch.testing.assert_close(original["loss_mask"], changed["loss_mask"], rtol=0, atol=0)
    torch.testing.assert_close(original["loss_dice"], changed["loss_dice"], rtol=0, atol=0)
    original["loss_total"].backward()
    gradient = outputs["pred_masks"].grad[0]
    assert torch.count_nonzero(gradient[:, ~target["valid"]]) == 0
    assert torch.count_nonzero(gradient[:, target["valid"]]) > 0


def test_duplicate_and_empty_queries_receive_no_object_but_unknown_only_does_not():
    instance_map = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
    target = target_from_map(instance_map, [0])
    positive = target["masks"][0].float() * 8 - 4
    unknown = -positive
    masks = torch.stack([positive, positive, unknown, torch.full_like(positive, -4)])[None]
    outputs = {"pred_logits": torch.tensor([[[8., 0., -4.], [6., 0., -4.],
                                             [6., 0., -4.], [6., 0., -4.]]], requires_grad=True),
               "pred_masks": masks.requires_grad_()}
    criterion = MaskSetCriterion(class_weight=1, mask_weight=0, dice_weight=0)
    losses = criterion(outputs, [target], seeded())
    assert losses["matched_count"].item() == 1
    assert losses["ignored_no_object"].item() == 1
    assert losses["supervised_no_object"].item() == 2
    losses["loss_total"].backward()
    gradient = outputs["pred_logits"].grad[0]
    assert gradient[1, 2] < 0  # 重复 query 应增加 no-object 概率。
    assert gradient[3, 2] < 0  # 空 query 也受到压力，不能全空逃避。
    assert torch.count_nonzero(gradient[2]) == 0  # unknown-only 不强制负标签。
    assert torch.count_nonzero(outputs["pred_masks"].grad) == 0  # 门控 detach。
    corrected = {"pred_logits": outputs["pred_logits"].detach().clone(), "pred_masks": masks.detach()}
    corrected["pred_logits"][0, 1] = torch.tensor([-4., 0., 6.])
    assert criterion(corrected, [target], seeded())["loss_class"] < losses["loss_class"]


def test_all_unknown_image_has_no_supervision_and_capacity_overflow_fails():
    target = target_from_map(torch.zeros(4, 4, dtype=torch.long), [])
    outputs = {"pred_logits": torch.randn(1, 2, 3, requires_grad=True),
               "pred_masks": torch.randn(1, 2, 4, 4, requires_grad=True)}
    criterion = MaskSetCriterion()
    losses = criterion(outputs, [target], seeded())
    assert losses["loss_total"].item() == 0
    assert losses["ignored_no_object"].item() == 2
    losses["loss_total"].backward()
    assert torch.count_nonzero(outputs["pred_logits"].grad) == 0
    assert torch.count_nonzero(outputs["pred_masks"].grad) == 0
    nonempty = target_from_map(torch.tensor([[1, 2], [1, 2]]), [0, 1])
    too_few = {"pred_logits": torch.zeros(1, 1, 3), "pred_masks": torch.zeros(1, 1, 2, 2)}
    with pytest.raises(ValueError, match="query capacity"):
        criterion(too_few, [nonempty], seeded())


def test_auxiliary_layer_loss_is_averaged_and_backpropagates():
    target = target_from_map(torch.tensor([[1, 1], [1, 1]]), [1])
    logits = torch.tensor([[[0., 2., -2.], [-2., 0., 2.]]])
    masks = torch.tensor([[[[2., 2.], [2., 2.]], [[-2., -2.], [-2., -2.]]]])
    criterion = MaskSetCriterion(aux_weight=0.7)
    main = {"pred_logits": logits.clone().requires_grad_(), "pred_masks": masks.clone().requires_grad_()}
    auxiliary = [{"pred_logits": logits.clone().requires_grad_(),
                  "pred_masks": masks.clone().requires_grad_()} for _ in range(2)]
    base = criterion(main, [target], seeded())["loss_total"]
    full = criterion({**main, "aux_outputs": auxiliary}, [target], seeded())
    torch.testing.assert_close(full["loss_aux"], base)
    torch.testing.assert_close(full["loss_total"], base * 1.7)
    full["loss_total"].backward()
    for layer in auxiliary:
        assert layer["pred_logits"].grad.abs().sum() > 0
        assert layer["pred_masks"].grad.abs().sum() > 0


def test_accurate_optional_content_mask_excludes_padding_from_unknown_support():
    target = target_from_map(torch.tensor([[1, 0, 0], [1, 0, 0]]), [0])
    criterion = MaskSetCriterion()
    masks = torch.tensor([[[-3., -3., 3.], [-3., -3., 3.]]])
    assert not criterion._no_object_support(masks, target).item()
    target["content_valid"] = torch.tensor([[True, True, False], [True, True, False]])
    # 有准确内容 mask 时，只有 padding 响应相当于内容内全空，应监督 no-object。
    assert criterion._no_object_support(masks, target).item()


def ownership_case():
    target = target_from_map(torch.tensor([[1, 1, 2, 2, 0], [1, 1, 2, 2, 0]]), [0, 1])
    masks = torch.stack([target["masks"][0], target["masks"][1], target["masks"][0]]).float()*6-3
    outputs = {"pred_logits": torch.tensor([[[8., -2., -4.], [-2., 8., -4.], [1., 0., 0.]]]),
               "pred_masks": masks[None].requires_grad_()}
    return outputs, target


def test_ownership_penalizes_duplicate_competitor_and_ignores_unknown():
    outputs, target = ownership_case()
    criterion = MaskSetCriterion(ownership_weight=.5)
    loss = criterion(outputs, [target], seeded())
    torch.testing.assert_close(loss["loss_total"], loss["loss_supervised"] + .5*loss["loss_ownership"])
    loss["loss_ownership"].backward()
    gradient = outputs["pred_masks"].grad[0]
    assert (gradient[0][target["masks"][0]] < 0).all()
    assert (gradient[2][target["masks"][0]] > 0).all()
    assert torch.count_nonzero(gradient[:, ~target["valid"]]) == 0
    changed = {"pred_logits": outputs["pred_logits"], "pred_masks": outputs["pred_masks"].detach().clone()}
    changed["pred_masks"][0, 2, target["valid"]] = -8
    assert criterion(changed, [target], seeded())["loss_ownership"] < loss["loss_ownership"]


def test_ownership_uses_matching_not_query_order_and_zero_weight_preserves_loss():
    outputs, target = ownership_case()
    criterion = MaskSetCriterion(ownership_weight=.5)
    original = criterion(outputs, [target], seeded())
    order = [2, 0, 1]
    permuted = {k: v[:, order] for k, v in outputs.items()}
    torch.testing.assert_close(criterion(permuted, [target], seeded())["loss_ownership"], original["loss_ownership"])
    base = MaskSetCriterion()(outputs, [target], seeded())
    disabled = MaskSetCriterion(ownership_weight=0)(outputs, [target], seeded())
    torch.testing.assert_close(base["loss_total"], disabled["loss_total"], rtol=0, atol=0)
    torch.testing.assert_close(base["loss_total"], original["loss_supervised"], rtol=0, atol=0)


def test_ownership_does_not_replace_absolute_coverage_loss():
    outputs, target = ownership_case()
    # 保留两个匹配query，避免重复槽位改变匈牙利选择。
    outputs = {k: v[:, :2] for k, v in outputs.items()}
    criterion = MaskSetCriterion(ownership_weight=.5)
    good = criterion(outputs, [target], seeded())
    empty = {"pred_logits": outputs["pred_logits"], "pred_masks": outputs["pred_masks"]-20}
    bad = criterion(empty, [target], seeded())
    torch.testing.assert_close(good["loss_ownership"], bad["loss_ownership"])
    assert bad["loss_mask"] > good["loss_mask"]
    assert bad["loss_total"] > good["loss_total"]
