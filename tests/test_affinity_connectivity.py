# -*- coding: utf-8 -*-
"""真实部署关系引导困难边采样：方向、未知域、实例均衡与兼容性。"""
import numpy as np
import pytest
import torch

from utils.affinity_connectivity import deployment_connectivity_loss
from utils.affinity_loss import build_affinity_targets_torch
from utils.post_process import boundary_watershed_separation


def inputs(gt, prediction, offsets=((0, 1),), probability=.5):
    gt = torch.as_tensor(gt, dtype=torch.long)
    prediction = torch.as_tensor(prediction, dtype=torch.long)
    if gt.ndim == 2:
        gt, prediction = gt[None], prediction[None]
    target, valid = build_affinity_targets_torch(gt, (gt > 0)[:, None], offsets=offsets)
    logits = torch.full(target.shape, float(torch.logit(torch.tensor(probability))), requires_grad=True)
    return dict(logits=logits, target=target, edge_valid=valid,
                predicted_instances=prediction, gt_instances=gt, offsets=offsets)


def test_negative_and_positive_gradients_have_opposite_correct_directions():
    neg = inputs([[1, 1, 2, 2], [1, 1, 2, 2]], [[7]*4]*2, probability=.8)
    pos = inputs([[1]*4]*2, [[7, 7, 8, 8]]*2, probability=.2)
    ln, mn = deployment_connectivity_loss(**neg)
    lp, mp = deployment_connectivity_loss(**pos)
    ln.backward(); lp.backward()
    assert torch.all(neg['logits'].grad[neg['target'] == 0] >= 0)
    assert torch.all(pos['logits'].grad <= 0)
    assert int((neg['logits'].grad > 0).sum()) == 2
    assert int((pos['logits'].grad < 0).sum()) == 2
    assert mn['negative']['selected_groups'] == mp['positive']['selected_groups'] == 1


def test_empty_satisfied_and_no_prediction_labels_have_differentiable_zero():
    for probability, prediction in [(.1, [[8]*4]*2), (.8, [[0]*4]*2)]:
        item = inputs([[1, 1, 2, 2]]*2, prediction, probability=probability)
        loss, metrics = deployment_connectivity_loss(**item)
        loss.backward()
        assert loss.item() == 0 and torch.count_nonzero(item['logits'].grad) == 0
        assert metrics['selected_edges'] == 0


def test_probability_equal_to_margin_is_satisfied_and_bfloat_target_is_accepted():
    negative = inputs([[1, 2]]*2, [[7, 7]]*2, probability=.3)
    positive = inputs([[1, 1]]*2, [[7, 8]]*2, probability=.7)
    for item in (negative, positive):
        item['target'] = item['target'].bfloat16()
        loss, metrics = deployment_connectivity_loss(**item)
        loss.backward()
        assert loss.item() == 0 and metrics['selected_edges'] == 0


def test_edge_ignore_unknown_and_untrusted_endpoints_never_receive_gradient():
    item = inputs([[1, 2, 0, 3, 4]]*3, [[8]*5]*3)
    item['edge_valid'][0, 0, 0, 0] = False
    trust = torch.ones_like(item['gt_instances'], dtype=torch.bool)
    trust[0, 1, 1] = False
    loss, metrics = deployment_connectivity_loss(**item, trusted_pixels=trust[:, None], minimum_group_edges=1)
    loss.backward()
    gradient = item['logits'].grad[0, 0]
    assert gradient[0, 0] == gradient[1, 0] == 0
    assert not gradient[:, 1:3].any()
    assert metrics['negative']['skipped_untrusted_endpoint_edges'] == 1


@pytest.mark.parametrize('unknown', [True, False])
def test_long_offset_cannot_cross_unknown_or_untrusted_pixels(unknown):
    gt = [[1, 1, 1, 2, 2]]*2
    if unknown:
        gt = [[1, 1, 0, 2, 2]]*2
    item = inputs(gt, [[7]*5]*2, offsets=((0, 4),))
    trust = torch.ones_like(item['gt_instances'], dtype=torch.bool)
    if not unknown:
        trust[:, :, 2] = False
    loss, metrics = deployment_connectivity_loss(**item, trusted_pixels=trust)
    loss.backward()
    assert loss.item() == 0 and not item['logits'].grad.any()
    field = 'skipped_unknown_path_edges' if unknown else 'skipped_untrusted_path_edges'
    assert metrics['negative'][field] == 2


def test_prediction_zero_seam_inside_long_edge_does_not_hide_true_split():
    item = inputs([[1]*5]*2, [[7, 7, 0, 8, 8]]*2, offsets=((0, 4),), probability=.2)
    loss, metrics = deployment_connectivity_loss(**item)
    loss.backward()
    assert metrics['positive']['selected_edges'] == 2
    assert (item['logits'].grad < 0).sum() == 2


def test_one_remaining_active_edge_is_not_dropped_from_supported_relation():
    item = inputs([[1, 2]]*2, [[7, 7]]*2, probability=.1)
    with torch.no_grad():
        item['logits'][0, 0, 0, 0] = torch.logit(torch.tensor(.8))
    loss, metrics = deployment_connectivity_loss(**item, minimum_group_edges=2)
    loss.backward()
    assert metrics['negative']['valid_edges'] == 2
    assert metrics['negative']['selected_edges'] == 1


def test_relation_mean_is_not_weighted_by_number_of_pixels():
    item = inputs([[1, 2, 2, 3]]*4, [[9]*4]*4)
    item['edge_valid'][0, 0, 1:, 2] = False
    with torch.no_grad():
        item['logits'][0, 0, :, 0] = torch.logit(torch.tensor(.8))
        item['logits'][0, 0, 0, 2] = torch.logit(torch.tensor(.4))
    loss, metrics = deployment_connectivity_loss(**item, minimum_group_edges=1)
    expected = (-np.log(1-.8)-np.log(1-.4))/2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert metrics['negative']['selected_groups'] == 2
    assert float(item['logits'].grad[0, 0, :, 0].sum()) == pytest.approx(.8/2)
    assert float(item['logits'].grad[0, 0, 0, 2]) == pytest.approx(.4/2)


def test_same_numeric_ids_in_different_batch_items_are_separate_groups():
    item = inputs([[[1, 2]], [[1, 2]]], [[[8, 8]], [[8, 8]]])
    loss, metrics = deployment_connectivity_loss(**item, minimum_group_edges=2)
    assert metrics['negative']['candidate_groups'] == 2
    assert metrics['negative']['skipped_small_groups'] == 2 and loss.item() == 0


def test_group_and_edge_budgets_preserve_both_directions_and_determinism():
    gt = [[1, 2, 2, 3, 3, 3, 3]]*4
    pred = [[9, 9, 9, 10, 10, 11, 11]]*4
    item = inputs(gt, pred)
    loss, metrics = deployment_connectivity_loss(**item, max_groups=2, edges_per_group=2)
    again, repeated = deployment_connectivity_loss(**item, max_groups=2, edges_per_group=2)
    assert loss.item() == again.item() and metrics == repeated
    assert metrics['selected_groups'] == 2 and metrics['selected_edges'] == 4
    assert metrics['negative']['selected_groups'] == metrics['positive']['selected_groups'] == 1
    loss.backward()
    assert int(torch.count_nonzero(item['logits'].grad)) == 4


@pytest.mark.parametrize('keyword,value', [('max_groups', 0), ('max_groups', True),
    ('edges_per_group', 1.5), ('minimum_group_edges', -1), ('negative_margin', -.1),
    ('positive_margin', 1.0), ('negative_margin', .8), ('positive_margin', float('nan'))])
def test_invalid_hyperparameters_fail(keyword, value):
    item = inputs([[1, 2]]*2, [[7, 7]]*2)
    with pytest.raises(ValueError):
        deployment_connectivity_loss(**item, **{keyword: value})


@pytest.mark.parametrize('problem', ['shape', 'float_ids', 'nan', 'nonbinary', 'trust_shape', 'target_mismatch', 'offset'])
def test_invalid_inputs_fail(problem):
    item = inputs([[1, 2]]*2, [[7, 7]]*2)
    if problem == 'shape': item['edge_valid'] = item['edge_valid'][..., :1]
    elif problem == 'float_ids': item['gt_instances'] = item['gt_instances'].float()
    elif problem == 'nan': item['logits'] = torch.full_like(item['logits'], float('nan'))
    elif problem == 'nonbinary': item['target'] = item['target'] + .1
    elif problem == 'trust_shape': item['trusted_pixels'] = torch.ones(3, dtype=torch.bool)
    elif problem == 'target_mismatch': item['target'] = torch.ones_like(item['target'])
    else: item['offsets'] = ((0, 0),)
    with pytest.raises(ValueError): deployment_connectivity_loss(**item)


def _deployed(boundary):
    semantic = np.ones(boundary.shape, np.uint8)
    return boundary_watershed_separation(semantic, boundary, dilate_width=1,
        min_area=20, bridge_width=0, marker_border_seal_width=2)[0]


def test_actual_deployment_leak_and_false_crack_generate_both_supervision_directions():
    height, width = 64, 80
    divided_gt = np.ones((height, width), np.int64)
    divided_gt[:, width//2:] = 2
    leak = np.zeros((height, width), np.uint8)
    leak[:25, width//2] = 1
    leak[39:, width//2] = 1
    merged = _deployed(leak)
    assert len(np.unique(merged[merged > 0])) == 1
    negative = inputs(divided_gt, merged, probability=.8)
    ln, mn = deployment_connectivity_loss(**negative)
    assert ln.item() > 0 and mn['negative']['selected_edges'] > 0
    closed = np.zeros_like(leak)
    closed[:, width//2] = 1
    split = _deployed(closed)
    assert len(np.unique(split[split > 0])) == 2
    positive = inputs(np.ones_like(divided_gt), split, offsets=((0, 4),), probability=.2)
    lp, mp = deployment_connectivity_loss(**positive)
    assert lp.item() > 0 and mp['positive']['selected_edges'] > 0
    # 这里只证明取样对应真实最终输出的两种错误，不声称一步梯度就可修复拓扑。
