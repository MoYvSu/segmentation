# -*- coding: utf-8 -*-
"""保留短测回执和真实梯度隔离的合成回归。"""
import json

import pytest
import torch

from tools.probe_affinity_weak_retain import component_gradient_record, scalar_record


def test_scalar_loss_receipt_is_json_serializable_without_graph():
    value = torch.tensor(1.25, requires_grad=True)
    result = scalar_record({'loss': value, 'rows': [value, 3], 'complete': True})
    assert json.loads(json.dumps(result)) == {'loss': 1.25, 'rows': [1.25, 3], 'complete': True}
    with pytest.raises(ValueError):
        scalar_record(torch.ones(2))


def test_component_probe_does_not_accumulate_parameter_optimizer_gradients():
    torch.manual_seed(4)
    head = torch.nn.Conv2d(2, 2, 1)
    x = torch.randn(1, 2, 3, 5)
    logits = head(x)
    logits.retain_grad()
    terms = dict(full=logits.square().mean(), local=(logits-1).square().mean(),
                 retention=(logits.detach()-logits).square().mean())
    report = component_gradient_record(terms, head)
    assert report['unweighted_parameter_gradient_norms']['full'] > 0
    assert report['unweighted_parameter_gradient_norms']['retention'] == 0
    assert report['full_local_retention_cosine'] is None
    assert all(p.grad is None for p in head.parameters())
    # 真正更新前清空中间logit诊断梯度，避免把三次probe累加到总目标。
    logits.grad = None
    total = terms['full'] + terms['local'] + terms['retention']
    expected = torch.autograd.grad(total, logits, retain_graph=True)[0]
    logits.grad = None
    total.backward()
    assert torch.equal(logits.grad, expected)
    actual = sum(float(p.grad.square().sum()) for p in head.parameters())**.5
    assert actual == pytest.approx(report['full_local_gradient_norm'], rel=1.e-6)
