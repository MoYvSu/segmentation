# -*- coding: utf-8 -*-
"""RGB 补边器共同初始层、分阶段梯度与严格权重格式；不使用赛题图。"""
from copy import deepcopy

import pytest
import torch

from models.boundary_restore import BoundaryRestoreNet, FORMAT as BASE_FORMAT, state_sha256
from models.boundary_context import BoundaryContextNet, FORMAT, load_boundary_context


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def e0_payload(model):
    return dict(format=BASE_FORMAT, architecture=model.architecture, epoch=0, global_step=0,
                model_state_dict=model.state_dict(), model_sha256=state_sha256(model),
                initial_state_sha256=model.initial_state_sha256)


@pytest.mark.parametrize('shape',[(2,1,32,32),(1,1,31,47),(1,1,1,3)])
def test_shared_random_initialization_and_exact_output_for_any_rgb(tmp_path,shape):
    state = torch.get_rng_state()
    base = BoundaryRestoreNet(width=4,seed=11)
    model = BoundaryContextNet(width=4,seed=11)
    assert torch.equal(state,torch.get_rng_state())
    path = tmp_path/'e0.pt'; torch.save(e0_payload(base),path)
    receipt = model.initialize_from_e0(path)
    assert receipt['source'] == 'sealed_old_random_e0'
    assert model.shared_initial_state_sha256 == state_sha256(base)
    for name,value in base.state_dict().items():
        actual = model.state_dict()[name]
        assert torch.equal(value,actual[:,:1] if name == 'stem.weight' else actual)
    assert not model.stem.weight[:,1:].any()
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters()) == 4*3*3*3
    heat = torch.rand(shape); rgb = torch.rand(shape[0],3,*shape[-2:])
    assert torch.equal(model(heat,rgb),heat) and torch.equal(base(heat),heat)
    assert not model.residual(heat,rgb).any()


def test_rgb_parameters_receive_gradient_after_zero_output_wakes_up():
    model = BoundaryContextNet(width=4,seed=7)
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    heat = torch.full((1,1,24,32),.2)
    rgb = torch.rand(1,3,24,32)
    target = .3+.4*rgb[:,:1]
    for step in range(2):
        optimizer.zero_grad()
        loss = (model(heat,rgb)-target).abs().mean(); loss.backward()
        if step == 0:
            assert model.correction.weight.grad.abs().sum() > 0
            assert model.stem.weight.grad[:,1:].abs().sum() == 0
        else:
            assert model.stem.weight.grad[:,1:].abs().sum() > 0
        optimizer.step()
    assert not torch.equal(model(heat,rgb),model(heat,torch.zeros_like(rgb)))


def test_e0_rejects_trained_and_forged_initialization(tmp_path):
    base = BoundaryRestoreNet(width=4,seed=8)
    model = BoundaryContextNet(width=4,seed=8)
    path = tmp_path/'bad.pt'
    payload = e0_payload(base); payload['epoch']=5; payload['global_step']=320
    torch.save(payload,path)
    with pytest.raises(ValueError,match='zero updates'):
        model.initialize_from_e0(path)
    payload = e0_payload(base)
    payload['model_state_dict']['encoder1.body.0.weight'].add_(.01)
    torch.save(payload,path)
    with pytest.raises(ValueError,match='exact shared random'):
        model.initialize_from_e0(path)


def test_context_strict_reload_and_reject_wrong_format_or_keys(tmp_path):
    model = BoundaryContextNet(width=4,seed=19)
    with torch.no_grad():
        model.correction.weight.fill_(.001); model.stem.weight[:,1:].fill_(.002)
    payload = dict(format=FORMAT, architecture=model.architecture,
        model_state_dict=model.state_dict(), model_sha256=state_sha256(model),
        initial_state_sha256=model.initial_state_sha256)
    path = tmp_path/'context.pt'; torch.save(payload,path)
    loaded = load_boundary_context(path)
    heat,rgb = torch.rand(1,1,25,33),torch.rand(1,3,25,33)
    assert torch.equal(loaded(heat,rgb),model(heat,rgb))
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    for mutation,error in [('key',RuntimeError),('format',ValueError),('architecture',ValueError)]:
        bad = deepcopy(payload)
        if mutation == 'key': bad['model_state_dict']['unknown.weight']=torch.ones(1)
        if mutation == 'format': bad['format']=BASE_FORMAT
        if mutation == 'architecture': bad['architecture']['context_channels']=2
        torch.save(bad,path)
        with pytest.raises(error): load_boundary_context(path)


@pytest.mark.parametrize('rgb',[torch.zeros(1,1,16,16),torch.zeros(1,3,15,16),
    torch.full((1,3,16,16),1.1),torch.full((1,3,16,16),float('nan'))])
def test_context_shape_range_must_match_heat(rgb):
    with pytest.raises(ValueError):
        BoundaryContextNet(width=4)(torch.zeros(1,1,16,16),rgb)
