# -*- coding: utf-8 -*-
"""用交付训练器的实际优化器语句，检查 CPU GradScaler 的溢出恢复。"""

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "submission/AIC-2026-59388388-代码与数据/training/source"
TRAINERS = {
    "stage2": ("train_stage2.py", "total_loss"),
    "affinity": ("train_affinity_geometry_g1.py", "loss"),
}


def statement_lists(node):
    for _, value in ast.iter_fields(node):
        if isinstance(value, list):
            if value and all(isinstance(item, ast.stmt) for item in value):
                yield value
            for item in value:
                if isinstance(item, ast.AST):
                    yield from statement_lists(item)
        elif isinstance(value, ast.AST):
            yield from statement_lists(value)


@lru_cache(None)
def optimizer_block(trainer):
    """直接编译真实循环中的 loss guard→step/EMA；不复制优化器实现。"""
    filename, loss_name = TRAINERS[trainer]
    path = SOURCE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    expected_guard = ast.dump(ast.parse(f"not torch.isfinite({loss_name})", mode="eval").body)
    for statements in statement_lists(tree):
        for start, node in enumerate(statements):
            if not isinstance(node, ast.If) or ast.dump(node.test) != expected_guard:
                continue
            for end in range(start + 1, len(statements)):
                candidate = statements[end]
                if trainer == "stage2":
                    last = isinstance(candidate, ast.If) and any(
                        isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                        and child.func.id == "update_ema" for child in ast.walk(candidate)
                    )
                else:
                    last = (isinstance(candidate, ast.AugAssign)
                            and ast.dump(candidate.target) == ast.dump(
                                ast.parse('totals["optimizer_steps"] += 1').body[0].target))
                if last:
                    return compile(ast.Module(body=statements[start:end + 1], type_ignores=[]),
                                   str(path), "exec")
    raise AssertionError(f"没有找到 {filename} 的真实 loss guard/optimizer 语句")


def run_actual_step(trainer, model, optimizer, scaler, loss, *, amp=True, ema=None):
    ema = [] if ema is None else ema
    state = dict(
        torch=torch, total_loss=loss, loss=loss, optimizer=optimizer, scaler=scaler,
        use_amp=amp, grad_clip=0.25, diagnostics_enabled=False,
        clip_params=list(model.parameters()), applied_steps=0,
        teacher_model=object(), student_model=model, update_teacher=True, ema_decay=0.99,
        update_ema=lambda *args: ema.append(True),
        system=SimpleNamespace(geometry_trainable_parameters=model.parameters),
        cfg={"grad_clip": 0.25}, totals={"optimizer_steps": 0},
    )
    exec(optimizer_block(trainer), state)
    return state["applied_steps"] if trainer == "stage2" else state["totals"]["optimizer_steps"]


def tiny_model():
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, -0.2]]))
        model.bias.fill_(0.1)
    return model


def finite_loss(model):
    return (model(torch.tensor([[2.0, -1.0]])) - 1.5).square().mean()


def assert_state_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("amp", [True, False])
def test_finite_step_and_clipping_equal_previous_behavior(trainer, amp):
    actual, reference = tiny_model(), tiny_model()
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.1, momentum=0.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cpu", enabled=amp, init_scale=8.0)
    reference_scaler = torch.amp.GradScaler("cpu", enabled=amp, init_scale=8.0)
    loss, expected_loss = finite_loss(actual), finite_loss(reference)
    # 旧行为只在有限梯度下作数值对照：相同 scale/unscale/裁剪/更新顺序。
    reference_scaler.scale(expected_loss).backward()
    reference_scaler.unscale_(reference_optimizer)
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25,
                                                   error_if_nonfinite=True)
    assert expected_norm > 0.25
    reference_scaler.step(reference_optimizer)
    reference_scaler.update()
    ema = []
    assert run_actual_step(trainer, actual, optimizer, scaler, loss, amp=amp, ema=ema) == 1
    assert_state_equal(actual.state_dict(), reference.state_dict())
    assert scaler.state_dict() == reference_scaler.state_dict()
    for actual_parameter, reference_parameter in zip(actual.parameters(), reference.parameters()):
        assert torch.equal(actual_parameter.grad, reference_parameter.grad)
        assert_state_equal(optimizer.state[actual_parameter],
                           reference_optimizer.state[reference_parameter])
    if trainer == "stage2":
        assert len(ema) == 1


@pytest.mark.parametrize("trainer", TRAINERS)
def test_inf_gradient_skips_backs_off_and_next_finite_batch_recovers(trainer):
    model = tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, backoff_factor=0.5)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    # loss 数值有限，但反向梯度溢出，区分于真正 NaN/Inf loss。
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    ema = []
    try:
        loss = finite_loss(model)
        assert torch.isfinite(loss)
        assert run_actual_step(trainer, model, optimizer, scaler, loss, ema=ema) == 0
    finally:
        hook.remove()
    assert_state_equal(model.state_dict(), before)
    assert not optimizer.state
    assert scaler.get_scale() == 4.0
    assert not ema
    optimizer.zero_grad(set_to_none=True)
    assert run_actual_step(trainer, model, optimizer, scaler, finite_loss(model), ema=ema) == 1
    assert any(not torch.equal(model.state_dict()[key], value) for key, value in before.items())
    assert scaler.get_scale() == 4.0
    if trainer == "stage2":
        assert len(ema) == 1


@pytest.mark.parametrize("trainer", TRAINERS)
def test_fp32_nonfinite_gradient_still_raises(trainer):
    model = tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            run_actual_step(trainer, model, optimizer, scaler, finite_loss(model), amp=False)
    finally:
        hook.remove()
    assert_state_equal(model.state_dict(), before)


@pytest.mark.parametrize("trainer", TRAINERS)
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_truly_nonfinite_loss_raises_before_backward(trainer, amp, value):
    model = tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=amp, init_scale=8.0)
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    ema = []
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        run_actual_step(trainer, model, optimizer, scaler, finite_loss(model) * value,
                        amp=amp, ema=ema)
    assert_state_equal(model.state_dict(), before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not optimizer.state
    assert scaler.get_scale() == 8.0 if amp else scaler.get_scale() == 1.0
    assert not ema
