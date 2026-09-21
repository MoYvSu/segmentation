# -*- coding: utf-8 -*-
"""AMP跳步不能计为参数更新，也不能推动EMA教师。"""
import contextlib
import copy

import torch

import train_stage2
from utils.loss import BoundaryLoss


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.decoder = torch.nn.Module()
        self.decoder.seg_fpn = torch.nn.Identity()
        self.decoder.seg_branch = torch.nn.Conv2d(3, 1, 1)
        self.decoder.boundary_fpn = torch.nn.Identity()
        self.decoder.boundary_branch = torch.nn.Conv2d(3, 1, 1)
        self.decoder.boundary_branch.requires_grad_(False)

    def forward(self, images, output_size=None):
        return torch.cat((self.decoder.seg_branch(images),
                          self.decoder.boundary_branch(images)), dim=1)


class SkipFirstScaler:
    def __init__(self):
        self.scale_value = 256.0
        self.calls = 0

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        if self.calls:
            optimizer.step()

    def update(self):
        if not self.calls:
            self.scale_value /= 2
        self.calls += 1


def test_amp_counts_real_updates_and_teacher_only_follows_success(monkeypatch):
    monkeypatch.setattr(train_stage2, "autocast", lambda *a, **kw: contextlib.nullcontext())
    student = TinyStudent()
    before = student.decoder.seg_branch.weight.detach().clone()
    teacher = copy.deepcopy(student)
    teacher_calls = []
    monkeypatch.setattr(train_stage2, "update_ema", lambda *args: teacher_calls.append(1))
    batch = {"image": torch.ones(1, 3, 8, 8), "target": torch.ones(1, 2, 8, 8),
             "weight": torch.ones(1, 1, 8, 8)}
    result = train_stage2.train_one_epoch(
        student, [batch], None, num_steps=2, num_unlabeled_steps=0,
        criterion=BoundaryLoss(freeze_boundary=True, center_weight=0),
        unsup_weight=0, unsup_seg_weight=0, ema_decay=.99,
        optimizer=torch.optim.SGD(student.parameters(), lr=.01), scaler=SkipFirstScaler(),
        device="cpu", use_amp=True, teacher_model=teacher, freeze_boundary=True,
    )
    assert result["attempted_steps"] == 2
    assert result["optimizer_steps"] == 1
    assert result["amp_skipped_steps"] == 1
    assert len(teacher_calls) == 1
    assert not torch.equal(before, student.decoder.seg_branch.weight)
