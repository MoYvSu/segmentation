# -*- coding: utf-8 -*-
"""代表性尾部监督：隔离复用封存训练入口，仅替换响应秩采样。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import ModuleType

import torch

from train_affinity_balance import comparable
from train_backend_adaptation import sha
from utils.affinity_tail_loss import SAMPLING_FORMAT, tail_boundary_ranking_loss
from utils.affinity_fused_loss import fused_ranking_domains
from utils.affinity_fusion import affinity_boundary_probability
from utils.config import load_config

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = 'config/train/affinity_tail.yaml'
FORMAT = 'affinity_tail_ranking_v1'

# 函数的全局变量属于本入口的私有模块；不修改旧入口或其封存源码。
_path = ROOT / 'train_affinity_fused.py'
_bytes = _path.read_bytes()
FUSED_SOURCE_SHA256 = hashlib.sha256(_bytes).hexdigest()
_fused = ModuleType(__name__ + '._fused_core')
_fused.__file__, _fused.__package__ = str(_path), ''
exec(compile(_bytes, str(_path), 'exec'), _fused.__dict__)
_old_reference_config, _old_verify_run = _fused.reference_config, _fused.verify_run
_fused.CONFIG_PATH, _fused.FORMAT = CONFIG_PATH, FORMAT
_fused.EXPECTED_OPTIONS = dict(_fused.EXPECTED_OPTIONS, format=FORMAT, sampling=SAMPLING_FORMAT)
_fused.fused_boundary_ranking_loss = tail_boundary_ranking_loss
_core = _fused._core
REFERENCE, REFERENCE_SHA256 = _fused.REFERENCE, _fused.REFERENCE_SHA256


def reference_config(config):
    """同时核对旧fused及mixed控制，拒绝暗中修改第二个训练变量。"""
    base = _old_reference_config(config)
    candidate = deepcopy(config)
    candidate['affinity_fused'].pop('sampling')
    candidate['affinity_fused']['format'] = 'affinity_fused_ranking_v1'
    if _fused._identity(comparable(candidate)) != _fused._identity(
            comparable(load_config('config/train/affinity_fused.yaml'))):
        raise ValueError('A variable besides representative tail sampling changed')
    return base


def _gradient_audit(base, auxiliary, logits, shared, target, valid, fusion, weight):
    """冻结本次选位检验梯度方向；重抽秩只记录，不改实际优化步骤。"""
    left = torch.autograd.grad(base, shared, retain_graph=True)[0].detach().float()
    right = torch.autograd.grad(auxiliary, shared, retain_graph=True)[0].detach().float() * weight
    ln, rn = float(left.norm()), float(right.norm())
    cos = float((left * right).sum()) / (ln * rn) if ln and rn else None
    grad = torch.autograd.grad(auxiliary, logits, retain_graph=True)[0].detach()
    invalid = grad[~valid]
    if invalid.numel() and bool((invalid != 0).any()):
        raise RuntimeError('Added ranking gradient reached an unknown relation')
    with torch.no_grad():
        _, before = tail_boundary_ranking_loss(logits.detach(), target, valid, fusion_kwargs=fusion)
        boundary, interior = fused_ranking_domains(target, valid)
        scores = affinity_boundary_probability(logits.detach().float(), **fusion)[:, 0]
        selection = []
        for index, group in enumerate(before['groups']):
            if not group['total_pairs']:
                continue
            positions = []
            for mask, domain, ranks, largest in (
                    (boundary, 'true_boundary', 'selected_true_ranks', False),
                    (interior, 'deep_interior', 'selected_interior_ranks', True)):
                coordinates = mask[index].flatten().nonzero().flatten()
                pool = scores[index].flatten()[coordinates]
                tail = pool.topk(group['tail_' + domain], largest=largest, sorted=True).indices
                picked = torch.tensor(group[ranks], dtype=torch.long, device=logits.device)
                positions.append(coordinates[tail[picked]])
            selection.append((index, *positions))

    def fixed_objective(values):
        fused = affinity_boundary_probability(values.float(), **fusion)[:, 0]
        losses, positives, negatives = [], [], []
        for index, pi, ni in selection:
            positive, negative = fused[index].flatten()[pi], fused[index].flatten()[ni]
            losses.append(torch.relu(.1 + negative[None, :] - positive[:, None]).mean())
            positives.append(positive.detach())
            negatives.append(negative.detach())
        loss = torch.stack(losses).mean() if losses else values.reshape(-1)[0] * 0.
        return (loss, float(torch.cat(positives).mean()) if positives else None,
                float(torch.cat(negatives).mean()) if negatives else None)

    probe = logits.detach().clone().requires_grad_(True)
    probe_loss, _, _ = fixed_objective(probe)
    probe_grad = torch.autograd.grad(probe_loss, probe)[0].detach()
    probe_error = float((probe_grad - grad).abs().max())
    if not torch.allclose(probe_grad, grad, rtol=1.e-5, atol=1.e-9):
        raise RuntimeError('Frozen probe gradient differs from the actual ranking gradient')
    del probe, probe_loss, probe_grad
    values = fixed_objective(logits.detach())
    fixed_before = (float(values[0].detach()), *values[1:])
    if abs(fixed_before[0] - float(auxiliary.detach())) > 1.e-7:
        raise RuntimeError('Frozen probe positions differ from the current training objective')
    scale = .01 / max(float(grad.abs().max()), 1.e-12)
    for attempt in range(7):
        changed = logits.detach() - scale * grad
        values = fixed_objective(changed)
        fixed_after = (float(values[0].detach()), *values[1:])
        if fixed_after[0] <= fixed_before[0] + 1.e-6:
            break
        scale *= .1
    else:
        raise RuntimeError('Small fixed-selection descent increases its own objective')
    change_max = float((changed - logits.detach()).abs().max())
    gradient_active = bool(grad.count_nonzero())
    if gradient_active and change_max == 0.:
        raise RuntimeError('Active ranking probe rounded to an unchanged tensor')
    _, dynamic_after = tail_boundary_ranking_loss(changed, target, valid, fusion_kwargs=fusion)
    return dict(shared_parameter='affinity_decoder.affinity_head.0.weight',
        base_norm=ln, weighted_auxiliary_norm=rn, weighted_parameter_ratio=rn / max(ln, 1.e-12),
        cosine=cos, unknown_logit_gradient_max=float(invalid.abs().max()) if invalid.numel() else 0.,
        directional_loss_before=fixed_before[0], directional_loss_after=fixed_after[0],
        directional_selection='frozen_current_positions',
        frozen_selection_gradient_max_error=probe_error,
        directional_logit_change_max=change_max, directional_probe_skipped=not gradient_active,
        directional_loss_delta=fixed_after[0] - fixed_before[0],
        directional_probe_only=True, directional_probe_scale=scale, directional_probe_attempts=attempt + 1,
        dynamic_reselected_loss_after=dynamic_after['loss'], dynamic_reselected_is_gate=False,
        selected_true_delta=(fixed_after[1] - fixed_before[1]) if fixed_before[1] is not None else None,
        selected_interior_delta=(fixed_after[2] - fixed_before[2]) if fixed_before[2] is not None else None)


def verify_run(config, output, expected_updates):
    result = _old_verify_run(config, output, expected_updates)
    folder = Path(output)
    rows = [json.loads(line) for line in (folder / 'fused_loss.jsonl').read_text().splitlines()]
    for row in rows:
        metrics = row['ranking']
        if metrics.get('sampling_format') != SAMPLING_FORMAT:
            raise RuntimeError('A legacy extreme-top256 sampler was used')
        for group in metrics['groups']:
            if not group['total_pairs']:
                continue
            for domain, ranks in (('true_boundary', 'selected_true_ranks'),
                                  ('deep_interior', 'selected_interior_ranks')):
                count = group['tail_' + domain]
                selected = group[ranks]
                budget = min(256, count)
                expected = [count // 2] if budget == 1 else [i * (count - 1) // (budget - 1)
                                                            for i in range(budget)]
                if selected != expected:
                    raise RuntimeError('Incomplete or changed response-rank coverage')
    head = torch.load(folder / 'last_head.pt', map_location='cpu', weights_only=False)
    if (head['fused_experiment']['format'] != FORMAT
            or head['fused_experiment']['options'] != config['affinity_fused']):
        raise RuntimeError('Tail checkpoint lacks explicit version and recipe')
    result.update(sampling_format=SAMPLING_FORMAT, full_tail_rank_coverage=True,
                  ranking_parent_retrained=False, fused_source_sha256=FUSED_SOURCE_SHA256)
    return result


_fused.reference_config = reference_config
_fused._gradient_audit = _gradient_audit
_fused.verify_run = verify_run
verify_reference = _fused.verify_reference
ranking_objective = _fused.ranking_objective
train = _fused.train
_rng_state = _fused._rng_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    train(load_config(args.config), args.output, smoke=args.smoke)
