# -*- coding: utf-8 -*-
"""在显式参数空间量度附加监督；不把梯度比例当作AdamW最终更新比例。"""
from __future__ import annotations

import math
import torch


def norm(values):
    return torch.stack([v.detach().float().square().sum() for v in values]).sum().sqrt()


def gradient_pair(base, auxiliary):
    a, b = norm(base), norm(auxiliary)
    dot = torch.stack([(x.detach().float()*y.detach().float()).sum() for x,y in zip(base,auxiliary)]).sum()
    return dict(base_norm=float(a), auxiliary_norm=float(b),
                cosine=float(dot/(a*b).clamp_min(1e-12)))


def parameter_weight(base_gradients, auxiliary_gradients, *, requested, target_ratio):
    if not math.isfinite(requested) or requested < 0 or not math.isfinite(target_ratio) or target_ratio < 0:
        raise ValueError('Gradient budgets must be finite and nonnegative')
    stats = gradient_pair(base_gradients, auxiliary_gradients)
    if not all(math.isfinite(v) for v in stats.values()):
        raise FloatingPointError('Nonfinite parameter gradients')
    a, b = stats['base_norm'], stats['auxiliary_norm']
    weight = min(float(requested), float(target_ratio)*a/max(b,1e-12)) if a>0 and b>0 else 0.
    return weight, {**stats, 'requested_ratio':float(target_ratio),
                    'weighted_ratio':weight*b/max(a,1e-12), 'coefficient_limited':weight<requested}


@torch.no_grad()
def adamw_delta(optimizer, gradients, max_grad_norm):
    """同一步、同optimizer历史下的假想更新；不修改参数或动量，不是另一条训练轨迹。"""
    if len(optimizer.param_groups)!=1:
        raise ValueError('Probe supports one AdamW parameter group')
    group=optimizer.param_groups[0];params=group['params']
    if len(params)!=len(gradients) or group.get('amsgrad') or group.get('maximize'):
        raise ValueError('Unsupported optimizer contract')
    beta1,beta2=group['betas'];lr,eps,decay=group['lr'],group['eps'],group['weight_decay']
    magnitude=float(norm(gradients));clip=min(1.,float(max_grad_norm)/(magnitude+1e-6))
    deltas=[]
    for p,gradient in zip(params,gradients):
        state=optimizer.state.get(p,{})
        step=int(state.get('step',0))+1;g=gradient.detach()*clip
        m=state.get('exp_avg',torch.zeros_like(p))*beta1+g*(1-beta1)
        v=state.get('exp_avg_sq',torch.zeros_like(p))*beta2+g.square()*(1-beta2)
        denominator=v.sqrt()/math.sqrt(1-beta2**step)+eps
        deltas.append(-lr*decay*p.detach()-(lr/(1-beta1**step))*m/denominator)
    return deltas,dict(unclipped_norm=magnitude,clip_coefficient=clip)
