# -*- coding: utf-8 -*-
"""语义专用LoRA：共享冻结trunk，只替换本次语义前向的低秩权重。"""
from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from models.lora import LoRALinear


class SemanticLoRA(nn.Module):
    VERSION = 'independent_lora_v1'

    def __init__(self, trunk, *, gradient_checkpointing=True, parameter_names=None):
        super().__init__()
        names = []
        for name, module in trunk.named_modules():
            if isinstance(module, LoRALinear):
                prefix = name + '.' if name else ''
                names.extend([prefix + 'lora_A', prefix + 'lora_B'])
        if not names or parameter_names is not None and names != list(parameter_names):
            raise ValueError('Semantic LoRA parameter names do not match the shared trunk')
        self.parameter_names = tuple(names)
        original = dict(trunk.named_parameters())
        self.weights = nn.ParameterList([nn.Parameter(original[name].detach().clone()) for name in names])
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def architecture(self):
        return {'version': self.VERSION, 'parameter_names': list(self.parameter_names),
                'gradient_checkpointing': self.gradient_checkpointing}

    @classmethod
    def from_architecture(cls, trunk, config):
        if config.get('version') != cls.VERSION:
            raise ValueError('Unsupported semantic LoRA architecture version')
        return cls(trunk, gradient_checkpointing=config['gradient_checkpointing'],
                   parameter_names=config['parameter_names'])

    def forward(self, trunk, image):
        # 不把trunk注册到本模块：底座与affinity LoRA在state_dict中只保存一次。
        trunk.eval()
        replacements = dict(zip(self.parameter_names, self.weights))

        def run(value):
            # 重计算也必须在替换作用域内，否则反传会误用冻结的affinity LoRA。
            return functional_call(trunk, replacements, (value,), strict=False)

        if self.gradient_checkpointing and torch.is_grad_enabled() and any(p.requires_grad for p in self.weights):
            return checkpoint(run, image, use_reentrant=False)
        return run(image)


def semantic_features(model, image):
    """旧模型保持原路径；新模型的训练、诊断和推理共用此入口。"""
    adapter = getattr(model, 'semantic_lora', None)
    return model.encoder(image) if adapter is None else adapter(model.encoder.trunk, image)
