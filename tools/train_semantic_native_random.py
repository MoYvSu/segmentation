# -*- coding: utf-8 -*-
"""完整随机语义头的单组native1024零噪训练；复用冻结训练器的更新与保存流程。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.fpn_decoder import FPNDecoder
from tools import train_semantic_native_zero as base

FORMAT = 'semantic_native_random_delta_v1'
INITIALIZATION = 'random_full_v1'
INITIALIZATION_SEED = 1101
BASE_TRAINER_SHA = 'da9f9ceb681bf6688d29eeea24ab79cce1b6074382582cc39528c3f78b677a43'
FIXED = {**deepcopy(base.FIXED), 'learning_rate': 1e-4}


def learning_rate(epoch):
    if not 1 <= epoch <= 20:
        raise ValueError('epoch must be 1..20')
    factor = epoch / 2 if epoch <= 2 else .1 + .9 * (1 + math.cos(math.pi * (epoch-3) / 17)) / 2
    return 1e-4 * factor


def _rng_digest(state):
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def reset_semantic(model, *, seed=INITIALIZATION_SEED, expected_parameters=base.HEAD_PARAMETERS):
    """只重置三块语义参数；fork_rng让初始化不推进之后的dropout随机流。"""
    if seed != INITIALIZATION_SEED:
        raise ValueError('Only the single authorized initialization seed 1101 is supported')
    head = model.semantic_decoder
    names = ('seg_fpn', 'seg_branch', 'semantic_residual')
    if tuple(dict(head.named_children())) != names or getattr(head, 'semantic_residual_version', None) != 'highres_v1':
        raise ValueError('Expected exactly the existing full FPN/classifier/highres semantic head')
    if sum(p.numel() for p in head.parameters()) != expected_parameters:
        raise ValueError('The complete semantic parameter count differs')
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in head.modules()):
        raise ValueError('This architecture has GroupNorm; BatchNorm is not authorized')
    owned = {id(p) for module in head.modules() if isinstance(module, (nn.Conv2d, nn.GroupNorm))
             for p in module.parameters(recurse=False)}
    if owned != {id(p) for p in head.parameters()}:
        raise ValueError('A semantic parameter has no explicit Conv/GroupNorm reset rule')
    before = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    frozen_before = base.frozen_sha(model)
    state = torch.get_rng_state().clone()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    cuda_states = torch.cuda.get_rng_state_all() if devices else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        FPNDecoder._reset_modules([getattr(head, name) for name in names])
        head.semantic_residual.reset_output()
    after_state = torch.get_rng_state()
    after_cuda = torch.cuda.get_rng_state_all() if devices else []
    if not torch.equal(state, after_state) or any(not torch.equal(a,b) for a,b in zip(cuda_states, after_cuda)):
        raise RuntimeError('Semantic initialization changed the subsequent training RNG')
    if base.frozen_sha(model) != frozen_before:
        raise RuntimeError('Semantic reset changed a frozen parameter/buffer')
    if (torch.count_nonzero(head.semantic_residual.out.weight) != 0
            or torch.count_nonzero(head.semantic_residual.out.bias) != 0):
        raise RuntimeError('Highres residual ending must start at exact zero')
    for module in head.modules():
        if isinstance(module, nn.GroupNorm):
            if not torch.equal(module.weight, torch.ones_like(module.weight)) or torch.count_nonzero(module.bias) != 0:
                raise RuntimeError('GroupNorm initialization must be affine 1/0')
    after = head.state_dict()
    original_sha, initial_sha = base.tensor_sha(before.items()), base.tensor_sha(after.items())
    if original_sha == initial_sha:
        raise RuntimeError('The full semantic head was not reinitialized')
    receipt = dict(initialization=INITIALIZATION, initialization_seed=seed,
        parameter_count=expected_parameters, base_original_semantic_sha256=original_sha,
        random_initial_semantic_sha256=initial_sha,
        nonsemantic_before_sha256=frozen_before, nonsemantic_after_sha256=base.frozen_sha(model),
        torch_rng_preserved=True, cpu_rng_before_sha256=_rng_digest(state),
        cpu_rng_after_sha256=_rng_digest(after_state),
        cuda_rng_before_sha256=[_rng_digest(value) for value in cuda_states],
        cuda_rng_after_sha256=[_rng_digest(value) for value in after_cuda],
        conv_initialization='Kaiming normal fan_out ReLU; bias zero',
        norm_initialization='GroupNorm affine weight1/bias0; no BatchNorm',
        residual_ending='exact zero weight and bias after full residual reinitialization',
        submodules={name: dict(parameters=sum(p.numel() for p in getattr(head,name).parameters()),
            before_sha256=base.tensor_sha((key,value) for key,value in before.items() if key.startswith(name+'.')),
            after_sha256=base.tensor_sha((key,value) for key,value in after.items() if key.startswith(name+'.')))
            for name in names},
        parameters=[dict(name=name, numel=p.numel(), reset_applied=True,
            value_changed=not torch.equal(before[name], p.detach().cpu()),
            initial_sha256=base.tensor_sha([(name,p)])) for name,p in head.named_parameters()],
        previous_trained_delta_loaded=False)
    return receipt


def normalize_identity(value):
    """在旧engine已有的JSON正规化接点补新格式身份；重复调用结果不变。"""
    if not isinstance(value, dict) or value.get('format') != FORMAT or 'recipe' not in value:
        return value
    receipt = value['frozen']['init_receipt']
    original, initial = receipt['base_original_semantic_sha256'], receipt['random_initial_semantic_sha256']
    if value.get('base_semantic_sha256') not in (original, initial):
        raise ValueError('Random initialization and immutable base semantic identity disagree')
    value.update(base_semantic_sha256=original, base_original_semantic_sha256=original,
                 random_initial_semantic_sha256=initial, init_receipt=deepcopy(receipt),
                 semantic_initialization=INITIALIZATION, initialization_seed=INITIALIZATION_SEED)
    return value


class IdentityJSON:
    """只属于隔离engine的JSON对象；不替换Python全局json或任何旧模块。"""
    dumps = staticmethod(json.dumps)

    @staticmethod
    def loads(value, *args, **kwargs):
        return normalize_identity(json.loads(value, *args, **kwargs))


def make_engine():
    """独立载入已封存实现；复用循环、零噪、monitor、checkpoint及游标检查。"""
    if base.file_sha(base.__file__) != BASE_TRAINER_SHA:
        raise RuntimeError('The frozen native-zero trainer source changed; review the adapter before reuse')
    spec = importlib.util.spec_from_file_location('_semantic_native_random_engine', base.__file__)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.FORMAT, engine.FIXED, engine.learning_rate = FORMAT, deepcopy(FIXED), learning_rate
    engine.json = IdentityJSON
    original_validate = engine.validate_config
    original_load = engine.backend_adaptation.load_backend
    original_freeze = engine.freeze_contract
    original_sources = engine.source_seal
    original_delta = engine.load_head_delta
    context = {}

    def validate(config):
        cfg = config['semantic_native_random']
        if cfg != dict(initialization=INITIALIZATION, initialization_seed=INITIALIZATION_SEED):
            raise ValueError('Only one random_full_v1/seed1101 initialization recipe is authorized')
        result = original_validate(config)
        if result['seed'] != 20261031:
            raise ValueError('Dataset/dropout training seed must remain 20261031')
        return {**result, **cfg}

    def load_backend(path, config, device):
        if context:
            raise RuntimeError('Expected exactly one immutable base load per random training invocation')
        if base.file_sha(path) != base.BACKEND_SHA:
            raise ValueError('Random head starts only from immutable 13b3 backend, never the previous delta')
        model, bundle = original_load(path, config, device)
        context['init_receipt'] = reset_semantic(model)
        return model, bundle

    def freeze(model, restorer, **kwargs):
        summary = original_freeze(model, restorer, **kwargs)
        receipt = context['init_receipt']
        if summary['frozen_backend_sha256'] != receipt['nonsemantic_before_sha256']:
            raise RuntimeError('Frozen geometry identity changed during setup')
        summary.update(init_receipt=deepcopy(receipt),
            base_original_semantic_sha256=receipt['base_original_semantic_sha256'],
            random_initial_semantic_sha256=receipt['random_initial_semantic_sha256'])
        return summary

    def sources():
        value = original_sources()
        value['tools/train_semantic_native_random.py'] = base.file_sha(__file__)
        return dict(sorted(value.items()))

    def delta(model, path, *, expected_backend_sha256=base.BACKEND_SHA):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != FORMAT:
            raise ValueError('Random trainer accepts only explicit semantic_native_random_delta_v1')
        identity = checkpoint['identity']
        if normalize_identity(deepcopy(identity)) != identity:
            raise ValueError('Random checkpoint lacks the explicit original/random initialization identity')
        receipt = original_delta(model, path, expected_backend_sha256=expected_backend_sha256)
        receipt.update(base_original_semantic_sha256=identity['base_original_semantic_sha256'],
            random_initial_semantic_sha256=identity['random_initial_semantic_sha256'],
            initialization_seed=INITIALIZATION_SEED, initialization=INITIALIZATION,
            init_receipt=deepcopy(identity['init_receipt']))
        return receipt

    engine.validate_config, engine.freeze_contract, engine.source_seal, engine.load_head_delta = validate, freeze, sources, delta
    # 替换隔离engine的依赖引用，不修改共享backend_adaptation模块的load_backend。
    engine.backend_adaptation = SimpleNamespace(load_backend=load_backend)
    engine.no_restoration_rng = base.no_restoration_rng
    return engine


def load_head_delta(model, path, *, expected_backend_sha256=base.BACKEND_SHA):
    """新版本部署接口；不让随机初态/训练结果冒充旧warm-start增量格式。"""
    return make_engine().load_head_delta(model, path, expected_backend_sha256=expected_backend_sha256)


def run(args):
    return make_engine().run(args)


if __name__ == '__main__':
    run(base.parse_args())
