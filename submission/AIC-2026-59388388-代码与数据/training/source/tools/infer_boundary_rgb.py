# -*- coding: utf-8 -*-
"""本轮NAF G严格导出/重载；只包含结构、输入合同和状态检查。"""
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import torch
from models.boundary_rgb import BoundaryRGBNAFNet
from models.boundary_restore import state_sha256

FORMAT = 'boundary_rgb_naf_inference_v1'
CHECKPOINT_FORMAT = 'boundary_rgb_training_v1'
INPUT_CONTRACT = dict(input_mode='boundary', input_threshold=.5, tile_size=256,
    overlap=64, reflect_pad=32, batch_size=16, iterations=1, revert_unconnected=False,
    boundary_representation='binary', output_stitch='probability',
    input_normalization=dict(mode='skeleton_dilate', kernel_size=8))

def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def _exact_contract(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(_exact_contract(value[k], v) for k, v in expected.items())
    return value == expected

def validate_input_contract(contract):
    if not _exact_contract(contract, INPUT_CONTRACT):
        raise ValueError('Unknown width-normalized NAF input contract')
    return deepcopy(contract)

def load_naf_checkpoint(path, device='cuda', *, expected_model_sha=None):
    """只严格加载G；不恢复优化器、训练RNG或判别器。"""
    path = Path(path).resolve()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('format') != CHECKPOINT_FORMAT or type(saved.get('global_step')) is not int
            or saved['global_step'] < 1):
        raise ValueError('Expected formal boundary_rgb_training_v1 checkpoint')
    cfg, architecture = saved.get('config'), saved.get('architecture')
    if (not isinstance(cfg, dict) or not isinstance(cfg.get('model'), dict)
            or cfg.get('input_mode') != 'boundary'
            or architecture != dict(cfg['model'], seed=cfg.get('seed'), input_mode='boundary')):
        raise ValueError('NAF checkpoint architecture/config mismatch')
    contract = validate_input_contract(saved.get('input_contract'))
    if (contract != cfg.get('inference')
            or cfg.get('input_normalization') != INPUT_CONTRACT['input_normalization']):
        raise ValueError('NAF checkpoint input_contract/config mismatch')
    numerical = saved.get('identity', {}).get('numerical_contract')
    digest = saved.get('model_sha256')
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise ValueError('NAF checkpoint model SHA256 missing')
    if expected_model_sha is not None and digest != expected_model_sha:
        raise ValueError('NAF model SHA256 is not the selected candidate')
    model = BoundaryRGBNAFNet(**architecture)
    if model.architecture != architecture:
        raise ValueError('NAF architecture changed during construction')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != digest:
        raise RuntimeError('NAF loaded state SHA256 differs')
    model = model.to(device).eval().requires_grad_(False)
    provenance = dict(format=FORMAT, checkpoint_format=CHECKPOINT_FORMAT,
        checkpoint=str(path), checkpoint_sha256=file_sha(path), checkpoint_bytes=path.stat().st_size,
        global_step=saved['global_step'], architecture=deepcopy(architecture), model_sha256=digest,
        parameter_count=sum(p.numel() for p in model.parameters()), input_contract=contract,
        numerical_contract=deepcopy(numerical), no_optimizer_or_rng_restored=True)
    return model, provenance
