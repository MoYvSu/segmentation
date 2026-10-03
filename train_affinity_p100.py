# -*- coding: utf-8 -*-
"""P100独立入口：复用封存来源实验逻辑，只将SAM2相对系数改为1。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType

from train_affinity_balance import comparable
from utils.config import load_config


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = 'config/train/affinity_source_p100.yaml'
SOURCE_FORMAT = 'affinity_source_p100_v1'
REFERENCE = 'outputs/affinity_balance/mixed'
REFERENCE_SHA256 = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
NOMINAL_COEFFICIENT_SUM = 1.5

# 不导入后修改train_affinity_source的全局变量：旧P025分析/恢复仍使用原命名空间。
# 私有模块的函数共享自己的globals，仅在独立训练进程内暂时包装native loss。
_source_path = ROOT / 'train_affinity_source.py'
_source_bytes = _source_path.read_bytes()
SOURCE_IMPLEMENTATION_SHA256 = hashlib.sha256(_source_bytes).hexdigest()
_core = ModuleType(__name__ + '._source_core')
_core.__file__ = str(_source_path)
_core.__package__ = ''
exec(compile(_source_bytes, str(_source_path), 'exec'), _core.__dict__)


def _number(value, expected):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value == expected)


def source_coefficients(config):
    """总名义系数仍为1.5；人工/SAM2实际反传系数均为0.75。"""
    if not isinstance(config, dict) or not isinstance(config.get('backend_adaptation'), dict):
        raise ValueError('P100 requires a complete backend configuration')
    beta = config['backend_adaptation'].get('pseudo_weight')
    if not _number(beta, 1.0):
        raise ValueError('P100 only permits numeric beta=1; reuse the existing complete control')
    alpha = NOMINAL_COEFFICIENT_SUM / (1.0 + beta)
    return dict(beta=float(beta), alpha=alpha, manual_coefficient=alpha,
                pseudo_coefficient=alpha * beta, nominal_coefficient_sum=NOMINAL_COEFFICIENT_SUM)


def reference_config(config):
    """去掉来源设置后必须还原现有mixed balance，拒绝任何隐含第二变量。"""
    source_coefficients(config)
    result = deepcopy(config)
    options = result.pop('affinity_source', None)
    if (not isinstance(options, dict) or set(options) != {'reference', 'reference_sha256'}
            or options['reference'] != REFERENCE or options['reference_sha256'] != REFERENCE_SHA256):
        raise ValueError('P100 must reuse the registered mixed balance control without extra options')
    try:
        loss = result['direct_semantic_affinity']['affinity_loss']
        cfg = result['backend_adaptation']
        if (not _number(loss['negative_weight'], 1.0)
                or not _number(loss['hard_negative_weight'], 1.0)
                or not _number(loss['hard_negative_gamma'], 2.0)
                or type(cfg['epochs']) is not int or cfg['epochs'] != 20
                or type(cfg['manual_repeats']) is not int or cfg['manual_repeats'] != 2
                or result['affinity_native']['penalty_sampling'] != 'mixed'):
            raise ValueError('P100 retains r1/h1/gamma2, mixed sampling and 20 x 64 updates')
        cfg['pseudo_weight'] = .5
        expected = load_config(str(ROOT / 'config/train/affinity_balance_mixed.yaml'))
        # 合成CPU回执允许临时仓库根；其余输入/冻结依赖不获得豁免。
        expected['paths']['project_root'] = result['paths']['project_root']
    except (KeyError, TypeError) as error:
        raise ValueError('P100 requires the complete registered configuration') from error
    if (json.dumps(comparable(result), sort_keys=True)
            != json.dumps(comparable(expected), sort_keys=True)):
        raise ValueError('P100 configuration changed beyond output directory and authorized source ratio')
    return result


def verify_config(config):
    reference_config(config)
    return dict(format=SOURCE_FORMAT, **source_coefficients(config),
                reference_beta=.5, reference_sha256=REFERENCE_SHA256, control_retrained=False)


# 只替换私有命名空间中的系数/配置核验，保留原梯度hook与逐步/冻结/保存/monitor核验。
_core.source_coefficients = source_coefficients
_core.reference_config = reference_config
_core.SOURCE_FORMAT = SOURCE_FORMAT
verify_reference = _core.verify_reference
verify_run = _core.verify_run
source_loss_scaling = _core.source_loss_scaling
checkpoint_extras = _core.checkpoint_extras
train = _core.train
native = _core.native
SIDECAR = _core.SIDECAR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('train', 'verify'), default='train')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    configuration = load_config(args.config)
    verify_config(configuration)
    if args.mode == 'verify':
        print(json.dumps(verify_run(configuration, args.output, 8 if args.smoke else 1280), ensure_ascii=False))
    else:
        train(configuration, args.output, smoke=args.smoke)
