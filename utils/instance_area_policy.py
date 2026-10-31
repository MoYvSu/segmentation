# -*- coding: utf-8 -*-
"""解析种子核心面积与最终实例面积，保留旧配置的双阶段缺省。

面积均使用各自处理数组中的像素数，不按输入文件尺寸猜测物理倍率。
neighbor_merge_v1 的最终清理由入口在原图语义投票后执行；底层分水岭
只保留区域，不提前将小区丢为背景。
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral


AREA_POLICY_KEYS = (
    'seed_min_area', 'final_min_instance_area', 'final_small_instance_action',
)
FINAL_SMALL_INSTANCE_ACTIONS = ('legacy_drop', 'neighbor_merge_v1')


def has_explicit_area_policy(inference: Mapping) -> bool:
    """None 与省略等价；旧调用不被额外添加新关键字或回执字段。"""
    return any(inference.get(key) is not None for key in AREA_POLICY_KEYS)


def _explicit_area(inference: Mapping, name: str, fallback: int, minimum: int) -> int:
    value = inference.get(name)
    if value is None:
        return fallback
    # 不把 true、50.5 或 '50' 偷偷转换成新的面积参数。
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}')
    return int(value)


def resolve_instance_area_policy(inference: Mapping | None = None) -> dict:
    """各新字段独立优先，否则回落到 min_instance_area（默认50）。

    显式 seed_min_area >=1，final_min_instance_area >=0；最终门槛0表示
    禁用面积过滤。旧别名继续接受既有的 int 转换以兼容历史配置。
    pre_vote_final_min_area 是底层WS应使用的丢弃门槛；合并模式为0，
    真正最终门槛仍由 final_min_instance_area 声明，交外层延后处理。
    """
    if inference is None:
        inference = {}
    if not isinstance(inference, Mapping):
        raise ValueError('Instance area policy requires an inference mapping')
    try:
        legacy = int(inference.get('min_instance_area', 50))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('min_instance_area must be convertible to an integer') from error
    if legacy < 0:
        raise ValueError('min_instance_area must be >= 0')
    seed = _explicit_area(inference, 'seed_min_area', legacy, 1)
    final = _explicit_area(inference, 'final_min_instance_area', legacy, 0)
    action = inference.get('final_small_instance_action')
    if action is None:
        action = 'legacy_drop'
    if not isinstance(action, str) or action not in FINAL_SMALL_INSTANCE_ACTIONS:
        raise ValueError(
            f'final_small_instance_action must be one of {FINAL_SMALL_INSTANCE_ACTIONS}, '
            f'got {action!r}')
    return dict(
        seed_min_area=seed,
        final_min_instance_area=final,
        final_small_instance_action=action,
        pre_vote_final_min_area=final if action == 'legacy_drop' else 0,
    )


def instance_area_kwargs(inference: Mapping) -> dict:
    """向底层后处理传递新字段；无显式新字段时保持旧调用签名。"""
    policy = resolve_instance_area_policy(inference)
    if not has_explicit_area_policy(inference):
        return {}
    return {key: policy[key] for key in AREA_POLICY_KEYS}
