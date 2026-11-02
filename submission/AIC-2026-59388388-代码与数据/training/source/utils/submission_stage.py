# -*- coding: utf-8 -*-
"""提交数据阶段的轻量契约；只依赖标准库，不加载模型或赛题文件。"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DatasetContract:
    """固定编号全集；单图/子集推理仍使用全集内的零起点索引。"""

    name: str
    first: int
    last: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f'test_{index:03d}' for index in range(self.first, self.last + 1))

    @property
    def count(self) -> int:
        return self.last - self.first + 1

    def source_index(self, stem: str) -> int:
        # 禁止路径、扩展名、Unicode 数字、变长编号及跨阶段混入。
        if not isinstance(stem, str) or re.fullmatch(r'test_[0-9]{3}', stem) is None:
            raise ValueError('Expected a test_XXX stem with exactly three ASCII digits')
        index = int(stem[5:])
        if not self.first <= index <= self.last:
            raise ValueError(f'{self.name} requires test_{self.first:03d} through test_{self.last:03d}')
        return index - self.first


_CONTRACTS = {
    'quarter_final_100': DatasetContract('quarter_final_100', 69, 168),
    'semifinal_150': DatasetContract('semifinal_150', 169, 318),
}


def dataset_contract(name: str | None = None) -> DatasetContract:
    """省略阶段时保留历史 100 图契约；不按文件数量猜测或自动换阶段。"""
    if name is None:
        name = 'quarter_final_100'
    if not isinstance(name, str) or name not in _CONTRACTS:
        raise ValueError('Unknown submission dataset: ' + repr(name))
    return _CONTRACTS[name]
