# -*- coding: utf-8 -*-
"""当前训练链统一入口；提供重训配方，不承诺精确重放历史最优权重。"""
from __future__ import annotations

import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '-h' in arguments or '--help' in arguments:
        print(__doc__, flush=True)
    # 训练/只读检查均交给原入口；不改变配置、预算、资产校验或默认run目录。
    from repro.pipeline import main as pipeline_main
    return pipeline_main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
