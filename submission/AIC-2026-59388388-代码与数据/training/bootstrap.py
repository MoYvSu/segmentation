# -*- coding: utf-8 -*-
"""所有底层算法固定从本包training/source导入。"""
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent / "source"


def activate_source():
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    return SOURCE_ROOT
