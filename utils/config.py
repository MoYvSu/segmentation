# -*- coding: utf-8 -*-
"""Configuration loading helpers.

Small experiment configs may inherit the repository default config through
the ``_base`` key.  The merge is recursive so a new experiment can override
only the decoder, loss, and inference settings it actually changes.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML and recursively merge an optional relative ``_base`` file."""
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base_name = config.pop("_base", None)
    if not base_name:
        return config

    base_path = base_name
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(config_path), base_path)
    return _deep_merge(load_config(base_path), config)
