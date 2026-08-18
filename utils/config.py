# -*- coding: utf-8 -*-
"""Configuration loading helpers.

Small experiment configs may inherit the repository default config through
the ``_base`` key.  The merge is recursive so a new experiment can override
only the decoder, loss, and inference settings it actually changes.

``paths.project_root: auto`` makes a config portable between the local
Windows checkout and the Linux training server.  An explicit absolute path is
still accepted for archived run snapshots.  ``SEGMENTATION_PROJECT_ROOT`` can
be used as a temporary command-line override without editing YAML files.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
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


def _repository_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def _resolve_project_root(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.setdefault("paths", {})
    env_root = os.environ.get("SEGMENTATION_PROJECT_ROOT", "").strip()
    configured = str(paths.get("project_root", "auto")).strip()
    if env_root:
        root = env_root
    elif configured.lower() in {"", "auto", "repo"}:
        root = _repository_root()
    else:
        root = os.path.expandvars(os.path.expanduser(configured))
    paths["project_root"] = os.path.abspath(root)
    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML, recursively merge ``_base``, and resolve project root."""
    config_path = os.path.abspath(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base_name = config.pop("_base", None)
    if not base_name:
        return _resolve_project_root(config)

    base_path = base_name
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(config_path), base_path)
    merged = _deep_merge(load_config(base_path), config)
    return _resolve_project_root(merged)


def project_path(config: Dict[str, Any], *parts: str) -> str:
    """Resolve a path inside the configured repository.

    Absolute path arguments are returned unchanged.  This keeps archived
    configs usable while making current experiment configs host-independent.
    """
    if not parts:
        return config["paths"]["project_root"]
    candidate = os.path.join(*[str(part) for part in parts])
    if os.path.isabs(candidate):
        return candidate
    return os.path.join(config["paths"]["project_root"], candidate)
