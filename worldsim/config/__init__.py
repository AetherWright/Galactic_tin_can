"""Configuration file loaders for WorldSim.

This package provides helpers to lazily load configuration data stored as
plain text or JSON files. Each file is loaded only once and cached so that
multiple modules can share the same lists without duplicating memory usage.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
from typing import List, Dict, Any

_CONFIG_DIR = Path(__file__).resolve().parent

@lru_cache()
def load_list(name: str) -> List[str]:
    """Return a list of strings loaded from ``name``.txt inside the config folder."""
    path = _CONFIG_DIR / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]

@lru_cache()
def load_json(name: str) -> Dict[str, Any]:
    """Return a JSON object loaded from ``name``.json inside the config folder."""
    path = _CONFIG_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

__all__ = ["load_list", "load_json"]
