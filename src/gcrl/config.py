
# src/gcrl/config.py
# -*- coding: utf-8 -*-
"""
Lightweight config resolver for gCRL.

Priority for SERGIO path:
1) Explicit function argument `sergio_dir`
2) Environment variable GCRL_SERGIO_DIR
3) Project file `gcrl.toml` in CWD or parent tree
4) User file `~/.gcrl/config.toml`

Both TOML files may contain:
[paths]
sergio_dir = "/path/to/SERGIO"  # absolute or relative to the TOML file location
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import tomllib  # py>=3.11
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def find_file_upwards(start: Path, name: str) -> Optional[Path]:
    cur = start.resolve()
    while True:
        cand = cur / name
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _resolve_relative_to(file_path: Path, value: str) -> str:
    """Resolve `value` relative to the directory of `file_path` if it's not absolute."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    return str((file_path.parent / p).resolve())


def resolve_sergio_dir(sergio_dir: Optional[str] = None) -> Optional[str]:
    # 1) explicit
    if sergio_dir:
        return str(Path(sergio_dir).expanduser().resolve())
    # 2) env var
    env = os.environ.get("GCRL_SERGIO_DIR", "").strip()
    if env:
        return str(Path(env).expanduser().resolve())

    # 3) gcrl.toml in project tree
    proj = find_file_upwards(Path.cwd(), "gcrl.toml")
    print(proj)
    if proj:
        cfg = _load_toml(proj)
        val = cfg.get("paths", {}).get("sergio_dir")
        if val:
            return _resolve_relative_to(proj, val)

    # 4) ~/.gcrl/config.toml
    user_cfg = Path.home() / ".gcrl" / "config.toml"
    if user_cfg.exists():
        cfg = _load_toml(user_cfg)
        val = cfg.get("paths", {}).get("sergio_dir")
        if val:
            return _resolve_relative_to(user_cfg, val)

    return None
