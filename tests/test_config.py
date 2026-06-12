
# tests/test_config.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import contextlib

import pytest

from gcrl.config import resolve_sergio_dir


@contextlib.contextmanager
def pushd(path: Path):
    cur = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cur)


def test_resolve_from_project_toml_relative(tmp_path, monkeypatch):
    # Simulate repo layout:
    # tmp/gCRL/
    #   gcrl.toml -> [paths].sergio_dir = "ext_tools/SERGIO"
    #   ext_tools/SERGIO/
    #   sub/ (where we run from)
    repo = tmp_path / "gCRL"
    sergio_rel = "ext_tools/SERGIO"
    sergio_abs = repo / sergio_rel
    subdir = repo / "sub"
    (repo / "ext_tools").mkdir(parents=True)
    sergio_abs.mkdir(parents=True)
    subdir.mkdir(parents=True)

    # Write gcrl.toml at repo root
    (repo / "gcrl.toml").write_text("[paths]\nsergio_dir = \"ext_tools/SERGIO\"\n")

    # Ensure env var not set
    monkeypatch.delenv("GCRL_SERGIO_DIR", raising=False)

    with pushd(subdir):
        resolved = resolve_sergio_dir(None)
        assert Path(resolved) == sergio_abs.resolve()


def test_env_var_overrides_project(tmp_path, monkeypatch):
    repo = tmp_path / "gCRL"
    sergio_rel = "ext_tools/SERGIO"
    sergio_abs = repo / sergio_rel
    alt = tmp_path / "alt_SERGIO"
    (repo / "ext_tools").mkdir(parents=True)
    sergio_abs.mkdir(parents=True)
    alt.mkdir(parents=True)

    (repo / "gcrl.toml").write_text("[paths]\nsergio_dir = \"ext_tools/SERGIO\"\n")
    # Set env var to a different location
    monkeypatch.setenv("GCRL_SERGIO_DIR", str(alt))

    with pushd(repo):
        resolved = resolve_sergio_dir(None)
        assert Path(resolved) == alt.resolve()


def test_explicit_argument_overrides_all(tmp_path, monkeypatch):
    repo = tmp_path / "gCRL"
    sergio_rel = "ext_tools/SERGIO"
    sergio_abs = repo / sergio_rel
    alt = tmp_path / "alt_SERGIO"
    explicit = tmp_path / "explicit_SERGIO"
    (repo / "ext_tools").mkdir(parents=True)
    sergio_abs.mkdir(parents=True)
    alt.mkdir(parents=True)
    explicit.mkdir(parents=True)

    (repo / "gcrl.toml").write_text("[paths]\nsergio_dir = \"ext_tools/SERGIO\"\n")
    monkeypatch.setenv("GCRL_SERGIO_DIR", str(alt))

    with pushd(repo):
        resolved = resolve_sergio_dir(str(explicit))
        assert Path(resolved) == explicit.resolve()
