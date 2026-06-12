# src/gcrl/data/__init__.py
# -*- coding: utf-8 -*-
"""
Data utilities for gCRL package.

This module provides utilities for accessing shared reference data
(ontologies, gene sets, etc.) included with the package.
"""

from __future__ import annotations
from pathlib import Path

# Get the package root directory (gCRL/)
PACKAGE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Define reference data directories
DATA_DIR = PACKAGE_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
ONTOLOGIES_DIR = REFERENCE_DIR / "ontologies"

# Common reference file paths
GO_BASIC_OBO = ONTOLOGIES_DIR / "go-basic.obo"


def get_go_obo_path() -> Path:
    """
    Get the path to the GO basic OBO file.

    Returns
    -------
    Path
        Path to go-basic.obo file

    Raises
    ------
    FileNotFoundError
        If the GO OBO file is not found at the expected location
    """
    if not GO_BASIC_OBO.exists():
        raise FileNotFoundError(
            f"GO OBO file not found at {GO_BASIC_OBO}. "
            "Please ensure the file is present in gCRL/data/reference/ontologies/"
        )
    return GO_BASIC_OBO


def get_reference_data_dir() -> Path:
    """
    Get the path to the reference data directory.

    Returns
    -------
    Path
        Path to reference data directory (gCRL/data/reference/)
    """
    return REFERENCE_DIR


__all__ = [
    "DATA_DIR",
    "REFERENCE_DIR",
    "ONTOLOGIES_DIR",
    "GO_BASIC_OBO",
    "get_go_obo_path",
    "get_reference_data_dir",
]
