"""Where VoxShift keeps things, in a source checkout and in the frozen app.

Program files (read-only, replaced on update) are kept apart from per-user
data in %APPDATA% (settings, your own voices, logs).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "VoxShift"
FROZEN = bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    """Folder holding the executable (frozen) or the repository root (source)."""
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return install_dir() / "models"


def data_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    d = Path(base) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_models_dir() -> Path:
    """Downloaded AI voice packs (%LOCALAPPDATA%/VoxShift/models); survives app updates."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    d = Path(base) / APP_NAME / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_voices_dir() -> Path:
    d = data_dir() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def references_dir() -> Path:
    """Reference recordings behind the user's own AI voices."""
    d = data_dir() / "references"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
