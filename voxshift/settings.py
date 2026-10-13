"""Persistent user settings (%APPDATA%\\VoxShift\\settings.json)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

from .paths import data_dir


@dataclass
class Settings:
    input_device: str = ""
    output_device: str = "virtual"
    monitor_device: str = ""
    voice: str = "clean"
    voice_enabled: bool = True
    hear_myself: bool = False
    mic_muted: bool = False
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    monitor_gain_db: float = -6.0
    diffusion_steps: int = 6
    soundboard_dir: str = ""
    start_minimized: bool = False
    close_to_tray: bool = True
    first_run_done: bool = False
    hotkeys: dict = field(default_factory=lambda: {
        "toggle_voice": "ctrl+alt+f9",
        "toggle_hear_myself": "ctrl+alt+f10",
        "toggle_mute": "ctrl+alt+f11",
        "stop_sounds": "ctrl+alt+f12",
    })

    @classmethod
    def path(cls):
        return data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in raw.items() if k in known})
        s.hotkeys = {**cls().hotkeys, **(s.hotkeys or {})}
        return s

    def save(self) -> None:
        tmp = self.path().with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(self.path())
