"""Windows integration: single instance, global hotkeys, start with Windows.

Pure ctypes / winreg, no extra dependencies. Every function degrades to a
no-op on other platforms.
"""
from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

IS_WIN = sys.platform == "win32"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"

_MODS = {"alt": 0x0001, "ctrl": 0x0002, "control": 0x0002, "shift": 0x0004, "win": 0x0008}
_NAMED_VK = {**{f"f{i}": 0x6F + i for i in range(1, 25)}, "space": 0x20, "tab": 0x09, "enter": 0x0D,
             "esc": 0x1B, "home": 0x24, "end": 0x23, "insert": 0x2D, "delete": 0x2E, "pause": 0x13,
             **{f"num{i}": 0x60 + i for i in range(10)}}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+v' -> (modifiers, virtual-key)."""
    mods, vk = 0, None
    for part in spec.lower().replace(" ", "").split("+"):
        if part in _MODS:
            mods |= _MODS[part]
        elif part in _NAMED_VK:
            vk = _NAMED_VK[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        else:
            raise ValueError(f"unknown key {part!r} in hotkey {spec!r}")
    if vk is None:
        raise ValueError(f"hotkey {spec!r} has no key")
    return mods, vk


class SingleInstance:
    """Named mutex; `already_running` is True if another VoxShift holds it."""

    def __init__(self, name: str = "Local\\VoxShift.SingleInstance"):
        self.handle = None
        self.already_running = False
        if IS_WIN:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self.handle = k32.CreateMutexW(None, False, name)
            self.already_running = ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


class GlobalHotkeys:
    """RegisterHotKey on a dedicated thread with its own message loop.

    Callbacks run on that thread; GUI code should marshal them back (e.g.
    ``root.after``). Keys another program already owns are reported in
    ``failed`` instead of raising.
    """

    def __init__(self, bindings: dict[str, tuple[str, callable]]):
        self.bindings = bindings  # id-name -> (spec, callback)
        self.failed: dict[str, str] = {}
        self._thread_id = None
        self._ready = threading.Event()
        if IS_WIN and bindings:
            threading.Thread(target=self._loop, name="hotkeys", daemon=True).start()
            self._ready.wait(2)

    def _loop(self):
        user32, k32 = ctypes.WinDLL("user32", use_last_error=True), ctypes.WinDLL("kernel32")
        self._thread_id = k32.GetCurrentThreadId()
        actions = {}
        for i, (name, (spec, cb)) in enumerate(self.bindings.items(), start=1):
            try:
                mods, vk = parse_hotkey(spec)
            except ValueError as e:
                self.failed[name] = str(e)
                continue
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                actions[i] = cb
            else:
                self.failed[name] = f"{spec} is already used by another program"
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in actions:
                try:
                    actions[msg.wParam]()
                except Exception:  # a failing action must not kill the hotkey thread
                    import logging
                    logging.getLogger(__name__).exception("hotkey action failed")
        for i in actions:
            user32.UnregisterHotKey(None, i)

    def stop(self):
        if IS_WIN and self._thread_id:
            ctypes.WinDLL("user32").PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)


def autostart_command() -> str:
    from .paths import FROZEN, install_dir
    if FROZEN:
        return f'"{install_dir() / "VoxShift.exe"}" --minimized'
    return f'"{sys.executable.replace("python.exe", "pythonw.exe")}" -m voxshift gui --minimized'


def get_autostart(name: str = "VoxShift") -> bool:
    if not IS_WIN:
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, name)
            return True
    except OSError:
        return False


def set_autostart(enabled: bool, name: str = "VoxShift") -> None:
    if not IS_WIN:
        return
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if enabled:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(k, name)
            except FileNotFoundError:
                pass


def set_app_id(app_id: str = "VoxShift.App") -> None:
    """Own taskbar identity/icon instead of python.exe's."""
    if IS_WIN:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


class _SHELLEXECUTEINFOW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("fMask", ctypes.c_ulong), ("hwnd", wintypes.HWND),
                ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
                ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
                ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
                ("dwHotKey", wintypes.DWORD), ("hIconOrMonitor", wintypes.HANDLE), ("hProcess", wintypes.HANDLE)]


def run_elevated(path: str, params: str = "", hwnd: int | None = None) -> int:
    """Run ``path`` as administrator and wait; returns its exit code.

    Requesting elevation from our own (foreground) process, parented to our
    window, makes Windows show the UAC prompt on top. A hidden helper process
    only gets a flashing taskbar button, and the prompt times out unseen.
    Raises PermissionError when the user declines.
    """
    if not IS_WIN:
        raise OSError("elevation is Windows-only")
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    info = _SHELLEXECUTEINFOW(cbSize=ctypes.sizeof(_SHELLEXECUTEINFOW), fMask=0x00000040,  # NOCLOSEPROCESS
                              hwnd=hwnd, lpVerb="runas", lpFile=str(path), lpParameters=params or None,
                              lpDirectory=str(__import__("pathlib").Path(path).parent), nShow=1)
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        err = ctypes.get_last_error()
        if err == 1223:  # ERROR_CANCELLED
            raise PermissionError("administrator permission was declined")
        raise ctypes.WinError(err)
    try:
        k32.WaitForSingleObject(info.hProcess, 0xFFFFFFFF)
        code = wintypes.DWORD()
        k32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
        return code.value
    finally:
        k32.CloseHandle(info.hProcess)


def enable_dpi_awareness() -> None:
    """Crisp rendering on scaled displays (call before creating the Tk root)."""
    if IS_WIN:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


def message_box(text: str, title: str = "VoxShift") -> None:
    if IS_WIN:
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x40)
