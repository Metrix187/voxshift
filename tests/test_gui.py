"""Desktop app lifecycle (real Tk window; audio, tray and hotkeys stubbed out)."""
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from voxshift import gui, winsys

    class NoEvent:  # don't touch the named events a running VoxShift may own
        QUIT = SHOW = ""

        def __init__(self, name):
            pass

        def is_set(self):
            return False

    monkeypatch.setattr(winsys, "NamedEvent", NoEvent)
    for name in ("restart_engine", "register_hotkeys", "_start_tray", "_first_run"):
        monkeypatch.setattr(gui.App, name, lambda self: None)
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    return gui.App(root)


def test_tray_quit_during_tick_is_clean(app):
    # the tray's Quit is marshalled through the UI queue and runs inside _tick;
    # _tick must not touch the (now destroyed) window afterwards
    app.call_soon(app.quit)
    app._tick()
    assert app._closing
    with pytest.raises(tk.TclError):
        app.root.winfo_exists()
    app.quit()  # a second request (hotkey, installer) is a harmless no-op
