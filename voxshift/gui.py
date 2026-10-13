"""VoxShift desktop app: Tk + Sun Valley theme, tray icon, global hotkeys.

Starts the audio engine with the saved devices, routes the processed voice
into a virtual microphone, keeps running in the tray, and is controlled with
global hotkeys.
"""
from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__, winsys
from .audio.cable import find_cables, install_vbcable
from .audio.engine import AudioEngine, NoVirtualCable, default_hostapi, list_devices
from .paths import FROZEN, log_dir
from .presets import (NEEDS_DOWNLOAD, build_processor, create_ai_voice, delete_user_voice, is_ai, list_presets,
                      unavailable_reason)
from .settings import Settings

SR = 48000
ASSETS = Path(__file__).resolve().parent / "assets"
log = logging.getLogger("voxshift")


# ---------------------------------------------------------------------- dialogs
class NewAIVoiceDialog(tk.Toplevel):
    """Voicelab-style dialog: name + a reference clip (file or 10 s recording)."""

    def __init__(self, app: "App"):
        super().__init__(app.root)
        self.app, self.reference = app, None
        self.title("New AI voice")
        self.resizable(False, False)
        self.transient(app.root)
        f = ttk.Frame(self, padding=16)
        f.pack(fill="both")
        ttk.Label(f, text="Name").grid(row=0, column=0, sticky="w")
        self.name = ttk.Entry(f, width=36)
        self.name.grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(f, text="Target voice").grid(row=1, column=0, sticky="w")
        ttk.Button(f, text="Choose file…", command=self.pick).grid(row=1, column=1, sticky="ew", pady=4)
        self.rec_btn = ttk.Button(f, text="Record 10 s", command=self.record)
        self.rec_btn.grid(row=1, column=2, sticky="ew", padx=(6, 0), pady=4)
        self.ref_label = ttk.Label(f, text="No recording yet", foreground="#999")
        self.ref_label.grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Label(f, wraplength=360, foreground="#999", justify="left",
                  text="3–10 seconds of one person speaking clearly, without music or noise. "
                       "Only use voices you have permission to use.").grid(row=3, column=0, columnspan=3, pady=10)
        ttk.Button(f, text="Create voice", style="Accent.TButton", command=self.create).grid(
            row=4, column=2, sticky="e")
        self.name.focus_set()

    def pick(self):
        p = filedialog.askopenfilename(parent=self, filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3")])
        if p:
            self.reference = p
            self.ref_label.config(text=Path(p).name)
            if not self.name.get():
                self.name.insert(0, Path(p).stem.replace("_", " ").title())

    def record(self):
        import sounddevice as sd
        import soundfile as sf
        dev = self.app.engine.input_device if self.app.engine else None
        self.rec_btn.state(["disabled"])
        out = Path(tempfile.mkdtemp()) / "recording.wav"

        def run():
            try:
                data = sd.rec(int(10 * SR), samplerate=SR, channels=1, device=dev, dtype="float32", blocking=True)
                sf.write(out, data[:, 0], SR)
                self.app.call_soon(lambda: self._recorded(out))
            except Exception as e:  # shown to the user
                msg = str(e)
                self.app.call_soon(lambda: self.ref_label.config(text=f"Recording failed: {msg}"))
            finally:
                self.app.call_soon(lambda: self.rec_btn.state(["!disabled"]))

        for i in range(10, 0, -1):
            self.after((10 - i) * 1000, lambda i=i: self.ref_label.config(text=f"Recording… speak now ({i}s)"))
        threading.Thread(target=run, daemon=True).start()

    def _recorded(self, path):
        self.reference = str(path)
        self.ref_label.config(text="Recorded 10 s")

    def create(self):
        name = self.name.get().strip()
        if not name or not self.reference:
            messagebox.showwarning("New AI voice", "Give the voice a name and a recording.", parent=self)
            return
        key = create_ai_voice(name, self.reference)
        self.destroy()
        self.app.refresh_voices(select=key)


class SettingsDialog(tk.Toplevel):
    def __init__(self, app: "App"):
        super().__init__(app.root)
        self.app, s = app, app.s
        self.title("Settings")
        self.resizable(False, False)
        self.transient(app.root)
        nb = ttk.Notebook(self, padding=10)
        nb.pack(fill="both", expand=True)

        audio = ttk.Frame(nb, padding=12)
        nb.add(audio, text="Audio")
        api = app.hostapi
        ins = ["(system default)"] + [d["name"] for d in list_devices("input", api)]
        outs = [d["name"] for d in list_devices("output", api)]
        cables = find_cables(api)
        self.inp = self._combo(audio, 0, "Microphone", ins, s.input_device or "(system default)")
        out_values = ["virtual"] + outs
        self.out = self._combo(audio, 1, "Voice output", out_values, s.output_device or "virtual")
        ttk.Label(audio, foreground="#999", text="'virtual' = the virtual microphone other apps listen to"
                  ).grid(row=2, column=1, sticky="w")
        self.mon = self._combo(audio, 3, "Hear-myself device", ["(system default)"] + outs,
                               s.monitor_device or "(system default)")
        self.gains = {}
        for i, (key, label) in enumerate([("input_gain_db", "Mic gain (dB)"), ("output_gain_db", "Voice gain (dB)"),
                                          ("monitor_gain_db", "Hear-myself volume (dB)")]):
            ttk.Label(audio, text=label).grid(row=4 + i, column=0, sticky="w", pady=4)
            var = tk.DoubleVar(value=getattr(s, key))
            ttk.Scale(audio, from_=-30, to=20, variable=var).grid(row=4 + i, column=1, sticky="ew")
            self.gains[key] = var

        vmic = ttk.Frame(nb, padding=12)
        nb.add(vmic, text="Virtual microphone")
        text = ("\n".join(f"• {c.label}: apps select '{c.capture}'" for c in cables)
                or "No virtual microphone is installed, so other apps cannot hear VoxShift.")
        ttk.Label(vmic, text=text, wraplength=420, justify="left").pack(anchor="w")
        ttk.Button(vmic, text="Install VB-CABLE…", command=app.install_cable).pack(anchor="w", pady=10)
        ttk.Button(vmic, text="Test virtual microphone", command=app.test_cable).pack(anchor="w")

        general = ttk.Frame(nb, padding=12)
        nb.add(general, text="General")
        self.autostart = tk.BooleanVar(value=winsys.get_autostart())
        self.minimized = tk.BooleanVar(value=s.start_minimized)
        self.to_tray = tk.BooleanVar(value=s.close_to_tray)
        ttk.Checkbutton(general, text="Start with Windows", variable=self.autostart).pack(anchor="w", pady=2)
        ttk.Checkbutton(general, text="Start minimized to the tray", variable=self.minimized).pack(anchor="w", pady=2)
        ttk.Checkbutton(general, text="Closing the window keeps VoxShift running in the tray",
                        variable=self.to_tray).pack(anchor="w", pady=2)
        ttk.Label(general, text="Global hotkeys", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 2))
        self.hotkeys = {}
        for name, spec in s.hotkeys.items():
            row = ttk.Frame(general)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=name.replace("_", " ").capitalize(), width=22).pack(side="left")
            e = ttk.Entry(row, width=18)
            e.insert(0, spec)
            e.pack(side="left")
            if name in app.hotkey_errors:
                ttk.Label(row, text=app.hotkey_errors[name], foreground="#e66").pack(side="left", padx=6)
            self.hotkeys[name] = e

        bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        bar.pack(fill="x")
        ttk.Button(bar, text="Save", style="Accent.TButton", command=self.save).pack(side="right")
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="right", padx=6)

    @staticmethod
    def _combo(parent, row, label, values, current):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        c = ttk.Combobox(parent, values=values, width=48, state="readonly")
        c.set(current if current in values else values[0])
        c.grid(row=row, column=1, sticky="ew", padx=(8, 0))
        return c

    def save(self):
        s = self.app.s
        default = "(system default)"
        s.input_device = "" if self.inp.get() == default else self.inp.get()
        s.output_device = self.out.get()
        s.monitor_device = "" if self.mon.get() == default else self.mon.get()
        for key, var in self.gains.items():
            setattr(s, key, round(var.get(), 1))
        s.start_minimized, s.close_to_tray = self.minimized.get(), self.to_tray.get()
        s.hotkeys = {k: e.get().strip().lower() for k, e in self.hotkeys.items()}
        s.save()
        winsys.set_autostart(self.autostart.get())
        self.destroy()
        self.app.restart_engine()
        self.app.register_hotkeys()


# ---------------------------------------------------------------------- main window
class App:
    def __init__(self, root: tk.Tk, minimized: bool = False):
        self.root = root
        self.s = Settings.load()
        self.hostapi = default_hostapi()
        self.engine: AudioEngine | None = None
        self.presets: dict[str, dict] = {}
        self.current_key: str | None = None
        self.hotkeys = None
        self.hotkey_errors: dict[str, str] = {}
        self.tray = None
        self._load_token = 0
        self._ui_queue: queue.SimpleQueue = queue.SimpleQueue()
        root.title("VoxShift")
        root.geometry("1000x640")
        root.minsize(860, 560)
        try:
            root.iconbitmap(default=str(ASSETS / "voxshift.ico"))
        except tk.TclError:
            pass
        self._theme()
        self._build()
        self.refresh_voices(select=self.s.voice)
        self.restart_engine()
        self.register_hotkeys()
        self._start_tray()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        # Tk delivers WM_QUERYENDSESSION (Windows logoff / Restart Manager during updates) as
        # WM_SAVE_YOURSELF: really exit then, instead of hiding in the tray.
        root.protocol("WM_SAVE_YOURSELF", self.quit)
        self._quit_evt = winsys.NamedEvent(winsys.NamedEvent.QUIT)
        self._show_evt = winsys.NamedEvent(winsys.NamedEvent.SHOW)
        if minimized or self.s.start_minimized:
            root.withdraw() if self.tray else root.iconify()
        root.after(400, self._first_run)
        self._tick()

    # -------------------------------------------------------------- layout
    def _theme(self):
        try:
            import sv_ttk
            sv_ttk.set_theme("dark")
        except Exception:  # theme is cosmetic
            pass

    def _build(self):
        r = self.root
        head = ttk.Frame(r, padding=(14, 10))
        head.pack(fill="x")
        ttk.Label(head, text="VoxShift", font=("Segoe UI Semibold", 18)).pack(side="left")
        self.voice_var = tk.BooleanVar(value=self.s.voice_enabled)
        self.hear_var = tk.BooleanVar(value=self.s.hear_myself)
        self.mute_var = tk.BooleanVar(value=self.s.mic_muted)
        for text, var in (("Voice changer", self.voice_var), ("Hear myself", self.hear_var),
                          ("Mute mic", self.mute_var)):
            ttk.Checkbutton(head, text=text, variable=var, style="Switch.TCheckbutton",
                            command=self.apply_toggles).pack(side="left", padx=(24, 0))
        ttk.Button(head, text="Settings", command=lambda: SettingsDialog(self)).pack(side="right")
        self.meter = tk.Canvas(head, width=160, height=26, highlightthickness=0, bg="#1c1c1c")
        self.meter.pack(side="right", padx=12)

        self.banner = ttk.Frame(r, padding=(14, 6))
        self.banner_text = ttk.Label(self.banner, text="", foreground="#f0b429")
        self.banner_text.pack(side="left")
        self.banner_btn = ttk.Button(self.banner, text="Set up virtual microphone", command=self.install_cable)
        self.banner_btn.pack(side="right")

        # AI voice pack download bar (shown while downloading)
        self.dl_bar = ttk.Frame(r, padding=(14, 4))
        self.dl_label = ttk.Label(self.dl_bar, text="")
        self.dl_label.pack(side="left")
        ttk.Button(self.dl_bar, text="Cancel", command=self._cancel_download).pack(side="right")
        self.dl_progress = ttk.Progressbar(self.dl_bar, mode="determinate", length=320, maximum=1000)
        self.dl_progress.pack(side="right", padx=10)
        self._dl_cancel = None

        self.status = tk.StringVar(value="Starting…")  # packed before the body so it always gets space
        ttk.Label(r, textvariable=self.status, padding=(14, 6), foreground="#aaa").pack(fill="x", side="bottom")
        body = ttk.Frame(r, padding=(14, 4))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.search = tk.StringVar()
        self.search.trace_add("write", lambda *_: self._fill_tree())
        ttk.Label(left, text="Search voices", foreground="#999").pack(anchor="w")
        ttk.Entry(left, textvariable=self.search, width=32).pack(fill="x")
        ttk.Button(left, text="+  New AI voice", style="Accent.TButton",
                   command=lambda: self.ensure_ai_pack(lambda: NewAIVoiceDialog(self))).pack(fill="x", side="bottom")
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse", height=12)
        self.tree.column("#0", width=290)
        self.tree.pack(fill="both", expand=True, pady=6)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_select())

        right = ttk.Frame(body, padding=(18, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)
        self.v_name = ttk.Label(right, text="", font=("Segoe UI Semibold", 16))
        self.v_name.pack(anchor="w")
        self.v_desc = ttk.Label(right, text="", wraplength=560, justify="left", foreground="#aaa")
        self.v_desc.pack(anchor="w", pady=(2, 8))
        self.ai_box = ttk.Frame(right)
        ttk.Label(self.ai_box, text="Quality (diffusion steps)").pack(side="left")
        self.steps = tk.IntVar(value=self.s.diffusion_steps)
        ttk.Spinbox(self.ai_box, from_=2, to=12, textvariable=self.steps, width=4,
                    command=self._steps_changed).pack(side="left", padx=8)
        self.del_btn = ttk.Button(self.ai_box, text="Delete voice", command=self._delete_voice)
        self.progress = ttk.Progressbar(right, mode="indeterminate", length=240)

        sb = self.sb = ttk.LabelFrame(right, text="Soundboard", padding=10)
        sb.pack(fill="both", expand=True, pady=(12, 0))
        bar = ttk.Frame(sb)
        bar.pack(fill="x")
        ttk.Button(bar, text="Choose folder…", command=self._pick_sounds).pack(side="left")
        ttk.Button(bar, text="Stop sounds", command=self.stop_sounds).pack(side="left", padx=6)
        self.sb_grid = ttk.Frame(sb)
        self.sb_grid.pack(fill="both", expand=True, pady=(8, 0))
        self._fill_soundboard()


    # -------------------------------------------------------------- voices
    def refresh_voices(self, select: str | None = None):
        self.presets = list_presets()
        self._fill_tree()
        if select and select in self.presets:
            self.tree.selection_set(select)
            self.tree.see(select)

    def _fill_tree(self):
        q = self.search.get().lower().strip()
        self.tree.delete(*self.tree.get_children())
        groups: dict[str, str] = {}
        order = lambda k: (self.presets[k].get("category") != "my voices", self.presets[k].get("category", ""),  # noqa: E731
                           self.presets[k].get("name", k))
        for key in sorted(self.presets, key=order):
            spec = self.presets[key]
            if spec.get("template"):
                continue  # templates are reached through "New AI voice"
            if q and q not in spec.get("name", key).lower() and q not in spec.get("category", "").lower():
                continue
            cat = spec.get("category", "other").title()
            if cat not in groups:
                groups[cat] = self.tree.insert("", "end", text=cat, open=True)
            self.tree.insert(groups[cat], "end", iid=key, text=spec.get("name", key))
        if self.current_key and self.tree.exists(self.current_key):
            self.tree.selection_set(self.current_key)

    def _on_select(self):
        sel = self.tree.selection()
        if not sel or sel[0] not in self.presets or sel[0] == self.current_key:
            return
        self.select_voice(sel[0])

    def select_voice(self, key: str):
        spec = self.presets[key]
        self.current_key = key
        self.s.voice = key
        self.s.save()
        self.v_name.config(text=spec.get("name", key))
        self.v_desc.config(text=spec.get("description", ""))
        if is_ai(spec):
            self.ai_box.pack(anchor="w", pady=4, before=self.sb)
            self.del_btn.pack(side="left", padx=12) if spec.get("_user") else self.del_btn.pack_forget()
        else:
            self.ai_box.pack_forget()
        self._load_processor(spec)

    def _load_processor(self, spec):
        if not self.engine:
            return
        if unavailable_reason(spec) == NEEDS_DOWNLOAD:
            self.ensure_ai_pack(lambda: self._load_processor(spec))
            return
        self._load_token += 1
        token = self._load_token
        if not is_ai(spec):
            self._apply_processor(build_processor(spec, SR), spec)
            return
        self.progress.pack(anchor="w", pady=4, before=self.sb)
        self.progress.start(12)
        self.status.set(f"Loading {spec.get('name')}…")
        steps = int(self.steps.get())  # Tk state is read on the UI thread only

        def run():
            try:
                proc = build_processor(spec, SR, diffusion_steps=steps, ai_mode="threaded")
                proc.warmup()
                self.call_soon(lambda: token == self._load_token and self._apply_processor(proc, spec))
            except Exception as e:
                log.exception("AI voice failed to load")
                msg = str(e)
                self.call_soon(lambda: self.status.set(f"Could not load AI voice: {msg}"))
            finally:
                self.call_soon(self._stop_progress)
        threading.Thread(target=run, daemon=True).start()

    def _stop_progress(self):
        self.progress.stop()
        self.progress.pack_forget()

    def _apply_processor(self, proc, spec):
        if self.engine:
            self.engine.set_processor(proc)
            self._status_line()

    def _steps_changed(self):
        self.s.diffusion_steps = int(self.steps.get())
        self.s.save()
        if self.current_key:
            self._load_processor(self.presets[self.current_key])

    def _delete_voice(self):
        key = self.current_key
        if key and messagebox.askyesno("Delete voice", f"Delete '{self.presets[key].get('name')}'?"):
            delete_user_voice(key)
            self.current_key = None
            self.refresh_voices(select="clean")

    # -------------------------------------------------------------- engine
    def restart_engine(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        s = self.s
        try:
            monitor = s.monitor_device or self._default_monitor()
            self.engine = AudioEngine(s.input_device or None, s.output_device or "virtual", monitor,
                                      samplerate=SR, blocksize=512, hostapi=self.hostapi)
            self.apply_toggles(save=False)
            self.engine.start()
            self.banner.pack_forget()
        except NoVirtualCable:
            self.engine = None
            self._show_banner("No virtual microphone installed — Discord, OBS and games can't hear VoxShift yet.")
            self.status.set("Audio stopped: no virtual microphone")
            return
        except Exception as e:
            log.exception("audio engine failed to start")
            self.engine = None
            self._show_banner(f"Audio device problem: {e}", button=False)
            self.status.set("Audio stopped — check Settings")
            return
        if self.current_key:
            self._load_processor(self.presets[self.current_key])
        self._status_line()

    def _default_monitor(self) -> str | None:
        """Hear yourself on the system default output unless that is a virtual cable."""
        import sounddevice as sd
        h = next(h for h in sd.query_hostapis() if h["name"] == self.hostapi)
        if h["default_output_device"] < 0:
            return None
        name = sd.query_devices(h["default_output_device"])["name"]
        return None if any(name == c.playback for c in find_cables(self.hostapi)) else name

    def _show_banner(self, text, button=True):
        self.banner_text.config(text=text)
        self.banner_btn.pack(side="right") if button else self.banner_btn.pack_forget()
        self.banner.pack(fill="x", after=self.root.winfo_children()[0])

    def apply_toggles(self, save=True):
        s = self.s
        s.voice_enabled, s.hear_myself, s.mic_muted = self.voice_var.get(), self.hear_var.get(), self.mute_var.get()
        if self.engine:
            e = self.engine
            e.voice_enabled, e.hear_myself, e.mic_muted = s.voice_enabled, s.hear_myself, s.mic_muted
            e.input_gain_db, e.output_gain_db, e.monitor_gain_db = s.input_gain_db, s.output_gain_db, s.monitor_gain_db
        if save:
            s.save()
        self._update_tray()

    def _status_line(self):
        if not self.engine:
            return
        import sounddevice as sd
        out = sd.query_devices(self.engine.output_device)["name"]
        cable = next((c for c in find_cables(self.hostapi) if c.playback == out), None)
        mic = f"In Discord/OBS/games pick the microphone “{cable.capture}”" if cable else f"Output: {out}"
        lat = ""
        if self.engine.processor is not None:
            lat = f"   ·   latency ≈ {1000 * self.engine.processor.latency / SR + self.engine.stream_latency_ms:.0f} ms"
        self.status.set(mic + lat)

    # -------------------------------------------------------------- virtual mic
    def install_cable(self):
        if not messagebox.askokcancel(
                "Set up virtual microphone",
                "VoxShift will download the free VB-CABLE driver from vb-audio.com, check its signature "
                "and install it.\n\nWindows will ask for permission twice (admin + driver). "
                "A restart may be needed afterwards."):
            return
        self.status.set("Installing VB-CABLE…")
        hwnd = int(self.root.wm_frame(), 16)  # parent the UAC prompt to our window so it shows on top

        def run():
            try:
                install_vbcable(silent=True, hwnd=hwnd)
                ok = bool(find_cables(self.hostapi))
                msg = "Virtual microphone ready." if ok else "VB-CABLE installed — restart Windows to finish."
            except Exception as e:
                log.exception("VB-CABLE install failed")
                msg = f"VB-CABLE install failed: {e}"
            self.call_soon(lambda: (self.status.set(msg), self._after_install()))
        threading.Thread(target=run, daemon=True).start()

    def _after_install(self):
        import sounddevice as sd
        sd._terminate()  # re-enumerate devices so the new endpoints appear
        sd._initialize()
        self.restart_engine()

    def test_cable(self):
        from .audio.cable import self_test
        cables = find_cables(self.hostapi)
        if not cables:
            messagebox.showinfo("Virtual microphone", "No virtual microphone installed.")
            return
        ok, ratio = self_test(cables[0], self.hostapi)
        messagebox.showinfo("Virtual microphone", f"{cables[0].label}: {'working' if ok else 'NOT passing audio'} "
                                                  f"({ratio:.0f} dB tone-to-noise)")

    # -------------------------------------------------------------- AI voice pack
    def ensure_ai_pack(self, then):
        """Run ``then`` now if the AI voice pack is installed, else offer to download it first."""
        from .ai.model_pack import find_model
        if find_model():
            then()
            return
        if self._dl_cancel is not None:
            self.status.set("The AI voice pack is still downloading…")
            return
        if messagebox.askyesno("AI voices", "AI voices need a one-time download of about 820 MB.\n\n"
                               "Download them now? You can keep using VoxShift meanwhile, and an "
                               "interrupted download resumes where it stopped."):
            self.download_ai_pack(then)

    def download_ai_pack(self, then=None):
        from .ai import model_pack
        cancel = self._dl_cancel = threading.Event()
        self.dl_progress["value"] = 0
        self.dl_label.config(text="Downloading AI voices…")
        self.dl_bar.pack(fill="x", after=self.root.winfo_children()[0])
        last = [0.0]

        def progress(done, total):
            import time
            if time.monotonic() - last[0] > 0.2 or done >= total:
                last[0] = time.monotonic()
                self.call_soon(lambda: self._dl_update(done, total))

        def run():
            try:
                model_pack.download(progress, cancel)
                msg, ok = "AI voices installed.", True
            except InterruptedError:
                msg, ok = "AI voice download paused — it resumes next time.", False
            except Exception as e:
                log.exception("AI voice pack download failed")
                msg, ok = f"AI voice download failed: {e} (it resumes next time)", False
            self.call_soon(lambda: self._dl_finished(msg, ok, then))
        threading.Thread(target=run, name="ai-pack-download", daemon=True).start()

    def _dl_update(self, done, total):
        self.dl_progress["value"] = 1000 * done / max(total, 1)
        self.dl_label.config(text=f"Downloading AI voices… {done / 2**20:.0f} / {total / 2**20:.0f} MB")

    def _dl_finished(self, msg, ok, then):
        self._dl_cancel = None
        self.dl_bar.pack_forget()
        self.status.set(msg)
        if ok:
            self.refresh_voices(select=self.current_key)
            if then:
                then()

    def _cancel_download(self):
        if self._dl_cancel:
            self._dl_cancel.set()

    def _first_run(self):
        if self.s.first_run_done:
            return
        self.s.first_run_done = True
        self.s.save()
        if not find_cables(self.hostapi):
            self.install_cable()

    # -------------------------------------------------------------- soundboard
    def _pick_sounds(self):
        d = filedialog.askdirectory()
        if d:
            self.s.soundboard_dir = d
            self.s.save()
            self._fill_soundboard()

    def _fill_soundboard(self):
        for w in self.sb_grid.winfo_children():
            w.destroy()
        d = Path(self.s.soundboard_dir) if self.s.soundboard_dir else None
        if not d or not d.is_dir():
            ttk.Label(self.sb_grid, text="Choose a folder of sounds (wav, flac, ogg, mp3).",
                      foreground="#999").grid()
            return
        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3")]
        for i, p in enumerate(files[:30]):
            ttk.Button(self.sb_grid, text=p.stem[:18], width=18, command=lambda p=p: self.play(p)).grid(
                row=i // 4, column=i % 4, sticky="ew", padx=3, pady=3)

    def play(self, path: Path):
        import soundfile as sf
        if self.engine:
            data, sr = sf.read(str(path), dtype="float32")
            self.engine.play_sound(data, sr)

    def stop_sounds(self):
        if self.engine:
            self.engine.stop_sounds()

    # -------------------------------------------------------------- hotkeys / tray
    def register_hotkeys(self):
        if self.hotkeys:
            self.hotkeys.stop()
        actions = {"toggle_voice": lambda: self._flip(self.voice_var),
                   "toggle_hear_myself": lambda: self._flip(self.hear_var),
                   "toggle_mute": lambda: self._flip(self.mute_var),
                   "stop_sounds": self.stop_sounds}
        bindings = {k: (spec, (lambda a=actions[k]: self.call_soon(a)))
                    for k, spec in self.s.hotkeys.items() if spec and k in actions}
        self.hotkeys = winsys.GlobalHotkeys(bindings)
        self.hotkey_errors = dict(self.hotkeys.failed)
        for k, err in self.hotkey_errors.items():
            log.warning("hotkey %s: %s", k, err)

    def _flip(self, var: tk.BooleanVar):
        var.set(not var.get())
        self.apply_toggles()

    def _start_tray(self):
        try:
            import pystray
            from PIL import Image
        except ImportError:
            return
        item = pystray.MenuItem
        ui = lambda f: (lambda icon, it: self.call_soon(f))  # noqa: E731
        menu = pystray.Menu(
            item("Open VoxShift", ui(self.show), default=True),
            # pystray evaluates `checked` on its own thread: read plain settings, never Tk variables
            item("Voice changer", ui(lambda: self._flip(self.voice_var)), checked=lambda it: self.s.voice_enabled),
            item("Hear myself", ui(lambda: self._flip(self.hear_var)), checked=lambda it: self.s.hear_myself),
            item("Mute mic", ui(lambda: self._flip(self.mute_var)), checked=lambda it: self.s.mic_muted),
            pystray.Menu.SEPARATOR,
            item("Quit", ui(self.quit)))
        self.tray = pystray.Icon("VoxShift", Image.open(ASSETS / "voxshift.png"), "VoxShift", menu)
        self.tray.run_detached()

    def _update_tray(self):
        if self.tray:
            self.tray.update_menu()

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def on_close(self):
        if self.s.close_to_tray and self.tray:
            self.root.withdraw()
            if not getattr(self, "_told_tray", False):
                self._told_tray = True
                self.tray.notify("VoxShift keeps running in the tray. Right-click the icon to quit.", "VoxShift")
        else:
            self.quit()

    def quit(self):
        self.s.save()
        if self.hotkeys:
            self.hotkeys.stop()
        if self.engine:
            self.engine.stop()
        if self.tray:
            self.tray.stop()
        self.root.destroy()

    # -------------------------------------------------------------- threading
    def call_soon(self, fn):
        """Run ``fn`` on the Tk thread. Safe from any thread (Tk itself is not)."""
        self._ui_queue.put(fn)

    # -------------------------------------------------------------- meters
    def _tick(self):
        if self._quit_evt.is_set():  # the installer is about to replace our files
            self.quit()
            return
        if self._show_evt.is_set():  # VoxShift was launched again: surface this instance
            self.show()
        while not self._ui_queue.empty():
            try:
                self._ui_queue.get_nowait()()
            except Exception:
                log.exception("UI callback failed")
        c = self.meter
        c.delete("all")
        if self.engine:
            for i, db in enumerate((self.engine.meters.input_db, self.engine.meters.output_db)):
                frac = max(0.0, min(1.0, (db + 60) / 60))
                c.create_rectangle(0, 3 + i * 12, 160 * frac, 11 + i * 12,
                                   fill="#22c55e" if frac < 0.85 else "#ef4444", width=0)
        self.root.after(60, self._tick)


# ---------------------------------------------------------------------- entry
def _setup_logging():
    path = log_dir() / "voxshift.log"
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    if FROZEN and sys.stdout is None:  # windowed build: no console to print to
        sys.stdout = sys.stderr = open(path.with_name("stdout.log"), "a", encoding="utf-8", buffering=1)
    log.info("VoxShift %s starting", __version__)


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    _setup_logging()
    guard = winsys.SingleInstance()
    if guard.already_running:
        if not winsys.NamedEvent.signal(winsys.NamedEvent.SHOW):
            winsys.message_box("VoxShift is already running — look for its icon in the system tray.")
        return
    winsys.set_app_id()
    winsys.enable_dpi_awareness()
    root = tk.Tk()

    def report(exc, val, tb):
        log.error("unhandled UI error", exc_info=(exc, val, tb))
        messagebox.showerror("VoxShift", f"{exc.__name__}: {val}\n\nDetails were written to {log_dir()}")
    root.report_callback_exception = report
    App(root, minimized="--minimized" in argv)
    root.mainloop()


if __name__ == "__main__":
    main()
