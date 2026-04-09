"""
gui.py
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

log = logging.getLogger(__name__)

_BG      = "#1e1e2e"
_PANEL   = "#2a2a3e"
_PANEL2  = "#313147"
_ACCENT  = "#7c6af7"
_GREEN   = "#50cd89"
_RED     = "#f1416c"
_YELLOW  = "#ffc107"
_FG      = "#cdd6f4"
_FG_DIM  = "#6c7086"
_FONT    = ("Segoe UI", 10)
_FONT_B  = ("Segoe UI", 10, "bold")
_FONT_H  = ("Segoe UI", 14, "bold")
_FONT_M  = ("Segoe UI", 11, "bold")
_FONT_C  = ("Consolas",  9)


class _Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text   = text
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        if self._tip:
            return
        x = self._widget.winfo_rootx() + 24
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self._tip, text=self._text, font=("Segoe UI", 9),
            bg="#2e2e42", fg=_FG, relief="flat", padx=8, pady=4,
        ).pack()

    def _hide(self, _event=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


def _label(parent, text, bold=False, dim=False, **kw):
    fg   = _FG_DIM if dim else _FG
    font = _FONT_B if bold else _FONT
    return tk.Label(parent, text=text, font=font, bg=kw.pop("bg", _PANEL), fg=fg, **kw)


def _entry(parent, var, width=42, **kw):
    return tk.Entry(parent, textvariable=var, bg=_BG, fg=_FG, font=_FONT,
                    relief="flat", insertbackground=_FG, width=width, **kw)


def _btn(parent, text, cmd, color=_ACCENT, fg="white", padx=14, pady=6, **kw):
    return tk.Button(
        parent, text=text, command=cmd, font=_FONT_B,
        bg=color, fg=fg, activebackground=_GREEN,
        relief="flat", padx=padx, pady=pady, cursor="hand2", **kw,
    )


class ParkingGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("🅿  Unified Parking System")
        self.geometry("860x720")
        self.minsize(720, 580)
        self.configure(bg=_BG)

        self._video_var      = tk.StringVar()
        self._pos_var        = tk.StringVar(value="CarParkPos")
        self._output_var     = tk.StringVar()
        self._mode_var       = tk.StringVar(value="detect")
        self._yolo_var       = tk.BooleanVar(value=False)
        self._no_disp_var    = tk.BooleanVar(value=False)
        self._no_slots_var   = tk.BooleanVar(value=False)
        self._dwell_var      = tk.DoubleVar(value=0.5)
        self._conf_var       = tk.DoubleVar(value=0.35)
        self._yolo_model_var = tk.StringVar(value="yolov8n.pt")
        self._zone_var       = tk.StringVar()

        self._proc: Optional[subprocess.Popen] = None
        self._start_time: Optional[float] = None
        self._timer_id: Optional[str] = None

        self._build_ui()


    def _build_ui(self) -> None:
        # Title bar
        title_f = tk.Frame(self, bg=_BG)
        title_f.pack(fill="x", padx=0, pady=0)
        tk.Label(title_f, text="🅿  Unified Parking System",
                 font=_FONT_H, bg=_BG, fg=_ACCENT).pack(side="left", padx=24, pady=12)
        tk.Label(title_f, text="Detection · Tracking · Alerts",
                 font=_FONT, bg=_BG, fg=_FG_DIM).pack(side="left", pady=12)

        # Notebook
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",        background=_BG,    borderwidth=0)
        style.configure("TNotebook.Tab",    background=_PANEL, foreground=_FG_DIM,
                         font=_FONT_B, padding=[14, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", _PANEL2)],
                  foreground=[("selected", _FG)])
        style.configure("TFrame", background=_PANEL)

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=False, padx=20, pady=(0, 6))

        self._tab_files  = self._make_tab("📂  Files")
        self._tab_detect = self._make_tab("🔍  Detect")
        self._tab_slots  = self._make_tab("🅿  Slots")
        self._tab_about  = self._make_tab("ℹ  About")

        self._build_files_tab()
        self._build_detect_tab()
        self._build_slots_tab()
        self._build_about_tab()

        mode_f = tk.Frame(self, bg=_BG)
        mode_f.pack(fill="x", padx=20, pady=2)
        _label(mode_f, "Mode:", bold=True, bg=_BG).pack(side="left", padx=(0, 8))
        for val, lbl, tip in (
            ("picker",    "🖱  Picker",    "Annotate slot positions on first video frame"),
            ("occupancy", "📊  Occupancy", "Detect slot occupancy only (no tracking)"),
            ("detect",    "🚗  Detect",    "Full pipeline: detection + tracking + alerts"),
        ):
            rb = tk.Radiobutton(
                mode_f, text=lbl, variable=self._mode_var, value=val,
                bg=_BG, fg=_FG, selectcolor=_ACCENT,
                activebackground=_BG, font=_FONT,
                command=self._on_mode_change,
            )
            rb.pack(side="left", padx=8)
            _Tooltip(rb, tip)

        # Run / Kill buttons
        btn_f = tk.Frame(self, bg=_BG)
        btn_f.pack(pady=6)
        self._run_btn = _btn(btn_f, "▶  Run", self._on_run, padx=32, pady=10)
        self._run_btn.pack(side="left", padx=8)
        self._kill_btn = _btn(btn_f, "⏹  Stop", self._on_kill,
                              color="#444460", padx=20, pady=10)
        self._kill_btn.pack(side="left", padx=4)
        self._kill_btn.configure(state="disabled")
        _btn(btn_f, "✕  Clear log", lambda: self._log.delete("1.0", "end"),
             color=_PANEL, fg=_FG_DIM, padx=16, pady=10).pack(side="left", padx=8)

        # Log panel
        log_hdr = tk.Frame(self, bg=_BG)
        log_hdr.pack(fill="x", padx=24, pady=(2, 0))
        _label(log_hdr, "Output log", bold=True, bg=_BG).pack(side="left")
        self._status_lbl = _label(log_hdr, "", dim=True, bg=_BG)
        self._status_lbl.pack(side="right", padx=8)

        self._log = scrolledtext.ScrolledText(
            self, height=12, bg="#12121e", fg=_FG, font=_FONT_C,
            insertbackground=_FG, relief="flat", bd=0,
            state="normal",
        )
        self._log.pack(fill="both", expand=True, padx=20, pady=(2, 14))
        self._log.tag_config("warn",  foreground=_YELLOW)
        self._log.tag_config("error", foreground=_RED)
        self._log.tag_config("cmd",   foreground=_ACCENT)

        self._on_mode_change()

    def _make_tab(self, title: str) -> ttk.Frame:
        f = ttk.Frame(self._nb)
        self._nb.add(f, text=title)
        return f

    def _build_files_tab(self) -> None:
        f = self._tab_files
        f.columnconfigure(1, weight=1)

        rows = [
            ("Video file",     self._video_var,  self._browse_video,
             "Input video for processing (.mp4 / .avi / .mov / .mkv)"),
            ("Slot positions", self._pos_var,    self._browse_pos,
             "Pickle file produced by the Slot Picker (default: CarParkPos)"),
            ("Output .mp4",    self._output_var, self._browse_output,
             "Optional: save annotated video to this path"),
        ]
        for i, (lbl, var, cmd, tip) in enumerate(rows):
            _label(f, lbl, bold=True).grid(
                row=i, column=0, padx=(14, 6), pady=8, sticky="w")
            e = _entry(f, var)
            e.grid(row=i, column=1, padx=4, pady=8, sticky="ew")
            _Tooltip(e, tip)
            b = _btn(f, "Browse", cmd, padx=10, pady=4)
            b.grid(row=i, column=2, padx=(4, 14), pady=8)


    def _build_detect_tab(self) -> None:
        f = self._tab_detect
        f.columnconfigure(1, weight=1)

        # YOLO
        _label(f, "Detector", bold=True).grid(row=0, column=0, padx=14, pady=8, sticky="w")
        yolo_f = tk.Frame(f, bg=_PANEL)
        yolo_f.grid(row=0, column=1, columnspan=2, sticky="w", padx=4, pady=8)
        cb = tk.Checkbutton(yolo_f, text="Use YOLOv8", variable=self._yolo_var,
                            bg=_PANEL, fg=_FG, selectcolor=_ACCENT,
                            activebackground=_PANEL, font=_FONT,
                            command=self._on_yolo_toggle)
        cb.pack(side="left")
        _Tooltip(cb, "Uncheck to use classical MOG2 background subtraction")
        _label(yolo_f, "  Model:", bold=False).pack(side="left")
        self._yolo_entry = _entry(yolo_f, self._yolo_model_var, width=24)
        self._yolo_entry.pack(side="left", padx=6)
        self._yolo_entry.configure(state="disabled")
        _Tooltip(self._yolo_entry, "Path to YOLOv8 weights, e.g. yolov8n.pt")

        # Dwell slider
        _label(f, "Dwell (min)", bold=True).grid(row=1, column=0, padx=14, pady=8, sticky="w")
        dwell_f = tk.Frame(f, bg=_PANEL)
        dwell_f.grid(row=1, column=1, sticky="w", padx=4)
        tk.Scale(dwell_f, variable=self._dwell_var, from_=0.1, to=30.0,
                 resolution=0.1, orient="horizontal", length=200,
                 bg=_PANEL, fg=_FG, troughcolor=_BG,
                 highlightthickness=0, font=_FONT).pack(side="left")
        _label(dwell_f, "min before alert fires", dim=True).pack(side="left", padx=6)

        # Confidence slider
        _label(f, "YOLO conf", bold=True).grid(row=2, column=0, padx=14, pady=8, sticky="w")
        conf_f = tk.Frame(f, bg=_PANEL)
        conf_f.grid(row=2, column=1, sticky="w", padx=4)
        tk.Scale(conf_f, variable=self._conf_var, from_=0.10, to=1.0,
                 resolution=0.01, orient="horizontal", length=200,
                 bg=_PANEL, fg=_FG, troughcolor=_BG,
                 highlightthickness=0, font=_FONT).pack(side="left")
        _label(conf_f, "detection confidence threshold", dim=True).pack(side="left", padx=6)

        # Zone JSON
        _label(f, "Zone JSON", bold=True).grid(row=3, column=0, padx=14, pady=8, sticky="w")
        ze = _entry(f, self._zone_var, width=46)
        ze.grid(row=3, column=1, columnspan=2, padx=(4, 14), pady=8, sticky="ew")
        _Tooltip(ze, 'Pre-defined zone as JSON: [[x,y],[x,y],[x,y]] — leave blank to draw interactively')

        # Checkboxes
        chk_f = tk.Frame(f, bg=_PANEL)
        chk_f.grid(row=4, column=0, columnspan=3, sticky="w", padx=14, pady=6)
        for var, lbl, tip in (
            (self._no_disp_var,  "Headless (no display window)", "Run without opening a GUI window"),
            (self._no_slots_var, "Disable slot overlay",         "Skip slot-occupancy overlay even if CarParkPos exists"),
        ):
            cb2 = tk.Checkbutton(chk_f, text=lbl, variable=var,
                                 bg=_PANEL, fg=_FG, selectcolor=_ACCENT,
                                 activebackground=_PANEL, font=_FONT)
            cb2.pack(side="left", padx=12)
            _Tooltip(cb2, tip)

    def _build_slots_tab(self) -> None:
        f = self._tab_slots
        lines = [
            ("Picker mode",    "Opens an interactive window on the first video frame."),
            ("",               "Left-click to place a slot rectangle."),
            ("",               "Right-click to remove the slot under the cursor."),
            ("",               "Press Q to save and quit."),
            ("Occupancy mode", "Runs slot-occupancy detection using saved positions."),
            ("",               "Green border = free slot, Red = occupied."),
        ]
        for i, (bold, rest) in enumerate(lines):
            row_f = tk.Frame(f, bg=_PANEL)
            row_f.pack(fill="x", padx=14, pady=1)
            if bold:
                _label(row_f, bold + "  ", bold=True).pack(side="left")
            _label(row_f, rest, dim=not bold).pack(side="left")

        sep = tk.Frame(f, bg=_PANEL2, height=1)
        sep.pack(fill="x", padx=14, pady=10)
        _label(f, "Quick-launch shortcuts:", bold=True, bg=_PANEL).pack(anchor="w", padx=14)
        btn_f = tk.Frame(f, bg=_PANEL)
        btn_f.pack(anchor="w", padx=14, pady=8)
        _btn(btn_f, "🖱  Launch Picker",
             lambda: self._quick_launch("picker"), padx=16).pack(side="left", padx=4)
        _btn(btn_f, "📊  Launch Occupancy",
             lambda: self._quick_launch("occupancy"), padx=16).pack(side="left", padx=4)

    def _quick_launch(self, mode: str) -> None:
        self._mode_var.set(mode)
        self._on_mode_change()
        self._nb.select(0)   # switch to Files tab so user can see the path
        self._on_run()

    def _build_about_tab(self) -> None:
        f = self._tab_about
        lines = [
            "Unified Parking System",
            "",
            "Modes",
            "  picker     — Annotate rectangular parking slots on the first frame.",
            "  occupancy  — Classical preprocessing slot-fill detector.",
            "  detect     — MOG2 or YOLOv8 vehicle detection + SORT-Kalman tracking",
            "               + zone-based dwell-time illegal-parking alerts.",
            "",
            "Tracker",
            "  • Kalman filter per track (constant-velocity model)",
            "  • ByteTrack two-stage Hungarian matching",
            "  • Mahalanobis distance gating",
            "  • Adaptive IoU threshold (speed + age aware)",
            "  • Motion-gated zone entry (suppresses transient crossings)",
        ]
        for line in lines:
            bold = line and not line.startswith(" ") and line != "Unified Parking System"
            _label(f, line, bold=bold, dim=not line or line.startswith(" "),
                   anchor="w").pack(fill="x", padx=18, pady=0)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")],
        )
        if p:
            self._video_var.set(p)

    def _browse_pos(self) -> None:
        p = filedialog.askopenfilename(title="Select slot positions file")
        if p:
            self._pos_var.set(p)

    def _browse_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Save output video", defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")],
        )
        if p:
            self._output_var.set(p)

    def _on_mode_change(self) -> None:
        mode  = self._mode_var.get()
        state = "normal" if mode == "detect" else "disabled"
        tab_idx = {"picker": 2, "occupancy": 2, "detect": 1}
        self._nb.select(tab_idx.get(mode, 1))
        for child in self._tab_detect.winfo_children():
            _set_state_recursive(child, state)

    def _on_yolo_toggle(self) -> None:
        self._yolo_entry.configure(
            state="normal" if self._yolo_var.get() else "disabled"
        )

    def _on_run(self) -> None:
        video = self._video_var.get().strip()
        if not video or not Path(video).exists():
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        if self._proc and self._proc.poll() is None:
            messagebox.showwarning("Running", "A process is already running.")
            return
        self._run_btn.configure(state="disabled", text="⏳  Running…")
        self._kill_btn.configure(state="normal")
        self._start_time = time.time()
        self._tick_timer()
        threading.Thread(target=self._run_subprocess, daemon=True).start()

    def _on_kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log_write("[Process terminated by user]\n", tag="warn")

    def _tick_timer(self) -> None:
        if self._start_time is None:
            return
        elapsed = time.time() - self._start_time
        self._status_lbl.configure(text=f"Elapsed: {elapsed:.0f}s")
        self._timer_id = self.after(1000, self._tick_timer)

    def _stop_timer(self) -> None:
        if self._timer_id:
            self.after_cancel(self._timer_id)
            self._timer_id    = None
            self._start_time  = None


    def _run_subprocess(self) -> None:
        mode  = self._mode_var.get()
        video = self._video_var.get().strip()
        pos   = self._pos_var.get().strip()

        cmd = [sys.executable, "main.py", mode, video, "--pos-file", pos]

        if mode == "detect":
            cmd += ["--dwell", str(round(self._dwell_var.get(), 2))]
            if self._output_var.get().strip():
                cmd += ["--output", self._output_var.get().strip()]
            if self._no_disp_var.get():
                cmd.append("--no-display")
            if self._no_slots_var.get():
                cmd.append("--no-slots")
            if self._yolo_var.get():
                cmd += ["--yolo", "--yolo-model", self._yolo_model_var.get().strip(),
                        "--yolo-conf", str(round(self._conf_var.get(), 2))]
            if self._zone_var.get().strip():
                cmd += ["--zone", self._zone_var.get().strip()]

        self._log_write(f"$ {' '.join(cmd)}\n\n", tag="cmd")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._proc.stdout:
                tag = (
                    "warn"  if "WARNING" in line or "WARN" in line
                    else "error" if "ERROR" in line or "CRITICAL" in line
                    else None
                )
                self._log_write(line, tag=tag)
            self._proc.wait()
            self._log_write(f"\n[Process exited with code {self._proc.returncode}]\n",
                            tag="warn" if self._proc.returncode != 0 else None)
        except Exception as exc:
            self._log_write(f"\n[ERROR] {exc}\n", tag="error")
        finally:
            self.after(0, self._on_process_done)

    def _on_process_done(self) -> None:
        self._run_btn.configure(state="normal", text="▶  Run")
        self._kill_btn.configure(state="disabled")
        self._stop_timer()
        elapsed = ""
        if self._start_time:
            elapsed = f" | {time.time() - self._start_time:.1f}s"
        self._status_lbl.configure(text=f"Done{elapsed}")

    def _log_write(self, text: str, tag: Optional[str] = None) -> None:
        def _do():
            self._log.configure(state="normal")
            if tag:
                self._log.insert("end", text, tag)
            else:
                self._log.insert("end", text)
            self._log.see("end")
        self.after(0, _do)


def _set_state_recursive(widget: tk.Widget, state: str) -> None:
    try:
        widget.configure(state=state)
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        _set_state_recursive(child, state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = ParkingGUI()
    app.mainloop()