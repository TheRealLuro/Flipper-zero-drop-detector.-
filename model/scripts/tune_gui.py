"""Minimal tkinter UI around tuning.py.

Status panel auto-refreshes every 1.5 s, showing check / X for each required
file and each user_data/<class>/ folder. Pick params via radio buttons.
Start Tuning launches tuning.py as a subprocess and streams its stdout live
into the output panel.

Run:
    python tune_gui.py
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from pathlib import Path


CLASS_NAMES = ["idle", "walking", "fidget", "drop"]
EPOCH_CHOICES = [3, 5, 10, 15]
LR_CHOICES = [0.001, 0.005, 0.01]
REFRESH_MS = 1500
DRAIN_MS = 80

CHECK = "✓"
CROSS = "✗"


class TunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flipper Drop Detector — Tuner")
        self.root.geometry("680x720")
        self.root.minsize(620, 600)

        self.script_dir = Path(__file__).resolve().parent
        self.proc = None
        self.proc_thread = None
        self.out_queue = queue.Queue()

        self.status_labels = {}       # filename -> Label
        self.user_data_labels = {}    # class_name -> Label

        # Tk variables for radio buttons
        self.epochs_var = tk.IntVar(value=5)
        self.lr_var = tk.DoubleVar(value=0.005)
        self.keep_old_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._refresh_status()
        self._drain_queue()

    # ---------- layout ----------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}

        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Required files")
        status_frame.pack(fill="x", **pad)
        for fname in ("model.json", "TRAIN.bin", "TEST.bin"):
            row = ttk.Frame(status_frame)
            row.pack(fill="x", padx=8, pady=2)
            mark = tk.Label(row, text="?", width=2, font=("Segoe UI", 11, "bold"))
            mark.pack(side="left")
            txt = tk.Label(row, text=fname, anchor="w")
            txt.pack(side="left", fill="x", expand=True)
            note = tk.Label(row, text="", fg="#666", anchor="e")
            note.pack(side="right")
            self.status_labels[fname] = (mark, note)

        # User data frame
        ud_frame = ttk.LabelFrame(self.root, text="User data  (drop CSVs into user_data/<class>/)")
        ud_frame.pack(fill="x", **pad)
        for cls in CLASS_NAMES:
            row = ttk.Frame(ud_frame)
            row.pack(fill="x", padx=8, pady=2)
            mark = tk.Label(row, text="?", width=2, font=("Segoe UI", 11, "bold"))
            mark.pack(side="left")
            txt = tk.Label(row, text=f"user_data/{cls}", anchor="w")
            txt.pack(side="left", fill="x", expand=True)
            note = tk.Label(row, text="", fg="#666", anchor="e")
            note.pack(side="right")
            self.user_data_labels[cls] = (mark, note)

        # Params frame
        params_frame = ttk.LabelFrame(self.root, text="Parameters")
        params_frame.pack(fill="x", **pad)

        epochs_row = ttk.Frame(params_frame)
        epochs_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(epochs_row, text="Epochs:", width=10).pack(side="left")
        for n in EPOCH_CHOICES:
            ttk.Radiobutton(epochs_row, text=str(n), value=n, variable=self.epochs_var).pack(side="left", padx=4)

        lr_row = ttk.Frame(params_frame)
        lr_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(lr_row, text="Learn rate:", width=10).pack(side="left")
        for v in LR_CHOICES:
            ttk.Radiobutton(lr_row, text=f"{v:g}", value=v, variable=self.lr_var).pack(side="left", padx=4)

        out_row = ttk.Frame(params_frame)
        out_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(out_row, text="Output:", width=10).pack(side="left")
        ttk.Radiobutton(out_row, text="Overwrite model.json",
                        value=False, variable=self.keep_old_var).pack(side="left", padx=4)
        ttk.Radiobutton(out_row, text="Keep old → model_tuned.json",
                        value=True,  variable=self.keep_old_var).pack(side="left", padx=4)

        # Controls
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", **pad)
        self.start_btn = ttk.Button(ctrl, text="Start Tuning", command=self._start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.clear_btn = ttk.Button(ctrl, text="Clear output", command=self._clear_output)
        self.clear_btn.pack(side="left", padx=4)
        self.state_label = tk.Label(ctrl, text="Idle.", fg="#666")
        self.state_label.pack(side="right", padx=4)

        # Output
        out_frame = ttk.LabelFrame(self.root, text="Output")
        out_frame.pack(fill="both", expand=True, **pad)
        self.output = ScrolledText(out_frame, wrap="word", font=("Consolas", 9), height=18)
        self.output.pack(fill="both", expand=True, padx=4, pady=4)
        self.output.configure(state="disabled")

    # ---------- helpers ----------

    def _set_mark(self, mark_label, present):
        mark_label.configure(text=CHECK if present else CROSS,
                             fg="#1a8a1a" if present else "#b50000")

    def _count_csvs(self, dirpath):
        if not dirpath.is_dir():
            return -1
        return sum(1 for f in dirpath.iterdir() if f.is_file() and f.suffix.lower() == ".csv")

    def _can_start(self):
        # model.json + TRAIN.bin required; TEST.bin is optional.
        if not (self.script_dir / "model.json").is_file():
            return False
        if not (self.script_dir / "TRAIN.bin").is_file():
            return False
        # Need at least one user CSV somewhere.
        any_csv = False
        for cls in CLASS_NAMES:
            if self._count_csvs(self.script_dir / "user_data" / cls) > 0:
                any_csv = True
                break
        return any_csv

    def _running(self):
        return self.proc is not None and self.proc.poll() is None

    # ---------- periodic tasks ----------

    def _refresh_status(self):
        # Required files
        for fname, (mark, note) in self.status_labels.items():
            path = self.script_dir / fname
            present = path.is_file()
            self._set_mark(mark, present)
            note.configure(text=("" if present else "missing"))

        # User data
        total_user_csvs = 0
        for cls in CLASS_NAMES:
            mark, note = self.user_data_labels[cls]
            d = self.script_dir / "user_data" / cls
            n = self._count_csvs(d)
            if n < 0:
                self._set_mark(mark, False)
                note.configure(text="folder missing")
            elif n == 0:
                self._set_mark(mark, False)
                note.configure(text="no csv")
            else:
                self._set_mark(mark, True)
                note.configure(text=f"{n} file{'s' if n != 1 else ''}")
                total_user_csvs += n

        # Enable / disable Start
        if self._running():
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        else:
            self.start_btn.configure(state="normal" if self._can_start() else "disabled")
            self.stop_btn.configure(state="disabled")

        # Schedule next refresh
        self.root.after(REFRESH_MS, self._refresh_status)

    def _drain_queue(self):
        # Pull any queued stdout lines and append to output widget.
        any_lines = False
        while True:
            try:
                line = self.out_queue.get_nowait()
            except queue.Empty:
                break
            self._append_output(line)
            any_lines = True

        # If proc just finished, append a final status line.
        if self.proc is not None and self.proc.poll() is not None and self.proc_thread is not None:
            if not self.proc_thread.is_alive():
                code = self.proc.poll()
                self._append_output(f"\n[tuning.py exited with code {code}]\n")
                self.state_label.configure(
                    text=("Done." if code == 0 else f"Failed (exit {code})."),
                    fg=("#1a8a1a" if code == 0 else "#b50000"))
                self.proc = None
                self.proc_thread = None

        self.root.after(DRAIN_MS, self._drain_queue)

    # ---------- output panel ----------

    def _append_output(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _clear_output(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    # ---------- subprocess control ----------

    def _start(self):
        if self._running():
            return
        if not self._can_start():
            self._append_output("Cannot start: need model.json + TRAIN.bin and at least one user CSV.\n")
            return

        cmd = [sys.executable, "-u", str(self.script_dir / "tuning.py"),
               "--epochs", str(self.epochs_var.get()),
               "--lr", str(self.lr_var.get()),
               # Archive into model/runs/ alongside the C++ train runs.
               "--runs-dir", str(self.script_dir.parent / "runs")]
        if self.keep_old_var.get():
            cmd.append("--keep_old")

        self._append_output(f"$ {' '.join(cmd)}\n")
        self.state_label.configure(text="Running...", fg="#a06000")

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            self._append_output(f"Failed to start subprocess: {e}\n")
            self.state_label.configure(text="Failed.", fg="#b50000")
            self.proc = None
            return

        self.proc_thread = threading.Thread(target=self._reader, daemon=True)
        self.proc_thread.start()

    def _reader(self):
        assert self.proc is not None
        try:
            for line in self.proc.stdout:
                self.out_queue.put(line)
        except Exception as e:
            self.out_queue.put(f"[reader error: {e}]\n")
        finally:
            try:
                self.proc.stdout.close()
            except Exception:
                pass

    def _stop(self):
        if not self._running():
            return
        self._append_output("\n[stopping...]\n")
        try:
            self.proc.terminate()
        except Exception as e:
            self._append_output(f"terminate() failed: {e}\n")


def main():
    root = tk.Tk()
    try:
        # Slightly less jagged on Windows.
        from tkinter import font as tkfont
        default = tkfont.nametofont("TkDefaultFont")
        default.configure(size=10)
    except Exception:
        pass
    TunerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
