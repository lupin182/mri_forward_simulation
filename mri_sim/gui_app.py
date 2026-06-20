from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import codecs
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mri_sim.phantom_database import (
    DATA_FILE_MAP,
    create_phantom_database_entry,
    delete_phantom_database_data,
    delete_phantom_database_entry,
    import_phantom_database_file,
    list_phantom_database,
)
from mri_sim.sequence_database import (
    delete_sequence_database_entry,
    list_sequence_database,
    load_sequence_database_file,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT_DIR / "main.py"
OUTPUT_DIR = ROOT_DIR / "output"
ENV_PATH = ROOT_DIR / ".env"
PROGRESS_PREFIX = "__MRI_PROGRESS__ "

PHANTOMS = ("asymmetric", "sphere", "ring", "database")
SEQUENCES = ("gre", "gre_label", "se", "tse", "epi", "epi_se", "epi_label", "database")
PHANTOM_DATA_NAMES = tuple(DATA_FILE_MAP.keys())
HARDWARE_DEFAULTS = {
    "MRI_SYSTEM_MAX_GRAD": "32",
    "MRI_SYSTEM_GRAD_UNIT": "mT/m",
    "MRI_SYSTEM_MAX_SLEW": "130",
    "MRI_SYSTEM_SLEW_UNIT": "T/m/s",
    "MRI_SYSTEM_RF_RINGDOWN_TIME": "20e-6",
    "MRI_SYSTEM_RF_DEAD_TIME": "100e-6",
    "MRI_SYSTEM_ADC_DEAD_TIME": "10e-6",
}
SEQUENCE_SPECIFIC_FIELDS = {
    "gre": {"seq_flip_angle_deg", "seq_rf_spoiling_inc_deg", "seq_dummy_scans", "seq_ideal_spoiling_reset"},
    "gre_label": {
        "seq_flip_angle_deg",
        "seq_rf_spoiling_inc_deg",
        "seq_dummy_scans",
        "seq_ideal_spoiling_reset",
        "seq_readout_duration",
    },
    "se": {
        "seq_excitation_flip_angle_deg",
        "seq_refocusing_flip_angle_deg",
        "seq_rf_excitation_duration",
        "seq_rf_refocusing_duration",
        "seq_readout_time",
        "seq_prephase_duration",
    },
    "tse": {"seq_n_echo", "seq_rf_flip_deg"},
    "epi": set(),
    "epi_se": set(),
    "epi_label": {"seq_n_reps", "seq_n_navigator"},
    "database": set(),
}


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")


class MriGuiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MRI Forward Simulation")
        self.geometry("1280x820")
        self.minsize(1080, 720)

        self.output_queue: queue.Queue[tuple[str, str | dict]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.process: subprocess.Popen | None = None
        self.process_output_buffer = ""
        self.current_summary: dict | None = None

        self.sim_vars: dict[str, tk.Variable] = {}
        self.seq_specific_widgets: dict[str, list[tk.Widget]] = {}
        self.env_vars: dict[str, tk.StringVar] = {}

        self._configure_style()
        self._build_ui()
        self._poll_queue()
        self.refresh_phantom_database()
        self.refresh_sequence_database()
        self.refresh_history()
        self.load_hardware_config()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", padding=(8, 4))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", foreground="#555555")

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self._build_simulation_tab()
        self._build_phantom_database_tab()
        self._build_sequence_database_tab()
        self._build_hardware_tab()
        self._build_results_tab()

    def _build_simulation_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Simulation")

        paned = ttk.PanedWindow(tab, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ScrollFrame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        controls = left.inner
        row = 0
        row = self._section(controls, row, "Phantom")
        self._combo(controls, row, "Type", "phantom", PHANTOMS, "asymmetric", self._refresh_simulation_form)
        self._entry(controls, row, 2, "Database name", "phantom_name", "")
        row += 1
        self._entry(controls, row, 0, "nx", "nx", "64")
        self._entry(controls, row, 2, "ny", "ny", "64")
        row += 1
        self._entry(controls, row, 0, "nz", "nz", "1")
        self._entry(controls, row, 2, "fov-x", "fov_x", "")
        row += 1
        self._entry(controls, row, 0, "fov-y", "fov_y", "")
        self._entry(controls, row, 2, "slice-thickness", "slice_thickness", "")
        row += 1
        self._entry(controls, row, 0, "sphere radius", "radius", "16")
        self._entry(controls, row, 2, "ring inner", "inner_radius", "10")
        row += 1
        self._entry(controls, row, 0, "ring outer", "outer_radius", "20")
        row += 1

        row = self._section(controls, row, "Sequence")
        self._combo(controls, row, "Type", "sequence", SEQUENCES, "gre_label", self._refresh_simulation_form)
        self._entry(controls, row, 2, "Database name", "sequence_name", "")
        row += 1
        self._entry(controls, row, 0, "seq-nx", "seq_nx", "")
        self._entry(controls, row, 2, "seq-ny", "seq_ny", "")
        row += 1
        self._entry(controls, row, 0, "seq-n-slices", "seq_n_slices", "")
        self._entry(controls, row, 2, "seq-fov-x", "seq_fov_x", "")
        row += 1
        self._entry(controls, row, 0, "seq-fov-y", "seq_fov_y", "")
        self._entry(controls, row, 2, "seq-slice-thickness", "seq_slice_thickness", "")
        row += 1
        self._entry(controls, row, 0, "TR / seq-tr", "seq_tr", "0.1")
        self._entry(controls, row, 2, "TE / seq-te", "seq_te", "0.02")
        row += 1

        row = self._section(controls, row, "Sequence-specific")
        row = self._seq_entry(controls, row, "seq_flip_angle_deg", "flip angle deg", "seq_rf_spoiling_inc_deg", "RF spoiling inc deg")
        row = self._seq_entry(controls, row, "seq_dummy_scans", "dummy scans", "seq_readout_duration", "readout duration")
        row = self._seq_entry(controls, row, "seq_excitation_flip_angle_deg", "excitation flip deg", "seq_refocusing_flip_angle_deg", "refocusing flip deg")
        row = self._seq_entry(controls, row, "seq_rf_excitation_duration", "RF excitation duration", "seq_rf_refocusing_duration", "RF refocusing duration")
        row = self._seq_entry(controls, row, "seq_readout_time", "readout time", "seq_prephase_duration", "prephase duration")
        row = self._seq_entry(controls, row, "seq_n_echo", "TSE n echo", "seq_rf_flip_deg", "TSE RF flip deg")
        row = self._seq_entry(controls, row, "seq_n_reps", "EPI reps", "seq_n_navigator", "EPI navigators")
        self._combo(
            controls,
            row,
            "ideal spoiling reset",
            "seq_ideal_spoiling_reset",
            ("default", "true", "false"),
            "default",
            None,
        )
        self.seq_specific_widgets.setdefault("seq_ideal_spoiling_reset", []).extend(
            [controls.grid_slaves(row=row, column=0)[0], controls.grid_slaves(row=row, column=1)[0]]
        )
        row += 1

        row = self._section(controls, row, "Artifact and Output")
        self._check(controls, row, "RF artifact", "rf_artifact", False)
        self._entry(controls, row, 2, "RF noise freq", "rf_noise_freq", "127700000.0")
        row += 1
        self._entry(controls, row, 0, "RF noise amp", "rf_noise_amp", "5.0")
        self._entry(controls, row, 2, "background noise amp", "bg_noise_amp", "1.0")
        row += 1
        self._check(controls, row, "B0 artifact", "b0_artifact", False)
        self._combo(controls, row, "B0 mode", "b0_mode", ("linear", "parabolic"), "linear", None, start_col=2)
        row += 1
        self._entry(controls, row, 0, "B0 delta ppm", "b0_delta_ppm", "0.5")
        self._combo(controls, row, "B0 axis", "b0_axis", ("x", "y"), "x", None, start_col=2)
        row += 1
        self._entry(controls, row, 0, "fine dt", "fine_dt", "1e-5")
        self._entry(controls, row, 2, "seed", "seed", "")
        row += 1
        self._combo(controls, row, "CuPy mode", "cupy_mode", ("auto", "disabled"), "auto", None)
        row += 1
        self._entry(controls, row, 0, "output dir", "output_dir", str(OUTPUT_DIR))
        ttk.Button(controls, text="Browse", command=self.choose_output_dir).grid(row=row, column=2, sticky="ew", padx=4, pady=3)
        row += 1

        buttons = ttk.Frame(controls)
        buttons.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(10, 4))
        self.run_button = ttk.Button(buttons, text="Run Simulation", command=self.start_simulation)
        self.run_button.pack(side="left", padx=(0, 8))
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel_simulation, state="disabled")
        self.cancel_button.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var, style="Status.TLabel").grid(
            row=row + 1, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )

        self.log_text = tk.Text(right, height=10, wrap="word")
        self.log_text.pack(fill="x", padx=8, pady=(0, 8))
        progress_frame = ttk.Frame(right)
        progress_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.progress_label = ttk.Label(progress_frame, text="0/0 | 0.0% | 0.00 it/s", width=28)
        self.progress_label.pack(side="right", padx=(8, 0))
        self.figure_frame = ttk.Frame(right)
        self.figure_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.sim_canvas: FigureCanvasTkAgg | None = None
        self._refresh_simulation_form()

    def _build_phantom_database_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Phantom Database")

        left = ttk.Frame(tab)
        right = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        right.pack(side="right", fill="y", padx=8, pady=8)

        self.phantom_tree = self._tree(left, ("name", "description"), ("Name", "Description"))
        ttk.Button(left, text="Refresh", command=self.refresh_phantom_database).pack(anchor="w", pady=(6, 0))

        ttk.Label(right, text="Create Entry", style="Header.TLabel").pack(anchor="w")
        self.db_phantom_name = tk.StringVar()
        self.db_phantom_description = tk.StringVar()
        self._pack_labeled_entry(right, "Name", self.db_phantom_name)
        self._pack_labeled_entry(right, "Description", self.db_phantom_description)
        ttk.Button(right, text="Create", command=self.create_phantom_entry).pack(fill="x", pady=(4, 12))

        ttk.Label(right, text="Load Data File", style="Header.TLabel").pack(anchor="w")
        self.db_phantom_data = tk.StringVar(value="rho")
        self.db_phantom_file = tk.StringVar()
        ttk.Combobox(right, textvariable=self.db_phantom_data, values=PHANTOM_DATA_NAMES, state="readonly").pack(fill="x", pady=3)
        self._pack_labeled_entry(right, "File path", self.db_phantom_file)
        ttk.Button(right, text="Choose File", command=lambda: self._choose_file(self.db_phantom_file)).pack(fill="x", pady=3)
        ttk.Button(right, text="Import Data", command=self.import_phantom_data).pack(fill="x", pady=(4, 12))

        ttk.Label(right, text="Delete", style="Header.TLabel").pack(anchor="w")
        ttk.Button(right, text="Delete Selected Data", command=self.delete_phantom_data).pack(fill="x", pady=3)
        ttk.Button(right, text="Delete Selected Phantom", command=self.delete_phantom_entry).pack(fill="x", pady=3)

    def _build_sequence_database_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Sequence Database")

        left = ttk.Frame(tab)
        right = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        right.pack(side="right", fill="y", padx=8, pady=8)

        self.sequence_tree = self._tree(left, ("name", "description"), ("Name", "Description"))
        ttk.Button(left, text="Refresh", command=self.refresh_sequence_database).pack(anchor="w", pady=(6, 0))

        ttk.Label(right, text="Load .seq", style="Header.TLabel").pack(anchor="w")
        self.db_sequence_name = tk.StringVar()
        self.db_sequence_description = tk.StringVar()
        self.db_sequence_file = tk.StringVar()
        self._pack_labeled_entry(right, "Name", self.db_sequence_name)
        self._pack_labeled_entry(right, "Description", self.db_sequence_description)
        self._pack_labeled_entry(right, "File path", self.db_sequence_file)
        ttk.Button(right, text="Choose .seq", command=lambda: self._choose_file(self.db_sequence_file, [("Pulseq sequence", "*.seq")])).pack(fill="x", pady=3)
        ttk.Button(right, text="Load Sequence", command=self.load_sequence_entry).pack(fill="x", pady=(4, 12))

        ttk.Label(right, text="Delete", style="Header.TLabel").pack(anchor="w")
        ttk.Button(right, text="Delete Selected Sequence", command=self.delete_sequence_entry).pack(fill="x", pady=3)

    def _build_hardware_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Hardware Config")

        panel = ttk.Frame(tab)
        panel.pack(anchor="nw", fill="x", padx=14, pady=14)
        ttk.Label(panel, text="MRI_SYSTEM_* values in root .env", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        for row, (key, default) in enumerate(HARDWARE_DEFAULTS.items(), start=1):
            ttk.Label(panel, text=key, width=34).grid(row=row, column=0, sticky="w", padx=4, pady=4)
            var = tk.StringVar(value=default)
            self.env_vars[key] = var
            ttk.Entry(panel, textvariable=var, width=28).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
            ttk.Label(panel, text=f"default {default}", style="Status.TLabel").grid(row=row, column=2, sticky="w", padx=4)
        ttk.Button(panel, text="Reload", command=self.load_hardware_config).grid(row=len(HARDWARE_DEFAULTS) + 1, column=0, pady=12, sticky="w")
        ttk.Button(panel, text="Save to .env", command=self.save_hardware_config).grid(row=len(HARDWARE_DEFAULTS) + 1, column=1, pady=12, sticky="w")
        self.hardware_status = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.hardware_status, style="Status.TLabel").grid(
            row=len(HARDWARE_DEFAULTS) + 2, column=0, columnspan=3, sticky="w"
        )

    def _build_results_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Results")

        left = ttk.Frame(tab)
        right = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=8, pady=8)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        ttk.Label(left, text="Output History", style="Header.TLabel").pack(anchor="w")
        self.history_list = tk.Listbox(left, width=48, height=28)
        self.history_list.pack(fill="y", expand=True, pady=6)
        ttk.Button(left, text="Refresh", command=self.refresh_history).pack(fill="x", pady=3)
        ttk.Button(left, text="Load Selected", command=self.load_selected_history).pack(fill="x", pady=3)

        self.summary_text = tk.Text(right, height=12, wrap="none")
        self.summary_text.pack(fill="x", pady=(0, 8))
        self.history_figure_frame = ttk.Frame(right)
        self.history_figure_frame.pack(fill="both", expand=True)
        self.history_canvas: FigureCanvasTkAgg | None = None

    def _section(self, parent: ttk.Frame, row: int, title: str) -> int:
        ttk.Label(parent, text=title, style="Header.TLabel").grid(row=row, column=0, columnspan=4, sticky="w", pady=(12, 4))
        return row + 1

    def _entry(self, parent: ttk.Frame, row: int, start_col: int, label: str, name: str, default: str) -> None:
        var = tk.StringVar(value=default)
        self.sim_vars[name] = var
        label_widget = ttk.Label(parent, text=label)
        entry = ttk.Entry(parent, textvariable=var, width=18)
        label_widget.grid(row=row, column=start_col, sticky="w", padx=4, pady=3)
        entry.grid(row=row, column=start_col + 1, sticky="ew", padx=4, pady=3)
        if name.startswith("seq_") and name not in {"seq_nx", "seq_ny", "seq_n_slices", "seq_fov_x", "seq_fov_y", "seq_slice_thickness", "seq_tr", "seq_te"}:
            self.seq_specific_widgets.setdefault(name, []).extend([label_widget, entry])

    def _seq_entry(self, parent: ttk.Frame, row: int, name1: str, label1: str, name2: str, label2: str) -> int:
        self._entry(parent, row, 0, label1, name1, "")
        self._entry(parent, row, 2, label2, name2, "")
        return row + 1

    def _combo(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        name: str,
        values: tuple[str, ...],
        default: str,
        callback,
        start_col: int = 0,
    ) -> None:
        var = tk.StringVar(value=default)
        self.sim_vars[name] = var
        label_widget = ttk.Label(parent, text=label)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=16)
        label_widget.grid(row=row, column=start_col, sticky="w", padx=4, pady=3)
        combo.grid(row=row, column=start_col + 1, sticky="ew", padx=4, pady=3)
        if callback is not None:
            combo.bind("<<ComboboxSelected>>", lambda _event: callback())

    def _check(self, parent: ttk.Frame, row: int, label: str, name: str, default: bool) -> None:
        var = tk.BooleanVar(value=default)
        self.sim_vars[name] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=3)

    def _tree(self, parent: ttk.Frame, columns: tuple[str, ...], headings: tuple[str, ...]) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=24)
        for column, heading in zip(columns, headings):
            tree.heading(column, text=heading)
            tree.column(column, width=180 if column == "name" else 520, anchor="w")
        tree.pack(fill="both", expand=True)
        return tree

    def _pack_labeled_entry(self, parent: ttk.Frame, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=var, width=42).pack(fill="x", pady=3)

    def _refresh_simulation_form(self) -> None:
        sequence = self.sim_vars.get("sequence", tk.StringVar(value="gre_label")).get()
        enabled = SEQUENCE_SPECIFIC_FIELDS.get(sequence, set())
        for name, widgets in self.seq_specific_widgets.items():
            state = "normal" if name in enabled else "disabled"
            if name == "seq_ideal_spoiling_reset":
                state = "readonly" if name in enabled else "disabled"
            for widget in widgets:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(ROOT_DIR))
        if path:
            self.sim_vars["output_dir"].set(path)

    def _choose_file(self, var: tk.StringVar, filetypes=None) -> None:
        path = filedialog.askopenfilename(initialdir=str(ROOT_DIR), filetypes=filetypes or [("All files", "*.*")])
        if path:
            var.set(path)

    def start_simulation(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            command = self._build_simulation_command()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.log_text.delete("1.0", "end")
        self.process_output_buffer = ""
        self.progress_var.set(0.0)
        self.progress_label.configure(text="0/0 | 0.0% | 0.00 it/s")
        self.status_var.set("Running simulation...")
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.worker = threading.Thread(target=self._run_subprocess, args=(command,), daemon=True)
        self.worker.start()

    def cancel_simulation(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status_var.set("Cancelling...")

    def _run_subprocess(self, command: list[str]) -> None:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            assert self.process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while True:
                chunk = self.process.stdout.read(1024)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if text:
                    self.output_queue.put(("log", text))
            remainder = decoder.decode(b"", final=True)
            if remainder:
                self.output_queue.put(("log", remainder))
            return_code = self.process.wait()
            if return_code == 0:
                self.output_queue.put(("done", {"status": "success", "output_dir": self.sim_vars["output_dir"].get()}))
            else:
                self.output_queue.put(("done", {"status": "error", "return_code": return_code}))
        except Exception as exc:
            self.output_queue.put(("done", {"status": "error", "message": str(exc)}))
        finally:
            self.process = None

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "log":
                    self._handle_process_output(str(payload))
                elif kind == "done":
                    self._flush_process_output_buffer()
                    self._handle_simulation_done(payload if isinstance(payload, dict) else {})
        except queue.Empty:
            pass
        self.after(200, self._poll_queue)

    def _handle_process_output(self, text: str) -> None:
        self.process_output_buffer += text.replace("\r", "\n")
        while "\n" in self.process_output_buffer:
            line, self.process_output_buffer = self.process_output_buffer.split("\n", 1)
            content = line.rstrip("\r")
            if content.startswith(PROGRESS_PREFIX):
                self._update_progress_from_line(content)
            elif content:
                self._append_log(f"{content}\n")

    def _flush_process_output_buffer(self) -> None:
        content = self.process_output_buffer.strip("\r\n")
        self.process_output_buffer = ""
        if not content:
            return
        if content.startswith(PROGRESS_PREFIX):
            self._update_progress_from_line(content)
        else:
            self._append_log(f"{content}\n")

    def _update_progress_from_line(self, line: str) -> None:
        try:
            payload = json.loads(line[len(PROGRESS_PREFIX):])
            current = int(payload.get("current", 0))
            total = int(payload.get("total", 0))
            percent = float(payload.get("percent", 0.0))
            rate = float(payload.get("rate", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._append_log(f"{line}\n")
            return
        self.progress_var.set(max(0.0, min(100.0, percent)))
        self.progress_label.configure(text=f"{current}/{total} | {percent:.1f}% | {rate:.2f} it/s")

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _handle_simulation_done(self, payload: dict) -> None:
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if payload.get("status") == "success":
            self.status_var.set("Simulation completed.")
            self.progress_var.set(100.0)
            summary_path = Path(str(payload["output_dir"])) / "summary.json"
            if summary_path.exists():
                self.load_result_summary(summary_path, target="simulation")
            self.refresh_history()
        else:
            self.status_var.set(f"Simulation failed or cancelled: {payload}")

    def _build_simulation_command(self) -> list[str]:
        args = [sys.executable, "-u", str(MAIN_PATH), "simulate"]
        self._append_choice(args, "--phantom", "phantom")
        if self.sim_vars["phantom"].get() == "database":
            self._append_required(args, "--phantom-name", "phantom_name")
        self._append_choice(args, "--sequence", "sequence")
        if self.sim_vars["sequence"].get() == "database":
            self._append_required(args, "--sequence-name", "sequence_name")

        for option, key in [
            ("--nx", "nx"),
            ("--ny", "ny"),
            ("--nz", "nz"),
            ("--fov-x", "fov_x"),
            ("--fov-y", "fov_y"),
            ("--slice-thickness", "slice_thickness"),
            ("--seq-nx", "seq_nx"),
            ("--seq-ny", "seq_ny"),
            ("--seq-n-slices", "seq_n_slices"),
            ("--seq-fov-x", "seq_fov_x"),
            ("--seq-fov-y", "seq_fov_y"),
            ("--seq-slice-thickness", "seq_slice_thickness"),
            ("--seq-tr", "seq_tr"),
            ("--seq-te", "seq_te"),
            ("--seq-flip-angle-deg", "seq_flip_angle_deg"),
            ("--seq-rf-spoiling-inc-deg", "seq_rf_spoiling_inc_deg"),
            ("--seq-dummy-scans", "seq_dummy_scans"),
            ("--seq-readout-duration", "seq_readout_duration"),
            ("--seq-excitation-flip-angle-deg", "seq_excitation_flip_angle_deg"),
            ("--seq-refocusing-flip-angle-deg", "seq_refocusing_flip_angle_deg"),
            ("--seq-rf-excitation-duration", "seq_rf_excitation_duration"),
            ("--seq-rf-refocusing-duration", "seq_rf_refocusing_duration"),
            ("--seq-readout-time", "seq_readout_time"),
            ("--seq-prephase-duration", "seq_prephase_duration"),
            ("--seq-n-echo", "seq_n_echo"),
            ("--seq-rf-flip-deg", "seq_rf_flip_deg"),
            ("--seq-n-reps", "seq_n_reps"),
            ("--seq-n-navigator", "seq_n_navigator"),
            ("--fine-dt", "fine_dt"),
            ("--radius", "radius"),
            ("--inner-radius", "inner_radius"),
            ("--outer-radius", "outer_radius"),
            ("--rf-noise-freq", "rf_noise_freq"),
            ("--rf-noise-amp", "rf_noise_amp"),
            ("--bg-noise-amp", "bg_noise_amp"),
            ("--b0-mode", "b0_mode"),
            ("--b0-delta-ppm", "b0_delta_ppm"),
            ("--b0-axis", "b0_axis"),
            ("--output-dir", "output_dir"),
            ("--seed", "seed"),
        ]:
            self._append_optional(args, option, key)

        reset = self.sim_vars["seq_ideal_spoiling_reset"].get()
        if reset == "true":
            args.append("--seq-ideal-spoiling-reset")
        elif reset == "false":
            args.append("--no-seq-ideal-spoiling-reset")
        if bool(self.sim_vars["rf_artifact"].get()):
            args.append("--rf-artifact")
        if bool(self.sim_vars["b0_artifact"].get()):
            args.append("--b0-artifact")
        args.extend(["--cupy-mode", str(self.sim_vars["cupy_mode"].get())])
        args.append("--progress-json")
        return args

    def _append_choice(self, args: list[str], option: str, key: str) -> None:
        args.extend([option, str(self.sim_vars[key].get())])

    def _append_required(self, args: list[str], option: str, key: str) -> None:
        value = str(self.sim_vars[key].get()).strip()
        if not value:
            raise ValueError(f"{option} is required.")
        args.extend([option, value])

    def _append_optional(self, args: list[str], option: str, key: str) -> None:
        value = str(self.sim_vars[key].get()).strip()
        if value:
            args.extend([option, value])

    def refresh_phantom_database(self) -> None:
        self._clear_tree(self.phantom_tree)
        try:
            for item in list_phantom_database():
                self.phantom_tree.insert("", "end", values=(item["name"], item["description"]))
        except Exception as exc:
            messagebox.showerror("Phantom database", str(exc))

    def create_phantom_entry(self) -> None:
        self._run_db_action(
            lambda: create_phantom_database_entry(self.db_phantom_name.get(), self.db_phantom_description.get()),
            self.refresh_phantom_database,
        )

    def import_phantom_data(self) -> None:
        name = self._selected_name(self.phantom_tree) or self.db_phantom_name.get()
        self._run_db_action(
            lambda: import_phantom_database_file(name, self.db_phantom_file.get(), self.db_phantom_data.get()),
            self.refresh_phantom_database,
        )

    def delete_phantom_data(self) -> None:
        name = self._selected_name(self.phantom_tree)
        if not name:
            messagebox.showwarning("Delete data", "Select a phantom first.")
            return
        data_name = self.db_phantom_data.get()
        if messagebox.askyesno("Delete data", f"Delete {data_name} from {name}?"):
            self._run_db_action(lambda: delete_phantom_database_data(name, data_name), self.refresh_phantom_database)

    def delete_phantom_entry(self) -> None:
        name = self._selected_name(self.phantom_tree)
        if not name:
            messagebox.showwarning("Delete phantom", "Select a phantom first.")
            return
        if messagebox.askyesno("Delete phantom", f"Delete entire phantom '{name}'?"):
            self._run_db_action(lambda: delete_phantom_database_entry(name), self.refresh_phantom_database)

    def refresh_sequence_database(self) -> None:
        self._clear_tree(self.sequence_tree)
        try:
            for item in list_sequence_database():
                self.sequence_tree.insert("", "end", values=(item["name"], item["description"]))
        except Exception as exc:
            messagebox.showerror("Sequence database", str(exc))

    def load_sequence_entry(self) -> None:
        self._run_db_action(
            lambda: load_sequence_database_file(
                self.db_sequence_name.get(),
                self.db_sequence_description.get(),
                self.db_sequence_file.get(),
            ),
            self.refresh_sequence_database,
        )

    def delete_sequence_entry(self) -> None:
        name = self._selected_name(self.sequence_tree)
        if not name:
            messagebox.showwarning("Delete sequence", "Select a sequence first.")
            return
        if messagebox.askyesno("Delete sequence", f"Delete sequence '{name}'?"):
            self._run_db_action(lambda: delete_sequence_database_entry(name), self.refresh_sequence_database)

    def _run_db_action(self, action, refresh) -> None:
        try:
            result = action()
            refresh()
            messagebox.showinfo("Success", json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _selected_name(self, tree: ttk.Treeview) -> str | None:
        selection = tree.selection()
        if not selection:
            return None
        values = tree.item(selection[0], "values")
        return str(values[0]) if values else None

    def load_hardware_config(self) -> None:
        values = HARDWARE_DEFAULTS.copy()
        if ENV_PATH.exists():
            for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                if "=" not in raw_line or raw_line.strip().startswith("#"):
                    continue
                key, value = raw_line.split("=", 1)
                key = key.strip()
                if key in values:
                    values[key] = value.strip()
        for key, value in values.items():
            self.env_vars[key].set(value)
        self.hardware_status.set(f"Loaded from {ENV_PATH}")

    def save_hardware_config(self) -> None:
        try:
            for key, var in self.env_vars.items():
                value = var.get().strip()
                if key not in {"MRI_SYSTEM_GRAD_UNIT", "MRI_SYSTEM_SLEW_UNIT"}:
                    float(value)
            existing = []
            if ENV_PATH.exists():
                existing = ENV_PATH.read_text(encoding="utf-8").splitlines()
            retained = [
                line
                for line in existing
                if not any(line.startswith(f"{key}=") for key in HARDWARE_DEFAULTS)
            ]
            additions = [f"{key}={self.env_vars[key].get().strip()}" for key in HARDWARE_DEFAULTS]
            ENV_PATH.write_text("\n".join(retained + additions) + "\n", encoding="utf-8")
            self.hardware_status.set(f"Saved to {ENV_PATH}")
        except Exception as exc:
            messagebox.showerror("Hardware config", str(exc))

    def refresh_history(self) -> None:
        self.history_list.delete(0, "end")
        if not OUTPUT_DIR.exists():
            return
        for summary_path in sorted(OUTPUT_DIR.rglob("summary.json"), key=lambda path: path.stat().st_mtime, reverse=True):
            self.history_list.insert("end", str(summary_path.parent))

    def load_selected_history(self) -> None:
        selection = self.history_list.curselection()
        if not selection:
            messagebox.showwarning("Results", "Select an output directory first.")
            return
        summary_path = Path(self.history_list.get(selection[0])) / "summary.json"
        self.load_result_summary(summary_path, target="history")

    def load_result_summary(self, summary_path: Path, target: str) -> None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.current_summary = summary
            self._display_summary(summary, target)
            self._display_result_figure(summary_path.parent, target)
        except Exception as exc:
            messagebox.showerror("Results", str(exc))

    def _display_summary(self, summary: dict, target: str) -> None:
        text = self.summary_text if target == "history" else self.log_text
        if target == "history":
            text.delete("1.0", "end")
        text.insert("end", "\nSummary\n")
        text.insert("end", json.dumps(summary, ensure_ascii=False, indent=2))
        text.insert("end", "\n")
        text.see("end")

    def _display_result_figure(self, output_dir: Path, target: str) -> None:
        frame = self.history_figure_frame if target == "history" else self.figure_frame
        canvas_attr = "history_canvas" if target == "history" else "sim_canvas"
        old_canvas = getattr(self, canvas_attr)
        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(9, 6), dpi=100)
        comparison_png = output_dir / "reconstruction.png"

        if target == "simulation":
            ax = fig.subplots(1, 1)
            if comparison_png.exists():
                self._plot_image_file(ax, comparison_png, "Reconstruction / Phantom")
            else:
                self._plot_array_file(ax, output_dir / "reconstruction_magnitude.npy", "Reconstruction")
        else:
            axes = fig.subplots(1, 3)
            if comparison_png.exists():
                self._plot_image_file(axes[0], comparison_png, "Reconstruction / Phantom")
            else:
                self._plot_array_file(axes[0], output_dir / "reconstruction_magnitude.npy", "Reconstruction")
            self._plot_kspace(axes[1], output_dir / "kspace.npy")
            artifact_path = output_dir / "reconstruction_rf_artifact_magnitude.npy"
            if artifact_path.exists():
                self._plot_array_file(axes[2], artifact_path, "RF Artifact")
            else:
                axes[2].axis("off")
                axes[2].set_title("No RF artifact")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, canvas_attr, canvas)

    def _plot_image_file(self, ax, path: Path, title: str) -> None:
        ax.set_title(title)
        ax.axis("off")
        if not path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            return
        ax.imshow(mpimg.imread(path))

    def _plot_array_file(self, ax, path: Path, title: str) -> None:
        ax.set_title(title)
        ax.axis("off")
        if not path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            return
        data = self._to_2d(np.load(path))
        ax.imshow(data, cmap="gray")

    def _plot_kspace(self, ax, path: Path) -> None:
        ax.set_title("k-space magnitude")
        if not path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            return
        data = np.abs(np.asarray(np.load(path)).reshape(-1))
        ax.plot(data)
        ax.set_xlabel("Sample")

    def _to_2d(self, data) -> np.ndarray:
        array = np.abs(np.asarray(data))
        array = np.squeeze(array)
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return array[0]
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array.reshape(array.shape[-2], array.shape[-1])


def main() -> None:
    app = MriGuiApp()
    app.mainloop()
