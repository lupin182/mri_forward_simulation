"""Simulation tool and in-memory k-space cache."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.tools.base_tool import MRISimulationBaseTool
from agent.tools.phantom_tool import get_cached_phantom
from mri_sim.sequences import get_sequence
from mri_sim.simulation import SimulationConfig, simulate


_kspace_cache = None
_seq_cache = None
_kspace_figure_cache = None


class RunSimulationTool(MRISimulationBaseTool):
    name = "run_simulation"
    description = (
        "Run MRI forward simulation after a phantom exists. JSON params: "
        "sequence_type, tr, te, fine_dt, return_figure."
    )

    def _run(self, query: str) -> str:
        global _kspace_cache, _seq_cache

        cached = get_cached_phantom()
        if cached is None:
            return json.dumps({"status": "error", "message": "Generate or load a phantom first."})

        phantom, _, _, _ = cached
        params = json.loads(query or "{}")
        sequence_type = params.get("sequence_type", "gre_label")
        fine_dt = float(params.get("fine_dt", 1e-5))
        return_figure = bool(params.get("return_figure", True))

        sequence_kwargs = {
            "fov": (phantom.fov_x, phantom.fov_y),
            "n_x": phantom.Nx,
            "n_y": phantom.Ny,
            "slice_thickness": phantom.slice_thickness,
        }
        if sequence_type in {"gre_label", "se"}:
            sequence_kwargs["n_slices"] = phantom.Nz
        if sequence_type in {"gre", "gre_label", "se", "epi_se"}:
            sequence_kwargs["te"] = float(params.get("te", 0.02))
        if sequence_type in {"gre", "gre_label", "se"}:
            sequence_kwargs["tr"] = float(params.get("tr", 0.1))
        if sequence_type in {"epi", "epi_label"}:
            sequence_kwargs["n_slices"] = phantom.Nz

        try:
            seq = get_sequence(sequence_type, **sequence_kwargs)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)

        kspace = simulate(phantom, seq, SimulationConfig(fine_dt=fine_dt))
        _kspace_cache = kspace
        _seq_cache = seq

        if return_figure:
            _set_kspace_figure(kspace, phantom.Nx, phantom.Ny)

        return json.dumps(
            {
                "status": "success",
                "sequence_type": sequence_type,
                "phantom_shape": [phantom.Nz, phantom.Nx, phantom.Ny],
                "k_space_shape": list(np.asarray(kspace).shape),
            },
            ensure_ascii=False,
        )


def _set_kspace_figure(kspace, nx: int, ny: int) -> None:
    global _kspace_figure_cache
    magnitude = np.abs(np.asarray(kspace).squeeze())
    total_len = nx * ny

    if magnitude.size >= total_len:
        kspace_2d = magnitude.reshape(-1)[:total_len].reshape(ny, nx)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].set_title("k-space Magnitude")
        axes[0].imshow(kspace_2d, cmap="gray")
        axes[0].axis("off")
        axes[1].set_title("k-space Log Magnitude")
        axes[1].imshow(np.log1p(kspace_2d), cmap="gray")
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title("k-space Signal")
        ax.plot(magnitude.reshape(-1))
        ax.set_xlabel("Sample")
        ax.set_ylabel("Magnitude")
    fig.tight_layout()
    _kspace_figure_cache = fig


def get_cached_kspace():
    return _kspace_cache


def get_cached_seq():
    return _seq_cache


def get_cached_kspace_figure():
    return _kspace_figure_cache


def clear_simulation_cache() -> None:
    global _kspace_cache, _seq_cache, _kspace_figure_cache
    _kspace_cache = None
    _seq_cache = None
    _kspace_figure_cache = None
