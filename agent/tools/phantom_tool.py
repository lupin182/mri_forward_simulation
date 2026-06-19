"""Phantom generation tool and in-memory phantom cache."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.tools.base_tool import MRISimulationBaseTool
from mri_sim.phantom import (
    Phantom,
    generate_simple_asymmetric_phantom,
    generate_simple_ring_phantom,
    generate_simple_sphere_phantom,
)


phantom_cache: tuple[Phantom, Any, Any, Any] | None = None
_phantom_figure_cache = None


class GeneratePhantomTool(MRISimulationBaseTool):
    name = "generate_phantom"
    description = (
        "Generate an MRI phantom. JSON params: phantom_type "
        "(asymmetric, ring, sphere), Nz, Nx, Ny, fov_x, fov_y, slice_thickness, "
        "radius, inner_radius, outer_radius, return_figure."
    )

    def _run(self, query: str) -> str:
        params = json.loads(query or "{}")
        phantom_type = params.get("phantom_type", "asymmetric")
        nz = int(params.get("Nz", params.get("nz", 1)))
        nx = int(params.get("Nx", params.get("nx", 64)))
        ny = int(params.get("Ny", params.get("ny", 64)))
        fov_x = float(params.get("fov_x", 0.256))
        fov_y = float(params.get("fov_y", 0.256))
        slice_thickness = float(params.get("slice_thickness", 0.004))
        return_figure = bool(params.get("return_figure", True))

        if phantom_type == "asymmetric":
            rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=nz, Nx=nx, Ny=ny)
        elif phantom_type == "ring":
            rho, t1, t2 = generate_simple_ring_phantom(
                Nz=nz,
                Nx=nx,
                Ny=ny,
                inner_radius=int(params.get("inner_radius", 10)),
                outer_radius=int(params.get("outer_radius", 20)),
            )
        elif phantom_type == "sphere":
            rho, t1, t2 = generate_simple_sphere_phantom(
                Nz=nz,
                Nx=nx,
                Ny=ny,
                radius=int(params.get("radius", 16)),
            )
        else:
            return json.dumps({"status": "error", "message": f"Unknown phantom_type: {phantom_type}"})

        phantom = Phantom(rho, t1, t2, fov_x=fov_x, fov_y=fov_y, slice_thickness=slice_thickness)
        set_cached_phantom(phantom, rho, t1, t2)

        if return_figure:
            _set_phantom_figure(rho, t1, t2)

        return json.dumps(
            {
                "status": "success",
                "phantom_type": phantom_type,
                "shape": [phantom.Nz, phantom.Nx, phantom.Ny],
                "fov": [fov_x, fov_y],
                "slice_thickness": slice_thickness,
            },
            ensure_ascii=False,
        )


def _set_phantom_figure(rho, t1, t2) -> None:
    global _phantom_figure_cache
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, title, data, cmap in [
        (axes[0], "Proton Density", rho[0, 0, 0], "gray"),
        (axes[1], "T1", t1[0, 0, 0], "viridis"),
        (axes[2], "T2", t2[0, 0, 0], "plasma"),
    ]:
        ax.set_title(title)
        ax.imshow(data, cmap=cmap)
        ax.axis("off")
    fig.tight_layout()
    _phantom_figure_cache = fig


def get_cached_phantom():
    return phantom_cache


def get_cached_phantom_figure():
    return _phantom_figure_cache


def set_cached_phantom(phantom, rho, t1, t2) -> None:
    global phantom_cache
    phantom_cache = (phantom, rho, t1, t2)


def clear_phantom_cache() -> None:
    global phantom_cache, _phantom_figure_cache
    phantom_cache = None
    _phantom_figure_cache = None
