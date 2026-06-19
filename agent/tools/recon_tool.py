"""Image reconstruction tool and in-memory reconstruction cache."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.tools.base_tool import MRISimulationBaseTool
from agent.tools.phantom_tool import get_cached_phantom
from agent.tools.simulation_tool import get_cached_kspace, get_cached_seq
from mri_sim.reconstruction import reconstruct_3d_cartesian_fft_multichannel, sos_reconstruction


_image_cache = None
_figure_cache = None


class ReconstructImageTool(MRISimulationBaseTool):
    name = "reconstruct_image"
    description = "Reconstruct MRI image after simulation. JSON params: output_path, return_figure."

    def _run(self, query: str) -> str:
        global _image_cache, _figure_cache

        cached_phantom = get_cached_phantom()
        if cached_phantom is None:
            return json.dumps({"status": "error", "message": "Generate or load a phantom first."})

        kspace = get_cached_kspace()
        seq = get_cached_seq()
        if kspace is None or seq is None:
            return json.dumps({"status": "error", "message": "Run simulation first."})

        phantom, rho, _, _ = cached_phantom
        params = json.loads(query or "{}")
        return_figure = bool(params.get("return_figure", True))
        output_path = Path(params.get("output_path", "output/mri_result.png"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        k_traj_adc, _, _, _, _ = seq.calculate_kspace()
        signal = np.asarray(kspace)
        recon_input = signal.T if signal.ndim == 2 else signal.squeeze()
        coil_images, _ = reconstruct_3d_cartesian_fft_multichannel(
            recon_input,
            k_traj_adc,
            Ny=phantom.Ny,
            Nx=phantom.Nx,
            Nz=phantom.Nz,
        )
        image = sos_reconstruction(coil_images) if coil_images.ndim == 4 else coil_images
        _image_cache = image

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_title("Reconstruction")
        axes[0].imshow(_to_2d_magnitude(image), cmap="gray")
        axes[0].axis("off")
        axes[1].set_title("Original Phantom")
        axes[1].imshow(rho[0, 0, 0], cmap="gray")
        axes[1].axis("off")
        fig.tight_layout()

        if return_figure:
            _figure_cache = fig
        else:
            fig.savefig(output_path)
            plt.close(fig)

        return json.dumps(
            {
                "status": "success",
                "phantom_shape": [phantom.Nz, phantom.Nx, phantom.Ny],
                "image_shape": list(np.asarray(image).shape),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
        )


def get_cached_image():
    return _image_cache


def get_cached_figure():
    return _figure_cache


def clear_recon_cache() -> None:
    global _image_cache, _figure_cache
    _image_cache = None
    _figure_cache = None


def _to_2d_magnitude(image) -> np.ndarray:
    data = np.abs(np.asarray(image))
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[0]
    raise ValueError(f"Expected 2D or 3D image data, got shape {data.shape}.")
