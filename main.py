from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mri_sim.generate_artifact import generate_B0_inhomogeneity, generate_rf_artifact_real
from mri_sim.phantom import (
    Phantom,
    generate_simple_asymmetric_phantom,
    generate_simple_ring_phantom,
    generate_simple_sphere_phantom,
)
from mri_sim.phantom_database import load_phantom_dataset
from mri_sim.phantom_database import (
    create_phantom_database_entry,
    delete_phantom_database_data,
    delete_phantom_database_entry,
    import_phantom_database_file,
    list_phantom_database,
)
from mri_sim.reconstruction import reconstruct_3d_cartesian_fft_multichannel, sos_reconstruction
from mri_sim.sequences import get_sequence
from mri_sim.simulation import SimulationConfig, simulate


DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_FOV_X = 0.256
DEFAULT_FOV_Y = 0.256
DEFAULT_SLICE_THICKNESS = 0.004


def _parse_csv_floats(value: str) -> list[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _build_phantom(args: argparse.Namespace) -> tuple[Phantom, np.ndarray, np.ndarray, np.ndarray]:
    if args.phantom == "asymmetric":
        rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=args.nz, Nx=args.nx, Ny=args.ny)
    elif args.phantom == "sphere":
        rho, t1, t2 = generate_simple_sphere_phantom(Nz=args.nz, Nx=args.nx, Ny=args.ny, radius=args.radius)
    elif args.phantom == "ring":
        rho, t1, t2 = generate_simple_ring_phantom(
            Nz=args.nz,
            Nx=args.nx,
            Ny=args.ny,
            inner_radius=args.inner_radius,
            outer_radius=args.outer_radius,
        )
    elif args.phantom == "database":
        if not args.phantom_name:
            raise ValueError("--phantom-name is required when --phantom database is used.")
        dataset = load_phantom_dataset(args.phantom_name)
        phantom = dataset.build_phantom(
            fov_x=args.fov_x,
            fov_y=args.fov_y,
            slice_thickness=args.slice_thickness,
        )
        if args.b0_artifact:
            generate_B0_inhomogeneity(
                phantom,
                mode=args.b0_mode,
                delta_B0_ppm=args.b0_delta_ppm,
                axis=args.b0_axis,
            )
        return phantom, dataset.rho, dataset.t1, dataset.t2
    else:
        raise ValueError(f"Unsupported phantom type: {args.phantom}")

    phantom = Phantom(
        rho,
        t1,
        t2,
        fov_x=args.fov_x if args.fov_x is not None else DEFAULT_FOV_X,
        fov_y=args.fov_y if args.fov_y is not None else DEFAULT_FOV_Y,
        slice_thickness=args.slice_thickness if args.slice_thickness is not None else DEFAULT_SLICE_THICKNESS,
    )
    if args.b0_artifact:
        generate_B0_inhomogeneity(
            phantom,
            mode=args.b0_mode,
            delta_B0_ppm=args.b0_delta_ppm,
            axis=args.b0_axis,
        )
    return phantom, rho, t1, t2


def _build_sequence(args: argparse.Namespace, phantom: Phantom):
    base_kwargs: dict[str, Any] = {
        "fov": (phantom.fov_x, phantom.fov_y),
        "n_x": phantom.Nx,
        "n_y": phantom.Ny,
        "slice_thickness": phantom.slice_thickness,
    }
    if args.sequence == "gre":
        base_kwargs.update({"tr": args.tr, "te": args.te})
    elif args.sequence == "gre_label":
        base_kwargs.update({"n_slices": phantom.Nz, "tr": args.tr, "te": args.te})
    elif args.sequence == "se":
        base_kwargs.update({"n_slices": phantom.Nz, "tr": args.tr, "te": args.te})
    elif args.sequence == "epi":
        base_kwargs.update({"n_slices": phantom.Nz})
    elif args.sequence == "epi_se":
        base_kwargs.update({"te": args.te})
    elif args.sequence == "epi_label":
        base_kwargs.update({"n_slices": phantom.Nz})
    return get_sequence(args.sequence, **base_kwargs)


def _as_reconstruction_input(kspace: np.ndarray) -> np.ndarray:
    signal = np.asarray(kspace)
    if signal.ndim == 2:
        return signal.T
    return signal.squeeze()


def _reconstruct(kspace: np.ndarray, k_traj_adc: np.ndarray, phantom: Phantom) -> tuple[np.ndarray, np.ndarray]:
    coil_images, k_grid = reconstruct_3d_cartesian_fft_multichannel(
        _as_reconstruction_input(kspace),
        k_traj_adc,
        Ny=phantom.Ny,
        Nx=phantom.Nx,
        Nz=phantom.Nz,
    )
    if coil_images.ndim == 4:
        image = sos_reconstruction(coil_images)
    else:
        image = coil_images
    return image, k_grid


def _add_rf_artifact(args: argparse.Namespace, t_adc: np.ndarray, kspace: np.ndarray) -> np.ndarray:
    freqs = _parse_csv_floats(args.rf_noise_freq)
    amps = _parse_csv_floats(args.rf_noise_amp)
    if len(freqs) != len(amps):
        raise ValueError("--rf-noise-freq and --rf-noise-amp must contain the same number of values.")

    signal = np.asarray(kspace).squeeze()
    if signal.ndim == 1:
        return generate_rf_artifact_real(
            t_adc,
            signal,
            rf_noise_freq=freqs,
            rf_noise_amp=amps,
            bg_noise_amp=args.bg_noise_amp,
        )

    artifact_channels = [
        generate_rf_artifact_real(
            t_adc,
            signal[:, coil_idx],
            rf_noise_freq=freqs,
            rf_noise_amp=amps,
            bg_noise_amp=args.bg_noise_amp,
        )
        for coil_idx in range(signal.shape[1])
    ]
    return np.stack(artifact_channels, axis=1)


def _save_png(path: Path, image: np.ndarray, reference: np.ndarray, title: str) -> None:
    image_2d = _to_2d_magnitude(image)
    reference_2d = _to_2d_magnitude(reference)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title(title)
    axes[0].imshow(image_2d, cmap="gray")
    axes[0].axis("off")
    axes[1].set_title("Phantom Rho")
    axes[1].imshow(reference_2d, cmap="gray")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _reference_rho(rho: np.ndarray) -> np.ndarray:
    data = np.asarray(rho)
    if data.ndim == 5:
        return data[0, 0]
    if data.ndim in {2, 3}:
        return data
    raise ValueError(f"Expected rho shape (Nz, Nx, Ny) or (TypeNum, SpinNum, Nz, Nx, Ny), got {data.shape}.")


def _to_2d_magnitude(image: np.ndarray) -> np.ndarray:
    data = np.abs(np.asarray(image))
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[0]
    raise ValueError(f"Expected 2D or 3D image data, got shape {data.shape}.")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phantom, rho, _, _ = _build_phantom(args)
    sequence = _build_sequence(args, phantom)

    kspace = simulate(phantom, sequence, SimulationConfig(fine_dt=args.fine_dt))
    k_traj_adc, _, _, _, _ = sequence.calculate_kspace()
    image, _ = _reconstruct(kspace, k_traj_adc, phantom)

    np.save(output_dir / "kspace.npy", kspace)
    np.save(output_dir / "reconstruction.npy", image)
    np.save(output_dir / "reconstruction_magnitude.npy", np.abs(image))
    if not args.no_plot:
        _save_png(output_dir / "reconstruction.png", image, _reference_rho(rho), "Reconstruction")

    summary: dict[str, Any] = {
        "phantom": args.phantom,
        "phantom_name": args.phantom_name,
        "sequence": args.sequence,
        "shape": {"nz": phantom.Nz, "nx": phantom.Nx, "ny": phantom.Ny},
        "kspace_shape": list(np.asarray(kspace).shape),
        "reconstruction_shape": list(np.asarray(image).shape),
        "output_dir": str(output_dir.resolve()),
        "rf_artifact": False,
        "b0_artifact": bool(args.b0_artifact),
    }

    if args.rf_artifact:
        _, _, _, t_adc, _ = sequence.waveforms_and_times()
        artifact_kspace = _add_rf_artifact(args, np.asarray(t_adc), np.asarray(kspace))
        artifact_image, _ = _reconstruct(artifact_kspace, k_traj_adc, phantom)
        np.save(output_dir / "kspace_rf_artifact.npy", artifact_kspace)
        np.save(output_dir / "reconstruction_rf_artifact.npy", artifact_image)
        np.save(output_dir / "reconstruction_rf_artifact_magnitude.npy", np.abs(artifact_image))
        if not args.no_plot:
            _save_png(output_dir / "reconstruction_rf_artifact.png", artifact_image, _reference_rho(rho), "RF Artifact")
        summary["rf_artifact"] = True
        summary["artifact_kspace_shape"] = list(np.asarray(artifact_kspace).shape)
        summary["artifact_reconstruction_shape"] = list(np.asarray(artifact_image).shape)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return summary


def build_simulation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MRI forward simulation pipeline without a UI.")
    parser.add_argument("--phantom", choices=["asymmetric", "sphere", "ring", "database"], default="asymmetric")
    parser.add_argument("--phantom-name", default=None)
    parser.add_argument("--sequence", choices=["gre", "gre_label", "se", "epi", "epi_se", "epi_label"], default="gre_label")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--fov-y", type=float, default=None)
    parser.add_argument("--slice-thickness", type=float, default=None)
    parser.add_argument("--tr", type=float, default=0.1)
    parser.add_argument("--te", type=float, default=0.02)
    parser.add_argument("--fine-dt", type=float, default=1e-5)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--inner-radius", type=int, default=10)
    parser.add_argument("--outer-radius", type=int, default=20)
    parser.add_argument("--rf-artifact", action="store_true")
    parser.add_argument("--rf-noise-freq", default="127700000.0")
    parser.add_argument("--rf-noise-amp", default="5.0")
    parser.add_argument("--bg-noise-amp", type=float, default=1.0)
    parser.add_argument("--b0-artifact", action="store_true")
    parser.add_argument("--b0-mode", choices=["linear", "parabolic"], default="linear")
    parser.add_argument("--b0-delta-ppm", type=float, default=0.5)
    parser.add_argument("--b0-axis", choices=["x", "y"], default="x")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def build_database_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the local MRI phantom database.")
    subparsers = parser.add_subparsers(dest="database_command", required=True)

    subparsers.add_parser("list", help="List available database phantoms.")

    create_parser = subparsers.add_parser("create", help="Create a new phantom database entry.")
    create_parser.add_argument("--name", required=True, help="New phantom name and directory name.")
    create_parser.add_argument("--description", required=True, help="Description written to phantom_depository/info.txt.")

    load_parser = subparsers.add_parser("load", help="Import one legal phantom data file into an existing database entry.")
    load_parser.add_argument("--name", required=True, help="Target phantom name.")
    load_parser.add_argument("--file-path", required=True, help="Source file to copy into the phantom directory.")
    load_parser.add_argument(
        "--data",
        required=True,
        help="Data name: rho, t1, t2, dB0, CS, dWRnd, txCoilmg, txCoilpe, rxCoilmg, rxCoilpe, or info.",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete one phantom data file or an entire phantom entry.")
    delete_parser.add_argument("--name", required=True, help="Target phantom name.")
    delete_group = delete_parser.add_mutually_exclusive_group(required=True)
    delete_group.add_argument("--data", help="Delete one data file by legal data name.")
    delete_group.add_argument("--all", action="store_true", help="Delete the entire phantom directory and index entry.")
    return parser


def run_database_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.database_command == "list":
        return {"status": "success", "phantoms": list_phantom_database()}
    if args.database_command == "create":
        result = create_phantom_database_entry(args.name, args.description)
        return {"status": "success", **result}
    if args.database_command == "load":
        result = import_phantom_database_file(args.name, args.file_path, args.data)
        return {"status": "success", **result}
    if args.database_command == "delete":
        if args.all:
            result = delete_phantom_database_entry(args.name)
        else:
            result = delete_phantom_database_data(args.name, args.data)
        return {"status": "success", **result}
    raise ValueError(f"Unsupported database command: {args.database_command}")


def build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MRI forward simulation and agent platform.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("simulate", help="Run the MRI forward simulation CLI.")
    subparsers.add_parser("agent-cli", help="Run the ReAct agent in an interactive terminal.")
    agent_ui = subparsers.add_parser("agent-ui", help="Start the Streamlit agent UI.")
    agent_ui.add_argument("--server-port", type=int, default=None)
    agent_ui.add_argument("--server-address", default=None)
    return parser


def run_agent_ui(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Start the Streamlit MRI agent UI.")
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--server-address", default=None)
    args = parser.parse_args(argv)

    streamlit_app = Path(__file__).resolve().parent / "agent" / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(streamlit_app)]
    if args.server_port is not None:
        command.extend(["--server.port", str(args.server_port)])
    if args.server_address is not None:
        command.extend(["--server.address", args.server_address])
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "simulate":
        if len(argv) > 1 and argv[1] == "database":
            args = build_database_parser().parse_args(argv[2:])
            result = run_database_command(args)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        args = build_simulation_parser().parse_args(argv[1:])
    elif argv and argv[0] == "agent-cli":
        from agent.react_agent import run_interactive_cli

        run_interactive_cli()
        return
    elif argv and argv[0] == "agent-ui":
        run_agent_ui(argv[1:])
        return
    elif argv in (["-h"], ["--help"]):
        build_root_parser().print_help()
        return
    elif argv and not argv[0].startswith("-"):
        build_root_parser().parse_args(argv)
        return
    else:
        args = build_simulation_parser().parse_args(argv)

    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
