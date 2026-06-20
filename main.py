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
from mri_sim.sequence_database import (
    delete_sequence_database_entry,
    list_sequence_database,
    load_sequence_database_file,
    load_sequence_from_database,
)
from mri_sim.simulation import SimulationConfig, simulate


DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_FOV_X = 0.256
DEFAULT_FOV_Y = 0.256
DEFAULT_SLICE_THICKNESS = 0.004
SEQUENCE_SPECIFIC_ARGUMENTS = {
    "seq_tr",
    "seq_te",
    "seq_flip_angle_deg",
    "seq_rf_spoiling_inc_deg",
    "seq_dummy_scans",
    "seq_ideal_spoiling_reset",
    "seq_readout_duration",
    "seq_excitation_flip_angle_deg",
    "seq_refocusing_flip_angle_deg",
    "seq_rf_excitation_duration",
    "seq_rf_refocusing_duration",
    "seq_readout_time",
    "seq_prephase_duration",
    "seq_n_reps",
    "seq_n_navigator",
    "seq_n_echo",
    "seq_rf_flip_deg",
}
SEQUENCE_SUPPORTED_ARGUMENTS = {
    "gre": {
        "seq_tr",
        "seq_te",
        "seq_flip_angle_deg",
        "seq_rf_spoiling_inc_deg",
        "seq_dummy_scans",
        "seq_ideal_spoiling_reset",
    },
    "gre_label": {
        "seq_tr",
        "seq_te",
        "seq_flip_angle_deg",
        "seq_rf_spoiling_inc_deg",
        "seq_dummy_scans",
        "seq_ideal_spoiling_reset",
        "seq_readout_duration",
    },
    "se": {
        "seq_tr",
        "seq_te",
        "seq_excitation_flip_angle_deg",
        "seq_refocusing_flip_angle_deg",
        "seq_rf_excitation_duration",
        "seq_rf_refocusing_duration",
        "seq_readout_time",
        "seq_prephase_duration",
    },
    "tse": {"seq_tr", "seq_te", "seq_n_echo", "seq_rf_flip_deg"},
    "epi": set(),
    "epi_se": {"seq_te"},
    "epi_label": {"seq_n_reps", "seq_n_navigator"},
    "database": set(),
}
SEQUENCE_OPTION_DESTS = {
    "--seq-n-slices": "seq_n_slices",
    "--seq-tr": "seq_tr",
    "--seq-te": "seq_te",
    "--seq-flip-angle-deg": "seq_flip_angle_deg",
    "--seq-rf-spoiling-inc-deg": "seq_rf_spoiling_inc_deg",
    "--seq-dummy-scans": "seq_dummy_scans",
    "--seq-ideal-spoiling-reset": "seq_ideal_spoiling_reset",
    "--no-seq-ideal-spoiling-reset": "seq_ideal_spoiling_reset",
    "--seq-readout-duration": "seq_readout_duration",
    "--seq-excitation-flip-angle-deg": "seq_excitation_flip_angle_deg",
    "--seq-refocusing-flip-angle-deg": "seq_refocusing_flip_angle_deg",
    "--seq-rf-excitation-duration": "seq_rf_excitation_duration",
    "--seq-rf-refocusing-duration": "seq_rf_refocusing_duration",
    "--seq-readout-time": "seq_readout_time",
    "--seq-prephase-duration": "seq_prephase_duration",
    "--seq-n-reps": "seq_n_reps",
    "--seq-n-navigator": "seq_n_navigator",
    "--seq-n-echo": "seq_n_echo",
    "--seq-rf-flip-deg": "seq_rf_flip_deg",
}
SEQUENCE_N_SLICES_SUPPORTED = {"gre_label", "se", "tse", "epi", "epi_label"}


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


def _explicit_args(args: argparse.Namespace) -> set[str]:
    return getattr(args, "_explicit_args", set())


def _effective_sequence_geometry(args: argparse.Namespace, phantom: Phantom) -> dict[str, int | float]:
    if args.sequence == "database":
        return {
            "nx": int(args.seq_nx if args.seq_nx is not None else phantom.Nx),
            "ny": int(args.seq_ny if args.seq_ny is not None else phantom.Ny),
            "n_slices": int(args.seq_n_slices if args.seq_n_slices is not None else phantom.Nz),
            "fov_x": float(args.seq_fov_x if args.seq_fov_x is not None else phantom.fov_x),
            "fov_y": float(args.seq_fov_y if args.seq_fov_y is not None else phantom.fov_y),
            "slice_thickness": float(
                args.seq_slice_thickness if args.seq_slice_thickness is not None else phantom.slice_thickness
            ),
        }
    if "seq_n_slices" in _explicit_args(args) and args.sequence not in SEQUENCE_N_SLICES_SUPPORTED:
        raise ValueError(f"{args.sequence} does not support sequence argument(s): --seq-n-slices")
    n_slices = args.seq_n_slices if args.seq_n_slices is not None else phantom.Nz
    if args.sequence not in SEQUENCE_N_SLICES_SUPPORTED:
        n_slices = 1
    return {
        "nx": int(args.seq_nx if args.seq_nx is not None else phantom.Nx),
        "ny": int(args.seq_ny if args.seq_ny is not None else phantom.Ny),
        "n_slices": int(n_slices),
        "fov_x": float(args.seq_fov_x if args.seq_fov_x is not None else phantom.fov_x),
        "fov_y": float(args.seq_fov_y if args.seq_fov_y is not None else phantom.fov_y),
        "slice_thickness": float(
            args.seq_slice_thickness if args.seq_slice_thickness is not None else phantom.slice_thickness
        ),
    }


def _validate_sequence_specific_args(args: argparse.Namespace) -> None:
    explicit_specific = getattr(args, "_explicit_seq_args", set()) & SEQUENCE_SPECIFIC_ARGUMENTS
    unsupported = sorted(explicit_specific - SEQUENCE_SUPPORTED_ARGUMENTS[args.sequence])
    if unsupported:
        options = ", ".join(f"--{name.replace('_', '-')}" for name in unsupported)
        raise ValueError(f"{args.sequence} does not support sequence argument(s): {options}")


def _add_if_explicit(kwargs: dict[str, Any], args: argparse.Namespace, attr: str, target: str) -> None:
    if attr in _explicit_args(args):
        kwargs[target] = getattr(args, attr)


def _build_sequence(args: argparse.Namespace, phantom: Phantom):
    _validate_sequence_specific_args(args)
    seq_geometry = _effective_sequence_geometry(args, phantom)
    if args.sequence == "database":
        if not args.sequence_name:
            raise ValueError("--sequence-name is required when --sequence database is used.")
        return load_sequence_from_database(args.sequence_name), seq_geometry

    base_kwargs: dict[str, Any] = {
        "fov": (seq_geometry["fov_x"], seq_geometry["fov_y"]),
        "n_x": seq_geometry["nx"],
        "n_y": seq_geometry["ny"],
        "slice_thickness": seq_geometry["slice_thickness"],
    }
    if args.sequence == "gre":
        base_kwargs.update({"tr": args.seq_tr, "te": args.seq_te})
        _add_if_explicit(base_kwargs, args, "seq_flip_angle_deg", "flip_angle_deg")
        _add_if_explicit(base_kwargs, args, "seq_rf_spoiling_inc_deg", "rf_spoiling_inc_deg")
        _add_if_explicit(base_kwargs, args, "seq_dummy_scans", "dummy_scans")
        _add_if_explicit(base_kwargs, args, "seq_ideal_spoiling_reset", "ideal_spoiling_reset")
    elif args.sequence == "gre_label":
        base_kwargs.update({"n_slices": seq_geometry["n_slices"], "tr": args.seq_tr, "te": args.seq_te})
        _add_if_explicit(base_kwargs, args, "seq_flip_angle_deg", "flip_angle_deg")
        _add_if_explicit(base_kwargs, args, "seq_rf_spoiling_inc_deg", "rf_spoiling_inc_deg")
        _add_if_explicit(base_kwargs, args, "seq_dummy_scans", "dummy_scans")
        _add_if_explicit(base_kwargs, args, "seq_ideal_spoiling_reset", "ideal_spoiling_reset")
        _add_if_explicit(base_kwargs, args, "seq_readout_duration", "readout_duration")
    elif args.sequence == "se":
        base_kwargs.update({"n_slices": seq_geometry["n_slices"], "tr": args.seq_tr, "te": args.seq_te})
        _add_if_explicit(base_kwargs, args, "seq_excitation_flip_angle_deg", "excitation_flip_angle_deg")
        _add_if_explicit(base_kwargs, args, "seq_refocusing_flip_angle_deg", "refocusing_flip_angle_deg")
        _add_if_explicit(base_kwargs, args, "seq_rf_excitation_duration", "rf_excitation_duration")
        _add_if_explicit(base_kwargs, args, "seq_rf_refocusing_duration", "rf_refocusing_duration")
        _add_if_explicit(base_kwargs, args, "seq_readout_time", "readout_time")
        _add_if_explicit(base_kwargs, args, "seq_prephase_duration", "prephase_duration")
    elif args.sequence == "tse":
        base_kwargs.update({"n_slices": seq_geometry["n_slices"], "tr": args.seq_tr, "te": args.seq_te})
        _add_if_explicit(base_kwargs, args, "seq_n_echo", "n_echo")
        _add_if_explicit(base_kwargs, args, "seq_rf_flip_deg", "rf_flip_deg")
    elif args.sequence == "epi":
        base_kwargs.update({"n_slices": seq_geometry["n_slices"]})
    elif args.sequence == "epi_se":
        base_kwargs.update({"te": args.seq_te})
    elif args.sequence == "epi_label":
        base_kwargs.update({"n_slices": seq_geometry["n_slices"]})
        _add_if_explicit(base_kwargs, args, "seq_n_reps", "n_reps")
        _add_if_explicit(base_kwargs, args, "seq_n_navigator", "n_navigator")
    return get_sequence(args.sequence, **base_kwargs), seq_geometry


def _as_reconstruction_input(kspace: np.ndarray) -> np.ndarray:
    signal = np.asarray(kspace)
    if signal.ndim == 2:
        return signal.T
    return signal.squeeze()


def _reconstruct(
    kspace: np.ndarray,
    k_traj_adc: np.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]:
    coil_images, k_grid = reconstruct_3d_cartesian_fft_multichannel(
        _as_reconstruction_input(kspace),
        k_traj_adc,
        Ny=ny,
        Nx=nx,
        Nz=nz,
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
    sequence, sequence_geometry = _build_sequence(args, phantom)

    kspace = simulate(phantom, sequence, SimulationConfig(fine_dt=args.fine_dt))
    k_traj_adc, _, _, _, _ = sequence.calculate_kspace()
    image, _ = _reconstruct(
        kspace,
        k_traj_adc,
        nx=int(sequence_geometry["nx"]),
        ny=int(sequence_geometry["ny"]),
        nz=int(sequence_geometry["n_slices"]),
    )

    np.save(output_dir / "kspace.npy", kspace)
    np.save(output_dir / "reconstruction.npy", image)
    np.save(output_dir / "reconstruction_magnitude.npy", np.abs(image))
    if not args.no_plot:
        _save_png(output_dir / "reconstruction.png", image, _reference_rho(rho), "Reconstruction")

    summary: dict[str, Any] = {
        "phantom": args.phantom,
        "phantom_name": args.phantom_name,
        "sequence": args.sequence,
        "shape": {"nz": int(sequence_geometry["n_slices"]), "nx": int(sequence_geometry["nx"]), "ny": int(sequence_geometry["ny"])},
        "phantom_shape": {"nz": phantom.Nz, "nx": phantom.Nx, "ny": phantom.Ny},
        "sequence_shape": {
            "nz": int(sequence_geometry["n_slices"]),
            "nx": int(sequence_geometry["nx"]),
            "ny": int(sequence_geometry["ny"]),
        },
        "sequence_fov": {
            "fov_x": float(sequence_geometry["fov_x"]),
            "fov_y": float(sequence_geometry["fov_y"]),
            "slice_thickness": float(sequence_geometry["slice_thickness"]),
        },
        "kspace_shape": list(np.asarray(kspace).shape),
        "reconstruction_shape": list(np.asarray(image).shape),
        "output_dir": str(output_dir.resolve()),
        "rf_artifact": False,
        "b0_artifact": bool(args.b0_artifact),
    }

    if args.rf_artifact:
        _, _, _, t_adc, _ = sequence.waveforms_and_times()
        artifact_kspace = _add_rf_artifact(args, np.asarray(t_adc), np.asarray(kspace))
        artifact_image, _ = _reconstruct(
            artifact_kspace,
            k_traj_adc,
            nx=int(sequence_geometry["nx"]),
            ny=int(sequence_geometry["ny"]),
            nz=int(sequence_geometry["n_slices"]),
        )
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
    parser.add_argument(
        "--sequence",
        choices=["gre", "gre_label", "se", "tse", "epi", "epi_se", "epi_label", "database"],
        default="gre_label",
    )
    parser.add_argument("--sequence-name", default=None)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--fov-y", type=float, default=None)
    parser.add_argument("--slice-thickness", type=float, default=None)
    parser.add_argument("--seq-nx", type=int, default=None)
    parser.add_argument("--seq-ny", type=int, default=None)
    parser.add_argument("--seq-n-slices", type=int, default=None)
    parser.add_argument("--seq-fov-x", type=float, default=None)
    parser.add_argument("--seq-fov-y", type=float, default=None)
    parser.add_argument("--seq-slice-thickness", type=float, default=None)
    parser.add_argument("--tr", "--seq-tr", dest="seq_tr", type=float, default=0.1)
    parser.add_argument("--te", "--seq-te", dest="seq_te", type=float, default=0.02)
    parser.add_argument("--seq-flip-angle-deg", type=float, default=None)
    parser.add_argument("--seq-rf-spoiling-inc-deg", type=float, default=None)
    parser.add_argument("--seq-dummy-scans", type=int, default=None)
    parser.add_argument("--seq-ideal-spoiling-reset", dest="seq_ideal_spoiling_reset", action="store_true", default=None)
    parser.add_argument("--no-seq-ideal-spoiling-reset", dest="seq_ideal_spoiling_reset", action="store_false")
    parser.add_argument("--seq-readout-duration", type=float, default=None)
    parser.add_argument("--seq-excitation-flip-angle-deg", type=float, default=None)
    parser.add_argument("--seq-refocusing-flip-angle-deg", type=float, default=None)
    parser.add_argument("--seq-rf-excitation-duration", type=float, default=None)
    parser.add_argument("--seq-rf-refocusing-duration", type=float, default=None)
    parser.add_argument("--seq-readout-time", type=float, default=None)
    parser.add_argument("--seq-prephase-duration", type=float, default=None)
    parser.add_argument("--seq-n-reps", type=int, default=None)
    parser.add_argument("--seq-n-navigator", type=int, default=None)
    parser.add_argument("--seq-n-echo", type=int, default=None)
    parser.add_argument("--seq-rf-flip-deg", type=int, default=None)
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


def parse_simulation_args(argv: list[str]) -> argparse.Namespace:
    parser = build_simulation_parser()
    args = parser.parse_args(argv)
    explicit_options = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    args._explicit_args = {dest for option, dest in SEQUENCE_OPTION_DESTS.items() if option in explicit_options}
    args._explicit_seq_args = set(args._explicit_args)
    return args


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


def build_sequence_database_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the local MRI sequence database.")
    subparsers = parser.add_subparsers(dest="sequence_database_command", required=True)

    subparsers.add_parser("list", help="List available database sequences.")

    load_parser = subparsers.add_parser("load", help="Import a .seq file into the sequence database.")
    load_parser.add_argument("--name", required=True, help="Sequence name and target directory name.")
    load_parser.add_argument("--description", required=True, help="Description written to seq_depository/info.txt.")
    load_parser.add_argument("--file-path", required=True, help="Source .seq file to copy into the sequence database.")

    delete_parser = subparsers.add_parser("delete", help="Delete an entire database sequence entry.")
    delete_parser.add_argument("--name", required=True, help="Target sequence name.")
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


def run_sequence_database_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.sequence_database_command == "list":
        return {"status": "success", "sequences": list_sequence_database()}
    if args.sequence_database_command == "load":
        result = load_sequence_database_file(args.name, args.description, args.file_path)
        return {"status": "success", **result}
    if args.sequence_database_command == "delete":
        result = delete_sequence_database_entry(args.name)
        return {"status": "success", **result}
    raise ValueError(f"Unsupported sequence database command: {args.sequence_database_command}")


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
            try:
                result = run_database_command(args)
            except (ValueError, FileNotFoundError, AssertionError, OSError, RuntimeError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        if len(argv) > 1 and argv[1] == "sequence-database":
            args = build_sequence_database_parser().parse_args(argv[2:])
            try:
                result = run_sequence_database_command(args)
            except (ValueError, FileNotFoundError, AssertionError, OSError, RuntimeError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        args = parse_simulation_args(argv[1:])
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
        args = parse_simulation_args(argv)

    try:
        summary = run_pipeline(args)
    except (ValueError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
