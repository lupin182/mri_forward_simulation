"""Tools for listing and loading stored phantoms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agent.tools.base_tool import MRISimulationBaseTool
from agent.tools.phantom_tool import set_cached_phantom
from mri_sim.phantom import Phantom


DEPOSITORY_PATH = Path(__file__).resolve().parents[2] / "mri_sim" / "phantom_depository"


class ListPhantomDatabaseTool(MRISimulationBaseTool):
    name = "list_phantom_database"
    description = "List available phantoms in the local phantom database."

    def _run(self, query: str) -> str:
        if not DEPOSITORY_PATH.exists():
            return json.dumps({"status": "error", "message": f"Database path does not exist: {DEPOSITORY_PATH}"})

        phantoms = []
        index_files = [path for path in DEPOSITORY_PATH.glob("*.txt") if path.is_file()]
        if index_files:
            for line in index_files[0].read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    name, description = line.split(":", 1)
                    phantoms.append({"name": name.strip(), "description": description.strip()})

        if not phantoms:
            for path in DEPOSITORY_PATH.iterdir():
                if path.is_dir() and all((path / name).exists() for name in ("rho.npy", "t1.npy", "t2.npy")):
                    phantoms.append({"name": path.name, "description": "MRI phantom dataset"})

        return json.dumps({"status": "success", "phantoms": phantoms, "count": len(phantoms)}, ensure_ascii=False)


class LoadPhantomFromDatabaseTool(MRISimulationBaseTool):
    name = "load_phantom_from_database"
    description = "Load a phantom from the local phantom database. JSON params: phantom_name."

    def _run(self, query: str) -> str:
        params = json.loads(query or "{}")
        phantom_name = params.get("phantom_name")
        if not phantom_name:
            return json.dumps({"status": "error", "message": "phantom_name is required."})

        phantom_dir = DEPOSITORY_PATH / str(phantom_name)
        if not phantom_dir.exists():
            return json.dumps({"status": "error", "message": f"Unknown phantom: {phantom_name}"})

        required = {name: phantom_dir / f"{name}.npy" for name in ("rho", "t1", "t2")}
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            return json.dumps({"status": "error", "message": f"Missing phantom files: {', '.join(missing)}"})

        rho = np.load(required["rho"])
        t1 = np.load(required["t1"])
        t2 = np.load(required["t2"])
        metadata = _load_metadata(phantom_dir)
        optional = _load_optional_arrays(phantom_dir)

        phantom = Phantom(
            rho=rho,
            t1=t1,
            t2=t2,
            fov_x=float(metadata.get("fov_x", 0.256)),
            fov_y=float(metadata.get("fov_y", 0.256)),
            slice_thickness=float(metadata.get("slice_thickness", 0.004)),
            RxCoilNum=int(metadata.get("RxCoilNum", 1)),
            TxCoilNum=int(metadata.get("TxCoilNum", 1)),
            B0=float(metadata.get("B0", 3.0)),
            **optional,
        )
        set_cached_phantom(phantom, rho, t1, t2)

        return json.dumps(
            {
                "status": "success",
                "phantom_name": phantom_name,
                "shape": [phantom.Nz, phantom.Nx, phantom.Ny],
                "fov": [phantom.fov_x, phantom.fov_y],
                "slice_thickness": phantom.slice_thickness,
            },
            ensure_ascii=False,
        )


def _load_metadata(phantom_dir: Path) -> dict[str, str]:
    metadata = {}
    for path in phantom_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
    return metadata


def _load_optional_arrays(phantom_dir: Path) -> dict[str, np.ndarray]:
    mapping = {
        "dB0.npy": "dB0",
        "CS.npy": "CS",
        "dWRnd.npy": "dWRnd",
        "txCoilmg.npy": "txCoilmg",
        "rxCoilmg.npy": "rxCoilmg",
        "txCoilpe.npy": "txCoilpe",
        "rxCoilpe.npy": "rxCoilpe",
    }
    return {argument: np.load(phantom_dir / filename) for filename, argument in mapping.items() if (phantom_dir / filename).exists()}
