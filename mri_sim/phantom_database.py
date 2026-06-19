from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

from mri_sim.phantom import Phantom


PHANTOM_DATABASE_DIR = Path(__file__).resolve().parent / "phantom_depository"
REQUIRED_ARRAY_FILES = {"rho": "rho.npy", "t1": "t1.npy", "t2": "t2.npy"}
OPTIONAL_ARRAY_FILES = {
    "dB0": "dB0.npy",
    "CS": "CS.npy",
    "dWRnd": "dWRnd.npy",
    "txCoilmg": "txCoilmg.npy",
    "txCoilpe": "txCoilpe.npy",
    "rxCoilmg": "rxCoilmg.npy",
    "rxCoilpe": "rxCoilpe.npy",
}
DATA_FILE_MAP = {
    **REQUIRED_ARRAY_FILES,
    **OPTIONAL_ARRAY_FILES,
    "info": "info.txt",
}


@dataclass(frozen=True)
class PhantomDataset:
    name: str
    directory: Path
    rho: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    metadata: dict[str, str]
    optional_arrays: dict[str, np.ndarray]

    def build_phantom(
        self,
        *,
        fov_x: float | None = None,
        fov_y: float | None = None,
        slice_thickness: float | None = None,
        b0: float | None = None,
    ) -> Phantom:
        return Phantom(
            rho=self.rho,
            t1=self.t1,
            t2=self.t2,
            fov_x=_float_value(fov_x, self.metadata, "fov_x", 0.256),
            fov_y=_float_value(fov_y, self.metadata, "fov_y", 0.256),
            slice_thickness=_float_value(slice_thickness, self.metadata, "slice_thickness", 0.004),
            RxCoilNum=_coil_count(self.metadata, self.optional_arrays, "RxCoilNum", "rxCoilmg", "rxCoilpe"),
            TxCoilNum=_coil_count(self.metadata, self.optional_arrays, "TxCoilNum", "txCoilmg", "txCoilpe"),
            B0=_float_value(b0, self.metadata, "B0", 3.0),
            **self.optional_arrays,
        )


def list_phantom_database(repository_path: Path = PHANTOM_DATABASE_DIR) -> list[dict[str, str]]:
    if not repository_path.exists():
        return []

    index_files = [path for path in repository_path.glob("*.txt") if path.is_file()]
    if index_files:
        phantoms = []
        for line in index_files[0].read_text(encoding="utf-8").splitlines():
            if ":" in line:
                name, description = line.split(":", 1)
                phantoms.append({"name": name.strip(), "description": description.strip()})
        if phantoms:
            return phantoms

    phantoms = []
    for path in sorted(repository_path.iterdir()):
        if path.is_dir() and _has_required_arrays(path):
            metadata = load_phantom_metadata(path)
            phantoms.append({"name": path.name, "description": metadata.get("description", "MRI phantom dataset")})
    return phantoms


def create_phantom_database_entry(
    name: str,
    description: str,
    repository_path: Path = PHANTOM_DATABASE_DIR,
) -> dict[str, str]:
    phantom_name = _validate_phantom_name(name)
    repository_path.mkdir(parents=True, exist_ok=True)
    phantom_dir = _phantom_directory(phantom_name, repository_path)
    if phantom_dir.exists():
        raise FileExistsError(f"Phantom already exists: {phantom_name}")

    phantom_dir.mkdir()
    _upsert_index_entry(repository_path, phantom_name, description)
    return {"name": phantom_name, "description": description, "directory": str(phantom_dir)}


def import_phantom_database_file(
    name: str,
    file_path: str | Path,
    data_name: str,
    repository_path: Path = PHANTOM_DATABASE_DIR,
) -> dict[str, str]:
    phantom_name = _validate_phantom_name(name)
    canonical_data_name = _validate_data_name(data_name)
    phantom_dir = _phantom_directory(phantom_name, repository_path)
    if not phantom_dir.exists():
        raise FileNotFoundError(f"Unknown phantom: {phantom_name}. Create it before importing files.")

    source = Path(file_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")

    target = phantom_dir / DATA_FILE_MAP[canonical_data_name]
    if canonical_data_name == "info":
        source.read_text(encoding="utf-8")
    else:
        np.load(source)

    shutil.copy2(source, target)
    return {"name": phantom_name, "data": canonical_data_name, "path": str(target)}


def delete_phantom_database_data(
    name: str,
    data_name: str,
    repository_path: Path = PHANTOM_DATABASE_DIR,
) -> dict[str, str]:
    phantom_name = _validate_phantom_name(name)
    canonical_data_name = _validate_data_name(data_name)
    phantom_dir = _phantom_directory(phantom_name, repository_path)
    if not phantom_dir.exists():
        raise FileNotFoundError(f"Unknown phantom: {phantom_name}")

    target = phantom_dir / DATA_FILE_MAP[canonical_data_name]
    if not target.exists():
        raise FileNotFoundError(f"Phantom data does not exist: {target}")

    target.unlink()
    return {"name": phantom_name, "deleted_data": canonical_data_name, "path": str(target)}


def delete_phantom_database_entry(
    name: str,
    repository_path: Path = PHANTOM_DATABASE_DIR,
) -> dict[str, str]:
    phantom_name = _validate_phantom_name(name)
    phantom_dir = _phantom_directory(phantom_name, repository_path)
    if not phantom_dir.exists():
        raise FileNotFoundError(f"Unknown phantom: {phantom_name}")

    resolved_repo = repository_path.resolve()
    resolved_dir = phantom_dir.resolve()
    if resolved_dir == resolved_repo or resolved_repo not in resolved_dir.parents:
        raise ValueError(f"Refusing to delete outside the phantom database: {resolved_dir}")

    shutil.rmtree(resolved_dir)
    _remove_index_entry(repository_path, phantom_name)
    return {"name": phantom_name, "deleted_directory": str(resolved_dir)}


def load_phantom_dataset(name: str, repository_path: Path = PHANTOM_DATABASE_DIR) -> PhantomDataset:
    phantom_name = _validate_phantom_name(name)
    phantom_dir = _phantom_directory(phantom_name, repository_path)
    if not phantom_dir.exists():
        raise FileNotFoundError(f"Unknown phantom: {phantom_name}")

    missing = [str(phantom_dir / filename) for filename in REQUIRED_ARRAY_FILES.values() if not (phantom_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing phantom database files: {', '.join(missing)}")

    arrays = {key: np.load(phantom_dir / filename) for key, filename in REQUIRED_ARRAY_FILES.items()}
    optional = {
        key: np.load(phantom_dir / filename)
        for key, filename in OPTIONAL_ARRAY_FILES.items()
        if (phantom_dir / filename).exists()
    }
    return PhantomDataset(
        name=phantom_name,
        directory=phantom_dir,
        rho=arrays["rho"],
        t1=arrays["t1"],
        t2=arrays["t2"],
        metadata=load_phantom_metadata(phantom_dir),
        optional_arrays=optional,
    )


def load_phantom_metadata(phantom_dir: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for path in sorted(phantom_dir.glob("*.txt")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def _has_required_arrays(phantom_dir: Path) -> bool:
    return all((phantom_dir / filename).exists() for filename in REQUIRED_ARRAY_FILES.values())


def _validate_phantom_name(name: str) -> str:
    phantom_name = str(name).strip()
    if not phantom_name:
        raise ValueError("phantom name is required.")
    if phantom_name in {".", ".."} or any(separator in phantom_name for separator in ("/", "\\")):
        raise ValueError("phantom name must be a single directory name without path separators.")
    return phantom_name


def _validate_data_name(data_name: str) -> str:
    canonical = str(data_name).strip()
    if canonical.endswith(".npy"):
        canonical = canonical[:-4]
    elif canonical == "info.txt":
        canonical = "info"
    if canonical not in DATA_FILE_MAP:
        allowed = ", ".join(sorted(DATA_FILE_MAP))
        raise ValueError(f"Unsupported phantom data: {data_name}. Allowed values: {allowed}")
    return canonical


def _phantom_directory(name: str, repository_path: Path) -> Path:
    repository_path.mkdir(parents=True, exist_ok=True)
    phantom_dir = (repository_path / name).resolve()
    resolved_repo = repository_path.resolve()
    if resolved_repo not in phantom_dir.parents:
        raise ValueError(f"Phantom path escapes database directory: {phantom_dir}")
    return phantom_dir


def _upsert_index_entry(repository_path: Path, name: str, description: str) -> None:
    index_path = repository_path / "info.txt"
    lines = []
    if index_path.exists():
        lines = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        lines = [line for line in lines if line.split(":", 1)[0].strip() != name]
    lines.append(f"{name}:{description}")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _remove_index_entry(repository_path: Path, name: str) -> None:
    index_path = repository_path / "info.txt"
    if not index_path.exists():
        return
    lines = [
        line
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split(":", 1)[0].strip() != name
    ]
    content = "\n".join(lines)
    index_path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def _float_value(override: float | None, metadata: dict[str, str], key: str, default: float) -> float:
    if override is not None:
        return float(override)
    return float(metadata.get(key, default))


def _coil_count(metadata: dict[str, str], optional_arrays: dict[str, np.ndarray], metadata_key: str, *array_keys: str) -> int:
    if metadata_key in metadata:
        return int(metadata[metadata_key])
    for key in array_keys:
        if key in optional_arrays:
            return int(optional_arrays[key].shape[0])
    return 1
