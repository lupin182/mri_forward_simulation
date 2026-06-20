from __future__ import annotations

from pathlib import Path
import shutil

import pypulseq as pp


SEQUENCE_DATABASE_DIR = Path(__file__).resolve().parent / "seq_depository"


def list_sequence_database(repository_path: Path = SEQUENCE_DATABASE_DIR) -> list[dict[str, str]]:
    if not repository_path.exists():
        return []

    index_path = repository_path / "info.txt"
    if index_path.exists():
        sequences = []
        for raw_line in index_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and ":" in line:
                name, description = line.split(":", 1)
                sequences.append({"name": name.strip(), "description": description.strip()})
        if sequences:
            return sequences

    sequences = []
    for path in sorted(repository_path.iterdir()):
        if path.is_dir() and (path / f"{path.name}.seq").exists():
            sequences.append({"name": path.name, "description": "MRI sequence dataset"})
    return sequences


def load_sequence_database_file(
    name: str,
    description: str,
    file_path: str | Path,
    repository_path: Path = SEQUENCE_DATABASE_DIR,
) -> dict[str, str]:
    sequence_name = _validate_sequence_name(name)
    source = Path(file_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source sequence file does not exist: {source}")
    if source.suffix.lower() != ".seq":
        raise ValueError(f"Sequence database only accepts .seq files: {source}")

    _read_sequence_file(source)
    sequence_dir = _sequence_directory(sequence_name, repository_path)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    target = sequence_dir / f"{sequence_name}.seq"
    shutil.copy2(source, target)
    _upsert_index_entry(repository_path, sequence_name, description)
    return {"name": sequence_name, "description": description, "path": str(target)}


def load_sequence_from_database(name: str, repository_path: Path = SEQUENCE_DATABASE_DIR):
    sequence_name = _validate_sequence_name(name)
    sequence_path = _sequence_directory(sequence_name, repository_path) / f"{sequence_name}.seq"
    if not sequence_path.exists():
        raise FileNotFoundError(f"Unknown database sequence: {sequence_name}")
    return _read_sequence_file(sequence_path)


def delete_sequence_database_entry(
    name: str,
    repository_path: Path = SEQUENCE_DATABASE_DIR,
) -> dict[str, str]:
    sequence_name = _validate_sequence_name(name)
    sequence_dir = _sequence_directory(sequence_name, repository_path)
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Unknown database sequence: {sequence_name}")

    resolved_repo = repository_path.resolve()
    resolved_dir = sequence_dir.resolve()
    if resolved_dir == resolved_repo or resolved_repo not in resolved_dir.parents:
        raise ValueError(f"Refusing to delete outside the sequence database: {resolved_dir}")

    shutil.rmtree(resolved_dir)
    _remove_index_entry(repository_path, sequence_name)
    return {"name": sequence_name, "deleted_directory": str(resolved_dir)}


def _read_sequence_file(path: Path):
    sequence = pp.Sequence()
    sequence.read(str(path))
    return sequence


def _validate_sequence_name(name: str) -> str:
    sequence_name = str(name).strip()
    if not sequence_name:
        raise ValueError("sequence name is required.")
    if sequence_name in {".", ".."} or any(separator in sequence_name for separator in ("/", "\\")):
        raise ValueError("sequence name must be a single directory name without path separators.")
    return sequence_name


def _sequence_directory(name: str, repository_path: Path) -> Path:
    repository_path.mkdir(parents=True, exist_ok=True)
    sequence_dir = (repository_path / name).resolve()
    resolved_repo = repository_path.resolve()
    if resolved_repo not in sequence_dir.parents:
        raise ValueError(f"Sequence path escapes database directory: {sequence_dir}")
    return sequence_dir


def _upsert_index_entry(repository_path: Path, name: str, description: str) -> None:
    repository_path.mkdir(parents=True, exist_ok=True)
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
