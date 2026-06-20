"""Sequence factories used by the simulator."""

from __future__ import annotations

from collections.abc import Callable

from .write_epi import write_epi_sequence
from .write_epi_label import write_epi_label_sequence
from .write_epi_se import write_epi_se_sequence
from .write_gre import write_gre_sequence
from .write_gre_label import write_gre_label_sequence
from .write_se import write_se_sequence
from .tse import write_tse_sequence

SEQUENCE_FACTORIES: dict[str, Callable[..., object]] = {
    "gre": write_gre_sequence,
    "gre_label": write_gre_label_sequence,
    "se": write_se_sequence,
    "tse": write_tse_sequence,
    "epi": write_epi_sequence,
    "epi_se": write_epi_se_sequence,
    "epi_label": write_epi_label_sequence,
}


def get_sequence(sequence_type: str, **kwargs):
    """Build a PyPulseq sequence by its public sequence type name."""
    try:
        factory = SEQUENCE_FACTORIES[sequence_type]
    except KeyError as exc:
        valid = ", ".join(sorted(SEQUENCE_FACTORIES))
        raise ValueError(f"Unknown sequence type '{sequence_type}'. Valid types: {valid}") from exc
    return factory(**kwargs)


__all__ = [
    "SEQUENCE_FACTORIES",
    "get_sequence",
    "write_epi_label_sequence",
    "write_epi_se_sequence",
    "write_epi_sequence",
    "write_gre_label_sequence",
    "write_gre_sequence",
    "write_se_sequence",
    "write_tse_sequence",
]
