"""PyPulseq hardware configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

import pypulseq as pp


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load root .env values without overriding existing process variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_pypulseq_system() -> pp.Opts:
    """Build a PyPulseq system object from MRI_SYSTEM_* environment values."""
    load_dotenv()
    return pp.Opts(
        max_grad=_get_float("MRI_SYSTEM_MAX_GRAD", 32.0),
        grad_unit=_get_str("MRI_SYSTEM_GRAD_UNIT", "mT/m"),
        max_slew=_get_float("MRI_SYSTEM_MAX_SLEW", 130.0),
        slew_unit=_get_str("MRI_SYSTEM_SLEW_UNIT", "T/m/s"),
        rf_ringdown_time=_get_float("MRI_SYSTEM_RF_RINGDOWN_TIME", 20e-6),
        rf_dead_time=_get_float("MRI_SYSTEM_RF_DEAD_TIME", 100e-6),
        adc_dead_time=_get_float("MRI_SYSTEM_ADC_DEAD_TIME", 10e-6),
    )


def _get_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)
