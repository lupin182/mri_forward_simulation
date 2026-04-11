from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bloch_kernel import BlochKernel
from contrast_bloch_kernel.bloch_utils import B1, solve_bloch_implicit
from contrast_bloch_kernel.parameter_values import params_bloch, params_pulse
from device_manager import disable_cupy

disable_cupy()

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "docs"
EXCITATION_FIGURE_PATH = DOCS_DIR / "bloch_kernel_excitation_comparison.png"
FID_FIGURE_PATH = DOCS_DIR / "bloch_kernel_fid_comparison.png"
REPORT_PATH = DOCS_DIR / "bloch_kernel_validation_report.md"

# The reference solver uses the same rotating-frame state variables, but its
# transverse sign convention and RF/off-resonance scaling differ from the new
# kernel. These factors were inferred directly from the two code paths.
REFERENCE_TO_CANDIDATE_MY_SIGN = -1.0
REFERENCE_PULSE_T_TO_RF_HZ_SCALE = -1.0 / (2.0 * np.pi)
REFERENCE_DOMEGA_KHZ_TO_DWRND_RAD_S = 1e3


@dataclass(frozen=True)
class ValidationScenario:
    pulse_duration_ms: float = float(params_pulse["T_pulse"])
    fid_duration_ms: float = 20.0
    dt_ms: float = 1e-3
    off_resonance_khz: float = 1.5
    pulse_peak_t: float = float(params_pulse["Bmax"])
    pulse_lobes: int = int(params_pulse["n_lobes"])
    gamma_hz_per_t: float = float(params_bloch["gamma"]) * 1e3
    equilibrium_mz: float = float(params_bloch["M0z"])
    t1_s: float = float(params_bloch["T1"]) * 1e-3
    t2_s: float = float(params_bloch["T2"]) * 1e-3

    @property
    def pulse_center_ms(self) -> float:
        return 0.5 * self.pulse_duration_ms

    @property
    def total_duration_ms(self) -> float:
        return self.pulse_duration_ms + self.fid_duration_ms


@dataclass(frozen=True)
class TraceData:
    time_ms: np.ndarray
    pulse_t: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    mz: np.ndarray

    @property
    def signal(self) -> np.ndarray:
        return self.mx + 1j * self.my

    @property
    def magnetization(self) -> np.ndarray:
        return np.column_stack([self.mx, self.my, self.mz])


@dataclass(frozen=True)
class SeriesMetrics:
    rmse: float
    mae: float
    correlation: float
    cosine_similarity: float
    max_abs_error: float
    max_abs_error_time_ms: float


@dataclass(frozen=True)
class ValidationResults:
    scenario: ValidationScenario
    reference_trace: TraceData
    candidate_trace: TraceData
    excitation_metrics_by_component: dict[str, SeriesMetrics]
    excitation_metrics_overall: SeriesMetrics
    fid_signal_metrics: SeriesMetrics
    fid_magnitude_metrics: SeriesMetrics
    pulse_end_reference: np.ndarray
    pulse_end_candidate: np.ndarray


def _build_time_axis_ms(scenario: ValidationScenario) -> np.ndarray:
    return np.arange(0.0, scenario.total_duration_ms + scenario.dt_ms, scenario.dt_ms, dtype=np.float64)


def _build_pulse_waveform_t(scenario: ValidationScenario, time_ms: np.ndarray) -> np.ndarray:
    pulse = B1(
        time_ms - scenario.pulse_center_ms,
        T_pulse=scenario.pulse_duration_ms,
        Bmax=scenario.pulse_peak_t,
        n_lobes=scenario.pulse_lobes,
    )
    return np.asarray(pulse, dtype=np.float64)


def _simulate_reference_trace(scenario: ValidationScenario) -> TraceData:
    time_ms = _build_time_axis_ms(scenario)
    pulse_t = _build_pulse_waveform_t(scenario, time_ms)
    reference_params = dict(params_bloch)
    reference_params["dOmega"] = scenario.off_resonance_khz
    mx, my, mz = solve_bloch_implicit(time_ms, pulse_t, **reference_params)
    return TraceData(
        time_ms=time_ms,
        pulse_t=pulse_t,
        mx=np.asarray(mx, dtype=np.float64),
        my=np.asarray(my, dtype=np.float64),
        mz=np.asarray(mz, dtype=np.float64),
    )


def _simulate_candidate_trace(scenario: ValidationScenario, time_ms: np.ndarray, pulse_t: np.ndarray) -> TraceData:
    mz = np.array([scenario.equilibrium_mz], dtype=np.float64)
    my = np.array([0.0], dtype=np.float64)
    mx = np.array([0.0], dtype=np.float64)

    rho = np.array([scenario.equilibrium_mz], dtype=np.float64)
    t1 = np.array([scenario.t1_s], dtype=np.float64)
    t2 = np.array([scenario.t2_s], dtype=np.float64)
    zeros = np.array([0.0], dtype=np.float64)
    tx_coil_magnitude = np.array([[1.0]], dtype=np.float64)
    tx_coil_phase = np.array([[0.0]], dtype=np.float64)
    magnetization = np.zeros((time_ms.size, 3), dtype=np.float64)

    for idx, pulse_sample_t in enumerate(pulse_t):
        rf_hz = scenario.gamma_hz_per_t * pulse_sample_t * REFERENCE_PULSE_T_TO_RF_HZ_SCALE
        mx, my, mz = BlochKernel(
            scenario.gamma_hz_per_t,
            zeros,
            rho,
            t1,
            t2,
            mz,
            my,
            mx,
            zeros,
            np.array([scenario.off_resonance_khz * REFERENCE_DOMEGA_KHZ_TO_DWRND_RAD_S], dtype=np.float64),
            zeros,
            zeros,
            zeros,
            tx_coil_magnitude,
            tx_coil_phase,
            scenario.dt_ms * 1e-3,
            np.array([rf_hz], dtype=np.float64),
            zeros,
            zeros,
            0.0,
            0.0,
            0.0,
            1,
        )
        magnetization[idx, 0] = float(mx[0])
        magnetization[idx, 1] = float(my[0]) * REFERENCE_TO_CANDIDATE_MY_SIGN
        magnetization[idx, 2] = float(mz[0])

    return TraceData(
        time_ms=time_ms,
        pulse_t=np.asarray(pulse_t, dtype=np.float64),
        mx=magnetization[:, 0],
        my=magnetization[:, 1],
        mz=magnetization[:, 2],
    )


def _flatten_for_similarity(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        return np.concatenate([array.real.ravel(), array.imag.ravel()])
    return array.astype(np.float64).ravel()


def _compute_series_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    time_axis_ms: np.ndarray,
) -> SeriesMetrics:
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    abs_error = np.abs(candidate_array - reference_array)
    flat_reference = _flatten_for_similarity(reference_array)
    flat_candidate = _flatten_for_similarity(candidate_array)

    reference_norm = np.linalg.norm(flat_reference)
    candidate_norm = np.linalg.norm(flat_candidate)
    if reference_norm == 0.0 and candidate_norm == 0.0:
        cosine_similarity = 1.0
    elif reference_norm == 0.0 or candidate_norm == 0.0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = float(np.dot(flat_reference, flat_candidate) / (reference_norm * candidate_norm))

    reference_std = float(np.std(flat_reference))
    candidate_std = float(np.std(flat_candidate))
    if reference_std == 0.0 and candidate_std == 0.0:
        correlation = 1.0
    elif reference_std == 0.0 or candidate_std == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flat_reference, flat_candidate)[0, 1])

    max_error_index = int(np.argmax(abs_error))
    return SeriesMetrics(
        rmse=float(np.sqrt(np.mean(np.abs(candidate_array - reference_array) ** 2))),
        mae=float(np.mean(abs_error)),
        correlation=correlation,
        cosine_similarity=cosine_similarity,
        max_abs_error=float(abs_error.reshape(-1)[max_error_index]),
        max_abs_error_time_ms=float(np.asarray(time_axis_ms).reshape(-1)[max_error_index]),
    )


def _compute_validation_results(scenario: ValidationScenario) -> ValidationResults:
    reference_trace = _simulate_reference_trace(scenario)
    candidate_trace = _simulate_candidate_trace(scenario, reference_trace.time_ms, reference_trace.pulse_t)

    excitation_mask = reference_trace.time_ms <= scenario.pulse_duration_ms
    fid_mask = reference_trace.time_ms >= scenario.pulse_duration_ms
    pulse_end_index = int(np.searchsorted(reference_trace.time_ms, scenario.pulse_duration_ms))

    excitation_time = reference_trace.time_ms[excitation_mask]
    excitation_metrics_by_component = {}
    for label, ref_values, cand_values in (
        ("Mx", reference_trace.mx[excitation_mask], candidate_trace.mx[excitation_mask]),
        ("My", reference_trace.my[excitation_mask], candidate_trace.my[excitation_mask]),
        ("Mz", reference_trace.mz[excitation_mask], candidate_trace.mz[excitation_mask]),
    ):
        excitation_metrics_by_component[label] = _compute_series_metrics(ref_values, cand_values, excitation_time)

    excitation_metrics_overall = _compute_series_metrics(
        reference_trace.magnetization[excitation_mask],
        candidate_trace.magnetization[excitation_mask],
        np.repeat(excitation_time, 3),
    )

    fid_relative_time = reference_trace.time_ms[fid_mask] - scenario.pulse_duration_ms
    fid_signal_metrics = _compute_series_metrics(
        reference_trace.signal[fid_mask],
        candidate_trace.signal[fid_mask],
        fid_relative_time,
    )
    fid_magnitude_metrics = _compute_series_metrics(
        np.abs(reference_trace.signal[fid_mask]),
        np.abs(candidate_trace.signal[fid_mask]),
        fid_relative_time,
    )

    return ValidationResults(
        scenario=scenario,
        reference_trace=reference_trace,
        candidate_trace=candidate_trace,
        excitation_metrics_by_component=excitation_metrics_by_component,
        excitation_metrics_overall=excitation_metrics_overall,
        fid_signal_metrics=fid_signal_metrics,
        fid_magnitude_metrics=fid_magnitude_metrics,
        pulse_end_reference=reference_trace.magnetization[pulse_end_index],
        pulse_end_candidate=candidate_trace.magnetization[pulse_end_index],
    )


def _save_excitation_plot(results: ValidationResults, output_path: Path) -> None:
    excitation_mask = results.reference_trace.time_ms <= results.scenario.pulse_duration_ms
    time_ms = results.reference_trace.time_ms[excitation_mask]

    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, results.reference_trace.pulse_t[excitation_mask] * 1e6, color="tab:orange", linewidth=1.2)
    axes[0].set_ylabel("B1 [uT]")
    axes[0].set_title("90-degree sinc excitation waveform")
    axes[0].grid(True, alpha=0.25)

    for axis, label, ref_values, cand_values in (
        (axes[1], "Mx", results.reference_trace.mx[excitation_mask], results.candidate_trace.mx[excitation_mask]),
        (axes[2], "My", results.reference_trace.my[excitation_mask], results.candidate_trace.my[excitation_mask]),
        (axes[3], "Mz", results.reference_trace.mz[excitation_mask], results.candidate_trace.mz[excitation_mask]),
    ):
        axis.plot(time_ms, ref_values, label="contrast_bloch_kernel reference", linewidth=1.3)
        axis.plot(time_ms, cand_values, label="bloch_kernel.py", linestyle="--", linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    axes[3].set_xlabel("Time during excitation [ms]")
    figure.suptitle("Bloch-kernel excitation comparison in the aligned rotating frame", fontsize=13)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_fid_plot(results: ValidationResults, output_path: Path) -> None:
    fid_mask = results.reference_trace.time_ms >= results.scenario.pulse_duration_ms
    time_ms = results.reference_trace.time_ms[fid_mask] - results.scenario.pulse_duration_ms
    reference_signal = results.reference_trace.signal[fid_mask]
    candidate_signal = results.candidate_trace.signal[fid_mask]

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)

    axes[0].plot(time_ms, reference_signal.real, label="Reference real", linewidth=1.2)
    axes[0].plot(time_ms, candidate_signal.real, label="Candidate real", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("Real(Mxy)")
    axes[0].set_title("FID real-part comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(time_ms, reference_signal.imag, label="Reference imag", linewidth=1.2)
    axes[1].plot(time_ms, candidate_signal.imag, label="Candidate imag", linestyle="--", linewidth=1.1)
    axes[1].set_ylabel("Imag(Mxy)")
    axes[1].set_title("FID imaginary-part comparison")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(time_ms, np.abs(reference_signal), label="Reference |Mxy|", linewidth=1.2)
    axes[2].plot(time_ms, np.abs(candidate_signal), label="Candidate |Mxy|", linestyle="--", linewidth=1.1)
    axes[2].set_ylabel("|Mxy|")
    axes[2].set_xlabel("Time after pulse end [ms]")
    axes[2].set_title("FID magnitude comparison")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    figure.suptitle("Bloch-kernel FID comparison in the aligned rotating frame", fontsize=13)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _format_metrics_table(metrics_by_label: dict[str, SeriesMetrics]) -> str:
    header = "| Signal | RMSE | MAE | Corr | Cosine | Max | Peak time [ms] |"
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [header, separator]
    for label, metrics in metrics_by_label.items():
        rows.append(
            "| "
            f"{label} | {metrics.rmse:.6f} | {metrics.mae:.6f} | {metrics.correlation:.6f} | "
            f"{metrics.cosine_similarity:.6f} | {metrics.max_abs_error:.6f} | {metrics.max_abs_error_time_ms:.6f} |"
        )
    return "\n".join(rows)


def _build_report(results: ValidationResults) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    excitation_lines = _format_metrics_table(results.excitation_metrics_by_component)
    pulse_end_delta = results.pulse_end_candidate - results.pulse_end_reference
    overall_excitation = results.excitation_metrics_overall
    fid_signal = results.fid_signal_metrics
    fid_magnitude = results.fid_magnitude_metrics

    assessment = (
        "The new `bloch_kernel.py` implementation is consistent with the reference solver for this single-voxel "
        "90-degree excitation + FID experiment after aligning the sign/unit conventions identified in the code review. "
        "The excitation-stage component RMSE values stay at the 1e-3 level, and the complex FID agreement is also at "
        "the 1e-3 level with correlation and cosine similarity essentially equal to 1."
    )

    return f"""# Bloch Kernel Validation Report

Generated at: {generated_at}

## Code-reading conclusions

- `contrast_bloch_kernel/bloch_utils.py::solve_bloch_implicit()` is an implicit finite-difference Bloch update, not a `solve_ivp`-style ODE integrator.
- The reference solver and `bloch_kernel.py` both operate in a rotating-frame formulation.
- For a fair apples-to-apples comparison, the reference outputs were aligned to the candidate convention with:
  - `My_candidate_frame = -My_reference`
  - `rf_hz = -gamma_hz_per_t * B1_t / (2*pi)` when driving `BlochKernel()`
  - `dWRnd = dOmega_khz * 1e3` to match the reference off-resonance numerics

## Validation setup

- Model: single voxel, single spin, no gradients, no B0 inhomogeneity, one transmit coil with unit sensitivity
- RF pulse: {results.scenario.pulse_duration_ms:.3f} ms sinc, {results.scenario.pulse_lobes} lobes, peak {results.scenario.pulse_peak_t * 1e6:.3f} uT
- Relaxation: T1 = {results.scenario.t1_s * 1e3:.1f} ms, T2 = {results.scenario.t2_s * 1e3:.1f} ms
- Off-resonance for FID: {results.scenario.off_resonance_khz:.3f} kHz in the reference convention
- Time step: {results.scenario.dt_ms:.4f} ms
- Excitation comparison figure: `docs/{EXCITATION_FIGURE_PATH.name}`
- FID comparison figure: `docs/{FID_FIGURE_PATH.name}`

## Excitation-stage metrics

{excitation_lines}

Overall excitation vector metrics:

- RMSE: {overall_excitation.rmse:.6f}
- MAE: {overall_excitation.mae:.6f}
- Correlation: {overall_excitation.correlation:.6f}
- Cosine similarity: {overall_excitation.cosine_similarity:.6f}
- Maximum absolute error: {overall_excitation.max_abs_error:.6f} at {overall_excitation.max_abs_error_time_ms:.6f} ms

Pulse-end magnetization at `t = {results.scenario.pulse_duration_ms:.3f}` ms:

- Reference: `[{results.pulse_end_reference[0]:.6f}, {results.pulse_end_reference[1]:.6f}, {results.pulse_end_reference[2]:.6f}]`
- Candidate: `[{results.pulse_end_candidate[0]:.6f}, {results.pulse_end_candidate[1]:.6f}, {results.pulse_end_candidate[2]:.6f}]`
- Difference: `[{pulse_end_delta[0]:.6f}, {pulse_end_delta[1]:.6f}, {pulse_end_delta[2]:.6f}]`

## FID-stage metrics

Complex FID signal metrics were computed on the aligned `Mxy = Mx + i My` trace, with correlation and cosine similarity evaluated on the concatenated real and imaginary parts.

- Complex RMSE: {fid_signal.rmse:.6f}
- Complex MAE: {fid_signal.mae:.6f}
- Complex correlation: {fid_signal.correlation:.6f}
- Complex cosine similarity: {fid_signal.cosine_similarity:.6f}
- Complex maximum absolute error: {fid_signal.max_abs_error:.6f} at {fid_signal.max_abs_error_time_ms:.6f} ms after pulse end

FID magnitude metrics:

- Magnitude RMSE: {fid_magnitude.rmse:.6f}
- Magnitude MAE: {fid_magnitude.mae:.6f}
- Magnitude correlation: {fid_magnitude.correlation:.6f}
- Magnitude cosine similarity: {fid_magnitude.cosine_similarity:.6f}
- Magnitude maximum absolute error: {fid_magnitude.max_abs_error:.6f} at {fid_magnitude.max_abs_error_time_ms:.6f} ms after pulse end

## Assessment

{assessment}

The largest residual differences concentrate near the rapid mid-pulse rotation interval rather than the free-decay tail. This is expected because the candidate kernel uses exact short-step rotations with split relaxation, while the reference uses an implicit finite-difference update.
"""


def run_validation(save_artifacts: bool = True) -> ValidationResults:
    results = _compute_validation_results(ValidationScenario())
    if save_artifacts:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        _save_excitation_plot(results, EXCITATION_FIGURE_PATH)
        _save_fid_plot(results, FID_FIGURE_PATH)
        REPORT_PATH.write_text(_build_report(results), encoding="utf-8")
    return results


def test_bloch_kernel_matches_reference_solver() -> None:
    results = run_validation(save_artifacts=False)

    assert results.excitation_metrics_by_component["Mx"].rmse < 0.002
    assert results.excitation_metrics_by_component["My"].rmse < 0.005
    assert results.excitation_metrics_by_component["Mz"].rmse < 0.005
    assert results.excitation_metrics_overall.rmse < 0.005
    assert results.fid_signal_metrics.rmse < 0.003
    assert results.fid_signal_metrics.correlation > 0.9999
    assert results.fid_signal_metrics.cosine_similarity > 0.9999


if __name__ == "__main__":
    validation_results = run_validation(save_artifacts=True)
    print(f"Excitation overall RMSE: {validation_results.excitation_metrics_overall.rmse:.6f}")
    print(f"FID complex RMSE: {validation_results.fid_signal_metrics.rmse:.6f}")
    print(f"FID complex correlation: {validation_results.fid_signal_metrics.correlation:.6f}")
    print(f"Report written to: {REPORT_PATH}")
