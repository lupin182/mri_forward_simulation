from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from tqdm import tqdm
import numpy as np
import pypulseq as pp
from .bloch_kernel import TWO_PI, apply_bloch_step, build_off_resonance_rad_s
from .phantom import Phantom
from .device_manager import get_xp
from .utils import (
    build_time_grid,
    get_adc_sample_times,
    get_block_gradient_areas,
    get_gradient_amplitude,
    get_rf_frequency_offset_hz,
    preprocess_phantom,
    sample_rf_signal,
)

xp = get_xp()

BlockSolvePath = Literal["fine", "fast"]


@dataclass(slots=True)
class SimulationConfig:
    fine_dt: float = 1e-6
    demodulate_adc: bool = True
    reset_transverse_on_slice_change: bool = True
    reset_transverse_on_ideal_spoil_label: bool = True


@dataclass(slots=True)
class BlockSummary:
    index: int
    duration: float
    has_rf: bool
    has_adc: bool
    solve_path: BlockSolvePath
    num_steps: int = 0
    num_adc_samples: int = 0


@dataclass(slots=True)
class SimulationResult:
    signal: np.ndarray
    adc_times: np.ndarray
    block_summaries: list[BlockSummary]


def classify_block(block) -> BlockSolvePath:
    """Route RF/ADC blocks to the fine path and all others to the fast path."""
    return "fine" if getattr(block, "rf", None) is not None or getattr(block, "adc", None) is not None else "fast"


def analyze_sequence_blocks(sequence: pp.Sequence) -> list[BlockSummary]:
    """Return per-block metadata used by the simulator and by tests."""
    summaries: list[BlockSummary] = []
    for block_idx in range(1, len(sequence.block_durations) + 1):
        block = sequence.get_block(block_idx)
        summaries.append(
            BlockSummary(
                index=block_idx,
                duration=float(block.block_duration),
                has_rf=getattr(block, "rf", None) is not None,
                has_adc=getattr(block, "adc", None) is not None,
                solve_path=classify_block(block),
            )
        )
    return summaries


def _build_static_off_resonance(phantom: Phantom, gamma_hz: float) -> xp.ndarray:
    """构建静态偏共振项，使用CuPy加速当GPU可用时。"""
    return build_off_resonance_rad_s(
        gamma_hz=gamma_hz,
        chemical_shift_hz=phantom.CS,
        dB0_t=phantom.dB0,
        dwrnd_rad_s=phantom.dWRnd,
        x_m=phantom.x,
        y_m=phantom.y,
        z_m=phantom.z,
    )


def _rotate_transverse_state(mx: xp.ndarray, my: xp.ndarray, phase_rad: float) -> None:
    """Rotate transverse magnetization in-place around the z-axis.
    
    使用CuPy加速此计算，当GPU可用时。
    """
    # NEW: Used to enter and leave the RF-carrier rotating frame during RF support.
    if abs(phase_rad) <= 1e-15:
        return

    mxy = (mx + 1j * my) * xp.exp(1j * phase_rad)
    mx[:] = xp.real(mxy)
    my[:] = xp.imag(mxy)


def _is_excitation_block(block) -> bool:
    """Return True when the block contains an excitation RF event."""
    rf = getattr(block, "rf", None)
    return rf is not None and getattr(rf, "use", "undefined") in ("excitation", "undefined")


def _block_has_label(block, label_name: str) -> bool:
    """Return True when the block contains a PyPulseq label with the requested name."""
    labels = getattr(block, "label", None)
    if labels is None:
        return False
    return any(getattr(label, "label", None) == label_name for label in labels.values())


def _maybe_reset_transverse_for_ideal_spoiling(
    phantom: Phantom,
    block,
    *,
    enabled: bool,
) -> None:
    """Apply an idealized spoiler when the sequence explicitly marks the block.

    GRE sequences in this project use one isochromat per voxel, so gradient and
    RF spoiling cannot fully reproduce intravoxel dephasing. A `TRID` label on a
    spoiler block is treated as an explicit request to zero transverse
    magnetization at that time point.
    """
    if not enabled or not _block_has_label(block, "TRID"):
        return

    phantom.Mx[:] = 0.0
    phantom.My[:] = 0.0


def _apply_fast_block(phantom: Phantom, block, gamma_hz: float) -> None:
    """Exact free precession and relaxation for a block without RF or ADC.
    
    使用CuPy加速此计算，当GPU可用时。这是快速路径的核心计算函数。
    """
    duration = float(block.block_duration)
    if duration <= 0:
        return

    gx_area, gy_area, gz_area = get_block_gradient_areas(block)
    total_phase = duration * phantom.dWRnd
    total_phase += TWO_PI * duration * phantom.CS
    total_phase += TWO_PI * duration * gamma_hz * phantom.dB0
    total_phase += TWO_PI * (gx_area * phantom.x + gy_area * phantom.y + gz_area * phantom.z)

    e2 = xp.exp(-duration / xp.maximum(phantom.t2, 1e-12))
    e1 = xp.exp(-duration / xp.maximum(phantom.t1, 1e-12))
    mxy = (phantom.Mx + 1j * phantom.My) * e2 * xp.exp(-1j * total_phase)

    phantom.Mx[:] = xp.real(mxy)
    phantom.My[:] = xp.imag(mxy)
    phantom.Mz[:] = phantom.Mz * e1 + phantom.rho * (1.0 - e1)


def _maybe_reset_transverse_for_slice_change(
    phantom: Phantom,
    block,
    previous_excitation_freq_hz: float | None,
    *,
    enabled: bool,
) -> float | None:
    """Reset residual transverse magnetization when the sequence switches slices.

    The current forward model tracks one isochromat per voxel, so gradient
    spoilers cannot fully represent the intravoxel dephasing that normally
    suppresses signal from earlier slices. When excitation RF blocks switch to a
    new `freq_offset`, clear the previous slice's residual transverse state to
    avoid cross-slice contamination in later readouts.
    """
    if not _is_excitation_block(block):
        return previous_excitation_freq_hz

    current_excitation_freq_hz = float(getattr(block.rf, "freq_offset", 0.0))
    if (
        enabled
        and previous_excitation_freq_hz is not None
        and not np.isclose(current_excitation_freq_hz, previous_excitation_freq_hz, atol=1e-9, rtol=0.0)
    ):
        phantom.Mx[:] = 0.0
        phantom.My[:] = 0.0
    return current_excitation_freq_hz


def _readout_signal(
    phantom: Phantom,
    adc,
    sample_idx: int,
    sample_time: float,
    *,
    demodulate_adc: bool,
) -> xp.ndarray:  # 修改1：返回类型改为 多通道信号数组 (n_coils,)
    """读取信号，使用CuPy加速当GPU可用时。【多通道适配版】
    
    注意：此函数在每次ADC采样时调用，对于大规模问题可能需要优化。
    返回：每个接收通道的独立复信号，形状 (n_coils,)
    """
    # 1. 计算横向磁化矢量（不变）
    mxy = phantom.Mx + 1j * phantom.My
    
    # 2. 多通道接收灵敏度权重（不变）
    rx_weight = phantom.rxCoilmg * xp.exp(-1j * phantom.rxCoilpe)
    
    # 3. 计算每个通道的信号 (n_coils,) （不变）
    coil_signal = xp.sum(rx_weight * mxy[None, :], axis=1)

    if not demodulate_adc or adc is None:
        return coil_signal  # 修改3：直接返回多通道信号

    # --------------------- 原有相位解调逻辑（完全保留，自动适配多通道） ---------------------
    phase = float(adc.phase_offset) + TWO_PI * float(adc.freq_offset) * sample_time
    if getattr(adc, "phase_modulation", None) is not None and len(adc.phase_modulation) > sample_idx:
        phase += float(adc.phase_modulation[sample_idx])
    
    # 修改4：对【所有通道信号】统一解调（广播运算，自动适配）
    return coil_signal * xp.exp(-1j * phase)


def _simulate_fine_block(
    phantom: Phantom,
    block,
    gamma_hz: float,
    system_b0_t: float,
    config: SimulationConfig,
) -> tuple[list[complex], int]:
    """Numerically integrate a block using a refined time grid.
    
    使用CuPy加速此计算，当GPU可用时。这是最计算密集的函数之一，
    包含精细时间步长上的Bloch方程求解。
    """
    time_grid = build_time_grid(block, config.fine_dt)
    adc_sample_times = get_adc_sample_times(getattr(block, "adc", None))
    adc_samples: list[complex] = []
    adc_cursor = 0
    num_steps = max(0, len(time_grid) - 1)

    static_off_resonance = _build_static_off_resonance(phantom, gamma_hz)
    x_scale = TWO_PI * phantom.x
    y_scale = TWO_PI * phantom.y
    z_scale = TWO_PI * phantom.z
    rf_event = getattr(block, "rf", None)
    # NEW: Read the RF carrier once per block and handle it as an effective detuning.
    rf_carrier_hz = get_rf_frequency_offset_hz(rf_event, gamma_hz=gamma_hz, system_b0_t=system_b0_t)
    rf_support_start = float(getattr(rf_event, "delay", 0.0)) if rf_event is not None else 0.0
    rf_support_stop = (
        float(getattr(rf_event, "delay", 0.0) + getattr(rf_event, "shape_dur", 0.0))
        if rf_event is not None
        else 0.0
    )

    for start, stop in zip(time_grid[:-1], time_grid[1:]):
        dt = float(stop - start)
        if dt <= 0:
            continue

        mid_time = 0.5 * (start + stop)
        gx_amp = get_gradient_amplitude(getattr(block, "gx", None), mid_time)
        gy_amp = get_gradient_amplitude(getattr(block, "gy", None), mid_time)
        gz_amp = get_gradient_amplitude(getattr(block, "gz", None), mid_time)

        off_resonance = static_off_resonance + gx_amp * x_scale + gy_amp * y_scale + gz_amp * z_scale
        # MOD: Sample the RF envelope in baseband so the carrier is represented by
        # MOD: the effective off-resonance term instead of a phase-modulated B1.
        rf_sample = sample_rf_signal(
            rf_event,
            mid_time,
            include_freq_offset_phase=False,
            gamma_hz=gamma_hz,
            system_b0_t=system_b0_t,
        )
        # NEW: Restrict the rotating-frame transform to the actual RF support window.
        rf_step_active = (
            rf_event is not None
            and start >= rf_support_start - 1e-15
            and stop <= rf_support_stop + 1e-15
        )
        if rf_step_active and abs(rf_carrier_hz) > 0.0:
            # NEW: Enter the RF carrier frame at the start of the sub-step.
            _rotate_transverse_state(
                phantom.Mx,
                phantom.My,
                -TWO_PI * rf_carrier_hz * (start - rf_support_start),
            )
            # NEW: In the RF frame, the effective detuning is spin offset minus RF carrier.
            off_resonance = off_resonance - TWO_PI * rf_carrier_hz

        apply_bloch_step(
            rho=phantom.rho,
            t1=phantom.t1,
            t2=phantom.t2,
            mx=phantom.Mx,
            my=phantom.My,
            mz=phantom.Mz,
            off_resonance_rad_s=off_resonance,
            dt_s=dt,
            rf_hz=rf_sample,
            tx_coil_magnitude=phantom.txCoilmg,
            tx_coil_phase=phantom.txCoilpe,
        )
        if rf_step_active and abs(rf_carrier_hz) > 0.0:
            # NEW: Return to the simulator's main rotating frame after the RF sub-step.
            _rotate_transverse_state(
                phantom.Mx,
                phantom.My,
                TWO_PI * rf_carrier_hz * (stop - rf_support_start),
            )

        while adc_cursor < len(adc_sample_times) and np.isclose(adc_sample_times[adc_cursor], stop, atol=1e-12, rtol=0.0):
            adc_samples.append(
                _readout_signal(
                    phantom,
                    getattr(block, "adc", None),
                    adc_cursor,
                    float(adc_sample_times[adc_cursor]),
                    demodulate_adc=config.demodulate_adc,
                )
            )
            adc_cursor += 1

    return adc_samples, num_steps


def simulate(
    phantom: Phantom,
    sequence: pp.Sequence,
    config: SimulationConfig | None = None,
    *,
    return_details: bool = False,
):
    """Run the forward simulation using block-wise routing.

    RF- and ADC-containing blocks use a refined numerical update. All remaining
    blocks use a closed-form free-precession update based on gradient areas.
    
    此函数已优化支持CuPy GPU加速，在GPU可用时自动使用GPU计算，
    无GPU时回退到NumPy CPU计算。输出始终为NumPy数组以保持接口兼容性。
    """
    if config is None:
        config = SimulationConfig()

    phantom_state = preprocess_phantom(phantom)
    gamma_hz = float(getattr(sequence.system, "gamma", getattr(phantom_state, "Gyro", 42.576e6)))
    # NEW: PyPulseq ppm offsets are defined relative to the sequence system B0.
    system_b0_t = float(getattr(sequence.system, "B0", 3.0))

    summaries = analyze_sequence_blocks(sequence)
    collected_signal: list[complex] = []
    previous_excitation_freq_hz: float | None = None

    for summary in tqdm(summaries):
        block = sequence.get_block(summary.index)
        _maybe_reset_transverse_for_ideal_spoiling(
            phantom_state,
            block,
            enabled=config.reset_transverse_on_ideal_spoil_label,
        )
        previous_excitation_freq_hz = _maybe_reset_transverse_for_slice_change(
            phantom_state,
            block,
            previous_excitation_freq_hz,
            enabled=config.reset_transverse_on_slice_change,
        )
        if summary.solve_path == "fine":
            block_signal, num_steps = _simulate_fine_block(phantom_state, block, gamma_hz, system_b0_t, config)
            collected_signal.extend(block_signal)
            summary.num_steps = num_steps
            summary.num_adc_samples = len(block_signal)
        else:
            _apply_fast_block(phantom_state, block, gamma_hz)
            summary.num_steps = 1 if summary.duration > 0 else 0

    signal = np.asarray(collected_signal, dtype=np.complex128)
    if return_details:
        adc_times, _ = sequence.adc_times()
        return SimulationResult(signal=signal, adc_times=adc_times, block_summaries=summaries)
    return signal
