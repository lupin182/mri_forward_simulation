"""Utility helpers for the forward MRI simulator.

此模块已优化支持CuPy GPU加速，在GPU可用时自动使用GPU计算。
"""

from __future__ import annotations

import copy

import numpy as np

from .phantom import Phantom
from .device_manager import get_xp, device_manager

xp = get_xp()


def preprocess_phantom(phantom: Phantom) -> Phantom:
    """Return a flattened copy of the phantom ready for vectorized simulation.
    
    使用CuPy加速此预处理步骤，当GPU可用时。
    """
    if getattr(phantom.rho, "ndim", 0) == 1:
        return phantom

    result = copy.copy(phantom)
    shape_target = (result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny)

    # 将数据移动到当前设备（CPU/GPU）
    result.x = device_manager.to_device(result.x)
    result.y = device_manager.to_device(result.y)
    result.z = device_manager.to_device(result.z)
    result.rho = device_manager.to_device(result.rho)
    result.t1 = device_manager.to_device(result.t1)
    result.t2 = device_manager.to_device(result.t2)
    result.Mz = device_manager.to_device(result.Mz)
    result.Mx = device_manager.to_device(result.Mx)
    result.My = device_manager.to_device(result.My)
    result.CS = device_manager.to_device(result.CS)
    result.dB0 = device_manager.to_device(result.dB0)
    result.dWRnd = device_manager.to_device(result.dWRnd)
    result.txCoilmg = device_manager.to_device(result.txCoilmg)
    result.txCoilpe = device_manager.to_device(result.txCoilpe)
    result.rxCoilmg = device_manager.to_device(result.rxCoilmg)
    result.rxCoilpe = device_manager.to_device(result.rxCoilpe)

    result.x = xp.broadcast_to(result.x, shape_target)
    result.y = xp.broadcast_to(result.y, shape_target)
    result.z = xp.broadcast_to(result.z, shape_target)

    result.rho = xp.asarray(result.rho).ravel()
    result.t1 = xp.asarray(result.t1).ravel()
    result.t2 = xp.asarray(result.t2).ravel()
    result.Mz = xp.asarray(result.Mz).ravel()
    result.Mx = xp.asarray(result.Mx).ravel()
    result.My = xp.asarray(result.My).ravel()
    result.CS = xp.asarray(result.CS).ravel()
    result.dB0 = xp.asarray(result.dB0).ravel()
    result.dWRnd = xp.asarray(result.dWRnd).ravel()

    result.x = xp.asarray(result.x).ravel()
    result.y = xp.asarray(result.y).ravel()
    result.z = xp.asarray(result.z).ravel()

    result.txCoilmg = xp.broadcast_to(
        result.txCoilmg[:, None, None, :, :, :],
        (result.TxCoilNum, result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny),
    ).reshape(result.TxCoilNum, -1)
    result.txCoilpe = xp.broadcast_to(
        result.txCoilpe[:, None, None, :, :, :],
        (result.TxCoilNum, result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny),
    ).reshape(result.TxCoilNum, -1)
    result.rxCoilmg = xp.broadcast_to(
        result.rxCoilmg[:, None, None, :, :, :],
        (result.RxCoilNum, result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny),
    ).reshape(result.RxCoilNum, -1)
    result.rxCoilpe = xp.broadcast_to(
        result.rxCoilpe[:, None, None, :, :, :],
        (result.RxCoilNum, result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny),
    ).reshape(result.RxCoilNum, -1)

    active_mask = result.rho > 1e-6
    result.rho = result.rho[active_mask]
    result.t1 = result.t1[active_mask]
    result.t2 = result.t2[active_mask]
    result.Mz = result.Mz[active_mask]
    result.Mx = result.Mx[active_mask]
    result.My = result.My[active_mask]
    result.CS = result.CS[active_mask]
    result.dB0 = result.dB0[active_mask]
    result.dWRnd = result.dWRnd[active_mask]
    result.x = result.x[active_mask]
    result.y = result.y[active_mask]
    result.z = result.z[active_mask]
    result.txCoilmg = result.txCoilmg[:, active_mask]
    result.txCoilpe = result.txCoilpe[:, active_mask]
    result.rxCoilmg = result.rxCoilmg[:, active_mask]
    result.rxCoilpe = result.rxCoilpe[:, active_mask]
    return result


def get_gradient_amplitude(grad, t):
    """Sample a PyPulseq gradient event at one or more times."""
    t_array = np.asarray(t, dtype=np.float64)
    g_amp = np.zeros_like(t_array)

    if grad is None:
        return g_amp if isinstance(t, np.ndarray) else 0.0

    if grad.type == "trap":
        amp = grad.amplitude
        t1 = grad.delay
        t2 = t1 + grad.rise_time
        t3 = t2 + grad.flat_time
        t4 = t3 + grad.fall_time

        if grad.rise_time > 0:
            mask_rise = (t_array >= t1) & (t_array < t2)
            g_amp[mask_rise] = amp * (t_array[mask_rise] - t1) / grad.rise_time

        mask_flat = (t_array >= t2) & (t_array < t3)
        g_amp[mask_flat] = amp

        if grad.fall_time > 0:
            mask_fall = (t_array >= t3) & (t_array <= t4)
            g_amp[mask_fall] = amp * (1.0 - (t_array[mask_fall] - t3) / grad.fall_time)

        return g_amp if isinstance(t, np.ndarray) else float(g_amp)

    if grad.type == "grad":
        sample_times = getattr(grad, "tt", getattr(grad, "t", None))
        if sample_times is None:
            raise ValueError("Arbitrary gradient is missing its time support.")
        grad_t_shifted = np.asarray(sample_times, dtype=np.float64) + float(grad.delay)
        g_amp = np.interp(t_array, grad_t_shifted, grad.waveform, left=0.0, right=0.0)
        return g_amp if isinstance(t, np.ndarray) else float(g_amp)

    raise ValueError(f"Unknown gradient type: {grad.type}")


def get_gradient_area(grad) -> float:
    """Return the integrated gradient area in Hz*s/m."""
    if grad is None:
        return 0.0
    if hasattr(grad, "area"):
        return float(grad.area)
    if grad.type == "trap":
        return float(grad.amplitude * (grad.flat_time + 0.5 * (grad.rise_time + grad.fall_time)))
    if grad.type == "grad":
        sample_times = np.asarray(getattr(grad, "tt", getattr(grad, "t")), dtype=np.float64)
        waveform = np.asarray(grad.waveform, dtype=np.float64)
        if sample_times.size < 2:
            return float(np.sum(waveform) * grad.shape_dur)
        return float(np.trapz(waveform, sample_times))
    raise ValueError(f"Unknown gradient type: {grad.type}")


def get_block_gradient_areas(block) -> tuple[float, float, float]:
    """Return gradient areas for gx, gy, gz in this order."""
    return (
        get_gradient_area(getattr(block, "gx", None)),
        get_gradient_area(getattr(block, "gy", None)),
        get_gradient_area(getattr(block, "gz", None)),
    )


def get_rf_frequency_offset_hz(
    rf,
    *,
    gamma_hz: float | None = None,
    system_b0_t: float | None = None,
) -> float:
    """Return the full RF carrier frequency offset in Hz."""
    # NEW: Read the explicit PyPulseq RF carrier frequency offset from the block RF event.
    freq_offset_hz = float(getattr(rf, "freq_offset", 0.0)) if rf is not None else 0.0
    # NEW: Match PyPulseq's ppm handling when the sequence system B0 is known.
    if rf is not None and gamma_hz is not None and system_b0_t is not None:
        freq_offset_hz += float(getattr(rf, "freq_ppm", 0.0)) * 1e-6 * float(gamma_hz) * float(system_b0_t)
    return freq_offset_hz


def get_rf_phase_offset_rad(
    rf,
    *,
    gamma_hz: float | None = None,
    system_b0_t: float | None = None,
) -> float:
    """Return the full RF phase offset in radians."""
    # NEW: Keep the static RF phase offset separate from the carrier-frequency term.
    phase_offset_rad = float(getattr(rf, "phase_offset", 0.0)) if rf is not None else 0.0
    # NEW: Match PyPulseq's ppm handling when the sequence system B0 is known.
    if rf is not None and gamma_hz is not None and system_b0_t is not None:
        phase_offset_rad += float(getattr(rf, "phase_ppm", 0.0)) * 1e-6 * float(gamma_hz) * float(system_b0_t)
    return phase_offset_rad


def sample_rf_signal(
    rf,
    t,
    *,
    include_freq_offset_phase: bool = True,
    gamma_hz: float | None = None,
    system_b0_t: float | None = None,
):
    """Sample the RF waveform at time ``t`` and apply RF phase controls."""
    t_array = np.asarray(t, dtype=np.float64)
    signal = np.zeros_like(t_array, dtype=np.complex128)
    is_array_input = np.ndim(t_array) > 0

    if rf is None:
        return signal if is_array_input else complex(signal.item())

    local_time = t_array - float(rf.delay)
    active_mask = (local_time >= 0.0) & (local_time < float(rf.shape_dur))
    if not np.any(active_mask):
        return signal if is_array_input else complex(signal.item())

    if len(rf.t) > 1:
        rf_dt = float(rf.t[1] - rf.t[0])
    else:
        rf_dt = float(rf.shape_dur)

    indices = np.floor(local_time[active_mask] / rf_dt).astype(int)
    indices = np.clip(indices, 0, len(rf.signal) - 1)
    # MOD: Keep RF phase offset handling explicit so the simulator can move the
    # MOD: RF carrier into the Bloch off-resonance term for slice selection.
    phase = get_rf_phase_offset_rad(rf, gamma_hz=gamma_hz, system_b0_t=system_b0_t)
    # NEW: Preserve the old behaviour for callers that still want the RF carrier
    # NEW: encoded directly in the complex RF waveform.
    if include_freq_offset_phase:
        phase += 2.0 * np.pi * get_rf_frequency_offset_hz(
            rf,
            gamma_hz=gamma_hz,
            system_b0_t=system_b0_t,
        ) * local_time[active_mask]
    signal[active_mask] = np.asarray(rf.signal, dtype=np.complex128)[indices] * np.exp(1j * phase)
    return signal if is_array_input else complex(signal.item())


def get_adc_sample_times(adc) -> np.ndarray:
    """Return ADC sample times relative to the start of a block."""
    if adc is None:
        return np.zeros(0, dtype=np.float64)
    return (np.arange(adc.num_samples, dtype=np.float64) + 0.5) * float(adc.dwell) + float(adc.delay)


def _get_gradient_boundaries(grad) -> list[float]:
    if grad is None:
        return []
    if grad.type == "trap":
        return [
            float(grad.delay),
            float(grad.delay + grad.rise_time),
            float(grad.delay + grad.rise_time + grad.flat_time),
            float(grad.delay + grad.rise_time + grad.flat_time + grad.fall_time),
        ]

    sample_times = np.asarray(getattr(grad, "tt", getattr(grad, "tt")), dtype=np.float64)
    if sample_times.size == 0:
        return [float(grad.delay), float(grad.delay + grad.shape_dur)]
    if sample_times.size == 1:
        return [float(grad.delay), float(grad.delay + grad.shape_dur)]

    midpoints = 0.5 * (sample_times[1:] + sample_times[:-1])
    edges = np.concatenate(([0.0], midpoints, [float(grad.shape_dur)]))
    return list(float(grad.delay) + edges)


def build_time_grid(block, max_dt: float) -> np.ndarray:
    """Create a refined time grid for the fine solve path."""
    if max_dt <= 0:
        raise ValueError("max_dt must be positive.")

    block_duration = float(block.block_duration)
    key_times = [0.0, block_duration]

    if getattr(block, "rf", None) is not None:
        rf = block.rf
        if len(rf.t) > 1:
            rf_dt = float(rf.t[1] - rf.t[0])
        else:
            rf_dt = float(rf.shape_dur)
        rf_edges = float(rf.delay) + np.arange(len(rf.signal) + 1, dtype=np.float64) * rf_dt
        key_times.extend(rf_edges.tolist())

    key_times.extend(_get_gradient_boundaries(getattr(block, "gx", None)))
    key_times.extend(_get_gradient_boundaries(getattr(block, "gy", None)))
    key_times.extend(_get_gradient_boundaries(getattr(block, "gz", None)))
    key_times.extend(get_adc_sample_times(getattr(block, "adc", None)).tolist())

    filtered = sorted(
        {
            round(time_point, 15)
            for time_point in key_times
            if -1e-15 <= time_point <= block_duration + 1e-15
        }
    )
    if not filtered:
        return np.array([0.0, block_duration], dtype=np.float64)

    filtered[0] = 0.0
    filtered[-1] = block_duration

    refined = [filtered[0]]
    for start, stop in zip(filtered[:-1], filtered[1:]):
        interval = stop - start
        if interval <= 0:
            continue
        num_steps = max(1, int(np.ceil(interval / max_dt)))
        step = interval / num_steps
        for step_idx in range(1, num_steps + 1):
            refined.append(start + step_idx * step)

    refined_array = np.asarray(refined, dtype=np.float64)
    refined_array[0] = 0.0
    refined_array[-1] = block_duration
    return refined_array
