"""Utility helpers for the forward MRI simulator."""

from __future__ import annotations

import copy

import numpy as np

from phantom.make_phantom import Phantom


def preprocess_phantom(phantom: Phantom) -> Phantom:
    """Return a flattened copy of the phantom ready for vectorized simulation."""
    if getattr(phantom.rho, "ndim", 0) == 1:
        return phantom

    result = copy.copy(phantom)
    shape_target = (result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny)

    result.x = np.broadcast_to(result.x, shape_target)
    result.y = np.broadcast_to(result.y, shape_target)
    result.z = np.broadcast_to(result.z, shape_target)

    result.rho = np.asarray(result.rho).ravel()
    result.t1 = np.asarray(result.t1).ravel()
    result.t2 = np.asarray(result.t2).ravel()
    result.Mz = np.asarray(result.Mz).ravel()
    result.Mx = np.asarray(result.Mx).ravel()
    result.My = np.asarray(result.My).ravel()
    result.CS = np.asarray(result.CS).ravel()
    result.dB0 = np.asarray(result.dB0).ravel()
    result.dWRnd = np.asarray(result.dWRnd).ravel()

    result.x = np.asarray(result.x).ravel()
    result.y = np.asarray(result.y).ravel()
    result.z = np.asarray(result.z).ravel()

    result.txCoilmg = np.broadcast_to(
        result.txCoilmg[:, None, None, :, :, :],
        (result.TxCoilNum, result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny),
    ).reshape(result.TxCoilNum, -1)
    result.txCoilpe = np.broadcast_to(
        result.txCoilpe[:, None, None, :, :, :],
        (result.TxCoilNum, result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny),
    ).reshape(result.TxCoilNum, -1)
    result.rxCoilmg = np.broadcast_to(
        result.rxCoilmg[:, None, None, :, :, :],
        (result.RxCoilNum, result.TypeNum, result.SpinNum, result.Nz, result.Nx, result.Ny),
    ).reshape(result.RxCoilNum, -1)
    result.rxCoilpe = np.broadcast_to(
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


def sample_rf_signal(rf, t):
    """Sample the RF waveform at time ``t`` and apply phase/frequency offsets."""
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
    phase = float(rf.phase_offset) + 2.0 * np.pi * float(rf.freq_offset) * local_time[active_mask]
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
