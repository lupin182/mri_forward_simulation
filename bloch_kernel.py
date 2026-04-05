"""Bloch update helpers used by the block-driven simulator.

The new simulator keeps the magnetization in the main rotating frame and uses
PyPulseq-native units:

- RF waveform samples are expressed in Hz.
- Gradient amplitudes are expressed in Hz/m.
- Chemical shift is expressed in Hz.
- ``dB0`` is expressed in Tesla and converted with ``gamma_hz``.
- ``dWRnd`` is treated as an angular off-resonance term in rad/s.

The public ``BlochKernel`` name is kept for backward compatibility, while the
new simulator uses ``apply_bloch_step`` and ``build_off_resonance_rad_s``.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def build_off_resonance_rad_s(
    *,
    gamma_hz: float,
    chemical_shift_hz: np.ndarray,
    dB0_t: np.ndarray,
    dwrnd_rad_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    gx_hz_per_m: float = 0.0,
    gy_hz_per_m: float = 0.0,
    gz_hz_per_m: float = 0.0,
) -> np.ndarray:
    """Build the angular off-resonance term for the current time step."""
    off_resonance = np.asarray(dwrnd_rad_s, dtype=np.float64).copy()
    off_resonance += TWO_PI * np.asarray(chemical_shift_hz, dtype=np.float64)
    off_resonance += TWO_PI * gamma_hz * np.asarray(dB0_t, dtype=np.float64)
    off_resonance += TWO_PI * (
        gz_hz_per_m * np.asarray(z_m, dtype=np.float64)
        + gy_hz_per_m * np.asarray(y_m, dtype=np.float64)
        + gx_hz_per_m * np.asarray(x_m, dtype=np.float64)
    )
    return off_resonance


def combine_transmit_field_hz(
    rf_hz: complex | np.ndarray,
    tx_coil_magnitude: np.ndarray | None,
    tx_coil_phase: np.ndarray | None,
) -> np.ndarray:
    """Combine a PyPulseq RF sample with transmit coil sensitivities."""
    rf_vector = np.atleast_1d(np.asarray(rf_hz, dtype=np.complex128))

    if tx_coil_magnitude is None or tx_coil_phase is None:
        if rf_vector.size != 1:
            raise ValueError("Per-coil RF input requires transmit sensitivity maps.")
        return np.full(1, rf_vector[0], dtype=np.complex128)

    tx_mag = np.asarray(tx_coil_magnitude, dtype=np.float64)
    tx_phase = np.asarray(tx_coil_phase, dtype=np.float64)

    if tx_mag.shape != tx_phase.shape:
        raise ValueError("Transmit magnitude and phase maps must share the same shape.")
    if tx_mag.ndim != 2:
        raise ValueError("Transmit sensitivity maps must have shape (coil, voxel).")

    num_coils = tx_mag.shape[0]
    if rf_vector.size == 1:
        rf_vector = np.repeat(rf_vector, num_coils)
    elif rf_vector.size != num_coils:
        raise ValueError("RF input must be a scalar or contain one value per transmit coil.")

    coil_weights = tx_mag * np.exp(1j * tx_phase)
    return np.sum(rf_vector[:, None] * coil_weights, axis=0)


def apply_bloch_step(
    *,
    rho: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    off_resonance_rad_s: np.ndarray,
    dt_s: float,
    rf_hz: complex | np.ndarray = 0.0j,
    tx_coil_magnitude: np.ndarray | None = None,
    tx_coil_phase: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance magnetization by one short Bloch step.

    The update uses an exact rigid rotation for the field term and then applies
    relaxation as a split operator. This is accurate for the small time steps
    used in the fine solve path.
    """
    if dt_s <= 0:
        return mx, my, mz

    wz = np.asarray(off_resonance_rad_s, dtype=np.float64)

    has_rf = np.any(np.abs(rf_hz) > 0)
    if has_rf:
        b1_hz = combine_transmit_field_hz(rf_hz, tx_coil_magnitude, tx_coil_phase)
        if b1_hz.size == 1 and wz.shape != (1,):
            b1_hz = np.full(wz.shape, b1_hz[0], dtype=np.complex128)
        wx = TWO_PI * np.real(b1_hz)
        wy = TWO_PI * np.imag(b1_hz)
    else:
        wx = np.zeros_like(wz)
        wy = np.zeros_like(wz)

    omega_mag = np.sqrt(wx * wx + wy * wy + wz * wz)
    active = omega_mag > 0.0

    if np.any(active):
        nx = wx[active] / omega_mag[active]
        ny = wy[active] / omega_mag[active]
        nz = wz[active] / omega_mag[active]

        theta = omega_mag[active] * dt_s
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        one_minus_cos = 1.0 - cos_theta

        mx0 = mx[active]
        my0 = my[active]
        mz0 = mz[active]

        # Bloch uses dM/dt = gamma * M x B, so the cross term is M x n.
        cross_x = my0 * nz - mz0 * ny
        cross_y = mz0 * nx - mx0 * nz
        cross_z = mx0 * ny - my0 * nx
        dot = mx0 * nx + my0 * ny + mz0 * nz

        mx[active] = mx0 * cos_theta + cross_x * sin_theta + nx * dot * one_minus_cos
        my[active] = my0 * cos_theta + cross_y * sin_theta + ny * dot * one_minus_cos
        mz[active] = mz0 * cos_theta + cross_z * sin_theta + nz * dot * one_minus_cos

    e2 = np.exp(-dt_s / np.maximum(np.asarray(t2, dtype=np.float64), 1e-12))
    e1 = np.exp(-dt_s / np.maximum(np.asarray(t1, dtype=np.float64), 1e-12))
    mx *= e2
    my *= e2
    mz[:] = mz * e1 + np.asarray(rho, dtype=np.float64) * (1.0 - e1)
    return mx, my, mz


def BlochKernel(
    Gyro,
    CS,
    Rho,
    T1,
    T2,
    Mz,
    My,
    Mx,
    dB0,
    dWRnd,
    Gzgrid,
    Gygrid,
    Gxgrid,
    TxCoilmg,
    TxCoilpe,
    dt,
    rfAmp,
    rfPhase,
    rfFreq,
    GzAmp,
    GyAmp,
    GxAmp,
    TxCoilNum,
):
    """Backward-compatible wrapper around the new Bloch step implementation.

    ``rfFreq`` is treated as an RF carrier detuning term and is subtracted from
    the effective spin off-resonance, matching the physical handling used by
    the block-driven simulator for PyPulseq RF ``freq_offset``.
    """
    del TxCoilNum

    rf_amp = np.atleast_1d(np.asarray(rfAmp, dtype=np.float64))
    rf_phase = np.atleast_1d(np.asarray(rfPhase, dtype=np.float64))
    rf_freq = np.atleast_1d(np.asarray(rfFreq, dtype=np.float64))

    if rf_amp.size == 0:
        rf_hz = 0.0j
    else:
        # MOD: Keep the RF waveform in baseband and represent the carrier in the
        # MOD: effective off-resonance term below.
        rf_hz = rf_amp * np.exp(1j * rf_phase)

    off_resonance = build_off_resonance_rad_s(
        gamma_hz=float(Gyro),
        chemical_shift_hz=np.asarray(CS, dtype=np.float64),
        dB0_t=np.asarray(dB0, dtype=np.float64),
        dwrnd_rad_s=np.asarray(dWRnd, dtype=np.float64),
        z_m=np.asarray(Gzgrid, dtype=np.float64),
        y_m=np.asarray(Gygrid, dtype=np.float64),
        x_m=np.asarray(Gxgrid, dtype=np.float64),
        gz_hz_per_m=float(GzAmp),
        gy_hz_per_m=float(GyAmp),
        gx_hz_per_m=float(GxAmp),
    )
    # NEW: Apply the RF carrier as a detuning shift in the legacy dense-kernel path.
    rf_carrier_hz = float(rf_freq[0]) if rf_freq.size > 0 else 0.0
    off_resonance = off_resonance - TWO_PI * rf_carrier_hz

    return apply_bloch_step(
        rho=np.asarray(Rho, dtype=np.float64),
        t1=np.asarray(T1, dtype=np.float64),
        t2=np.asarray(T2, dtype=np.float64),
        mx=np.asarray(Mx, dtype=np.float64),
        my=np.asarray(My, dtype=np.float64),
        mz=np.asarray(Mz, dtype=np.float64),
        off_resonance_rad_s=off_resonance,
        dt_s=float(dt),
        rf_hz=rf_hz,
        tx_coil_magnitude=np.asarray(TxCoilmg, dtype=np.float64),
        tx_coil_phase=np.asarray(TxCoilpe, dtype=np.float64),
    )
