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

此模块已优化支持CuPy GPU加速，在GPU可用时自动使用GPU计算，
GPU不可用时回退到NumPy CPU计算。
"""

from __future__ import annotations

from .device_manager import get_xp, device_manager

xp = get_xp()
TWO_PI = 2.0 * xp.pi


def build_off_resonance_rad_s(
    *,
    gamma_hz: float,
    chemical_shift_hz: xp.ndarray,
    dB0_t: xp.ndarray,
    dwrnd_rad_s: xp.ndarray,
    x_m: xp.ndarray,
    y_m: xp.ndarray,
    z_m: xp.ndarray,
    gx_hz_per_m: float = 0.0,
    gy_hz_per_m: float = 0.0,
    gz_hz_per_m: float = 0.0,
) -> xp.ndarray:
    """Build the angular off-resonance term for the current time step.
    
    使用CuPy加速此计算，当GPU可用时。
    """
    off_resonance = xp.asarray(dwrnd_rad_s, dtype=xp.float64).copy()
    off_resonance += TWO_PI * xp.asarray(chemical_shift_hz, dtype=xp.float64)
    off_resonance += TWO_PI * gamma_hz * xp.asarray(dB0_t, dtype=xp.float64)
    off_resonance += TWO_PI * (
        gz_hz_per_m * xp.asarray(z_m, dtype=xp.float64)
        + gy_hz_per_m * xp.asarray(y_m, dtype=xp.float64)
        + gx_hz_per_m * xp.asarray(x_m, dtype=xp.float64)
    )
    return off_resonance


def combine_transmit_field_hz(
    rf_hz: complex | xp.ndarray,
    tx_coil_magnitude: xp.ndarray | None,
    tx_coil_phase: xp.ndarray | None,
) -> xp.ndarray:
    """Combine a PyPulseq RF sample with transmit coil sensitivities.
    
    使用CuPy加速此计算，当GPU可用时。
    """
    rf_vector = xp.atleast_1d(xp.asarray(rf_hz, dtype=xp.complex128))

    if tx_coil_magnitude is None or tx_coil_phase is None:
        if rf_vector.size != 1:
            raise ValueError("Per-coil RF input requires transmit sensitivity maps.")
        return xp.full(1, rf_vector[0], dtype=xp.complex128)

    tx_mag = xp.asarray(tx_coil_magnitude, dtype=xp.float64)
    tx_phase = xp.asarray(tx_coil_phase, dtype=xp.float64)

    if tx_mag.shape != tx_phase.shape:
        raise ValueError("Transmit magnitude and phase maps must share the same shape.")
    if tx_mag.ndim != 2:
        raise ValueError("Transmit sensitivity maps must have shape (coil, voxel).")

    num_coils = tx_mag.shape[0]
    if rf_vector.size == 1:
        rf_vector = xp.repeat(rf_vector, num_coils)
    elif rf_vector.size != num_coils:
        raise ValueError("RF input must be a scalar or contain one value per transmit coil.")

    coil_weights = tx_mag * xp.exp(1j * tx_phase)
    return xp.sum(rf_vector[:, None] * coil_weights, axis=0)


def apply_bloch_step(
    *,
    rho: xp.ndarray,
    t1: xp.ndarray,
    t2: xp.ndarray,
    mx: xp.ndarray,
    my: xp.ndarray,
    mz: xp.ndarray,
    off_resonance_rad_s: xp.ndarray,
    dt_s: float,
    rf_hz: complex | xp.ndarray = 0.0j,
    tx_coil_magnitude: xp.ndarray | None = None,
    tx_coil_phase: xp.ndarray | None = None,
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """Advance magnetization by one short Bloch step.

    The update uses an exact rigid rotation for the field term and then applies
    relaxation as a split operator. This is accurate for the small time steps
    used in the fine solve path.
    
    使用CuPy加速此计算，当GPU可用时。这是计算最密集的核心函数，
    GPU加速将显著提升大规模仿真的性能。
    """
    if dt_s <= 0:
        return mx, my, mz

    wz = xp.asarray(off_resonance_rad_s, dtype=xp.float64)

    has_rf = xp.any(xp.abs(rf_hz) > 0)
    if has_rf:
        b1_hz = combine_transmit_field_hz(rf_hz, tx_coil_magnitude, tx_coil_phase)
        if b1_hz.size == 1 and wz.shape != (1,):
            b1_hz = xp.full(wz.shape, b1_hz[0], dtype=xp.complex128)
        wx = TWO_PI * xp.real(b1_hz)
        wy = TWO_PI * xp.imag(b1_hz)
    else:
        wx = xp.zeros_like(wz)
        wy = xp.zeros_like(wz)

    omega_mag = xp.sqrt(wx * wx + wy * wy + wz * wz)
    active = omega_mag > 0.0

    if xp.any(active):
        nx = wx[active] / omega_mag[active]
        ny = wy[active] / omega_mag[active]
        nz = wz[active] / omega_mag[active]

        theta = omega_mag[active] * dt_s
        sin_theta = xp.sin(theta)
        cos_theta = xp.cos(theta)
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

    e2 = xp.exp(-dt_s / xp.maximum(xp.asarray(t2, dtype=xp.float64), 1e-12))
    e1 = xp.exp(-dt_s / xp.maximum(xp.asarray(t1, dtype=xp.float64), 1e-12))
    mx *= e2
    my *= e2
    mz[:] = mz * e1 + xp.asarray(rho, dtype=xp.float64) * (1.0 - e1)
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
    
    使用CuPy加速此计算，当GPU可用时。
    """
    del TxCoilNum

    rf_amp = xp.atleast_1d(xp.asarray(rfAmp, dtype=xp.float64))
    rf_phase = xp.atleast_1d(xp.asarray(rfPhase, dtype=xp.float64))
    rf_freq = xp.atleast_1d(xp.asarray(rfFreq, dtype=xp.float64))

    if rf_amp.size == 0:
        rf_hz = 0.0j
    else:
        # MOD: Keep the RF waveform in baseband and represent the carrier in the
        # MOD: effective off-resonance term below.
        rf_hz = rf_amp * xp.exp(1j * rf_phase)

    off_resonance = build_off_resonance_rad_s(
        gamma_hz=float(Gyro),
        chemical_shift_hz=xp.asarray(CS, dtype=xp.float64),
        dB0_t=xp.asarray(dB0, dtype=xp.float64),
        dwrnd_rad_s=xp.asarray(dWRnd, dtype=xp.float64),
        z_m=xp.asarray(Gzgrid, dtype=xp.float64),
        y_m=xp.asarray(Gygrid, dtype=xp.float64),
        x_m=xp.asarray(Gxgrid, dtype=xp.float64),
        gz_hz_per_m=float(GzAmp),
        gy_hz_per_m=float(GyAmp),
        gx_hz_per_m=float(GxAmp),
    )
    # NEW: Apply the RF carrier as a detuning shift in the legacy dense-kernel path.
    rf_carrier_hz = float(rf_freq[0]) if rf_freq.size > 0 else 0.0
    off_resonance = off_resonance - TWO_PI * rf_carrier_hz

    return apply_bloch_step(
        rho=xp.asarray(Rho, dtype=xp.float64),
        t1=xp.asarray(T1, dtype=xp.float64),
        t2=xp.asarray(T2, dtype=xp.float64),
        mx=xp.asarray(Mx, dtype=xp.float64),
        my=xp.asarray(My, dtype=xp.float64),
        mz=xp.asarray(Mz, dtype=xp.float64),
        off_resonance_rad_s=off_resonance,
        dt_s=float(dt),
        rf_hz=rf_hz,
        tx_coil_magnitude=xp.asarray(TxCoilmg, dtype=xp.float64),
        tx_coil_phase=xp.asarray(TxCoilpe, dtype=xp.float64),
    )
