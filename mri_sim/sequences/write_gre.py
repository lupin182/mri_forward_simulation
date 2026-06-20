import numpy as np
import pypulseq as pp

from mri_sim.system_config import get_pypulseq_system


def write_gre_sequence(
    fov: float | tuple[float, float] = 256e-3,
    n_x: int = 64,
    n_y: int = 64,
    flip_angle_deg: float = 10,
    slice_thickness: float = 3e-3,
    tr: float = 12e-3,
    te: float = 5e-3,
    rf_spoiling_inc_deg: float = 117.0,
    dummy_scans: int = 0,
    ideal_spoiling_reset: bool = False,
):
    """Create a basic gradient echo (GRE) sequence.

    Parameters
    ----------
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 256e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    flip_angle_deg : float, optional
        Flip angle in degrees. Default is 10.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    tr : float, optional
        Repetition time in seconds. Default is 12e-3.
    te : float, optional
        Echo time in seconds. Default is 5e-3.
    ideal_spoiling_reset : bool, optional
        When True, add a PyPulseq label on the TR spoiler block so the simulator
        can enforce ideal spoiling by zeroing transverse magnetization there.

    Returns
    -------
    seq : pypulseq.Sequence
        The GRE sequence object.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov
    rf_spoiling_inc = rf_spoiling_inc_deg

    system = get_pypulseq_system()

    seq = pp.Sequence(system)

    # Create slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )

    # Define other gradients and ADC events
    delta_kx = 1 / fov_x
    delta_ky = 1 / fov_y
    gx = pp.make_trapezoid(channel='x', flat_area=n_x * delta_kx, flat_time=3.2e-3, system=system)
    adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
    phase_areas = (np.arange(n_y) - n_y / 2) * delta_ky

    # Gradient spoiling
    gx_spoil = pp.make_trapezoid(channel='x', area=2 * n_x * delta_kx, system=system)
    gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

    # Calculate timing
    te_delay = (
        te
        - (pp.calc_duration(gz, rf) - pp.calc_rf_center(rf)[0] - rf.delay)
        - pp.calc_duration(gx_pre)
        - pp.calc_duration(gx) / 2
        - pp.eps
    )
    te_delay = np.ceil(te_delay / seq.grad_raster_time) * seq.grad_raster_time

    tr_delay = tr - pp.calc_duration(gz, rf) - pp.calc_duration(gx_pre) - pp.calc_duration(gx) - te_delay
    tr_delay = np.ceil(tr_delay / seq.grad_raster_time) * seq.grad_raster_time

    assert np.all(te_delay >= 0)
    assert np.all(tr_delay >= pp.calc_duration(gx_spoil, gz_spoil))

    rf_phase = 0
    rf_inc = 0

    def add_tr(phase_area: float, acquire: bool) -> None:
        nonlocal rf_phase, rf_inc
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        seq.add_block(rf, gz)
        gy_pre = pp.make_trapezoid(
            channel='y',
            area=phase_area,
            duration=pp.calc_duration(gx_pre),
            system=system,
        )
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(pp.make_delay(te_delay))
        if acquire:
            seq.add_block(gx, adc)
        else:
            seq.add_block(gx)
        gy_pre.amplitude = -gy_pre.amplitude
        spoil_events = [pp.make_delay(tr_delay), gx_spoil, gy_pre, gz_spoil]
        if ideal_spoiling_reset:
            spoil_events.append(pp.make_label(label='TRID', type='INC', value=1))
        seq.add_block(*spoil_events)

    for _ in range(dummy_scans):
        add_tr(0.0, acquire=False)

    # Loop over phase encodes and define sequence blocks
    for i_phase in range(n_y):
        add_tr(float(phase_areas[i_phase]), acquire=True)

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness])
    seq.set_definition(key='Name', value='gre')

    return seq

