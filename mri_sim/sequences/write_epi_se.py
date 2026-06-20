import numpy as np
import pypulseq as pp

from mri_sim.system_config import get_pypulseq_system

def write_epi_se_sequence(
    fov: float | tuple[float, float] = 256e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 3e-3,
    te: float = 200e-3,
):
    """Create a spin-echo EPI sequence.

    Parameters
    ----------
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 256e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    te : float, optional
        Echo time in seconds. Default is 60e-3.

    Returns
    -------
    seq : pypulseq.Sequence
        The EPI sequence object.
    """
    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    system = get_pypulseq_system()

    seq = pp.Sequence(system)

    # Create 90 degree slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        system=system,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )

    # Define other gradients and ADC events
    delta_kx = 1 / fov_x
    delta_ky = 1 / fov_y
    k_width = n_x * delta_kx
    readout_time = 3.2e-4
    gx = pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
    adc = pp.make_adc(num_samples=n_x, system=system, duration=gx.flat_time, delay=gx.rise_time)

    # Pre-phasing gradients
    pre_time = 8e-4
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
    # Do not need minus for in-plane prephasers because of the spin-echo (position reflection in k-space)
    gx_pre = pp.make_trapezoid(channel='x', system=system, area=gx.area / 2 - delta_kx / 2, duration=pre_time)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=n_y / 2 * delta_ky, duration=pre_time)

    # Phase blip in shortest possible time
    gy_blip_duration = 2 * np.sqrt(delta_ky / system.max_slew)
    gy_blip_duration = np.ceil(gy_blip_duration / seq.grad_raster_time) * seq.grad_raster_time
    while True:
        try:
            gy = pp.make_trapezoid(channel='y', system=system, area=delta_ky, duration=gy_blip_duration)
            break
        except AssertionError:
            gy_blip_duration += seq.grad_raster_time

    # Refocusing pulse with spoiling gradients
    rf180 = pp.make_block_pulse(
        flip_angle=np.pi,
        delay=system.rf_dead_time,
        system=system,
        duration=500e-6,
        use='refocusing',
    )
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz.area * 2, duration=3 * pre_time)

    # Calculate delay time
    duration_to_center = (n_x / 2 + 0.5) * pp.calc_duration(gx) + n_y / 2 * pp.calc_duration(gy)
    rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
    rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]
    te_delay_1 = (
        te / 2
        - pp.calc_duration(gz)
        + rf_center_incl_delay
        - pre_time
        - pp.calc_duration(gz_spoil)
        - rf180_center_incl_delay
    )
    te_delay_2 = (
        te / 2 - pp.calc_duration(rf180) + rf180_center_incl_delay - pp.calc_duration(gz_spoil) - duration_to_center
    )

    # Construct sequence
    seq.add_block(rf, gz)
    seq.add_block(gx_pre, gy_pre, gz_reph)
    seq.add_block(pp.make_delay(te_delay_1))
    seq.add_block(gz_spoil)
    seq.add_block(rf180)
    seq.add_block(gz_spoil)
    seq.add_block(pp.make_delay(te_delay_2))
    for _ in range(n_y):
        seq.add_block(gx, adc)  # Read one line of k-space
        seq.add_block(gy)  # Phase blip
        gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient
    seq.add_block(pp.make_delay(1e-4))

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness])
    seq.set_definition(key='Name', value='epi_se')

    return seq

