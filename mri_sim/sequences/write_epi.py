import numpy as np
import pypulseq as pp

from mri_sim.system_config import get_pypulseq_system


def write_epi_sequence(
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 3e-3,
    n_slices: int = 1,
):
    """Create a basic EPI sequence without ramp-sampling.

    Parameters
    ----------
    fov : float or tuple of float, optional
        Field of view in meters. If a single value, it is used for both x and y.
        If a tuple, it is (fov_x, fov_y). Default is 220e-3.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase encoding steps. Default is 64.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 3e-3.
    n_slices : int, optional
        Number of slices. Default is 3.

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
    adc_dwell = 4e-6
    adc_duration = n_x * adc_dwell
    gx_flat_time = adc_duration
    gx_flat_time = np.ceil(gx_flat_time * 1e5) * 1e-5  # Round-up to the gradient raster
    gx = pp.make_trapezoid(
        channel='x',
        system=system,
        amplitude=k_width / adc_duration,
        flat_time=gx_flat_time,
    )
    adc = pp.make_adc(
        num_samples=n_x,
        duration=adc_duration,
        delay=gx.rise_time + gx_flat_time / 2 - (adc_duration - adc_dwell) / 2,
    )

    # Pre-phasing gradients
    pre_time = 8e-4
    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=pre_time)
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=pre_time)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=-n_y / 2 * delta_ky, duration=pre_time)

    # Phase blip in the shortest possible time
    gy_blip_duration = 2 * np.sqrt(delta_ky / system.max_slew)
    gy_blip_duration = np.ceil(gy_blip_duration / 10e-6) * 10e-6
    gy = pp.make_trapezoid(channel='y', system=system, area=delta_ky, duration=gy_blip_duration)

    # Loop over slices
    for i_slice in range(n_slices):
        rf.freq_offset = gz.amplitude * slice_thickness * (i_slice - (n_slices - 1) / 2)
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        for _ in range(n_y):
            seq.add_block(gx, adc)  # Read one line of k-space
            seq.add_block(gy)  # Phase blip
            gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='epi')

    return seq


