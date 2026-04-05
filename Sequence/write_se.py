import math

import numpy as np
import pypulseq as pp


def _round_up_to_raster(value: float, raster: float) -> float:
    if value < 0:
        return value
    return math.ceil(value / raster) * raster


def write_se_sequence(
    plot: bool = False,
    test_report: bool = False,
    write_seq: bool = False,
    seq_filename: str = 'se_pypulseq.seq',
    *,
    fov: float | tuple[float, float] = 220e-3,
    n_x: int = 64,
    n_y: int = 64,
    slice_thickness: float = 5e-3,
    n_slices: int = 1,
    tr: float = 1.0,
    te: float = 20e-3,
    excitation_flip_angle_deg: float = 90.0,
    refocusing_flip_angle_deg: float = 180.0,
    rf_excitation_duration: float = 3e-3,
    rf_refocusing_duration: float = 3e-3,
    readout_time: float = 4e-3,
    prephase_duration: float = 1.2e-3,
):
    """Create a standard Cartesian spin-echo sequence.

    The implementation follows the project sequence style and mirrors the
    timing structure of ``Sequence/se1.py``:

    ``90 deg excitation -> prephasers -> TE/2 delay -> 180 deg refocusing
    -> echo delay -> readout -> TR fill``.

    Parameters
    ----------
    plot : bool, optional
        Plot the sequence diagram. Default is False.
    test_report : bool, optional
        Print the PyPulseq test report. Default is False.
    write_seq : bool, optional
        Write the generated sequence to a ``.seq`` file. Default is False.
    seq_filename : str, optional
        Output filename for the ``.seq`` file. Default is ``'se_pypulseq.seq'``.
    fov : float or tuple of float, optional
        In-plane field of view in meters. If a single value is provided it is
        used for both readout and phase encoding.
    n_x : int, optional
        Number of readout samples. Default is 64.
    n_y : int, optional
        Number of phase-encoding lines. Default is 64.
    slice_thickness : float, optional
        Slice thickness in meters. Default is 5 mm.
    n_slices : int, optional
        Number of slices. Slices are acquired sequentially. Default is 1.
    tr : float, optional
        Requested repetition time in seconds. Default is 1.0 s.
    te : float, optional
        Requested echo time in seconds. Default is 20 ms.
    excitation_flip_angle_deg : float, optional
        Excitation flip angle in degrees. Default is 90.
    refocusing_flip_angle_deg : float, optional
        Refocusing flip angle in degrees. Default is 180.
    rf_excitation_duration : float, optional
        Excitation RF duration in seconds. Default is 3 ms.
    rf_refocusing_duration : float, optional
        Refocusing RF duration in seconds. Default is 3 ms.
    readout_time : float, optional
        Flat readout time in seconds. Default is 4 ms.
    prephase_duration : float, optional
        Duration of the readout/phase/slice prephasers in seconds. Default is
        1.2 ms.

    Returns
    -------
    seq : pypulseq.Sequence
        The spin-echo sequence object.

    Raises
    ------
    ValueError
        If the requested TE or TR is shorter than the minimum supported by the
        generated events.
    """
    if n_x <= 0 or n_y <= 0 or n_slices <= 0:
        raise ValueError('n_x, n_y and n_slices must all be positive.')

    fov_x, fov_y = (fov, fov) if isinstance(fov, (int, float)) else fov

    system = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )
    seq = pp.Sequence(system)

    rf_exc, gz_exc, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(excitation_flip_angle_deg),
        system=system,
        duration=rf_excitation_duration,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
        use='excitation',
    )
    rf_ref, gz_ref, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(refocusing_flip_angle_deg),
        system=system,
        duration=rf_refocusing_duration,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
        delay=system.rf_dead_time,
        phase_offset=np.pi / 2,
        use='refocusing',
    )

    delta_kx = 1 / fov_x
    delta_ky = 1 / fov_y
    readout_width = n_x * delta_kx

    gx = pp.make_trapezoid(channel='x', system=system, flat_area=readout_width, flat_time=readout_time)
    adc = pp.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=system)

    gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=prephase_duration)
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz_exc.area / 2, duration=prephase_duration)
    phase_areas = (np.arange(n_y) - n_y / 2) * delta_ky

    exc_center = rf_exc.delay + pp.calc_rf_center(rf_exc)[0]
    ref_center = rf_ref.delay + pp.calc_rf_center(rf_ref)[0]
    exc_block_duration = pp.calc_duration(rf_exc, gz_exc)
    ref_block_duration = pp.calc_duration(rf_ref, gz_ref)
    pre_block_duration = pp.calc_duration(gx_pre, gz_reph)
    readout_block_duration = pp.calc_duration(gx, adc)
    readout_center = gx.delay + gx.rise_time + gx.flat_time / 2

    min_half_te_1 = exc_block_duration - exc_center + pre_block_duration + ref_center
    min_half_te_2 = ref_block_duration - ref_center + readout_center
    min_te = 2 * max(min_half_te_1, min_half_te_2)
    if te < min_te:
        raise ValueError(
            f'Requested TE={te * 1e3:.2f} ms is too short for the generated SE events. '
            f'Minimum TE is {min_te * 1e3:.2f} ms.'
        )

    te_delay_1 = te / 2 - (exc_block_duration - exc_center) - pre_block_duration - ref_center
    te_delay_2 = te / 2 - (ref_block_duration - ref_center) - readout_center
    te_delay_1 = _round_up_to_raster(te_delay_1, system.grad_raster_time)
    te_delay_2 = _round_up_to_raster(te_delay_2, system.grad_raster_time)

    if te_delay_1 < 0 or te_delay_2 < 0:
        raise ValueError('Computed TE delays are negative. Check TE and event durations.')

    min_tr = exc_block_duration + pre_block_duration + te_delay_1 + ref_block_duration + te_delay_2 + readout_block_duration
    if tr < min_tr:
        raise ValueError(
            f'Requested TR={tr * 1e3:.2f} ms is too short for the generated SE events. '
            f'Minimum TR is {min_tr * 1e3:.2f} ms.'
        )

    tr_delay = _round_up_to_raster(
        tr - exc_block_duration - pre_block_duration - te_delay_1 - ref_block_duration - te_delay_2 - readout_block_duration,
        system.grad_raster_time,
    )
    if tr_delay < 0:
        raise ValueError('Computed TR delay is negative after raster alignment.')

    for i_slice in range(n_slices):
        slice_offset = i_slice - (n_slices - 1) / 2
        rf_exc.freq_offset = gz_exc.amplitude * slice_thickness * slice_offset
        rf_ref.freq_offset = gz_ref.amplitude * slice_thickness * slice_offset

        for phase_area in phase_areas:
            gy_pre = pp.make_trapezoid(
                channel='y',
                system=system,
                area=float(phase_area),
                duration=prephase_duration,
            )

            seq.add_block(rf_exc, gz_exc)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            if te_delay_1 > 0:
                seq.add_block(pp.make_delay(te_delay_1))
            seq.add_block(rf_ref, gz_ref)
            if te_delay_2 > 0:
                seq.add_block(pp.make_delay(te_delay_2))
            seq.add_block(gx, adc)
            if tr_delay > 0:
                seq.add_block(pp.make_delay(tr_delay))

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(err) for err in error_report]

    if test_report:
        print(seq.test_report())

    if plot:
        seq.plot(time_range=(0.0, exc_block_duration + pre_block_duration + te_delay_1 + ref_block_duration + te_delay_2 + readout_block_duration + tr_delay), stacked=True, show_guides=True)

    seq.set_definition(key='FOV', value=[fov_x, fov_y, slice_thickness * n_slices])
    seq.set_definition(key='Name', value='se')

    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    write_se_sequence(plot=True, write_seq=True)
