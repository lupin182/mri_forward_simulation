import numpy as np
from Sequence.write_gre_label import write_gre_label_sequence
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom
from simulate import SimulationConfig, simulate

def test_gre_end_to_end_simulation_returns_adc_aligned_signal():
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64)

    phantom = Phantom(rho, t1, t2, fov_x=0.256, fov_y=0.256, slice_thickness=3e-3)
    seq = write_gre_label_sequence(
        n_x=64,
        n_y=64,
        fov=0.256,
        slice_thickness=3e-3,
        tr=12e-3,
        te=5e-3,
        readout_duration=1e-3,
    )


    result = simulate(phantom, seq, SimulationConfig(fine_dt=5e-6), return_details=True)

    assert result.signal.shape == result.adc_times.shape
    assert result.signal.size == 32
    assert np.all(np.isfinite(result.signal))
    assert np.max(np.abs(result.signal)) > 0.0
    assert any(summary.solve_path == "fine" for summary in result.block_summaries)
    assert any(summary.solve_path == "fast" for summary in result.block_summaries)
    signal_only = simulate(phantom, seq, SimulationConfig(fine_dt=1e-6))
    assert np.allclose(signal_only, result.signal)
    k_traj_adc,_,_,_,_ = seq.calculate_kspace()
    return signal_only, k_traj_adc
