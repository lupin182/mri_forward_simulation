import numpy as np
import pypulseq as pp

from Sequence.write_epi import write_epi_sequence
from Sequence.write_gre_label import write_gre_label_sequence
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom
from simulate import (
    SimulationConfig,
    _apply_fast_block,
    _simulate_fine_block,
    analyze_sequence_blocks,
)
from utils import preprocess_phantom


def _make_phantom(nx: int = 8, ny: int = 8, fov: float = 0.22) -> Phantom:
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=nx, Ny=ny)
    return Phantom(rho, t1, t2, fov_x=fov, fov_y=fov, slice_thickness=3e-3)


def test_analyze_sequence_blocks_routes_rf_and_adc_blocks_to_fine_path():
    seq = write_epi_sequence(n_slices=1, n_x=8, n_y=4)
    summaries = analyze_sequence_blocks(seq)

    assert [summary.solve_path for summary in summaries] == [
        "fine",
        "fast",
        "fine",
        "fast",
        "fine",
        "fast",
        "fine",
        "fast",
        "fine",
        "fast",
    ]
    assert summaries[0].has_rf and not summaries[0].has_adc
    assert summaries[2].has_adc and not summaries[2].has_rf


def test_label_only_block_stays_on_fast_path():
    seq = write_gre_label_sequence(n_slices=1, n_x=8, n_y=4)
    first_block = analyze_sequence_blocks(seq)[0]

    assert first_block.solve_path == "fast"
    assert not first_block.has_rf
    assert not first_block.has_adc
    assert first_block.duration == 0.0


def test_fast_path_matches_fine_path_for_gradient_only_block():
    system = pp.Opts(max_grad=28, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    seq = pp.Sequence(system)
    gx = pp.make_trapezoid(channel="x", system=system, area=40.0, duration=1e-3)
    seq.add_block(gx)
    block = seq.get_block(1)

    phantom_fast = preprocess_phantom(_make_phantom(nx=6, ny=6))
    phantom_fine = preprocess_phantom(_make_phantom(nx=6, ny=6))

    rng = np.random.default_rng(7)
    mx_init = rng.normal(scale=0.3, size=phantom_fast.Mx.shape)
    my_init = rng.normal(scale=0.3, size=phantom_fast.My.shape)
    mz_init = rng.uniform(low=0.2, high=1.0, size=phantom_fast.Mz.shape)

    for phantom_state in (phantom_fast, phantom_fine):
        phantom_state.Mx[:] = mx_init
        phantom_state.My[:] = my_init
        phantom_state.Mz[:] = mz_init

    gamma_hz = float(seq.system.gamma)
    _apply_fast_block(phantom_fast, block, gamma_hz)
    adc_samples, _ = _simulate_fine_block(phantom_fine, block, gamma_hz, SimulationConfig(fine_dt=1e-6))

    assert adc_samples == []
    assert np.allclose(phantom_fast.Mx, phantom_fine.Mx, atol=5e-4, rtol=1e-4)
    assert np.allclose(phantom_fast.My, phantom_fine.My, atol=5e-4, rtol=1e-4)
    assert np.allclose(phantom_fast.Mz, phantom_fine.Mz, atol=5e-7, rtol=1e-6)
