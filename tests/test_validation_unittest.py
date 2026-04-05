import unittest

import numpy as np
import pypulseq as pp

from Sequence.write_epi import write_epi_sequence
from Sequence.write_gre import write_gre_sequence
from Sequence.write_gre_label import write_gre_label_sequence
from bloch_kernel import apply_bloch_step
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom
from simulate import (
    SimulationConfig,
    _apply_fast_block,
    _simulate_fine_block,
    analyze_sequence_blocks,
    simulate,
)
from utils import preprocess_phantom


def _make_phantom(nx: int = 8, ny: int = 8, fov: float = 0.22) -> Phantom:
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=nx, Ny=ny)
    return Phantom(rho, t1, t2, fov_x=fov, fov_y=fov, slice_thickness=3e-3)


class BlochKernelTests(unittest.TestCase):
    def test_rf_pulse_rotates_longitudinal_magnetization(self) -> None:
        mx = np.array([0.0], dtype=np.float64)
        my = np.array([0.0], dtype=np.float64)
        mz = np.array([1.0], dtype=np.float64)

        apply_bloch_step(
            rho=np.array([1.0]),
            t1=np.array([1e12]),
            t2=np.array([1e12]),
            mx=mx,
            my=my,
            mz=mz,
            off_resonance_rad_s=np.array([0.0]),
            dt_s=0.25,
            rf_hz=1.0,
        )

        self.assertTrue(np.allclose(mx, 0.0, atol=1e-8))
        self.assertTrue(np.allclose(my, 1.0, atol=1e-8))
        self.assertTrue(np.allclose(mz, 0.0, atol=1e-8))


class SimulationFrameworkTests(unittest.TestCase):
    def test_block_routing_matches_rf_adc_content(self) -> None:
        seq = write_epi_sequence(n_slices=1, n_x=8, n_y=4)
        summaries = analyze_sequence_blocks(seq)

        self.assertEqual(
            [summary.solve_path for summary in summaries],
            ["fine", "fast", "fine", "fast", "fine", "fast", "fine", "fast", "fine", "fast"],
        )
        self.assertTrue(summaries[0].has_rf)
        self.assertFalse(summaries[0].has_adc)
        self.assertTrue(summaries[2].has_adc)
        self.assertFalse(summaries[2].has_rf)

    def test_label_only_block_uses_fast_path(self) -> None:
        seq = write_gre_label_sequence(n_slices=1, n_x=8, n_y=4)
        summary = analyze_sequence_blocks(seq)[0]

        self.assertEqual(summary.solve_path, "fast")
        self.assertFalse(summary.has_rf)
        self.assertFalse(summary.has_adc)
        self.assertEqual(summary.duration, 0.0)

    def test_fast_path_matches_fine_reference_for_gradient_only_block(self) -> None:
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

        self.assertEqual(adc_samples, [])
        self.assertTrue(np.allclose(phantom_fast.Mx, phantom_fine.Mx, atol=5e-4, rtol=1e-4))
        self.assertTrue(np.allclose(phantom_fast.My, phantom_fine.My, atol=5e-4, rtol=1e-4))
        self.assertTrue(np.allclose(phantom_fast.Mz, phantom_fine.Mz, atol=5e-7, rtol=1e-6))


class EndToEndTests(unittest.TestCase):
    def test_gre_sequence_returns_adc_aligned_signal(self) -> None:
        phantom = _make_phantom(nx=8, ny=8)
        seq = write_gre_sequence(
            n_x=8,
            n_y=4,
            fov=(0.22, 0.22),
            slice_thickness=3e-3,
            tr=12e-3,
            te=5e-3,
        )

        result = simulate(phantom, seq, SimulationConfig(fine_dt=5e-6), return_details=True)
        signal_only = simulate(phantom, seq, SimulationConfig(fine_dt=5e-6))

        self.assertEqual(result.signal.shape, result.adc_times.shape)
        self.assertEqual(result.signal.size, 32)
        self.assertTrue(np.all(np.isfinite(result.signal)))
        self.assertGreater(np.max(np.abs(result.signal)), 0.0)
        self.assertTrue(any(summary.solve_path == "fine" for summary in result.block_summaries))
        self.assertTrue(any(summary.solve_path == "fast" for summary in result.block_summaries))
        self.assertTrue(np.allclose(signal_only, result.signal))


if __name__ == "__main__":
    unittest.main(verbosity=2)
