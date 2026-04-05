import numpy as np

from bloch_kernel import apply_bloch_step


def test_apply_bloch_step_rotates_magnetization_for_rf_pulse():
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

    assert np.allclose(mx, 0.0, atol=1e-8)
    assert np.allclose(my, 1.0, atol=1e-8)
    assert np.allclose(mz, 0.0, atol=1e-8)
