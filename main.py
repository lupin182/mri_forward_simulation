"""Project entry point for a Cartesian GRE forward-simulation demo."""

import matplotlib.pyplot as plt
import numpy as np

from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom
from reconn import reconstruct_image_fft
from Sequence.write_gre_label import write_gre_label_sequence
from simulate import SimulationConfig, simulate


def main() -> None:
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64)
    phantom = Phantom(rho, t1, t2, fov_x=0.22, fov_y=0.22, slice_thickness=3e-3)

    # The current forward model uses one isochromat per voxel, so RF spoiling
    # creates stronger artifacts than a scanner would. Disable it for the demo.
    seq = write_gre_label_sequence(
        n_x=64,
        n_y=64,
        fov=(0.22, 0.22),
        slice_thickness=3e-3,
        tr=12e-3,
        te=5e-3,
        n_slices=1,
        rf_spoiling_inc_deg=0.0,
    )
    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=2e-5))
    image_recon, _ = reconstruct_image_fft(k_space_signal, Ny=64, Nx=64)

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title("GRE Reconstruction")
    plt.imshow(np.abs(image_recon), cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.title("Original")
    plt.imshow(rho[0, 0, 0], cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
