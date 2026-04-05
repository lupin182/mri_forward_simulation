'''项目主入口'''
from reconn import reconstruct_image_fft, reconstruct_image
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom, load_simple_phantom
import pypulseq as pp
from Sequence.write_gre_label import write_gre_label_sequence
from Sequence.write_epi import write_epi_sequence
import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate, SimulationConfig
from tests.test_simulation_end_to_end import test_gre_end_to_end_simulation_returns_adc_aligned_signal
from reconn import reconstruct_image_fft, reconstruct_image
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom

rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64)

phantom = Phantom(rho, t1, t2, fov_x=0.22, fov_y=0.22, slice_thickness=3e-3)

seq = write_gre_label_sequence(
        n_x=64,
        n_y=64,
        fov=(0.22, 0.22),
        slice_thickness=3e-3,
        tr=12e-3,
        te=5e-3,
        n_slices=1,
    )
k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-6))
k_traj_adc,_,_,_,_ = seq.calculate_kspace()
image_recon = reconstruct_image(k_space_signal, k_traj_adc, fov_x=phantom.fov_x, fov_y=phantom.fov_y, Nx=64, Ny=64)
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.title("Reconstruction")
plt.imshow(np.abs(image_recon),cmap='gray')
plt.subplot(122)
plt.title("Original")
plt.imshow(rho[0,0,0,],cmap='gray')
plt.show()
