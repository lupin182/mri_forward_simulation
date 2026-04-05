import time
import numpy as np
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom
from Sequence.write_gre_label import write_gre_label_sequence
from simulate import simulate, SimulationConfig
from reconn import reconstruct_image_fft

N = 16
config = SimulationConfig(fine_dt=2e-5)
rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=N, Ny=N)
phantom = Phantom(rho, t1, t2, fov_x=0.22, fov_y=0.22, slice_thickness=3e-3)
print('phantom ready')
seq = write_gre_label_sequence(n_x=N, n_y=N, fov=(0.22,0.22), slice_thickness=3e-3, tr=12e-3, te=5e-3, n_slices=1)
print('seq ready')
start = time.perf_counter()
sig = simulate(phantom, seq, config)
print('simulate', time.perf_counter() - start)
start = time.perf_counter()
img_fft, k2d = reconstruct_image_fft(sig, Ny=N, Nx=N)
print('recon', time.perf_counter() - start)
print('done', np.abs(sig).max(), np.abs(k2d).mean(), np.abs(img_fft).mean())
