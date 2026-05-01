import matplotlib.pyplot as plt
import numpy as np
import pydicom
from phantom.make_phantom import generate_simple_sphere_phantom
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom, generate_multi_spin_sphere_phantom
from Sequence.write_gre_label import write_gre_label_sequence
from Sequence.write_epi import write_epi_sequence
from Sequence.write_se import write_se_sequence
from Sequence.write_epi_se import write_epi_se_sequence
from Sequence.write_epi_label import write_epi_label_sequence
from Sequence.write_gre import write_gre_sequence
from simulate import SimulationConfig, simulate
import numpy as np
import matplotlib.pyplot as plt
from generate_artifact import generate_rf_artifact, generate_rf_artifact_real
import pypulseq as pp
from recon import reconstruct_3d_cartesian_fft,reconstruct_3d_cartesian_fft_multichannel
from phantom.make_phantom import generate_coil_sensitivity_maps,generate_diff_coil_sensitivity_maps
from recon import sos_reconstruction
def main() -> None:

    #rho, t1, t2, dWRnd = generate_multi_spin_sphere_phantom(Nz=1, Nx=64, Ny=64, radius=16, Nspins=32)
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64)
    FOV_x =  0.256#dx*Nx # 单位：米
    FOV_y =  0.256#dy*Ny # 单位：米

    phantom = Phantom(rho, t1, t2, fov_x=FOV_x, fov_y=FOV_y, slice_thickness=0.004)
    #phantom.rxCoilmg, phantom.rxCoilpe, phantom.RxCoilNum = generate_diff_coil_sensitivity_maps(phantom.Nx, phantom.Ny, phantom.Nz, n_coils=4)

    #phantom.dWRnd = dWRnd
    # The current forward model uses one isochromat per voxel, so RF spoiling
    # creates stronger artifacts than a scanner would. Disable it for the demo.

    seq = pp.Sequence()
    seq.read('epi_se_pypulseq.seq')
    #seq=write_gre_sequence(ideal_spoiling_reset=True)
    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    _, _, _, t_adc, _ = seq.waveforms_and_times()

    k_space_signal = k_space_signal.squeeze()

    img_nonnoise, _ =  reconstruct_3d_cartesian_fft_multichannel(k_space_signal.T, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)

    k_space_signal = generate_rf_artifact_real(t_adc, k_space_signal, rf_noise_freq=[127.7e6], rf_noise_amp=[5.0], bg_noise_amp=1.0)

    

    image_recon,_ = reconstruct_3d_cartesian_fft_multichannel(k_space_signal.T, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)

    image_recon = sos_reconstruction(image_recon)

    #plot_color_overlay(image_recon[0], rho[0,0,0])
    #np.save('image_recon.npy', image_recon)
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title("Reconstruction with RF artifact")
    plt.imshow(np.abs(image_recon), cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.title("Reconstruction without RF artifact")
    plt.imshow(np.abs(img_nonnoise[0]),  cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
