"""Project entry point for a Cartesian GRE forward-simulation demo."""

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom, load_simple_phantom
from recon import reconstruct_3d_cartesian_fft
from Sequence.write_gre_label import write_gre_label_sequence
from Sequence.write_epi import write_epi_sequence
from Sequence.write_se import write_se_sequence
from Sequence.write_epi_se import write_epi_se_sequence
from Sequence.write_epi_label import write_epi_label_sequence
from Sequence.write_gre import write_gre_sequence
from simulate import SimulationConfig, simulate
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64)
    #rho, t1, t2 = load_simple_phantom("E:\毕业课题\old_version\mrisimulation_test\output\discrete_phantom_3.0T_miao.nii", 90)
    '''
    rho = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_PDMap_305_aligned/00000012.dcm').pixel_array
    t1 = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_T1Map_301_aligned/00000012.dcm').pixel_array
    t2 = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_T2Star_307_aligned/00000012.dcm').pixel_array
    rho = rho.astype(np.float32)
    t1 = t1.astype(np.float32)
    t2 = t2.astype(np.float32)
    rho = rho[np.newaxis, np.newaxis, np.newaxis, :, :]
    t1 = t1[np.newaxis, np.newaxis, np.newaxis, :, :]/1000
    t2 = t2[np.newaxis, np.newaxis, np.newaxis, :, :]/1000
    
    Nz, Nx, Ny = rho.shape[2:]
    dx = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_PDMap_305_aligned/00000012.dcm').PixelSpacing[0]/1000
    dy = pydicom.dcmread('E:\毕业课题/20260317\spinecho_test-1_132306/aligned_results\mtp_tra_1x0.8x2_MTP_PDMap_305_aligned/00000012.dcm').PixelSpacing[1]/1000
    '''
    FOV_x =  0.512 # 单位：米
    FOV_y = 0.512 # 单位：米

    phantom = Phantom(rho, t1, t2, fov_x=FOV_x, fov_y=FOV_y, slice_thickness=0.004)
    # The current forward model uses one isochromat per voxel, so RF spoiling
    # creates stronger artifacts than a scanner would. Disable it for the demo.
    seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
                            fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, 
                            tr=100e-3,te=20e-3)
    #seq = write_epi_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
    #                        fov=(phantom.fov_x, phantom.fov_y), te=200e-3)
    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title("Reconstruction")
    plt.imshow(np.abs(image_recon[0]), cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.title("Original")
    plt.imshow(rho[0, 0, 0],  cmap='gray')
    plt.axis('off')


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
