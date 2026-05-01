import matplotlib.pyplot as plt
import numpy as np
import pydicom
from phantom.make_phantom import Phantom, generate_simple_asymmetric_phantom, load_simple_phantom
from recon import reconstruct_3d_cartesian_fft, plot_color_overlay
from Sequence.write_gre_label import write_gre_label_sequence
from Sequence.write_epi import write_epi_sequence
from Sequence.write_se import write_se_sequence
from Sequence.write_epi_se import write_epi_se_sequence
from Sequence.write_epi_label import write_epi_label_sequence
from Sequence.write_gre import write_gre_sequence
from simulate import SimulationConfig, simulate
import numpy as np
import matplotlib.pyplot as plt
from generate_artifact import generate_rf_artifact
import pypulseq as pp

def main() -> None:

    rho=np.load('img_rho.npy')
    t1=np.load('img_T1.npy')
    t2=np.load('img_T2.npy')
    rho=np.moveaxis(rho, source=2, destination=0)
    t1=np.moveaxis(t1, source=2, destination=0)
    t2=np.moveaxis(t2, source=2, destination=0)

    FOV_x =  0.171#dx*Nx # 单位：米
    FOV_y =  0.204#dy*Ny # 单位：米

    phantom = Phantom(rho, t1, t2, fov_x=FOV_x, fov_y=FOV_y, slice_thickness=0.003)

    x_axis = np.arange(-0.0845, 0.0875, 0.001)
    y_axis = np.arange(-0.0985, 0.1065, 0.001)
    z_axis = np.array([0.0])

    phantom.z, phantom.x, phantom.y = np.meshgrid(z_axis, x_axis, y_axis, indexing='ij')
    '''
    phantom.CS=np.load('img_dw.npy')
    phantom.CS=np.moveaxis(phantom.CS, source=2, destination=0)
    phantom.CS=phantom.CS[np.newaxis,np.newaxis,:,:,:]

    phantom.dB0=np.load('img_dw.npy')
    phantom.dB0=np.moveaxis(phantom.dB0, source=2, destination=0)
    phantom.dB0=phantom.dB0[np.newaxis,np.newaxis,:,:,:]
    '''
    seq = pp.Sequence()
    seq.read("gre_pypulseq.seq")
    seq=write_gre_sequence(ideal_spoiling_reset=True)
    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))

    
    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=64, Nx=64, Nz=1)
    np.save('image_recon_ideal.npy', image_recon)
    
    plt.figure(figsize=(10, 10))
    plt.title("Reconstruction")
    plt.imshow(np.abs(image_recon[0]), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    main()