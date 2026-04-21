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
    rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64)
    rho=np.load('img_rho.npy')
    t1=np.load('img_T1.npy')
    t2=np.load('img_T2.npy')
    rho=np.moveaxis(rho, source=2, destination=0)
    t1=np.moveaxis(t1, source=2, destination=0)
    t2=np.moveaxis(t2, source=2, destination=0)
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
    FOV_x =  0.098#dx*Nx # 单位：米
    FOV_y =  0.098#dy*Ny # 单位：米

    phantom = Phantom(rho, t1, t2, fov_x=FOV_x, fov_y=FOV_y, slice_thickness=0.003)

    x_axis = np.arange(-0.0495, 0.0495, 0.001)
    y_axis = np.arange(-0.0495, 0.0495, 0.001)
    z_axis = np.array([0.0])

    phantom.z, phantom.x, phantom.y = np.meshgrid(z_axis, x_axis, y_axis, indexing='ij')
    '''
    phantom.CS=np.load('img_dw.npy')
    phantom.CS=np.moveaxis(phantom.CS, source=2, destination=0)
    phantom.CS=phantom.CS[np.newaxis,np.newaxis,:,:,:]
    '''
    phantom.dB0=np.load('img_dw.npy')
    phantom.dB0=np.moveaxis(phantom.dB0, source=2, destination=0)
    phantom.dB0=phantom.dB0[np.newaxis,np.newaxis,:,:,:]

    # The current forward model uses one isochromat per voxel, so RF spoiling
    # creates stronger artifacts than a scanner would. Disable it for the demo.
    #seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
    #                        fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, 
    #                        tr=100e-3,te=20e-3)
    seq = pp.Sequence()
    seq.read("epi_se_pypulseq.seq")
    #seq = write_gre_label_sequence(n_slices=1,fov=(0.171,0.204),n_y=205,n_x=172,slice_thickness=0.001)
    #seq = write_epi_se_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
    #                        fov=(phantom.fov_x, phantom.fov_y), te=200e-3)
    k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))
    _, _, _, t_adc, _ = seq.waveforms_and_times()
    #k_space_signal = generate_rf_artifact(t_adc, k_space_signal, rf_noise_freq=[127.7e6], rf_noise_amp=[1.0], bg_noise_amp=0.0)

    k_traj_adc, _, _, _, _ = seq.calculate_kspace()
    image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=64, Nx=64, Nz=1)
    np.save('image_recon.npy', image_recon)
    #plot_color_overlay(image_recon[0], rho[0,0,0])
    
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title("Reconstruction")
    plt.imshow(np.abs(image_recon[0]), cmap='gray')
    plt.axis('off')
    '''
    plt.subplot(122)
    plt.title("Original")
    plt.imshow(rho[0, 0, 0],  cmap='gray')
    plt.axis('off')
    '''

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 设备管理器会自动检测CuPy可用性
    import cupy as cp

    # 1. 获取当前正在使用的显卡 ID (默认通常是 0)
    current_device_id = cp.cuda.Device().id
    print(f"当前使用的显卡 ID: {current_device_id}")

    # 2. 获取当前显卡的详细名称
    # 注意：返回的 name 是 bytes 类型，需要解码为字符串
    props = cp.cuda.runtime.getDeviceProperties(current_device_id)
    device_name = props['name'].decode('utf-8')
    print(f"当前使用的显卡名称: {device_name}")
    main()