'''
测试伪影专用文件
'''

from device_manager import get_xp, device_manager
xp = get_xp()

def generate_complex_phantom_and_dB0(Nz=1, Nx=64, Ny=64, SpinNum=1, TypeNum=1):

    """
    生成一个复杂的体模数据,用于主磁场不均匀伪影验证,是你需要开发的函数，这里体模数据是离散的
    
    参数:
        SpinNum: 自旋池数
        TypeNum: 组织类型数
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 密度
        T1:  (type, spin_packet, Nz, Nx, Ny) T1值
        T2:  (type, spin_packet, Nz, Nx, Ny) T2值
        dB0:  (type, spin_packet, Nz, Nx, Ny) dB0主磁场不均匀值
    """
    pass


from phantom.make_phantom import Phantom
from Sequence.write_gre_label import write_gre_label_sequence
from simulate import SimulationConfig, simulate
from recon import reconstruct_3d_cartesian_fft

rho, t1, t2, dB0 = generate_complex_phantom_and_dB0()#参数需要你自己填
FOV_x = None #参数需要你自己填
FOV_y = None #参数需要你自己填
slice_thickness = None #参数需要你自己填

phantom = Phantom(rho, t1, t2, fov_x=FOV_x, fov_y=FOV_y, slice_thickness=slice_thickness)
phantom.dB0 = device_manager.to_device(dB0)

ideal_spoiling_reset = False #是否开启重置Mxy=0进行理想spoiling，单自旋池情况下建议开启，多自旋池情况下可选
dummy_scans = 0 #虚拟扫描 / 预扫描,单自旋 + 理想 spoiling，初始瞬态可忽略，可无需开启，多自旋+非理想spoiling（即ideal_spoiling_reset = False），建议开启

seq = write_gre_label_sequence(n_y=phantom.Ny, n_x=phantom.Nx,
                            fov=(phantom.fov_x, phantom.fov_y), n_slices=phantom.Nz, 
                            tr=100e-3,te=20e-3,
                            dummy_scans=dummy_scans,
                            ideal_spoiling_reset=ideal_spoiling_reset)

k_space_signal = simulate(phantom, seq, SimulationConfig(fine_dt=1e-5))
k_traj_adc, _, _, _, _ = seq.calculate_kspace()
image_recon, _ = reconstruct_3d_cartesian_fft(k_space_signal, k_traj_adc, Ny=phantom.Ny, Nx=phantom.Nx, Nz=phantom.Nz)