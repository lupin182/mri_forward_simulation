import nibabel as nib
import numpy as np
from .device_manager import get_xp, device_manager
xp = get_xp()

def generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64):
    """
    生成一个单层非对称体模，用于验证方向和坐标系。
    体模数据是离散的，并简化为每个体素只有单类型、单 spin packet。

    1. 中心有一个大圆柱区域，密度为 1.0。
    2. 右上角有一个小方块标记，密度为 2.0，用于判断方向。
    3. 背景保留弱信号，密度为 0.1。

    返回:
    Rho: (type, spin_packet, Nz, Nx, Ny) 质子密度
    T1:  (type, spin_packet, Nz, Nx, Ny) T1 值
    T2:  (type, spin_packet, Nz, Nx, Ny) T2 值
    """
    
    # 1. 初始化背景：密度 0.1，T1=2s，T2=0.005s。
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.005


    # 2. 生成坐标网格，严格匹配 (Nz, Nx, Ny) 顺序。
    # 这里的 x 对应 Nx 维度，y 对应 Ny 维度。
    _, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx) - Nx/2,  # 居中坐标。
        np.arange(Ny) - Ny/2, 
        indexing='ij'
    )

    # 3. 主体：中心大圆，半径 16。
    # x^2 + y^2 < r^2
    mask_main = (x**2 + y**2) <= 16**2
    
    # 4. 标记物：右上角小方块。
    # 放在 X 正半轴、Y 正半轴，用于辅助判断方向。
    # 这是一个 6x6 左右的小方块。
    mask_marker = (x > 10) & (x < 20) & (y > 10) & (y < 20)

    # 5. 赋值。
    # 主体：类似水的区域，Rho=1.0，T1=1.0，T2=0.1。
    rho[mask_main] = 1.0
    t1[mask_main]  = 1.0
    t2[mask_main]  = 0.1

    # 标记物：高亮点，Rho=2.0，T1=0.5，T2=0.05。
    rho[mask_marker] = 2.0 
    t1[mask_marker]  = 0.5 
    t2[mask_marker]  = 0.05

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2

def generate_simple_ring_phantom(Nz=1, Nx=64, Ny=64, inner_radius=10, outer_radius=20):
    """
    生成一个中心为圆环或圆环柱的体模数据。
    体模数据是离散的，并简化为每个体素只有单类型、单 spin packet。
    
    参数:
        inner_radius: 内圆半径，单位为像素。
        outer_radius: 外圆半径，单位为像素。
        
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 质子密度，圆环=1.0，背景=0.1。
        T1:  (type, spin_packet, Nz, Nx, Ny) T1 值。
        T2:  (type, spin_packet, Nz, Nx, Ny) T2 值。
    """
    
    # 1. 初始化背景 (Nz, Nx, Ny)。
    # 背景密度 0.1。
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    # 背景弛豫时间，模拟流体或长弛豫组织。
    t1 = np.ones((Nz, Nx, Ny)) * 2.0  # 2000ms
    t2 = np.ones((Nz, Nx, Ny)) * 0.5  # 500ms

    # 2. 生成坐标网格。
    # 使用 indexing='ij' 严格匹配 (Nz, Nx, Ny) 的矩阵形状。
    _, x_idx, y_idx = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx), 
        np.arange(Ny), 
        indexing='ij'
    )

    # 3. 定义中心点。
    cx, cy = Nx // 2, Ny // 2

    # 4. 在 Nx-Ny 平面计算距离平方。
    # 忽略 Z 轴距离，等价于圆柱状或 2D 圆环。
    dist_sq = (x_idx - cx)**2 + (y_idx - cy)**2

    # 5. 生成圆环掩膜。
    # 逻辑：距离平方位于内半径平方和外半径平方之间。
    ring_mask = (dist_sq >= inner_radius**2) & (dist_sq <= outer_radius**2)

    # 6. 给圆环区域赋值。
    # 模拟类似水的性质：高信号、较长 T1/T2。
    rho[ring_mask] = 1.0    # 密度 1。
    t1[ring_mask]  = 1.0    # T1 1000ms
    t2[ring_mask]  = 0.1    # T2 100ms，略短一些以模拟组织。

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2

def generate_simple_sphere_phantom(Nz=1, Nx=64, Ny=64, radius=16):
    """
    生成一个中心有球体或圆盘的体模数据。
    体模数据是离散的，并简化为每个体素只有单类型、单 spin packet。
    
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 质子密度，球体=1.0，背景=1e-7。
        T1:  (type, spin_packet, Nz, Nx, Ny) T1 值，球体=1.0s，背景=2.0s。
        T2:  (type, spin_packet, Nz, Nx, Ny) T2 值，球体=0.1s，背景=0.5s。
    """
    
    # 1. 初始化背景 (Nz, Nx, Ny)。
    # 背景密度接近 0。
    rho = np.ones((Nz, Nx, Ny)) * 1e-7
    # 背景 T1 = 2000ms，T2 = 500ms，模拟长弛豫背景。
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.5

    # 2. 生成坐标网格。
    # 使用 indexing='ij' 以匹配矩阵索引习惯。
    z, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx), 
        np.arange(Ny), 
        indexing='ij'
    )

    # 3. 定义球心坐标。
    cz, cx, cy = Nz // 2, Nx // 2, Ny // 2

    # 4. 计算距离平方。
    # 如果 Nz=1，Z 方向距离为 0，退化为 2D 圆盘。
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2

    # 5. 生成球体掩膜。
    mask = dist_sq <= radius**2

    # 6. 给球体区域赋值。
    rho[mask] = 1.0   # 密度 1。
    t1[mask]  = 1.0   # T1 1000ms。
    t2[mask]  = 0.1   # T2 100ms

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2


def generate_multi_spin_sphere_phantom(Nz=1, Nx=64, Ny=64, radius=16, Nspins=15):
    """
    生成单层多自旋包球体体模，用于 GRE 序列扰相梯度仿真实验。
    每个体素包含 Nspins 个独立自旋包，并通过 dWRnd 模拟 T2* 效应。
    
    参数:
        Nz=1: 单层 2D 仿真。
        Nx, Ny: 矩阵尺寸。
        radius: 球体半径。
        Nspins: 单个体素内的自旋包数量。
    返回:
        Rho:    (1, Nspins, Nz, Nx, Ny) 质子密度。
        T1:     (1, Nspins, Nz, Nx, Ny) T1 弛豫时间，单位秒。
        T2:     (1, Nspins, Nz, Nx, Ny) T2 弛豫时间，单位秒。
        dWRnd:  (1, Nspins, Nz, Nx, Ny) 随机局部磁场偏移，单位 rad/s，用于模拟 T2* 效应。
    """
    # 1. 基础体模：球体和背景。
    rho_base = np.ones((Nz, Nx, Ny)) * 1e-7
    t1_base = np.ones((Nz, Nx, Ny)) * 2.0
    t2_base = np.ones((Nz, Nx, Ny)) * 0.5

    # 坐标网格。
    z, x, y = np.meshgrid(np.arange(Nz), np.arange(Nx), np.arange(Ny), indexing='ij')
    cz, cx, cy = Nz // 2, Nx // 2, Ny // 2
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    mask = dist_sq <= radius**2

    # 给球体区域赋值。
    rho_base[mask] = 1.0
    t1_base[mask] = 1.0
    t2_base[mask] = 0.1

    # 2. 扩展为多自旋包维度，形状统一为 (1, Nspins, Nz, Nx, Ny)。
    rho = np.tile(rho_base[np.newaxis, np.newaxis, :, :, :], (1, Nspins, 1, 1, 1))
    T1  = np.tile(t1_base[np.newaxis, np.newaxis, :, :, :],  (1, Nspins, 1, 1, 1))
    T2  = np.tile(t2_base[np.newaxis, np.newaxis, :, :, :],  (1, Nspins, 1, 1, 1))

    # 3. 生成 dWRnd，严格满足形状要求。
    # 单位：rad/s。
    # 分布：正态分布，均值 0，较小标准差。
    # 每个自旋包和体素的偏移都相互独立。
    # 形状与 Rho/T1/T2 完全一致。
    dWRnd = np.random.normal(
        loc=0.0,        # 无偏，均值为 0。
        scale=35.0,      # 小随机数，单位 rad/s，作为 T2* 仿真的标准量级。
        size=rho.shape  # 形状完全匹配。
    )

    return rho, T1, T2, dWRnd


def generate_coil_sensitivity_maps(
    Nx=64, 
    Ny=64, 
    Nz=1, 
    n_coils=8,      # 线圈通道数。
    sigma=0.8       # 高斯平滑度。
):
    """
    生成 MRI 多通道接收线圈灵敏度图。
    输出拆分为幅值灵敏度和相位灵敏度。
    幅值采用环形分布的高斯模型，相位采用平滑随机相位。
    输出维度为 (n_coils, Nz, Nx, Ny)。
    
    参数:
        Nx, Ny: 图像矩阵尺寸。
        Nz: 层数，默认用于 2D 仿真。
        n_coils: 线圈通道数。
        sigma: 灵敏度平滑度。
    
    返回:
        coil_mag:   幅值灵敏度图，float32，形状为 (n_coils, Nz, Nx, Ny)。
        coil_phase: 相位灵敏度图，float32，形状为 (n_coils, Nz, Nx, Ny)，单位为弧度。
    """
    # 1. 归一化坐标网格 [-1, 1]。
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 初始化幅值和相位数组。
    coil_mag = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)
    coil_phase = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)

    # 2. 线圈位置：环形均匀分布。
    coil_positions = []
    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        cx = 0.85 * np.cos(angle)
        cy = 0.85 * np.sin(angle)
        coil_positions.append((cx, cy))

    # 3. 逐一生成每个线圈的幅值和相位。
    for c in range(n_coils):
        cx, cy = coil_positions[c]
        
        # 幅值：2D 高斯分布。
        amp = np.exp(-((X - cx)** 2 + (Y - cy)** 2) / (2 * sigma** 2))
        
        # 相位：平滑随机相位，避免高频噪声。
        phase = np.random.uniform(0, 2*np.pi) + 0.2 * (X + Y)
        
        # 赋值到对应维度。
        coil_mag[c, 0, :, :] = amp
        coil_phase[c, 0, :, :] = phase

    return coil_mag, coil_phase, n_coils


def generate_diff_coil_sensitivity_maps(
    Nx=64, 
    Ny=64, 
    Nz=1, 
    n_coils=8,      # 线圈通道数。
    sigma=0.6       # 高斯平滑度。
):
    """
    生成带空间位置差异的 MRI 多通道接收线圈灵敏度图。
    每个线圈位于不同空间位置，靠近线圈的区域灵敏度更高。
    输出拆分为幅值和相位，维度为 (n_coils, Nz, Nx, Ny)。
    """
    # 归一化坐标网格 [-1, 1]。
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    coil_mag = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)
    coil_phase = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)

    # 8 个线圈分布在 FOV 的不同位置，各自拥有专属高灵敏度区域。
    # 位置格式为 (cx, cy)，绝对值越大越靠近对应边缘。
    coil_positions = [
        (-0.75,  0.75),  # 0: 左上线圈，左上区域最灵敏。
        (-0.75, -0.75),  # 1: 左下线圈，左下区域最灵敏。
        ( 0.75,  0.75),  # 2: 右上线圈，右上区域最灵敏。
        ( 0.75, -0.75),  # 3: 右下线圈，右下区域最灵敏。
        (-0.85,  0.0),   # 4: 左侧线圈，左边区域最灵敏。
        ( 0.85,  0.0),   # 5: 右侧线圈，右边区域最灵敏。
        ( 0.0,  0.85),   # 6: 上部线圈，上边区域最灵敏。
        ( 0.0, -0.85),   # 7: 下部线圈，下边区域最灵敏。
    ]

    # 如果通道数不是 8，自动截断位置列表。
    coil_positions = coil_positions[:n_coils]

    # 为每个线圈生成灵敏度。
    for c in range(n_coils):
        cx, cy = coil_positions[c]
        
        # 高斯幅值：中心在 (cx, cy)，越靠近该位置数值越大。
        amp = np.exp(-((X - cx)** 2 + (Y - cy)** 2) / (2 * sigma** 2))
        
        # 平滑相位，保持物理上较平滑的空间变化。
        global_phase = np.random.uniform(0, 2 * np.pi)
        phase = global_phase + 0.15 * X + 0.15 * Y
        
        coil_mag[c, 0, :, :] = amp
        coil_phase[c, 0, :, :] = phase

    return coil_mag, coil_phase, n_coils

def load_simple_phantom(phantom_path, slice_num:int=None):
    """
    载入已有的体模数据。
    这里体模数据是离散的，并简化为每个体素只有单类型、单 spin packet。
    
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 质子密度。
        T1:  (type, spin_packet, Nz, Nx, Ny) T1 值。
        T2:  (type, spin_packet, Nz, Nx, Ny) T2 值。
    """
    phantom = nib.load(phantom_path)
    data = phantom.get_fdata()
    if slice_num is None: # 如果没有指定切片，返回所有切片。
        rho = data[:,:,:,0].transpose(2,0,1)
        t1 = data[:,:,:,1].transpose(2,0,1)
        t2 = data[:,:,:,2].transpose(2,0,1)
        rho = rho[np.newaxis,np.newaxis,:,:,:]
        t1 = t1[np.newaxis,np.newaxis,:,:,:]
        t2 = t2[np.newaxis,np.newaxis,:,:,:]
    else: # 如果指定切片，返回指定切片。
        rho = data[:,:,slice_num,0]
        t1 = data[:,:,slice_num,1]
        t2 = data[:,:,slice_num,2]
        rho = rho[np.newaxis,np.newaxis,np.newaxis,:,:]
        t1 = t1[np.newaxis,np.newaxis,np.newaxis,:,:]
        t2 = t2[np.newaxis,np.newaxis,np.newaxis,:,:]

    return rho, t1, t2

class Phantom:
    def __init__(self,rho:np.ndarray,t1:np.ndarray,t2:np.ndarray,fov_x:float=1.0,
                slice_thickness:float=1.0,fov_y:float=1.0,
                RxCoilNum=1,TxCoilNum=1,B0:float=3.0,dB0:np.ndarray=None,
                txCoilmg:np.ndarray=None,rxCoilmg:np.ndarray=None,
        txCoilpe:np.ndarray=None,rxCoilpe:np.ndarray=None,
                CS:np.ndarray=None,dWRnd:np.ndarray=None):
        """
        体模类，支持通过当前设备后端进行 CPU/GPU 计算。
        """
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.slice_thickness = slice_thickness
        self.B0 = B0
        if len(rho.shape) == 3:
            rho = rho[np.newaxis,np.newaxis,:,:,:]
        if len(t1.shape) == 3:
            t1 = t1[np.newaxis,np.newaxis,:,:,:]
        if len(t2.shape) == 3:
            t2 = t2[np.newaxis,np.newaxis,:,:,:]

        # 将数据移动到当前设备（CPU/GPU）。
        self.rho = device_manager.to_device(rho)
        self.t1 = device_manager.to_device(t1)
        self.t2 = device_manager.to_device(t2)

        self.Nz = rho.shape[2]
        self.Nx = rho.shape[3]
        self.Ny = rho.shape[4]

        self.dx = self.fov_x / self.Nx
        self.dy = self.fov_y / self.Ny

        z_axis = (xp.arange(self.Nz) - self.Nz / 2 + 0.5) * self.slice_thickness
        x_axis = (xp.arange(self.Nx) - self.Nx / 2 + 0.5) * self.dx
        y_axis = (xp.arange(self.Ny) - self.Ny / 2 + 0.5) * self.dy

        self.z, self.x, self.y = xp.meshgrid(z_axis, x_axis, y_axis, indexing='ij')

        assert self.rho.shape == self.t1.shape, "rho shape must be t1 shape"
        assert self.rho.shape == self.t2.shape, "rho shape must be t2 shape"
        assert len(self.rho.shape) == 5, "input rho shape must be (TypeNum, SpinNum, Nz, Nx, Ny) or (Nz, Nx, Ny)"

        self.SpinNum = self.rho.shape[1]     # 自旋包数量。
        self.TypeNum = self.rho.shape[0]     # 组织类型数量。
        self.RxCoilNum = RxCoilNum     # 接收线圈数量。
        self.TxCoilNum = TxCoilNum     # 发射线圈数量。


        # 高级环境属性默认值。
        self.txCoilmg = device_manager.to_device(xp.ones((TxCoilNum,self.Nz,self.Nx,self.Ny))) if txCoilmg is None else device_manager.to_device(txCoilmg)    # 发射场灵敏度幅值。
        self.txCoilpe = device_manager.to_device(xp.zeros((TxCoilNum,self.Nz,self.Nx,self.Ny))) if txCoilpe is None else device_manager.to_device(txCoilpe)    # 发射场灵敏度相位。
        self.rxCoilmg = device_manager.to_device(xp.ones((RxCoilNum,self.Nz,self.Nx,self.Ny))) if rxCoilmg is None else device_manager.to_device(rxCoilmg)    # 接收场灵敏度幅值。
        self.rxCoilpe = device_manager.to_device(xp.zeros((RxCoilNum,self.Nz,self.Nx,self.Ny))) if rxCoilpe is None else device_manager.to_device(rxCoilpe)    # 接收场灵敏度相位。
        
        assert self.txCoilmg.shape == (TxCoilNum, self.Nz, self.Nx, self.Ny), "txCoilmg shape must be (TxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.txCoilpe.shape == (TxCoilNum, self.Nz, self.Nx, self.Ny), "txCoilpe shape must be (TxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.rxCoilmg.shape == (RxCoilNum, self.Nz, self.Nx, self.Ny), "rxCoilmg shape must be (RxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.rxCoilpe.shape == (RxCoilNum, self.Nz, self.Nx, self.Ny), "rxCoilpe shape must be (RxCoilNum, self.Nz, self.Nx, self.Ny)"

        # 随时间演化的磁化状态，初始化为平衡状态。
        self.Mx = device_manager.to_device(xp.zeros_like(self.rho))       
        self.My = device_manager.to_device(xp.zeros_like(self.rho))
        self.Mz = device_manager.to_device(xp.copy(self.rho * self.B0))

        self.Gyro = 42.576e6    # gyromagnetic ratio
        # chemical shift array
        self.CS = device_manager.to_device(xp.zeros_like(self.rho)) if CS is None else device_manager.to_device(CS)
        # B0 inhomogeneity
        self.dB0 = device_manager.to_device(xp.zeros_like(self.rho)) if dB0 is None else device_manager.to_device(dB0)
        # random off-resonance for T2*
        self.dWRnd = device_manager.to_device(xp.zeros_like(self.rho)) if dWRnd is None else device_manager.to_device(dWRnd)

        assert self.dWRnd.shape == (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny), "dWRnd shape must be (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny)"
        assert self.CS.shape == (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny), "CS shape must be (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny)"
        assert self.dB0.shape == (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny), "dB0 shape must be (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny)"

