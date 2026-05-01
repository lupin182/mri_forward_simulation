import nibabel as nib
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from device_manager import get_xp, device_manager
import cupy as cp
xp = get_xp()

def generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64):
    """
    生成一个单切片非对称体模，用于验证方向和坐标系,这里体模数据是离散的,且简化为每个体素单类型单spin_packet
    1. 中心有一个大圆/圆柱 (密度 1.0)
    2. 有一个小方块/圆点 (密度 2.0) -> 用于定方向
    3. 背景有微弱信号 (密度 0.1)

    返回:
    Rho: (type, spin_packet, Nz, Nx, Ny) 密度
    T1:  (type, spin_packet, Nz, Nx, Ny) T1值
    T2:  (type, spin_packet, Nz, Nx, Ny) T2值
    """
    
    # 1. 初始化 (背景密度 0.1, T1=2s, T2=0.5s)
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.005


    # 2. 生成坐标网格 (注意这里为了生成数据，我们严格按照 Nz, Nx, Ny 顺序)
    # 这里的 x_idx 对应 Nx 维度，y_idx 对应 Ny 维度
    _, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx) - Nx/2,  # 居中坐标: -32 到 +32
        np.arange(Ny) - Ny/2, 
        indexing='ij'
    )

    # 3. 主体：中心大圆 (半径 16)
    # x^2 + y^2 < r^2
    mask_main = (x**2 + y**2) <= 16**2
    
    # 4. 标记物：右上角小点 (卫星)
    # 放在 X 正半轴, Y 正半轴 (例如 x=15, y=15 处)
    # 这是一个 6x6 的小方块
    mask_marker = (x > 10) & (x < 20) & (y > 10) & (y < 20)

    # 5. 赋值
    # 主体：水 (Rho=1.0, T1=1.0, T2=0.1)
    rho[mask_main] = 1.0
    t1[mask_main]  = 1.0
    t2[mask_main]  = 0.1

    # 标记物：高亮油点 (Rho=2.0, T1=0.5, T2=0.05 - 短T1更亮(在短TR下), 短T2)
    rho[mask_marker] = 2.0 
    t1[mask_marker]  = 0.5 
    t2[mask_marker]  = 0.05

    rho[0,10:15,50:55] = 2.0
    t1[0,10:15,50:55] = 0.5
    t2[0,10:15,50:55] = 0.05

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2

def generate_simple_ring_phantom(Nz=1, Nx=64, Ny=64, inner_radius=10, outer_radius=20):
    """
    生成一个中心为圆环/圆环柱的体模数据,这里体模数据是离散的,且简化为每个体素单类型单spin_packet
    
    参数:
        inner_radius: 内圆半径 (像素)
        outer_radius: 外圆半径 (像素)
        
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 密度, 圆环=1.0, 背景=0.1
        T1:  (type, spin_packet, Nz, Nx, Ny) T1值
        T2:  (type, spin_packet, Nz, Nx, Ny) T2值
    """
    
    # 1. 初始化背景 (Nz, Nx, Ny)
    # 背景密度 0.1
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    # 背景弛豫时间 (模拟流体或长弛豫组织)
    t1 = np.ones((Nz, Nx, Ny)) * 2.0  # 2000ms
    t2 = np.ones((Nz, Nx, Ny)) * 0.5  # 500ms

    # 2. 生成坐标网格
    # 使用 indexing='ij' 严格匹配 (Nz, Nx, Ny) 的矩阵形状
    _, x_idx, y_idx = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx), 
        np.arange(Ny), 
        indexing='ij'
    )

    # 3. 定义中心点
    cx, cy = Nx // 2, Ny // 2

    # 4. 计算距离平方 (在 Nx-Ny 平面上)
    # 忽略 Z 轴距离 (假设是圆柱状/2D圆环)
    dist_sq = (x_idx - cx)**2 + (y_idx - cy)**2

    # 5. 生成圆环掩膜 (Mask)
    # 逻辑：距离 >= 内半径平方  且  距离 <= 外半径平方
    ring_mask = (dist_sq >= inner_radius**2) & (dist_sq <= outer_radius**2)

    # 6. 赋值 (圆环部分)
    # 模拟类似水的性质 (高信号，长T1/T2)
    rho[ring_mask] = 1.0    # 密度 1
    t1[ring_mask]  = 1.0    # T1 1000ms
    t2[ring_mask]  = 0.1    # T2 100ms (稍微短一点，模拟组织)

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2

def generate_simple_sphere_phantom(Nz=1, Nx=64, Ny=64, radius=16):
    """
    生成一个中心有球体（或圆盘）的体模数据,这里体模数据是离散的,且简化为每个体素单类型单spin_packet
    
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 密度, 球=1.0, 背景=0.1
        T1:  (type, spin_packet, Nz, Nx, Ny) T1值, 球=1.0s (1000ms), 背景=2.0s
        T2:  (type, spin_packet, Nz, Nx, Ny) T2值, 球=0.1s (100ms),  背景=0.5s
    """
    
    # 1. 初始化背景 (Nz, Nx, Ny)
    # 背景密度 0.1
    rho = np.ones((Nz, Nx, Ny)) * 1e-7
    # 背景 T1 = 2000ms, T2 = 500ms (模拟类似脑脊液或水肿的长弛豫背景，方便对比)
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.5

    # 2. 生成坐标网格
    # 注意这里使用 indexing='ij' 以匹配矩阵索引习惯
    z, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx), 
        np.arange(Ny), 
        indexing='ij'
    )

    # 3. 定义球心坐标
    cz, cx, cy = Nz // 2, Nx // 2, Ny // 2

    # 4. 计算距离平方 (Distance Squared)
    # 如果 Nz=1，z方向的距离也就是0，退化为2D圆
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2

    # 5. 生成球体掩膜 (Mask)
    mask = dist_sq <= radius**2

    # 6. 赋值 (球体部分)
    rho[mask] = 1.0   # 密度 1
    t1[mask]  = 1.0   # T1 1000ms (典型水/脑实质)
    t2[mask]  = 0.1   # T2 100ms

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2


def generate_multi_spin_sphere_phantom(Nz=1, Nx=64, Ny=64, radius=16, Nspins=15):
    """
    生成【单层多自旋包】球体体模，专用于GRE序列扰相梯度仿真实验
    每个体素包含Nspins个独立自旋包，新增dWRnd模拟T2*效应
    
    参数：
        Nz=1: 单层（2D仿真）
        Nx,Ny: 矩阵尺寸
        radius: 球体半径
        Nspins: 单个体素内的自旋包数量
    返回：
        Rho:    (1, Nspins, Nz, Nx, Ny) 质子密度
        T1:     (1, Nspins, Nz, Nx, Ny) T1弛豫时间 (s)
        T2:     (1, Nspins, Nz, Nx, Ny) T2弛豫时间 (s)
        dWRnd:  (1, Nspins, Nz, Nx, Ny) 随机局部磁场偏移，单位：rad/s，模拟T2*效应
    """
    # 1. 基础体模（球体+背景）
    rho_base = np.ones((Nz, Nx, Ny)) * 1e-7
    t1_base = np.ones((Nz, Nx, Ny)) * 2.0
    t2_base = np.ones((Nz, Nx, Ny)) * 0.5

    # 坐标网格
    z, x, y = np.meshgrid(np.arange(Nz), np.arange(Nx), np.arange(Ny), indexing='ij')
    cz, cx, cy = Nz // 2, Nx // 2, Ny // 2
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    mask = dist_sq <= radius**2

    # 球体区域赋值
    rho_base[mask] = 1.0
    t1_base[mask] = 1.0
    t2_base[mask] = 0.1

    # 2. 扩展为多自旋包维度（形状统一：1, Nspins, Nz, Nx, Ny）
    rho = np.tile(rho_base[np.newaxis, np.newaxis, :, :, :], (1, Nspins, 1, 1, 1))
    T1  = np.tile(t1_base[np.newaxis, np.newaxis, :, :, :],  (1, Nspins, 1, 1, 1))
    T2  = np.tile(t2_base[np.newaxis, np.newaxis, :, :, :],  (1, Nspins, 1, 1, 1))

    # 3. 生成 dWRnd ✅ 严格满足所有要求
    # 单位：rad/s
    # 分布：正态分布，均值0（无偏），小标准差（真实物理量级）
    # 每个自旋包/体素的偏移都独立不同
    # 形状：与 Rho/T1/T2 完全一致
    dWRnd = np.random.normal(
        loc=0.0,        # 无偏（均值为0）
        scale=35.0,      # 小随机数，rad/s 单位（T2*仿真标准量级）
        size=rho.shape  # 形状完全匹配
    )

    return rho, T1, T2, dWRnd


def generate_coil_sensitivity_maps(
    Nx=64, 
    Ny=64, 
    Nz=1, 
    n_coils=8,      # 线圈通道数
    sigma=0.8       # 高斯平滑度
):
    """
    生成MRI多通道接收线圈灵敏度图
    ✅ 分离输出：幅值灵敏度 + 相位灵敏度
    ✅ 物理正确：环形分布高斯幅度 + 平滑随机相位
    ✅ 维度标准：(n_coils, Nz, Nx, Ny)
    
    参数：
        Nx, Ny: 图像矩阵尺寸
        Nz: 层数（默认1，2D仿真）
        n_coils: 线圈通道数
        sigma: 灵敏度平滑度
    
    返回：
        coil_mag:   幅值灵敏度图，float32，形状 (n_coils, Nz, Nx, Ny)
        coil_phase: 相位灵敏度图，float32，形状 (n_coils, Nz, Nx, Ny)，单位：弧度(rad)
    """
    # 1. 归一化坐标网格 [-1, 1]
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 初始化幅值和相位数组
    coil_mag = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)
    coil_phase = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)

    # 2. 线圈位置：环形均匀分布
    coil_positions = []
    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        cx = 0.85 * np.cos(angle)
        cy = 0.85 * np.sin(angle)
        coil_positions.append((cx, cy))

    # 3. 逐一生成每个线圈的幅值 & 相位
    for c in range(n_coils):
        cx, cy = coil_positions[c]
        
        # 幅值：2D高斯分布
        amp = np.exp(-((X - cx)** 2 + (Y - cy)** 2) / (2 * sigma** 2))
        
        # 相位：平滑随机相位（无高频噪声）
        phase = np.random.uniform(0, 2*np.pi) + 0.2 * (X + Y)
        
        # 赋值到对应维度
        coil_mag[c, 0, :, :] = amp
        coil_phase[c, 0, :, :] = phase

    return coil_mag, coil_phase, n_coils


def generate_diff_coil_sensitivity_maps(
    Nx=64, 
    Ny=64, 
    Nz=1, 
    n_coils=8,      # 线圈通道数
    sigma=0.6       # 高斯平滑度
):
    """
    【高度真实版】带空间位置差异的MRI多通道接收线圈灵敏度
    ✅ 每个线圈位于不同空间位置：左上/左下/右上/右下/左/右/上/下
    ✅ 靠近线圈的区域灵敏度显著更高（完全符合真实物理）
    ✅ 分离输出幅值 + 相位
    ✅ 维度：(n_coils, Nz, Nx, Ny)
    """
    # 归一化坐标 [-1, 1]
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    coil_mag = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)
    coil_phase = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)

    # ====================== 核心修改 ======================
    # 8 个线圈真实分布在 FOV 不同位置，各自有专属高灵敏区
    # 位置格式：(cx, cy)，越大越靠近对应边缘
    coil_positions = [
        (-0.75,  0.75),  # 0: 左上线圈 → 左上最灵敏
        (-0.75, -0.75),  # 1: 左下线圈 → 左下最灵敏
        ( 0.75,  0.75),  # 2: 右上线圈 → 右上最灵敏
        ( 0.75, -0.75),  # 3: 右下线圈 → 右下最灵敏
        (-0.85,  0.0),   # 4: 左侧线圈 → 左边最灵敏
        ( 0.85,  0.0),   # 5: 右侧线圈 → 右边最灵敏
        ( 0.0,  0.85),   # 6: 上部线圈 → 上边最灵敏
        ( 0.0, -0.85),   # 7: 下部线圈 → 下边最灵敏
    ]

    # 若通道数不是8，自动截断或循环取位置
    coil_positions = coil_positions[:n_coils]

    # ====================== 为每个线圈生成灵敏度 ======================
    for c in range(n_coils):
        cx, cy = coil_positions[c]
        
        # 高斯幅度：中心在 (cx,cy)，越靠近这里数值越大
        amp = np.exp(-((X - cx)** 2 + (Y - cy)** 2) / (2 * sigma** 2))
        
        # 平滑相位（保持物理真实）
        global_phase = np.random.uniform(0, 2 * np.pi)
        phase = global_phase + 0.15 * X + 0.15 * Y
        
        coil_mag[c, 0, :, :] = amp
        coil_phase[c, 0, :, :] = phase

    return coil_mag, coil_phase, n_coils

def load_simple_phantom(phantom_path, slice_num:int=None):
    """
    载入已有的体模数据,这里体模数据是离散的,且简化为每个体素单类型单spin_packet
    
    返回:
        Rho: (type, spin_packet, Nz, Nx, Ny) 密度
        T1:  (type, spin_packet, Nz, Nx, Ny) T1值
        T2:  (type, spin_packet, Nz, Nx, Ny) T2值
    """
    phantom = nib.load(phantom_path)
    data = phantom.get_fdata()
    if slice_num is None: # 如果没有指定切片，返回所有切片
        rho = data[:,:,:,0].transpose(2,0,1)
        t1 = data[:,:,:,1].transpose(2,0,1)
        t2 = data[:,:,:,2].transpose(2,0,1)
        rho = rho[np.newaxis,np.newaxis,:,:,:]
        t1 = t1[np.newaxis,np.newaxis,:,:,:]
        t2 = t2[np.newaxis,np.newaxis,:,:,:]
    else: # 如果指定切片，返回指定切片
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
        体模类，支持CuPy GPU加速。
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

        # 将数据移动到当前设备（CPU/GPU）
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

        self.SpinNum = self.rho.shape[1]     # 自旋数
        self.TypeNum = self.rho.shape[0]     # 类型数
        self.RxCoilNum = RxCoilNum     # 接收线圈数
        self.TxCoilNum = TxCoilNum     # 发射线圈数


        # 高级环境属性默认值
        self.txCoilmg = device_manager.to_device(xp.ones((TxCoilNum,self.Nz,self.Nx,self.Ny))) if txCoilmg is None else device_manager.to_device(txCoilmg)    # 发射场敏感度
        self.txCoilpe = device_manager.to_device(xp.zeros((TxCoilNum,self.Nz,self.Nx,self.Ny))) if txCoilpe is None else device_manager.to_device(txCoilpe)    # 发射场敏感度
        self.rxCoilmg = device_manager.to_device(xp.ones((RxCoilNum,self.Nz,self.Nx,self.Ny))) if rxCoilmg is None else device_manager.to_device(rxCoilmg)    # 接收场敏感度
        self.rxCoilpe = device_manager.to_device(xp.zeros((RxCoilNum,self.Nz,self.Nx,self.Ny))) if rxCoilpe is None else device_manager.to_device(rxCoilpe)    # 接收场敏感度
        
        assert self.txCoilmg.shape == (TxCoilNum, self.Nz, self.Nx, self.Ny), "txCoilmg shape must be (TxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.txCoilpe.shape == (TxCoilNum, self.Nz, self.Nx, self.Ny), "txCoilpe shape must be (TxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.rxCoilmg.shape == (RxCoilNum, self.Nz, self.Nx, self.Ny), "rxCoilmg shape must be (RxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.rxCoilpe.shape == (RxCoilNum, self.Nz, self.Nx, self.Ny), "rxCoilpe shape must be (RxCoilNum, self.Nz, self.Nx, self.Ny)"

        # 随时间演化的状态 (初始化平衡态)
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


if __name__ == '__main__':
    # 生成64x64，15个自旋包的体模
    rho, T1, T2, dWRnd = generate_multi_spin_sphere_phantom(Nspins=15)
    
    # 打印形状（验证兼容性）
    print(f"Rho 形状: {rho.shape}")
    print(f"T1 形状:  {T1.shape}")
    print(f"T2 形状:  {T2.shape}")
    print(f"dWRnd 形状: {dWRnd.shape}")
    
    # 验证dWRnd统计特性（无偏、小随机数）
    print(f"\ndWRnd 均值: {np.mean(dWRnd):.4f} rad/s (理论为0)")
    print(f"dWRnd 标准差: {np.std(dWRnd):.4f} rad/s")
    print(f"dWRnd 数值范围: {np.min(dWRnd):.2f} ~ {np.max(dWRnd):.2f} rad/s")