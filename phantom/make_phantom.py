import nibabel as nib
import numpy as np
from device_manager import get_xp, device_manager

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
    rho = np.ones((Nz, Nx, Ny)) * 0.1
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
                slice_thickness:float=1.0,fov_y:float=1.0,SpinNum=1,TypeNum=1,
                RxCoilNum=1,TxCoilNum=1):
        """
        体模类，支持CuPy GPU加速。
        """

        self.fov_x = fov_x
        self.fov_y = fov_y
        self.slice_thickness = slice_thickness

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

        self.SpinNum = SpinNum     # 自旋数
        self.TypeNum = TypeNum     # 类型数
        self.RxCoilNum = RxCoilNum     # 接收线圈数
        self.TxCoilNum = TxCoilNum     # 发射线圈数

        # 高级环境属性默认值
        self.txCoilmg = device_manager.to_device(xp.ones((TxCoilNum,self.Nz,self.Nx,self.Ny)))    # 发射场敏感度
        self.txCoilpe = device_manager.to_device(xp.zeros((TxCoilNum,self.Nz,self.Nx,self.Ny)))    # 发射场敏感度
        self.rxCoilmg = device_manager.to_device(xp.ones((RxCoilNum,self.Nz,self.Nx,self.Ny)))    # 接收场敏感度
        self.rxCoilpe = device_manager.to_device(xp.zeros((RxCoilNum,self.Nz,self.Nx,self.Ny)))    # 接收场敏感度
        
        # 随时间演化的状态 (初始化平衡态)
        self.Mx = device_manager.to_device(xp.zeros_like(self.rho))       
        self.My = device_manager.to_device(xp.zeros_like(self.rho))
        self.Mz = device_manager.to_device(xp.copy(self.rho))

        self.Gyro = 42.576e6    # gyromagnetic ratio
        self.CS = device_manager.to_device(xp.zeros_like(self.rho))      # chemical shift array
        self.dB0 = device_manager.to_device(xp.zeros_like(self.rho))     # B0 inhomogeneity
        self.dWRnd = device_manager.to_device(xp.zeros_like(self.rho))     # random off-resonance for T2*


    def generate_B0_inhomogeneity(self, mode: str, delta_B0_ppm: float, axis: str = 'x'):
        """
        生成3T主磁场不均匀性场图，输出单位：特斯拉（T）
        支持线性分布 / 抛物线分布两种建模方式
        
        参数
        ----------
        mode : str
            分布模式：'linear'（线性）/ 'parabolic'（抛物线/碗状）
        delta_B0_ppm : float
            主磁场不均匀度（ppm），例：0.5 → 0.5ppm
        axis : str
            线性模式的分布轴：'x' / 'y'，仅线性模式生效
        
        返回
        ----------
        B0_map : np.ndarray
            主磁场不均匀场，形状 (Nz, Nx, Ny)，单位：特斯拉（T）
        """
        # 主磁场强度 3T
        B0_nominal = 3.0  
        # ppm → 特斯拉 核心换算（1 ppm = 1e-6）
        ppm_to_T = B0_nominal * (delta_B0_ppm / 1e6)

        # ===================== 线性分布（沿X/Y轴）=====================
        if mode == "linear":
            if axis not in ["x", "y"]:
                raise ValueError("线性模式仅支持 x / y 轴")
            
            # 选取对应轴的中心化坐标
            coord = self.x if axis == "x" else self.y
            # 计算FOV总长度，归一化到 [-0.5, 0.5]
            fov_length = self.dx * self.Nx if axis == "x" else self.dy * self.Ny
            normalized_coord = coord / fov_length  
            # 输出：特斯拉（T）
            B0_map = ppm_to_T * normalized_coord

        # ===================== 抛物线分布（碗状，X-Y平面）=====================
        elif mode == "parabolic":
            # 到中心的径向距离平方（x²+y²）
            r_square = self.x ** 2 + self.y ** 2
            # 归一化分母（FOV对角最大值）
            half_fov_x = (self.Nx / 2) * self.dx
            half_fov_y = (self.Ny / 2) * self.dy
            norm_denominator = half_fov_x ** 2 + half_fov_y ** 2

            # 输出：特斯拉（T）
            B0_map = ppm_to_T * (r_square / norm_denominator)

        else:
            raise ValueError("模式仅支持 linear / parabolic")

        # 保存到类属性并返回
        self.dB0 = B0_map
        return B0_map


if __name__ == '__main__':
    pass