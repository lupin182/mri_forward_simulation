"""
设备管理模块：提供CuPy和NumPy的自动切换功能

该模块用于检测GPU可用性并在CuPy和NumPy之间自动切换，
确保代码在有GPU时使用GPU加速，无GPU时回退到CPU计算。
"""

import importlib
from typing import Any, Callable, Optional


class DeviceManager:
    """
    设备管理器类，封装CuPy和NumPy的自动切换逻辑
    """
    
    _instance: Optional['DeviceManager'] = None
    _xp = None
    _cupy_available = False
    _use_cupy = False
    
    def __new__(cls):
        """单例模式确保只有一个设备管理器实例"""
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化设备管理器，检测CuPy可用性"""
        try:
            # 尝试导入CuPy
            import cupy as cp
            # 检查是否有可用的GPU
            cp.cuda.Device(0).compute_capability
            self._xp = cp
            self._cupy_available = True
            self._use_cupy = True
            print("CuPy GPU加速已启用")
        except (ImportError, Exception):
            # 导入失败或无GPU，使用NumPy
            import numpy as np
            self._xp = np
            self._cupy_available = False
            self._use_cupy = False
            print("CuPy不可用，使用NumPy进行CPU计算")
    
    @property
    def xp(self):
        """获取当前使用的计算库（CuPy或NumPy）"""
        return self._xp
    
    @property
    def cupy_available(self):
        """检查CuPy是否可用"""
        return self._cupy_available
    
    @property
    def use_cupy(self):
        """检查当前是否在使用CuPy"""
        return self._use_cupy
    
    def set_use_cupy(self, use_cupy: bool):
        """
        手动设置是否使用CuPy
        
        参数:
            use_cupy: 是否使用CuPy
        """
        if use_cupy and not self._cupy_available:
            print("CuPy不可用，无法启用CuPy")
            return
        
        if use_cupy:
            import cupy as cp
            self._xp = cp
            self._use_cupy = True
            print("已切换到CuPy")
        else:
            import numpy as np
            self._xp = np
            self._use_cupy = False
            print("已切换到NumPy")
    
    def to_numpy(self, array: Any) -> Any:
        """
        将数组转换为NumPy数组
        
        参数:
            array: 输入数组（CuPy或NumPy）
            
        返回:
            NumPy数组
        """
        if self._use_cupy and hasattr(array, 'get'):
            return array.get()
        return array
    
    def to_device(self, array: Any) -> Any:
        """
        将数组移动到当前设备（GPU或CPU）
        
        参数:
            array: 输入数组
            
        返回:
            当前设备上的数组
        """
        if self._use_cupy:
            return self._xp.asarray(array)
        return self._xp.asarray(array)


def get_xp():
    """获取当前使用的计算库（CuPy或NumPy）"""
    return device_manager.xp


def enable_cupy():
    """启用CuPy（如果可用）"""
    device_manager.set_use_cupy(True)


def disable_cupy():
    """禁用CuPy，使用NumPy"""
    device_manager.set_use_cupy(False)


def is_cupy_available():
    """检查CuPy是否可用"""
    return device_manager.cupy_available


def is_using_cupy():
    """检查当前是否在使用CuPy"""
    return device_manager.use_cupy


# 全局设备管理器实例
device_manager = DeviceManager()
disable_cupy()
# 便捷的访问方式
xp = device_manager.xp
