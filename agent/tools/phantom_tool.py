
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import device_manager
device_manager.disable_cupy()

from agent.tools.base_tool import MRISimulationBaseTool
from phantom.make_phantom import (
    generate_simple_asymmetric_phantom,
    generate_simple_ring_phantom,
    generate_simple_sphere_phantom,
    Phantom
)
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

phantom_cache = None
_phantom_figure_cache = None

class GeneratePhantomTool(MRISimulationBaseTool):
    name: str = "generate_phantom"
    description: str = """生成MRI体模数据。
    参数说明：
    - phantom_type: 体模类型，可选 'asymmetric'（非对称，默认）、'ring'（圆环）、'sphere'（球体）
    - Nz: z轴切片数，默认1
    - Nx: x轴分辨率，默认64
    - Ny: y轴分辨率，默认64
    - inner_radius: 圆环内半径（仅ring类型），默认10
    - outer_radius: 圆环外半径（仅ring类型），默认20
    - radius: 球体半径（仅sphere类型），默认16
    - fov_x: x轴视场（米），默认0.256
    - fov_y: y轴视场（米），默认0.256
    - slice_thickness: 切片厚度（米），默认0.004
    - return_figure: 是否返回matplotlib figure对象用于Streamlit展示，默认True
    """

    def _run(self, query: str) -> str:
        global phantom_cache, _phantom_figure_cache
        
        params = json.loads(query)
        phantom_type = params.get('phantom_type', 'asymmetric')
        Nz = params.get('Nz', 1)
        Nx = params.get('Nx', 64)
        Ny = params.get('Ny', 64)
        fov_x = params.get('fov_x', 0.256)
        fov_y = params.get('fov_y', 0.256)
        slice_thickness = params.get('slice_thickness', 0.004)
        return_figure = params.get('return_figure', True)

        if phantom_type == 'asymmetric':
            rho, t1, t2 = generate_simple_asymmetric_phantom(Nz=Nz, Nx=Nx, Ny=Ny)
        elif phantom_type == 'ring':
            inner_radius = params.get('inner_radius', 10)
            outer_radius = params.get('outer_radius', 20)
            rho, t1, t2 = generate_simple_ring_phantom(Nz=Nz, Nx=Nx, Ny=Ny, inner_radius=inner_radius, outer_radius=outer_radius)
        elif phantom_type == 'sphere':
            radius = params.get('radius', 16)
            rho, t1, t2 = generate_simple_sphere_phantom(Nz=Nz, Nx=Nx, Ny=Ny, radius=radius)
        else:
            return json.dumps({"status": "error", "message": f"未知的体模类型 '{phantom_type}'"})

        phantom = Phantom(rho, t1, t2, fov_x=fov_x, fov_y=fov_y, slice_thickness=slice_thickness)
        set_cached_phantom(phantom, rho, t1, t2)
        
        if return_figure:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].set_title("Proton Density (Rho)")
            axes[0].imshow(rho[0, 0, 0], cmap='gray')
            axes[0].axis('off')
            
            axes[1].set_title("T1 Relaxation Time")
            axes[1].imshow(t1[0, 0, 0], cmap='viridis')
            axes[1].axis('off')
            
            axes[2].set_title("T2 Relaxation Time")
            axes[2].imshow(t2[0, 0, 0], cmap='plasma')
            axes[2].axis('off')
            
            plt.tight_layout()
            _phantom_figure_cache = fig
        
        result = {
            "status": "success",
            "phantom_type": phantom_type,
            "shape": (phantom.Nz, phantom.Nx, phantom.Ny),
            "fov": (fov_x, fov_y),
            "slice_thickness": slice_thickness
        }
        
        return json.dumps(result, ensure_ascii=False)

def get_cached_phantom():
    global phantom_cache
    return phantom_cache

def get_cached_phantom_figure():
    global _phantom_figure_cache
    return _phantom_figure_cache

def set_cached_phantom(phantom, rho, t1, t2):
    global phantom_cache
    phantom_cache = (phantom, rho, t1, t2)

def clear_phantom_cache():
    global phantom_cache, _phantom_figure_cache
    phantom_cache = None
    _phantom_figure_cache = None

