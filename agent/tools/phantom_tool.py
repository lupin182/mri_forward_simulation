
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
    """

    def _run(self, query: str) -> str:
        params = json.loads(query)
        phantom_type = params.get('phantom_type', 'asymmetric')
        Nz = params.get('Nz', 1)
        Nx = params.get('Nx', 64)
        Ny = params.get('Ny', 64)
        fov_x = params.get('fov_x', 0.256)
        fov_y = params.get('fov_y', 0.256)
        slice_thickness = params.get('slice_thickness', 0.004)

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
            return f"Error: 未知的体模类型 '{phantom_type}'"

        phantom = Phantom(rho, t1, t2, fov_x=fov_x, fov_y=fov_y, slice_thickness=slice_thickness)
        
        result = {
            "status": "success",
            "phantom_type": phantom_type,
            "shape": (phantom.Nz, phantom.Nx, phantom.Ny),
            "fov": (fov_x, fov_y),
            "slice_thickness": slice_thickness
        }
        
        return json.dumps(result)

