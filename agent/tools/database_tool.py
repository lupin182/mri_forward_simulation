
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mri_sim.device_manager import disable_cupy
disable_cupy()

from agent.tools.base_tool import MRISimulationBaseTool
from agent.tools.phantom_tool import set_cached_phantom
from mri_sim.phantom import Phantom
import json
import numpy as np
import re

DEPOSITORY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'mri_sim', 'phantom_depository')

_list_phantom_cache = None

class ListPhantomDatabaseTool(MRISimulationBaseTool):
    name: str = "list_phantom_database"
    description: str = """列出体模数据库中所有可用的体模。
    无需参数。
    """

    def _run(self, query: str) -> str:
        if not os.path.exists(DEPOSITORY_PATH):
            return json.dumps({"status": "error", "message": f"体模数据库路径不存在: {DEPOSITORY_PATH}"})
        
        phantoms = []
        
        index_file = None
        for filename in os.listdir(DEPOSITORY_PATH):
            if filename.endswith('.txt') and not os.path.isdir(os.path.join(DEPOSITORY_PATH, filename)):
                index_file = os.path.join(DEPOSITORY_PATH, filename)
                break
        
        if index_file:
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and ':' in line:
                            name, desc = line.split(':', 1)
                            phantoms.append({"name": name.strip(), "description": desc.strip()})
            except:
                pass
        
        if not phantoms:
            for item in os.listdir(DEPOSITORY_PATH):
                item_path = os.path.join(DEPOSITORY_PATH, item)
                if os.path.isdir(item_path):
                    has_rho = os.path.exists(os.path.join(item_path, 'rho.npy'))
                    has_t1 = os.path.exists(os.path.join(item_path, 't1.npy'))
                    has_t2 = os.path.exists(os.path.join(item_path, 't2.npy'))
                    if has_rho and has_t1 and has_t2:
                        phantoms.append({"name": item, "description": "体模数据"})
        
        result = {
            "status": "success",
            "phantoms": phantoms,
            "count": len(phantoms)
        }
        
        return json.dumps(result, ensure_ascii=False)

class LoadPhantomFromDatabaseTool(MRISimulationBaseTool):
    name: str = "load_phantom_from_database"
    description: str = """从体模数据库加载指定的体模。
    参数说明：
    - phantom_name: 体模名称（必填）
    """

    def _run(self, query: str) -> str:
        global _phantom_cache
        
        params = json.loads(query)
        phantom_name = params.get('phantom_name')
        
        if not phantom_name:
            return json.dumps({"status": "error", "message": "请提供体模名称"})
        
        phantom_dir = os.path.join(DEPOSITORY_PATH, phantom_name)
        
        if not os.path.exists(phantom_dir):
            return json.dumps({"status": "error", "message": f"体模 '{phantom_name}' 不存在"})
        
        rho_path = os.path.join(phantom_dir, 'rho.npy')
        t1_path = os.path.join(phantom_dir, 't1.npy')
        t2_path = os.path.join(phantom_dir, 't2.npy')
        
        if not all(os.path.exists(p) for p in [rho_path, t1_path, t2_path]):
            return json.dumps({"status": "error", "message": "体模数据不完整，缺少rho/t1/t2文件"})
        
        rho = np.load(rho_path)
        t1 = np.load(t1_path)
        t2 = np.load(t2_path)
        
        fov_x = 0.256
        fov_y = 0.256
        slice_thickness = 0.004
        RxCoilNum = 1
        TxCoilNum = 1
        B0 = 3.0
        dB0 = None
        txCoilmg = None
        rxCoilmg = None
        txCoilpe = None
        rxCoilpe = None
        CS = None
        dWRnd = None
        
        config_file = None
        for filename in os.listdir(phantom_dir):
            if filename.endswith('.txt') and filename != 'index.txt':
                config_file = os.path.join(phantom_dir, filename)
                break
        
        if config_file:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            if key == 'fov_x':
                                fov_x = float(value)
                            elif key == 'fov_y':
                                fov_y = float(value)
                            elif key == 'slice_thickness':
                                slice_thickness = float(value)
                            elif key == 'RxCoilNum':
                                RxCoilNum = int(value)
                            elif key == 'TxCoilNum':
                                TxCoilNum = int(value)
                            elif key == 'B0':
                                B0 = float(value)
            except:
                pass
        
        optional_files = {
            'dB0.npy': 'dB0',
            'CS.npy': 'CS',
            'dWRnd.npy': 'dWRnd',
            'txCoilmg.npy': 'txCoilmg',
            'rxCoilmg.npy': 'rxCoilmg',
            'txCoilpe.npy': 'txCoilpe',
            'rxCoilpe.npy': 'rxCoilpe'
        }
        
        for filename, var_name in optional_files.items():
            file_path = os.path.join(phantom_dir, filename)
            if os.path.exists(file_path):
                locals()[var_name] = np.load(file_path)
        
        phantom = Phantom(
            rho=rho, t1=t1, t2=t2,
            fov_x=fov_x, fov_y=fov_y, slice_thickness=slice_thickness,
            RxCoilNum=RxCoilNum, TxCoilNum=TxCoilNum, B0=B0,
            dB0=dB0,
            txCoilmg=txCoilmg, rxCoilmg=rxCoilmg,
            txCoilpe=txCoilpe, rxCoilpe=rxCoilpe,
            CS=CS, dWRnd=dWRnd
        )
        
        set_cached_phantom(phantom, rho, t1, t2)
        
        result = {
            "status": "success",
            "phantom_name": phantom_name,
            "shape": (phantom.Nz, phantom.Nx, phantom.Ny),
            "fov": (fov_x, fov_y),
            "slice_thickness": slice_thickness
        }
        
        return json.dumps(result, ensure_ascii=False)

