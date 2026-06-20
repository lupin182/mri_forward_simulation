# MRI Forward Simulation

本项目是一个磁共振成像前向模拟系统，支持体模生成与数据库管理、PyPulseq 序列生成、Bloch 方程前向模拟、k-space 采样、图像重建、RF/B0 伪影模拟，以及基于自然语言的智能代理工作台。

根目录唯一 Python 用户入口是 `main.py`。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CuPy 是可选依赖。未安装 CuPy 或没有可用 GPU 时，项目自动使用 NumPy/CPU。

## 基本模拟

默认运行完整前向模拟：

```powershell
python main.py
```

显式运行：

```powershell
python main.py simulate --sequence gre_label --phantom asymmetric --nx 64 --ny 64
```

输出目录默认是 `output/`，包含 `kspace.npy`、`reconstruction.npy`、`reconstruction_magnitude.npy`、`summary.json` 和可选 PNG。

## 体模参数与序列参数

体模几何参数：

```text
--nx --ny --nz
--fov-x --fov-y --slice-thickness
```

序列几何参数：

```text
--seq-nx --seq-ny --seq-n-slices
--seq-fov-x --seq-fov-y --seq-slice-thickness
```

规则：

- 体模构建只使用体模参数。
- 序列构建优先使用 `--seq-*` 参数。
- 如果没有传入对应 `--seq-*`，序列默认跟随体模几何。
- `gre` 和 `epi_se` 当前是单层序列，不支持 `--seq-n-slices`。
- 重建维度使用有效序列矩阵和有效序列层数。
- `summary.json` 同时记录 `phantom_shape` 和 `sequence_shape`。

示例：体模网格和序列采样矩阵相同：

```powershell
python main.py simulate --sequence gre_label --nx 64 --ny 64
```

示例：体模网格 128 x 128，序列采样矩阵 64 x 64：

```powershell
python main.py simulate --phantom asymmetric --nx 128 --ny 128 --sequence gre_label --seq-nx 64 --seq-ny 64
```

## 序列参数

通用时间参数：

```text
--tr / --seq-tr
--te / --seq-te
```

`--tr` 和 `--te` 是兼容旧用法的别名；推荐新命令使用 `--seq-tr` 和 `--seq-te`。

GRE 与 GRE label 支持：

```text
--seq-flip-angle-deg
--seq-rf-spoiling-inc-deg
--seq-dummy-scans
--seq-ideal-spoiling-reset
--no-seq-ideal-spoiling-reset
```

GRE label 额外支持：

```text
--seq-readout-duration
```

SE 支持：

```text
--seq-excitation-flip-angle-deg
--seq-refocusing-flip-angle-deg
--seq-rf-excitation-duration
--seq-rf-refocusing-duration
--seq-readout-time
--seq-prephase-duration
```

EPI label 支持：

```text
--seq-n-reps
--seq-n-navigator
```

如果显式传入当前序列不支持的 `--seq-*` 参数，程序会直接报错。例如：

```powershell
python main.py simulate --sequence epi --seq-n-reps 2
```

## 序列示例

GRE label：

```powershell
python main.py simulate --sequence gre_label --seq-flip-angle-deg 10 --seq-readout-duration 0.0032 --seq-dummy-scans 2 --no-seq-ideal-spoiling-reset
```

SE：

```powershell
python main.py simulate --sequence se --seq-excitation-flip-angle-deg 90 --seq-refocusing-flip-angle-deg 180 --seq-readout-time 0.004 --seq-prephase-duration 0.0012
```

EPI label：

```powershell
python main.py simulate --sequence epi_label --seq-n-reps 2 --seq-n-navigator 3
```

## 体模数据库

数据库根目录：

```text
mri_sim/phantom_depository/
```

每个体模是一个子目录：

```text
mri_sim/phantom_depository/<phantom_name>/
```

数据库索引文件：

```text
mri_sim/phantom_depository/info.txt
```

索引格式：

```text
name:description
```

管理命令只在 `simulate database` 下提供：

```powershell
python main.py simulate database list
python main.py simulate database create --name brain_demo --description "Brain phantom with coil maps"
python main.py simulate database load --name brain_demo --data rho --file-path E:\data\rho.npy
python main.py simulate database delete --name brain_demo --data rho
python main.py simulate database delete --name brain_demo --all
```

合法数据名：

```text
rho t1 t2 dB0 CS dWRnd txCoilmg txCoilpe rxCoilmg rxCoilpe info
```

除 `info` 外，其它数据必须是 `.npy` 文件。完整体模目录可以包含：

```text
info.txt
rho.npy
t1.npy
t2.npy
dB0.npy
CS.npy
dWRnd.npy
txCoilmg.npy
txCoilpe.npy
rxCoilmg.npy
rxCoilpe.npy
```

体模数组支持 `(Nz, Nx, Ny)` 或 `(TypeNum, SpinNum, Nz, Nx, Ny)`。线圈灵敏度图形状为 `(CoilNum, Nz, Nx, Ny)`。

使用数据库体模：

```powershell
python main.py simulate --phantom database --phantom-name test --sequence gre_label
```

## 伪影模拟

RF 伪影：

```powershell
python main.py simulate --rf-artifact --rf-noise-freq 127700000 --rf-noise-amp 5.0 --bg-noise-amp 1.0
```

B0 不均匀场：

```powershell
python main.py simulate --b0-artifact --b0-mode linear --b0-delta-ppm 0.5 --b0-axis x
```

如果数据库体模已经包含 `dB0.npy`，默认使用数据库中的 B0 不均匀场；显式传入 `--b0-artifact` 时会重新生成并覆盖当前 `phantom.dB0`。

## Agent

命令行 agent：

```powershell
python main.py agent-cli
```

Streamlit UI：

```powershell
python main.py agent-ui
```

Agent 只负责生成/加载体模、运行模拟和重建，不提供数据库创建、导入或删除能力。

## 验证

静态解析：

```powershell
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py')]; print('ok')"
```

小矩阵 smoke test：

```powershell
python main.py simulate --sequence gre_label --phantom asymmetric --nx 8 --ny 8 --fine-dt 1e-5 --no-plot
```

序列与体模几何拆分：

```powershell
python main.py simulate --sequence gre_label --phantom asymmetric --nx 16 --ny 16 --seq-nx 8 --seq-ny 8 --fine-dt 1e-5 --no-plot
```
