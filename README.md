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

## 硬件配置

所有序列生成函数都会通过根目录 `.env` 中的 `MRI_SYSTEM_*` 配置创建 PyPulseq `pp.Opts` 硬件对象。系统环境变量优先级高于 `.env`；如果二者都没有配置，则使用代码内保守默认值。

推荐配置：

```text
MRI_SYSTEM_MAX_GRAD=32
MRI_SYSTEM_GRAD_UNIT=mT/m
MRI_SYSTEM_MAX_SLEW=130
MRI_SYSTEM_SLEW_UNIT=T/m/s
MRI_SYSTEM_RF_RINGDOWN_TIME=20e-6
MRI_SYSTEM_RF_DEAD_TIME=100e-6
MRI_SYSTEM_ADC_DEAD_TIME=10e-6
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MRI_SYSTEM_MAX_GRAD` | `32` | 最大梯度强度。 |
| `MRI_SYSTEM_GRAD_UNIT` | `mT/m` | 最大梯度强度单位，传给 PyPulseq `grad_unit`。 |
| `MRI_SYSTEM_MAX_SLEW` | `130` | 最大 slew rate。 |
| `MRI_SYSTEM_SLEW_UNIT` | `T/m/s` | slew rate 单位，传给 PyPulseq `slew_unit`。 |
| `MRI_SYSTEM_RF_RINGDOWN_TIME` | `20e-6` | RF ringdown 时间，单位秒。 |
| `MRI_SYSTEM_RF_DEAD_TIME` | `100e-6` | RF dead time，单位秒。 |
| `MRI_SYSTEM_ADC_DEAD_TIME` | `10e-6` | ADC dead time，单位秒。 |

如果需要模拟更强硬件，可以在 `.env` 中改为例如：

```text
MRI_SYSTEM_MAX_GRAD=120
MRI_SYSTEM_MAX_SLEW=200
```

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

## 当前支持的序列

| 序列名 | Description |
| --- | --- |
| `gre` | 基础梯度回波序列，单层，支持 flip angle、RF spoiling 和 dummy scans。 |
| `gre_label` | 带 PyPulseq label 的多层 GRE，适合作为默认完整流程序列。 |
| `se` | 标准笛卡尔 spin echo 序列，支持 90/180 度脉冲和 readout/prephase 时长控制。 |
| `tse` | Turbo spin echo 序列，支持 echo train、refocusing flip angle 和多层采集。 |
| `epi` | 基础 EPI 序列，无 ramp sampling。 |
| `epi_se` | 单层 spin-echo EPI 序列。 |
| `epi_label` | 带 label、navigator、repetition 控制的 EPI 序列。 |
| `database` | 从本地序列数据库读取已导入的 `.seq` 文件，矩阵和重建维度由 CLI 几何参数控制。 |

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

TSE 支持：

```text
--seq-n-echo
--seq-rf-flip-deg
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

数据库序列：

```powershell
python main.py simulate --sequence database --sequence-name my_gre --phantom asymmetric --nx 64 --ny 64
```

## 序列数据库

序列数据库用于管理已经生成好的 PyPulseq `.seq` 文件。数据库根目录：

```text
mri_sim/seq_depository/
```

目录结构：

```text
mri_sim/seq_depository/
├── info.txt
└── <sequence_name>/
    └── <sequence_name>.seq
```

索引格式：

```text
name:description
```

每个序列条目是一个目录，目录内只要求存在同名 `.seq` 文件。导入时会先用 PyPulseq 读取源文件；如果读取失败，不会写入数据库。同名导入表示更新：覆盖 `<sequence_name>/<sequence_name>.seq`，并更新 `info.txt` 中的描述。

建议在 `load` 时把重要序列参数写入 `--description`，例如矩阵大小、FOV、层数、TR、TE、序列类型等，便于后续列出数据库序列时快速确认该 `.seq` 文件的采集配置。

管理命令只在 `simulate sequence-database` 下提供：

```powershell
python main.py simulate sequence-database list
python main.py simulate sequence-database load --name my_gre --description "Imported GRE sequence" --file-path E:\data\my_gre.seq
python main.py simulate sequence-database delete --name my_gre
```

使用数据库序列参与模拟：

```powershell
python main.py simulate --sequence database --sequence-name my_gre --phantom asymmetric --nx 64 --ny 64
```

数据库序列不会自动推断重建矩阵。默认使用体模几何；如果需要让体模网格和序列采样矩阵不同，显式传入：

```powershell
python main.py simulate --sequence database --sequence-name my_gre --nx 128 --ny 128 --seq-nx 64 --seq-ny 64 --seq-n-slices 1
```

`database` 序列只接受通用序列几何参数：

```text
--seq-nx --seq-ny --seq-n-slices
--seq-fov-x --seq-fov-y --seq-slice-thickness
```

它不接受内置序列的专属参数，例如 `--seq-n-reps`、`--seq-flip-angle-deg`、`--seq-n-echo`。如果显式传入这些参数，程序会报错。

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
