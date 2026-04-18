
# MRI模拟智能代理系统

基于LangChain框架开发的智能代理系统，用于通过自然语言操作MRI模拟项目。

## 目录结构

```
agent/
├── __init__.py          # 包初始化文件
├── config.py            # 配置文件
├── main.py             # 主入口文件
├── mri_agent.py       # 代理主类
├── example.py         # 使用示例
├── tools/             # 工具目录
│   ├── __init__.py
│   ├── base_tool.py   # 工具基类
│   ├── phantom_tool.py  # 体模生成工具
│   ├── simulation_tool.py  # 模拟运行工具
│   └── recon_tool.py  # 重建可视化工具
└── README.md          # 本文件
```

## 配置说明

在 `config.py` 中配置以下参数：

```python
API_KEY = ""      # API密钥
BASE_URL = ""     # API基础URL
MODEL = ""        # 模型名称
```

## 使用方法

### 方式1: 交互式运行

```bash
cd agent
python main.py
```

### 方式2: 编程使用

```python
from agent import MRIAgent

agent = MRIAgent()
agent.initialize(api_key="your_api_key", base_url="your_base_url", model="your_model")

response = agent.chat("生成一个非对称体模，分辨率64x64")
print(response)
```

## 可用功能

1. **生成MRI体模
   - 支持非对称、圆环、球体三种体模类型
   - 可自定义分辨率、视场等参数

2. **运行MRI模拟
   - 支持多种序列类型：GRE、SE、EPI等
   - 可自定义序列参数

3. **重建并可视化图像
   - 完整模拟流程
   - 图像重建和显示
   - 结果可视化

## 序列类型

- `gre`: 梯度回波
- `gre_label`: 带标签的梯度回波（默认）
- `se`: 自旋回波
- `epi`: 平面回波
- `epi_se`: 平面回波自旋回波
- `epi_label`: 带标签的平面回波

## 体模类型

- `asymmetric`: 非对称体模（默认）
- `ring`: 圆环体模
- `sphere`: 球体体模

