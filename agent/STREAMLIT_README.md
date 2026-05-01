
# MRI仿真 Streamlit GUI 使用说明

## 快速启动

1. 安装依赖（如需要）：
   ```bash
   cd agent
   pip install -r requirements.txt
   ```

2. 启动Streamlit应用：
   ```bash
   streamlit run streamlit_app.py
   ```

3. 在浏览器中打开显示的URL（通常是 http://localhost:8501）

## 功能说明

### 界面布局
- **左侧区域**：与MRI仿真智能代理对话交互
- **右侧区域**：仿真结果可视化（标签页展示体模、k空间、重建图像）

### 使用流程
1. 在对话框中输入您的需求，例如：
   - "生成一个球体体模，64x64大小"
   - "用gre_label序列进行模拟"
   - "重建图像并显示"

2. 系统会自动调用相应工具并在右侧展示结果

### 可视化功能
- **体模数据**：显示质子密度、T1、T2弛豫时间图
- **k空间**：显示k空间幅度和对数幅度图
- **重建结果**：显示重建图像和原始体模对比图

## 文件说明

- `streamlit_app.py` - Streamlit主界面文件
- `requirements.txt` - Streamlit所需依赖
- `tools/phantom_tool.py` - 修改后的体模生成工具（支持可视化）
- `tools/simulation_tool.py` - 修改后的仿真工具（支持k空间可视化）
- `tools/recon_tool.py` - 修改后的重建工具（支持可视化）

## 注意事项

- 所有代码修改均兼容原有功能
- 工具的`return_figure`参数用于Streamlit界面展示，默认不影响原有行为
