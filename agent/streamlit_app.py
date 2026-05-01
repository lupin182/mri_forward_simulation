
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import device_manager
device_manager.disable_cupy()
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent.main import ReActAgent
from agent.tools.phantom_tool import get_cached_phantom, get_cached_phantom_figure
from agent.tools.simulation_tool import get_cached_kspace, get_cached_kspace_figure
from agent.tools.recon_tool import get_cached_image, get_cached_figure

# ==================== 函数定义区 ====================

def display_image_for_view(view_type):
    """根据视图类型显示对应的图像"""
    if view_type == 'phantom':
        # 检查是否有缓存的体模图像
        phantom_fig = get_cached_phantom_figure()
        if phantom_fig is not None:
            st.pyplot(phantom_fig)
        else:
            # 如果没有缓存，则直接从数据创建
            cached_phantom = get_cached_phantom()
            if cached_phantom is not None:
                phantom, rho, t1, t2 = cached_phantom
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
                st.pyplot(fig)
    
    elif view_type == 'kspace':
        kspace_fig = get_cached_kspace_figure()
        if kspace_fig is not None:
            st.pyplot(kspace_fig)
        else:
            k_space_signal = get_cached_kspace()
            cached_phantom = get_cached_phantom()
            if k_space_signal is not None and cached_phantom is not None:
                phantom, _, _, _ = cached_phantom
                # 尝试将一维信号重塑为二维
                Nx, Ny = phantom.Nx, phantom.Ny
                try:
                    # 先尝试直接重塑
                    kspace_2d = np.abs(k_space_signal).reshape(Ny, Nx)
                except:
                    try:
                        # 尝试截断或填充
                        total_len = Nx * Ny
                        if len(k_space_signal) >= total_len:
                            kspace_2d = np.abs(k_space_signal[:total_len]).reshape(Ny, Nx)
                        else:
                            kspace_2d = np.zeros((Ny, Nx), dtype=np.float64)
                            kspace_2d.flat[:len(k_space_signal)] = np.abs(k_space_signal)
                    except:
                        # 如果都不行，显示一维信号
                        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                        ax.set_title("k-space Signal (1D)")
                        ax.plot(np.abs(k_space_signal))
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("Magnitude")
                        plt.tight_layout()
                        st.pyplot(fig)
                        return
                
                # 显示二维k空间
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].set_title("k-space Magnitude")
                axes[0].imshow(kspace_2d, cmap='gray')
                axes[0].axis('off')
                axes[1].set_title("k-space Log Magnitude")
                axes[1].imshow(np.log1p(kspace_2d), cmap='gray')
                axes[1].axis('off')
                plt.tight_layout()
                st.pyplot(fig)
    
    elif view_type == 'recon':
        recon_fig = get_cached_figure()
        if recon_fig is not None:
            st.pyplot(recon_fig)
        else:
            image_recon = get_cached_image()
            cached_phantom = get_cached_phantom()
            if image_recon is not None and cached_phantom is not None:
                phantom, rho, _, _ = cached_phantom
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].set_title("Reconstructed Image")
                axes[0].imshow(np.abs(image_recon[0]), cmap='gray')
                axes[0].axis('off')
                axes[1].set_title("Original Phantom")
                axes[1].imshow(rho[0, 0, 0], cmap='gray')
                axes[1].axis('off')
                plt.tight_layout()
                st.pyplot(fig)

def check_and_update_images():
    """检查是否有可用的图像数据"""
    # 这个函数现在主要用于标记状态变化
    # 实际的图像显示由display_image_for_view处理
    st.session_state.show_image = True

# ==================== 主界面区 ====================

# 页面配置
st.set_page_config(
    page_title="MRI仿真界面",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 初始化会话状态
if 'agent' not in st.session_state:
    st.session_state.agent = ReActAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_image' not in st.session_state:
    st.session_state.show_image = False
if 'image_data' not in st.session_state:
    st.session_state.image_data = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'recon'  # 'phantom', 'kspace', 'recon'
if 'thinking_history' not in st.session_state:
    st.session_state.thinking_history = []
if 'show_thinking' not in st.session_state:
    st.session_state.show_thinking = True

# 界面标题
st.title("🧲 MRI仿真智能代理系统")

# 主布局：左侧对话区，中间思考过程，右侧图像区
col1, col2, col3 = st.columns([1, 0.8, 1])

with col1:
    st.header("对话交互")
    
    # 显示聊天历史
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 输入框
    if prompt := st.chat_input("请输入您的需求..."):
        # 添加用户消息到历史
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 清空上一次的思考历史
        st.session_state.thinking_history = []
        
        # 获取代理响应
        with st.spinner("代理正在思考中..."):
            try:
                response = st.session_state.agent.chat(prompt)
                # 保存思考历史
                st.session_state.thinking_history = st.session_state.agent.thinking_history.copy()
                
                # 添加代理响应到历史
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # 检查是否有图像可以显示
                check_and_update_images()
                
            except Exception as e:
                error_msg = f"发生错误: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
    
    # 清除对话按钮
    if st.button("清空对话", key="clear_chat"):
        st.session_state.agent.clear_history()
        st.session_state.chat_history = []
        st.session_state.thinking_history = []
        st.session_state.show_image = False
        st.session_state.image_data = None
        st.rerun()

with col2:
    # 思考过程开关
    st.session_state.show_thinking = st.checkbox("显示思考过程", value=st.session_state.show_thinking)
    
    if st.session_state.show_thinking:
        st.header("思考过程")
        
        if st.session_state.thinking_history:
            for idx, item in enumerate(st.session_state.thinking_history):
                if item["type"] == "thought":
                    with st.expander(f"📝 迭代 {item['iteration']} - 思考"):
                        st.markdown(item["content"])
                elif item["type"] == "tool":
                    with st.expander(f"🛠️ 工具调用: {item['tool_name']}"):
                        st.subheader("参数")
                        st.json(item["params"])
                        st.subheader("结果")
                        st.markdown(f"```json\n{item['result']}\n```")
        else:
            st.info("等待用户输入...")
    else:
        st.header("思考过程")
        st.info("思考过程已隐藏，可勾选上方复选框显示")

with col3:
    st.header("仿真结果可视化")
    
    # 视图选择器
    view_options = []
    if get_cached_phantom() is not None:
        view_options.append(('体模数据', 'phantom'))
    if get_cached_kspace() is not None:
        view_options.append(('k空间', 'kspace'))
    if get_cached_image() is not None:
        view_options.append(('重建结果', 'recon'))
    
    if view_options:
        # 创建标签页
        tab_titles = [opt[0] for opt in view_options]
        tab_keys = [opt[1] for opt in view_options]
        tabs = st.tabs(tab_titles)
        
        # 显示每个标签页
        for i, (tab_title, tab_key) in enumerate(view_options):
            with tabs[i]:
                display_image_for_view(tab_key)
    else:
        st.info("运行仿真后，图像将在此处显示")

if __name__ == "__main__":
    pass
