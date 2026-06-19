import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from mri_sim.device_manager import disable_cupy
disable_cupy()
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
# （此区域保持你的原样，不做核心逻辑的更改）

def display_image_for_view(view_type):
    """根据视图类型显示对应的图像"""
    if view_type == 'phantom':
        phantom_fig = get_cached_phantom_figure()
        if phantom_fig is not None:
            st.pyplot(phantom_fig)
        else:
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
                Nx, Ny = phantom.Nx, phantom.Ny
                try:
                    kspace_2d = np.abs(k_space_signal).reshape(Ny, Nx)
                except:
                    try:
                        total_len = Nx * Ny
                        if len(k_space_signal) >= total_len:
                            kspace_2d = np.abs(k_space_signal[:total_len]).reshape(Ny, Nx)
                        else:
                            kspace_2d = np.zeros((Ny, Nx), dtype=np.float64)
                            kspace_2d.flat[:len(k_space_signal)] = np.abs(k_space_signal)
                    except:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                        ax.set_title("k-space Signal (1D)")
                        ax.plot(np.abs(k_space_signal))
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("Magnitude")
                        plt.tight_layout()
                        st.pyplot(fig)
                        return
                
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
    st.session_state.show_image = True


# ==================== 主界面区 ====================

# 1. 页面级全局配置
st.set_page_config(
    page_title="MRI仿真智能代理",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded" # 默认展开侧边栏
)

# 2. 注入自定义 CSS 提升质感
st.markdown("""
    <style>
    /* 隐藏默认菜单和页脚，使界面更像独立App */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* 优化顶部空白 */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    /* 美化按钮 */
    div.stButton > button { border-radius: 6px; font-weight: 500; }
    /* 标签页文本加粗 */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 初始化会话状态
if 'agent' not in st.session_state:
    st.session_state.agent = ReActAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_image' not in st.session_state:
    st.session_state.show_image = False
if 'thinking_history' not in st.session_state:
    st.session_state.thinking_history = []
if 'show_thinking' not in st.session_state:
    st.session_state.show_thinking = True

# 4. 侧边栏配置 (控制面板)
with st.sidebar:
    st.title("⚙️ 系统控制台")
    st.markdown("---")
    
    # 将开关变成 Toggle，更现代
    st.session_state.show_thinking = st.toggle("🔍 实时显示代理思考过程", value=st.session_state.show_thinking)
    
    st.markdown("---")
    st.caption("操作区")
    if st.button("🗑️ 清空当前对话", type="primary", use_container_width=True):
        st.session_state.agent.clear_history()
        st.session_state.chat_history = []
        st.session_state.thinking_history = []
        st.session_state.show_image = False
        st.rerun()
        
    st.markdown("---")
    st.caption("💡 **Tips:** \n这是一个基于 ReAct 架构的 MRI 前向仿真智能代理。您可以输入自然语言指令（例如：“帮我生成一个体模数据” 或 “对刚刚的数据进行 k 空间仿真”）。")

# 5. 顶部标题区域
st.title("🧲 MRI 仿真智能交互平台")
st.markdown("通过自然语言驱动磁共振前向仿真、k空间数据生成及图像重建。")
st.divider()

# 6. 主内容区：使用 4:6 比例拆分左右两侧
col_chat, col_vis = st.columns([4, 6], gap="large")

# ================= 左侧：对话与思考区 =================
with col_chat:
    st.subheader("💬 指令与交互")
    
    # 聊天记录显示区 (使用带边框容器包裹，显得干净整洁)
    chat_container = st.container(border=True, height=550) # 限制高度，超出自动滚动
    
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 欢迎！我是您的 MRI 仿真助手。请输入指令开始仿真任务。")
            
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # 思考过程现在紧跟在最近一次对话下面展示
        if st.session_state.show_thinking and st.session_state.thinking_history:
            st.caption("⚙️ 最新一轮代理思考记录：")
            for idx, item in enumerate(st.session_state.thinking_history):
                if item["type"] == "thought":
                    with st.expander(f"📝 步骤 {item['iteration']} - 逻辑推理", expanded=False):
                        st.markdown(item["content"])
                elif item["type"] == "tool":
                    with st.expander(f"🛠️ 动作调用: {item['tool_name']}", expanded=False):
                        st.markdown("**传入参数:**")
                        st.json(item["params"])
                        st.markdown("**返回结果:**")
                        st.code(item["result"], language="json")

    # 对话输入框 (吸附在聊天记录下方)
    if prompt := st.chat_input("请输入您的指令，例如：生成体模数据..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.thinking_history = [] # 清空上一次的思考
        st.rerun() # 触发页面重绘，使刚输入的问题立刻显示到容器中

# 处理后台逻辑 (在rerun之后捕获最后一条未回复的用户消息)
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    user_prompt = st.session_state.chat_history[-1]["content"]
    
    with col_chat:
        with st.spinner("🤖 代理正在推演仿真链路，请稍候..."):
            try:
                response = st.session_state.agent.chat(user_prompt)
                st.session_state.thinking_history = st.session_state.agent.thinking_history.copy()
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                check_and_update_images()
                st.rerun() # 再次重绘以展示最终结果和图表
            except Exception as e:
                error_msg = f"❌ 发生错误: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()

# ================= 右侧：可视化结果区 =================
with col_vis:
    st.subheader("📊 仿真结果可视化")
    
    # 同样使用带边框的容器，在视觉上与左侧对称呼应
    vis_container = st.container(border=True, height=620)
    
    with vis_container:
        view_options = []
        if get_cached_phantom() is not None:
            view_options.append(('🔲 体模数据', 'phantom'))
        if get_cached_kspace() is not None:
            view_options.append(('🌊 k空间数据', 'kspace'))
        if get_cached_image() is not None:
            view_options.append(('🖼️ 重建结果', 'recon'))
        
        if view_options:
            tab_titles = [opt[0] for opt in view_options]
            tab_keys = [opt[1] for opt in view_options]
            
            # 渲染标签页
            tabs = st.tabs(tab_titles)
            for i, (tab_title, tab_key) in enumerate(view_options):
                with tabs[i]:
                    st.markdown("<br>", unsafe_allow_html=True) # 增加一点顶部留白
                    display_image_for_view(tab_key)
        else:
            # 空白状态时的占位提示
            st.empty()
            st.markdown("""
                <div style='display: flex; justify-content: center; align-items: center; height: 400px; color: #6b7280; flex-direction: column;'>
                    <h1 style='font-size: 3rem;'>🩻</h1>
                    <p style='font-size: 1.2rem; margin-top: 1rem;'>暂无图像数据</p>
                    <p style='font-size: 0.9rem;'>请在左侧通过指令生成体模或运行仿真步骤</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    pass
