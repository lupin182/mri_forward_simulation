import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

import device_manager

device_manager.disable_cupy()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent.main import ReActAgent
from agent.tools.phantom_tool import get_cached_phantom, get_cached_phantom_figure
from agent.tools.recon_tool import get_cached_figure, get_cached_image
from agent.tools.simulation_tool import get_cached_kspace, get_cached_kspace_figure


st.set_page_config(
    page_title="MRI Agent Console",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    :root {
        --bg: #f6f8fb;
        --panel: #ffffff;
        --panel-soft: #f9fbfd;
        --ink: #17202a;
        --muted: #667085;
        --line: #dce4ef;
        --teal: #0f766e;
        --blue: #2563eb;
        --amber: #b45309;
        --rose: #be123c;
        --shadow: 0 18px 45px rgba(29, 41, 57, 0.08);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(15, 118, 110, 0.12), transparent 26%),
            radial-gradient(circle at 82% 4%, rgba(37, 99, 235, 0.10), transparent 22%),
            var(--bg);
        color: var(--ink);
    }

    .block-container {
        padding: 1.4rem 2rem 2.3rem;
        max-width: 1540px;
    }

    [data-testid="stSidebar"] {
        background: #101828;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.16);
        color: #ffffff;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(45, 212, 191, 0.75);
        color: #ffffff;
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(16, 24, 40, 0.98), rgba(21, 94, 117, 0.94)),
            linear-gradient(90deg, rgba(250, 204, 21, 0.12), transparent);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 1.3rem 1.45rem;
        margin-bottom: 1.05rem;
    }

    .hero-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.1rem;
    }

    .eyebrow {
        color: #99f6e4;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.05rem;
        line-height: 1.15;
        margin: 0.15rem 0 0.35rem;
        letter-spacing: 0;
    }

    .hero p {
        color: #d9e5ee;
        margin: 0;
        max-width: 850px;
        font-size: 0.98rem;
    }

    .status-pill {
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        color: #e6fffb;
        padding: 0.42rem 0.72rem;
        font-size: 0.82rem;
        white-space: nowrap;
        background: rgba(255, 255, 255, 0.08);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.72rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 8px;
        padding: 0.85rem;
        min-height: 84px;
    }

    .metric-label {
        color: #b6c9d6;
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.34rem;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 760;
        line-height: 1.15;
    }

    .metric-note {
        color: #cde7ef;
        font-size: 0.74rem;
        margin-top: 0.22rem;
    }

    .panel {
        background: rgba(255, 255, 255, 0.90);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.72rem;
    }

    .panel-title h2 {
        color: var(--ink);
        font-size: 1.02rem;
        margin: 0;
        letter-spacing: 0;
    }

    .panel-title span {
        color: var(--muted);
        font-size: 0.78rem;
        white-space: nowrap;
    }

    .workflow {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin-bottom: 0.85rem;
    }

    .step {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel-soft);
        padding: 0.75rem;
        min-height: 82px;
    }

    .step.ready {
        border-color: rgba(15, 118, 110, 0.42);
        background: #effcf9;
    }

    .step-name {
        color: var(--ink);
        font-size: 0.88rem;
        font-weight: 740;
        margin-bottom: 0.18rem;
    }

    .step-desc {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.38;
    }

    .empty-state {
        border: 1px dashed #b8c4d2;
        border-radius: 8px;
        background: #f8fafc;
        min-height: 352px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem;
    }

    .empty-state h3 {
        margin: 0 0 0.35rem;
        font-size: 1.15rem;
        color: var(--ink);
    }

    .empty-state p {
        color: var(--muted);
        margin: 0;
        line-height: 1.55;
    }

    .tool-row {
        display: grid;
        grid-template-columns: 128px 1fr;
        gap: 0.75rem;
        border-top: 1px solid var(--line);
        padding: 0.78rem 0;
    }

    .tool-row:first-child {
        border-top: 0;
        padding-top: 0;
    }

    .tool-badge {
        width: fit-content;
        border-radius: 999px;
        border: 1px solid rgba(37, 99, 235, 0.25);
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.74rem;
        font-weight: 720;
        padding: 0.24rem 0.55rem;
        white-space: nowrap;
    }

    .tool-body {
        color: var(--ink);
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .stChatMessage {
        border-radius: 8px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.75);
        padding: 0.15rem 0.35rem;
    }

    .stButton > button {
        border-radius: 7px;
        min-height: 2.35rem;
        font-weight: 680;
        border: 1px solid #cdd7e3;
    }

    .stButton > button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
    }

    .small-caption {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }

    @media (max-width: 980px) {
        .block-container { padding: 1rem; }
        .hero-top { align-items: flex-start; flex-direction: column; }
        .metric-grid, .workflow { grid-template-columns: 1fr; }
        .tool-row { grid-template-columns: 1fr; }
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


EXAMPLE_PROMPTS = {
    "完整流程：球体 GRE": "生成一个 1*64*64 的 sphere 体模，用 gre_label 序列运行仿真并重建图像",
    "数据库体模流程": "列出体模数据库，加载 test 体模，然后运行 gre_label 仿真并重建",
    "只生成体模": "生成一个 1*64*64 的 ring 体模，内半径 10，外半径 22",
    "只做重建": "对当前已经模拟得到的 k 空间数据进行图像重建",
}


def init_state():
    defaults = {
        "chat_history": [],
        "thinking_history": [],
        "show_thinking": True,
        "show_raw_json": False,
        "pending_prompt": None,
        "last_run_at": None,
        "last_error": None,
    }
    if "agent" not in st.session_state:
        st.session_state.agent = ReActAgent()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_shape(shape):
    if shape is None:
        return "Not ready"
    return " x ".join(str(part) for part in shape)


def get_snapshot():
    cached_phantom = get_cached_phantom()
    kspace = get_cached_kspace()
    image = get_cached_image()

    phantom_shape = None
    fov = None
    if cached_phantom is not None:
        phantom, _, _, _ = cached_phantom
        phantom_shape = (phantom.Nz, phantom.Nx, phantom.Ny)
        fov = f"{phantom.fov_x:.3f} m x {phantom.fov_y:.3f} m"

    kspace_samples = None
    if kspace is not None:
        kspace_samples = int(np.size(kspace))

    image_shape = None
    if image is not None:
        image_shape = tuple(image.shape)

    return {
        "phantom_ready": cached_phantom is not None,
        "kspace_ready": kspace is not None,
        "recon_ready": image is not None,
        "phantom_shape": phantom_shape,
        "fov": fov,
        "kspace_samples": kspace_samples,
        "image_shape": image_shape,
    }


def render_hero(snapshot):
    ready_count = sum(
        [snapshot["phantom_ready"], snapshot["kspace_ready"], snapshot["recon_ready"]]
    )
    last_run = st.session_state.last_run_at or "本轮尚未运行"
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-top">
                <div>
                    <div class="eyebrow">MRI Forward Simulation Agent</div>
                    <h1>磁共振前向仿真智能控制台</h1>
                    <p>用自然语言组织体模生成、序列仿真、k-space 采样与图像重建，并把代理行动轨迹和结果数据放在同一个工作台里。</p>
                </div>
                <div class="status-pill">Pipeline {ready_count}/3 · {last_run}</div>
            </div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">体模状态</div>
                    <div class="metric-value">{'Ready' if snapshot['phantom_ready'] else 'Waiting'}</div>
                    <div class="metric-note">{format_shape(snapshot['phantom_shape'])}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">k-space 采样</div>
                    <div class="metric-value">{snapshot['kspace_samples'] if snapshot['kspace_samples'] else 'Waiting'}</div>
                    <div class="metric-note">signal samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">重建图像</div>
                    <div class="metric-value">{'Ready' if snapshot['recon_ready'] else 'Waiting'}</div>
                    <div class="metric-note">{format_shape(snapshot['image_shape'])}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">视场范围</div>
                    <div class="metric-value">{snapshot['fov'] or 'Not set'}</div>
                    <div class="metric-note">field of view</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_workflow(snapshot):
    steps = [
        ("01 体模", "生成或加载 rho / T1 / T2 体模数据。", snapshot["phantom_ready"]),
        ("02 仿真", "写入序列并执行 Bloch 前向模拟。", snapshot["kspace_ready"]),
        ("03 重建", "FFT 重建并对照原始体模。", snapshot["recon_ready"]),
    ]
    html_parts = ['<div class="workflow">']
    for name, desc, ready in steps:
        klass = "step ready" if ready else "step"
        state = "已完成" if ready else "等待"
        html_parts.append('<div class="%s"><div class="step-name">%s · %s</div><div class="step-desc">%s</div></div>' % (klass, name, state, desc))
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def build_fallback_figure(view_type):
    if view_type == "phantom":
        cached_phantom = get_cached_phantom()
        if cached_phantom is None:
            return None
        _, rho, t1, t2 = cached_phantom
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        plots = [
            ("Proton Density", rho[0, 0, 0], "gray"),
            ("T1 Relaxation", t1[0, 0, 0], "viridis"),
            ("T2 Relaxation", t2[0, 0, 0], "plasma"),
        ]
        for ax, (title, data, cmap) in zip(axes, plots):
            ax.set_title(title)
            ax.imshow(data, cmap=cmap)
            ax.axis("off")
        fig.tight_layout()
        return fig

    if view_type == "kspace":
        kspace = get_cached_kspace()
        cached_phantom = get_cached_phantom()
        if kspace is None:
            return None
        if cached_phantom is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.plot(np.abs(kspace))
            ax.set_title("k-space Signal")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Magnitude")
            fig.tight_layout()
            return fig

        phantom, _, _, _ = cached_phantom
        total_len = phantom.Nx * phantom.Ny
        kspace_abs = np.abs(kspace)
        if np.size(kspace_abs) >= total_len:
            kspace_2d = kspace_abs[:total_len].reshape(phantom.Ny, phantom.Nx)
        else:
            kspace_2d = np.zeros((phantom.Ny, phantom.Nx), dtype=np.float64)
            kspace_2d.flat[: np.size(kspace_abs)] = kspace_abs

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].set_title("Magnitude")
        axes[0].imshow(kspace_2d, cmap="gray")
        axes[0].axis("off")
        axes[1].set_title("Log Magnitude")
        axes[1].imshow(np.log1p(kspace_2d), cmap="magma")
        axes[1].axis("off")
        fig.tight_layout()
        return fig

    if view_type == "recon":
        image = get_cached_image()
        cached_phantom = get_cached_phantom()
        if image is None or cached_phantom is None:
            return None
        _, rho, _, _ = cached_phantom
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].set_title("Reconstruction")
        axes[0].imshow(np.abs(image[0]), cmap="gray")
        axes[0].axis("off")
        axes[1].set_title("Original Phantom")
        axes[1].imshow(rho[0, 0, 0], cmap="gray")
        axes[1].axis("off")
        fig.tight_layout()
        return fig

    return None


def display_image_for_view(view_type):
    cached_figure = {
        "phantom": get_cached_phantom_figure,
        "kspace": get_cached_kspace_figure,
        "recon": get_cached_figure,
    }[view_type]()
    fig = cached_figure or build_fallback_figure(view_type)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)


def available_views():
    views = []
    if get_cached_phantom() is not None:
        views.append(("体模参数图", "phantom"))
    if get_cached_kspace() is not None:
        views.append(("k-space", "kspace"))
    if get_cached_image() is not None:
        views.append(("重建结果", "recon"))
    return views


def run_agent(prompt):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.thinking_history = []
    st.session_state.last_error = None
    try:
        response = st.session_state.agent.chat(prompt)
        st.session_state.thinking_history = st.session_state.agent.thinking_history.copy()
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")
    except Exception as exc:
        message = f"发生错误：{exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": message})
        st.session_state.last_error = message


def reset_session():
    st.session_state.agent.clear_history()
    st.session_state.chat_history = []
    st.session_state.thinking_history = []
    st.session_state.pending_prompt = None
    st.session_state.last_run_at = None
    st.session_state.last_error = None


def render_sidebar():
    with st.sidebar:
        st.markdown("## 控制台")
        st.caption("面向 MRI 前向仿真的代理工作台")
        st.divider()

        st.session_state.show_thinking = st.toggle(
            "显示代理轨迹", value=st.session_state.show_thinking
        )
        st.session_state.show_raw_json = st.toggle(
            "展开工具 JSON", value=st.session_state.show_raw_json
        )

        st.divider()
        st.markdown("### 快捷任务")
        for label, prompt in EXAMPLE_PROMPTS.items():
            if st.button(label, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()

        st.divider()
        if st.button("清空会话与缓存", type="primary", use_container_width=True):
            reset_session()
            st.rerun()

        st.caption(
            "提示：完整流程通常按“生成/加载体模 -> 运行仿真 -> 重建图像”的顺序执行。"
        )


def render_chat_panel():
    st.markdown(
        """
        <div class="panel-title">
            <h2>自然语言指令</h2>
            <span>Agent chat</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(height=470, border=True):
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="empty-state">
                    <div>
                        <h3>从一句实验意图开始</h3>
                        <p>例如：生成 sphere 体模，运行 gre_label 序列，然后完成重建。左侧也准备了几个可直接执行的快捷任务。</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("输入你的 MRI 仿真任务，例如：生成 ring 体模并运行 EPI 仿真")
    if prompt:
        st.session_state.pending_prompt = prompt
        st.rerun()


def render_visual_panel():
    views = available_views()
    st.markdown(
        """
        <div class="panel-title">
            <h2>结果可视化</h2>
            <span>Phantom / k-space / Recon</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not views:
        st.markdown(
            """
            <div class="empty-state">
                <div>
                    <h3>尚无可视化数据</h3>
                    <p>运行体模生成、仿真或重建后，这里会自动出现对应图像。</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    tabs = st.tabs([title for title, _ in views])
    for tab, (_, key) in zip(tabs, views):
        with tab:
            display_image_for_view(key)


def render_agent_trace():
    if not st.session_state.show_thinking:
        return

    st.markdown(
        """
        <div class="panel-title">
            <h2>代理执行轨迹</h2>
            <span>Reasoning and tool calls</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.thinking_history:
        st.info("等待下一次代理执行。")
        return

    for index, item in enumerate(st.session_state.thinking_history, start=1):
        if item.get("type") == "thought":
            with st.expander(f"迭代 {item.get('iteration', index)} · 推理输出", expanded=False):
                st.markdown(item.get("content", ""))
        elif item.get("type") == "tool":
            st.markdown(
                f"""
                <div class="tool-row">
                    <div><span class="tool-badge">{item.get('tool_name', 'tool')}</span></div>
                    <div class="tool-body">工具已执行，参数和结果如下。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("参数与返回结果", expanded=st.session_state.show_raw_json):
                st.markdown("**参数**")
                st.json(item.get("params", {}))
                st.markdown("**结果**")
                result = item.get("result", "")
                try:
                    st.json(json.loads(result))
                except Exception:
                    st.code(result, language="json")


def main():
    init_state()
    render_sidebar()

    if st.session_state.pending_prompt:
        prompt_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        with st.spinner("代理正在执行仿真链路，请稍候..."):
            run_agent(prompt_to_run)
        st.rerun()

    snapshot = get_snapshot()
    render_hero(snapshot)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    render_workflow(snapshot)

    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        render_chat_panel()
    with right:
        render_visual_panel()

    render_agent_trace()


if __name__ == "__main__":
    main()
