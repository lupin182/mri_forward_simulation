"""Streamlit UI for the MRI simulation agent."""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from agent.react_agent import ReActAgent
from agent.tools.phantom_tool import get_cached_phantom, get_cached_phantom_figure
from agent.tools.recon_tool import get_cached_figure, get_cached_image
from agent.tools.simulation_tool import get_cached_kspace, get_cached_kspace_figure


EXAMPLE_PROMPTS = {
    "Full GRE workflow": "Generate a 1x64x64 sphere phantom, run gre_label simulation, then reconstruct the image.",
    "Load database phantom": "List the phantom database, load the test phantom, run gre_label simulation, and reconstruct.",
    "Generate ring phantom": "Generate a 1x64x64 ring phantom with inner_radius 10 and outer_radius 22.",
    "Reconstruct current data": "Reconstruct the image from the current simulated k-space data.",
}


def init_state() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = ReActAgent()
    defaults = {
        "chat_history": [],
        "thinking_history": [],
        "pending_prompt": None,
        "show_trace": True,
        "last_run_at": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_session() -> None:
    st.session_state.agent.clear_history()
    st.session_state.chat_history = []
    st.session_state.thinking_history = []
    st.session_state.pending_prompt = None
    st.session_state.last_run_at = None
    st.session_state.last_error = None


def run_agent(prompt: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.thinking_history = []
    st.session_state.last_error = None
    try:
        response = st.session_state.agent.chat(prompt)
        st.session_state.thinking_history = st.session_state.agent.thinking_history.copy()
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.last_run_at = datetime.now().strftime("%H:%M:%S")
    except Exception as exc:
        message = f"Agent error: {exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": message})
        st.session_state.last_error = message


def get_snapshot() -> dict[str, object]:
    cached_phantom = get_cached_phantom()
    kspace = get_cached_kspace()
    image = get_cached_image()

    phantom_shape = None
    if cached_phantom is not None:
        phantom, _, _, _ = cached_phantom
        phantom_shape = (phantom.Nz, phantom.Nx, phantom.Ny)

    return {
        "phantom_ready": cached_phantom is not None,
        "phantom_shape": phantom_shape,
        "kspace_ready": kspace is not None,
        "kspace_samples": int(np.size(kspace)) if kspace is not None else None,
        "recon_ready": image is not None,
        "image_shape": tuple(image.shape) if image is not None else None,
    }


def render_sidebar() -> None:
    with st.sidebar:
        st.title("MRI Agent")
        st.caption("Natural-language control for phantom generation, forward simulation, and reconstruction.")
        st.session_state.show_trace = st.toggle("Show execution trace", value=st.session_state.show_trace)
        st.divider()
        st.subheader("Quick tasks")
        for label, prompt in EXAMPLE_PROMPTS.items():
            if st.button(label, use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()
        st.divider()
        if st.button("Clear session", type="primary", use_container_width=True):
            reset_session()
            st.rerun()


def render_status(snapshot: dict[str, object]) -> None:
    st.title("MRI Forward Simulation Agent")
    st.caption("Generate phantoms, run PyPulseq/Bloch forward simulation, and reconstruct MRI images.")
    cols = st.columns(4)
    cols[0].metric("Phantom", "Ready" if snapshot["phantom_ready"] else "Waiting")
    cols[1].metric("Shape", str(snapshot["phantom_shape"] or "-"))
    cols[2].metric("k-space samples", str(snapshot["kspace_samples"] or "-"))
    cols[3].metric("Reconstruction", str(snapshot["image_shape"] or "-"))
    if st.session_state.last_error:
        st.error(st.session_state.last_error)


def render_chat() -> None:
    st.subheader("Agent chat")
    with st.container(height=440, border=True):
        if not st.session_state.chat_history:
            st.info("Enter a task, or use a quick task from the sidebar.")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Describe the MRI simulation task")
    if prompt:
        st.session_state.pending_prompt = prompt
        st.rerun()


def render_visuals() -> None:
    st.subheader("Results")
    views = []
    if get_cached_phantom() is not None:
        views.append(("Phantom", "phantom"))
    if get_cached_kspace() is not None:
        views.append(("k-space", "kspace"))
    if get_cached_image() is not None:
        views.append(("Reconstruction", "recon"))

    if not views:
        st.info("No result data yet.")
        return

    tabs = st.tabs([label for label, _ in views])
    for tab, (_, view_type) in zip(tabs, views):
        with tab:
            display_image_for_view(view_type)


def display_image_for_view(view_type: str) -> None:
    figure = {
        "phantom": get_cached_phantom_figure,
        "kspace": get_cached_kspace_figure,
        "recon": get_cached_figure,
    }[view_type]()
    if figure is None:
        figure = build_fallback_figure(view_type)
    if figure is not None:
        st.pyplot(figure, use_container_width=True)


def build_fallback_figure(view_type: str):
    if view_type == "phantom":
        cached = get_cached_phantom()
        if cached is None:
            return None
        _, rho, t1, t2 = cached
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        for ax, title, data, cmap in [
            (axes[0], "Proton Density", rho[0, 0, 0], "gray"),
            (axes[1], "T1", t1[0, 0, 0], "viridis"),
            (axes[2], "T2", t2[0, 0, 0], "plasma"),
        ]:
            ax.set_title(title)
            ax.imshow(data, cmap=cmap)
            ax.axis("off")
        fig.tight_layout()
        return fig

    if view_type == "kspace":
        kspace = get_cached_kspace()
        if kspace is None:
            return None
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(np.abs(np.asarray(kspace).reshape(-1)))
        ax.set_title("k-space magnitude")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Magnitude")
        fig.tight_layout()
        return fig

    if view_type == "recon":
        image = get_cached_image()
        cached = get_cached_phantom()
        if image is None or cached is None:
            return None
        _, rho, _, _ = cached
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        axes[0].set_title("Reconstruction")
        axes[0].imshow(_to_2d_magnitude(image), cmap="gray")
        axes[0].axis("off")
        axes[1].set_title("Phantom")
        axes[1].imshow(rho[0, 0, 0], cmap="gray")
        axes[1].axis("off")
        fig.tight_layout()
        return fig

    return None


def _to_2d_magnitude(image) -> np.ndarray:
    data = np.abs(np.asarray(image))
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[0]
    raise ValueError(f"Expected 2D or 3D image data, got shape {data.shape}.")


def render_trace() -> None:
    if not st.session_state.show_trace:
        return
    st.subheader("Execution trace")
    if not st.session_state.thinking_history:
        st.info("No trace for the current session yet.")
        return

    for index, item in enumerate(st.session_state.thinking_history, start=1):
        if item.get("type") == "thought":
            with st.expander(f"Model response {item.get('iteration', index)}", expanded=False):
                st.markdown(str(item.get("content", "")))
        elif item.get("type") == "tool":
            with st.expander(f"Tool call: {item.get('tool_name')}", expanded=False):
                st.json(item.get("params", {}))
                result = str(item.get("result", ""))
                try:
                    st.json(json.loads(result))
                except json.JSONDecodeError:
                    st.code(result)


def main() -> None:
    st.set_page_config(page_title="MRI Agent", layout="wide")
    init_state()
    render_sidebar()

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        with st.spinner("Agent is running the simulation workflow..."):
            run_agent(prompt)
        st.rerun()

    render_status(get_snapshot())
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        render_chat()
    with right:
        render_visuals()
    render_trace()


if __name__ == "__main__":
    main()
