"""MRI simulation agent package."""

__all__ = ["ReActAgent"]


def __getattr__(name: str):
    if name == "ReActAgent":
        from agent.react_agent import ReActAgent

        return ReActAgent
    raise AttributeError(name)
