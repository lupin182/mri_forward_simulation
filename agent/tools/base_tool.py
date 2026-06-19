"""Minimal tool interface used by the ReAct agent."""

from __future__ import annotations

from typing import Any


class MRISimulationBaseTool:
    name: str = ""
    description: str = ""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
