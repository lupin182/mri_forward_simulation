
from langchain_core.tools import BaseTool
from typing import Any

class MRISimulationBaseTool(BaseTool):
    name: str = ""
    description: str = ""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

