"""ReAct-style agent for driving the MRI simulation tools."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any

import requests

from agent.config import API_KEY, BASE_URL, MODEL, TEMPERATURE, require_api_key
from agent.tools.database_tool import ListPhantomDatabaseTool, LoadPhantomFromDatabaseTool
from agent.tools.phantom_tool import GeneratePhantomTool, clear_phantom_cache
from agent.tools.recon_tool import ReconstructImageTool, clear_recon_cache
from agent.tools.simulation_tool import RunSimulationTool, clear_simulation_cache


SYSTEM_PROMPT = """You are an MRI forward-simulation assistant.

You can call one tool at a time by returning exactly one JSON object:
{"tool": "tool_name", "params": {...}}

Available tools:
- list_phantom_database: list available stored phantoms.
- load_phantom_from_database: load a stored phantom by phantom_name.
- generate_phantom: create an asymmetric, ring, or sphere phantom.
- run_simulation: run a PyPulseq/Bloch simulation after a phantom exists.
- reconstruct_image: reconstruct an image after simulation exists.

Valid phantom_type values: asymmetric, ring, sphere.
Valid sequence_type values: gre, gre_label, se, epi, epi_se, epi_label.

When the task is complete, respond with:
Finish: concise final answer for the user.
"""


class ReActAgent:
    """Small ReAct loop around the MRI simulation tools."""

    def __init__(self, max_iterations: int = 10):
        self.conversation_history: list[dict[str, str]] = []
        self.thinking_history: list[dict[str, Any]] = []
        self.max_iterations = max_iterations
        self.tools = {
            "generate_phantom": GeneratePhantomTool(),
            "list_phantom_database": ListPhantomDatabaseTool(),
            "load_phantom_from_database": LoadPhantomFromDatabaseTool(),
            "run_simulation": RunSimulationTool(),
            "reconstruct_image": ReconstructImageTool(),
        }

    def _build_api_url(self) -> str:
        url = BASE_URL.rstrip("/")
        if url.endswith("/chat/completions") or url.endswith("/start"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return url

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        response = requests.post(
            self._build_api_url(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {require_api_key()}",
            },
            json={"model": MODEL, "messages": messages, "temperature": TEMPERATURE},
            timeout=60,
        )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = response.text[:500].strip()
            raise RuntimeError(f"Model API request failed: {exc}. {detail}") from exc

        try:
            result = response.json()
        except JSONDecodeError as exc:
            preview = response.text[:500].strip()
            raise RuntimeError(f"Model API returned non-JSON content: {preview}") from exc

        if result.get("success") and "result" in result:
            return str(result["result"])
        if result.get("choices"):
            return str(result["choices"][0]["message"]["content"])
        if "message" in result:
            return str(result["message"])
        return str(result)

    @staticmethod
    def _extract_tool_call(response: str) -> dict[str, Any] | None:
        match = re.search(r"\{[\s\S]*\}", response)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def chat(self, user_input: str) -> str:
        self.thinking_history = []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})

        for iteration in range(1, self.max_iterations + 1):
            response = self._call_api(messages)
            self.thinking_history.append({"type": "thought", "iteration": iteration, "content": response})

            finish_match = re.search(r"Finish:\s*(.+)", response, re.DOTALL)
            if finish_match:
                final_answer = finish_match.group(1).strip()
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                return final_answer

            tool_call = self._extract_tool_call(response)
            if not tool_call or tool_call.get("tool") not in self.tools:
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": "Return a valid tool JSON object, or use Finish: when the task is complete.",
                    }
                )
                continue

            tool_name = str(tool_call["tool"])
            params = tool_call.get("params", {})
            if not isinstance(params, dict):
                params = {}

            result = self.tools[tool_name]._run(json.dumps(params, ensure_ascii=False))
            self.thinking_history.append(
                {"type": "tool", "tool_name": tool_name, "params": params, "result": result}
            )
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {result}"})

        return "The agent reached the iteration limit before completing the task."

    def clear_history(self) -> None:
        self.conversation_history = []
        self.thinking_history = []
        clear_phantom_cache()
        clear_simulation_cache()
        clear_recon_cache()


def run_interactive_cli() -> None:
    """Run a terminal chat loop for the ReAct agent."""
    agent = ReActAgent()
    print("MRI simulation agent. Type 'quit' or 'exit' to stop.")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        try:
            print(f"Agent: {agent.chat(user_input)}")
        except Exception as exc:
            print(f"Agent error: {exc}")
