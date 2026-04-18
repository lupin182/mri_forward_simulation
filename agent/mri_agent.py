
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import device_manager
device_manager.disable_cupy()

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from agent.config import API_KEY, BASE_URL, MODEL, TEMPERATURE, MAX_TOKENS
from agent.tools.phantom_tool import GeneratePhantomTool
from agent.tools.simulation_tool import RunSimulationTool
from agent.tools.recon_tool import ReconstructAndVisualizeTool
import json

class MRIAgent:
    def __init__(self):
        self.llm = None
        self.tools: List[BaseTool] = []
        self.tool_map: Dict[str, BaseTool] = {}
        self.conversation_history: List = []

    def initialize(self, api_key: str = None, base_url: str = None, model: str = None):
        api_key = api_key or API_KEY
        base_url = base_url or BASE_URL
        model = model or MODEL

        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        self.tools = [
            GeneratePhantomTool(),
            RunSimulationTool(),
            ReconstructAndVisualizeTool()
        ]
        
        self.tool_map = {tool.name: tool for tool in self.tools}

    def _get_tool_schemas(self) -> List[Dict]:
        schemas = []
        for tool in self.tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "工具参数，JSON格式字符串"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        return schemas

    def chat(self, user_input: str) -> str:
        if not self.llm:
            return "请先调用 initialize() 方法初始化代理。"

        system_message = SystemMessage(content="""你是一个专业的MRI模拟助手，可以帮助用户生成体模、运行模拟和可视化结果。
请根据用户的需求选择合适的工具来完成任务。可用的工具：
1. generate_phantom - 生成MRI体模
2. run_simulation - 运行MRI模拟
3. reconstruct_and_visualize - 完整流程并可视化

工具的query参数是一个JSON字符串，包含具体的参数。
如果用户没有指定具体参数，请使用合理的默认值。""")

        messages = [system_message] + self.conversation_history + [HumanMessage(content=user_input)]

        bound_llm = self.llm.bind_tools(self._get_tool_schemas())
        response = bound_llm.invoke(messages)

        if response.tool_calls:
            tool_result = ""
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    query = tool_args.get("query", "{}")
                    result = tool._run(query)
                    tool_result += f"\n工具 {tool_name} 执行结果:\n{result}\n"
            
            self.conversation_history.append(HumanMessage(content=user_input))
            
            follow_up_messages = messages + [
                AIMessage(content=response.content, tool_calls=response.tool_calls),
                HumanMessage(content=f"工具执行完成，结果如下：\n{tool_result}\n请根据结果给用户一个总结。")
            ]
            
            final_response = self.llm.invoke(follow_up_messages)
            self.conversation_history.append(AIMessage(content=final_response.content))
            return final_response.content
        else:
            self.conversation_history.append(HumanMessage(content=user_input))
            self.conversation_history.append(AIMessage(content=response.content))
            return response.content

    def clear_history(self):
        self.conversation_history = []

