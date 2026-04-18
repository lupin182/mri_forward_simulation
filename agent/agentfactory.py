'''本代码是从别的项目中搬过来的，仅有参考意义，不建议直接使用。'''

import yaml
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
import re

class AbstractAgent:
    def __init__(self):
        self.llm = None
        self.conversation_history = []
        self.mode = 'quick'

    def initialize_llm(self, model_name: str, api_key: str, base_url: str, 
            temperature: float = 0.7, max_tokens: int = 1024, top_p: float = 0.95, 
            frequency_penalty: float = 0.0, presence_penalty: float = 0.0):

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
    
    def clear_conversation_history(self):
        self.conversation_history = []
    
    def load_system_prompt(self, system_prompt_key: str, dynamic_params: dict = None):
        with open(os.path.join(os.path.dirname(__file__), '../configs/agent_prompts.yml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            system_prompt = config[system_prompt_key]

        if dynamic_params:
            if self.mode == 'quick':
                self.system_prompt = system_prompt['quick_response_prompt'].format(**dynamic_params)
            elif self.mode == 'react':
                self.system_prompt = system_prompt['react_response_prompt'].format(**dynamic_params)
        else:
            if self.mode == 'quick':
                self.system_prompt = system_prompt['quick_response_prompt']
            elif self.mode == 'react':
                self.system_prompt = system_prompt['react_response_prompt']
    
    def load_model_config(self, model_config_key: str):
        with open(os.path.join(os.path.dirname(__file__), '../configs/agent_model.yml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            model_config = config[model_config_key]
        self.model_config = model_config
    
    def load_tools(self, tool_names: list):
        """
        加载指定名称的工具
        
        Args:
            tool_names: 字符串列表，每个元素对应 agent_tools 文件夹中的工具文件名（不含.py后缀）
        """
        import importlib.util
        import sys
        from pathlib import Path
        
        tools = []
        for tool_name in tool_names:
            # 构建工具文件的路径
            tool_path = Path(__file__).parent.parent / "agent_tools" / f"{tool_name}_tool.py"
            if tool_path.exists():
                # 动态导入模块
                module_name = f"agent_tools.{tool_name}_tool"
                spec = importlib.util.spec_from_file_location(module_name, tool_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # 检查并获取 ToolClass
                if hasattr(module, 'ToolClass'):
                    tool_class = module.ToolClass
                    # 实例化工具
                    tool_instance = tool_class()
                    tools.append(tool_instance)
        
        # 生成工具描述
        self.tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" for tool in tools
        ])
        self.tool_names = [tool.name for tool in tools]
        self.tools = tools

    def react_respond(self, question: str) -> str:
        
        if not self.llm:
            raise ValueError("LLM not initialized. Call initialize_llm first.")

        # 初始化消息列表
        system_message = SystemMessage(content=self.system_prompt)
        messages = [system_message]
        messages.extend(self.conversation_history)
        messages.append(HumanMessage(content=question))

        # 2. ReAct 循环
        for i in range(self.model_config['react_max_iteration']):

            # 调用 LLM 获取思考/行动
            response = self.llm.invoke(messages)
            response_text = response.content
            print(f"第 {i+1} 轮思考: {response_text}")
            
            # 将 LLM 的这次思考加入历史
            #messages.append(AIMessage(content=response_text))

            # 3. 解析 LLM 输出 (简单的正则解析)
            # 提取 Action
            action_match = re.search(r"Action:\s*(.+)", response_text)
            # 提取 Action Input
            input_match = re.search(r"Action Input:\s*([\s\S]*?)(?=\n\w+:|$)", response_text)

            if not action_match:
                # 如果格式不对，直接返回或重试
                return response_text

            action_name = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""

            # 4. 判断是否结束
            if action_name == "Finish":
                final_answer = action_input
                # 更新会话历史
                self.conversation_history.append(HumanMessage(content=question))
                self.conversation_history.append(AIMessage(content=final_answer))
                return final_answer

            # 5. 执行工具
            # 找到对应的工具
            tool_to_use = next((t for t in self.tools if t.name == action_name), None)

            if not tool_to_use:
                observation = f"Error: Tool '{action_name}' not found."
            else:
                try:
                    # 处理输入参数，支持字符串和 self 变量
                    if action_input.startswith("self."):
                        # 提取属性名
                        attr_name = action_input.split(".")[1]
                        # 尝试获取 self 的属性
                        if hasattr(self, attr_name):
                            tool_input = getattr(self, attr_name)
                        else:
                            observation = f"Error: Attribute '{attr_name}' not found in self"
                            # 6. 将观察结果反馈给 LLM
                            observation_message = HumanMessage(content=f"Observation: {observation}")
                            messages.append(observation_message)
                            continue
                    else:
                        # 直接使用字符串作为输入
                        tool_input = action_input
                    
                    # 调用工具的 _run 方法执行工具
                    observation = str(tool_to_use._run(tool_input))
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"

            # 6. 将观察结果反馈给 LLM
            observation_message = HumanMessage(content=f"Observation: {observation}")
            messages.append(observation_message)

        # 如果超过最大迭代次数
        return "抱歉，我思考了太久仍未找到答案，请尝试重新提问。"

    def quick_respond(self, question: str) -> str:

        system_message = SystemMessage(content=self.system_prompt)
        human_message = HumanMessage(content=question)
        
        messages = [system_message]
        messages.extend(self.conversation_history)
        messages.append(human_message)
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        self.conversation_history.append(human_message)
        self.conversation_history.append(AIMessage(content=answer))
        
        return answer
    
    def respond(self, question: str) -> str:

        if self.mode == 'quick':
            return self.quick_respond(question)
        elif self.mode == 'react':
            return self.react_respond(question)
        else:
            raise ValueError("Invalid mode. Choose 'quick' or 'react'.")



