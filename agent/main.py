
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import device_manager
device_manager.disable_cupy()

import requests
import json
import re
from agent.config import API_KEY, BASE_URL, MODEL
from agent.tools.phantom_tool import GeneratePhantomTool, clear_phantom_cache
from agent.tools.simulation_tool import RunSimulationTool, clear_simulation_cache
from agent.tools.recon_tool import ReconstructImageTool, clear_recon_cache
from agent.tools.database_tool import ListPhantomDatabaseTool, LoadPhantomFromDatabaseTool

class ReActAgent:
    def __init__(self):
        self.conversation_history = []
        self.thinking_history = []  # 记录思考过程
        self.tools = {
            "generate_phantom": GeneratePhantomTool(),
            "list_phantom_database": ListPhantomDatabaseTool(),
            "load_phantom_from_database": LoadPhantomFromDatabaseTool(),
            "run_simulation": RunSimulationTool(),
            "reconstruct_image": ReconstructImageTool()
        }
        self.max_iterations = 10

    def _call_api(self, messages):
        url = BASE_URL
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        result = response.json()
        
        if result.get('success') and 'result' in result:
            return result['result']
        elif 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content']
        elif 'message' in result:
            return result['message']
        else:
            return str(result)

    def chat(self, user_input):
        # 清空上一次的思考历史
        self.thinking_history = []
        
        system_prompt = """你是一个专业的MRI模拟助手，采用ReAct（推理-行动）框架工作。

可用工具：
1. list_phantom_database - 列出体模数据库中所有可用的体模
   无需参数
   
2. load_phantom_from_database - 从体模数据库加载指定的体模
   参数: phantom_name（体模名称）
   
3. generate_phantom - 生成MRI体模（如果用户需要生成体模则调用，否则不调用该工具）
   参数: phantom_type, Nz, Nx, Ny, fov_x, fov_y, slice_thickness
   
4. run_simulation - 运行MRI模拟（需要先生成或加载体模）
   参数: sequence_type, tr, te, fine_dt
   
5. reconstruct_image - 重建MRI图像（需要先运行模拟）
   参数: output_path

工作流程（每次只能调用一个工具）：
1. Thought: 思考当前需要做什么
2. Action: 调用一个工具，格式为JSON: {"tool": "工具名", "params": {...}}
3. Observation: 等待工具执行结果
4. 重复上述步骤，直到任务完成

重要规则：
- phantom_type 只能是: asymmetric, ring, sphere
- sequence_type 只能是: gre, gre_label, se, epi, epi_se, epi_label
- 每次只能调用一个工具
- 如果用户需要完整流程，需要依次调用工具
- 如果用户提到数据库中的体模，先用 list_phantom_database 查看可用体模
- 任务完成后，使用 Finish: [最终答案] 结束

示例1（使用数据库体模）：
Thought: 用户想使用数据库中的体模，我需要先列出可用体模
Action: {"tool": "list_phantom_database", "params": {}}

（收到工具结果后）
Thought: 数据库中有test体模，我需要加载它
Action: {"tool": "load_phantom_from_database", "params": {"phantom_name": "test"}}

（收到工具结果后）
Thought: 体模已加载，现在需要运行模拟
Action: {"tool": "run_simulation", "params": {"sequence_type": "gre_label"}}

（收到工具结果后）
Thought: 模拟已完成，现在需要重建图像
Action: {"tool": "reconstruct_image", "params": {}}

（收到工具结果后）
Thought: 所有步骤完成，总结结果
Finish: 任务完成！已成功从数据库加载test体模、运行模拟并重建图像。图像已保存到output目录。

请严格按照这个格式回答！"""

        messages = [
            {"role": "system", "content": system_prompt}
        ] + self.conversation_history + [
            {"role": "user", "content": user_input}
        ]
        
        for iteration in range(self.max_iterations):
            print(f"\n[迭代 {iteration + 1}]")
            
            response = self._call_api(messages)
            print(f"{response}")
            
            # 记录思考过程
            self.thinking_history.append({
                "type": "thought",
                "iteration": iteration + 1,
                "content": response
            })
            
            finish_match = re.search(r'Finish:\s*(.+)', response, re.DOTALL)
            if finish_match:
                final_answer = finish_match.group(1).strip()
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                return final_answer
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    tool_call = json.loads(json_match.group())
                    if 'tool' in tool_call and tool_call['tool'] in self.tools:
                        tool_name = tool_call['tool']
                        tool = self.tools[tool_name]
                        params = tool_call.get('params', {})
                        
                        print(f"执行工具: {tool_name}")
                        tool_result = tool._run(json.dumps(params))
                        print(f"观察结果: {tool_result}")
                        
                        # 记录工具执行
                        self.thinking_history.append({
                            "type": "tool",
                            "tool_name": tool_name,
                            "params": params,
                            "result": tool_result
                        })
                        
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": f"Observation: {tool_result}"})
                    else:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": "Error: 未知的工具，请重新选择"})
                except Exception as e:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Error: 解析工具调用失败 - {str(e)}"})
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "请按照ReAct格式回答，使用Action: {...} 调用工具，或使用Finish: [...]结束"})
        
        return "抱歉，思考次数已达上限，请重新提问。"

    def clear_history(self):
        self.conversation_history = []
        self.thinking_history = []
        clear_phantom_cache()
        clear_simulation_cache()
        clear_recon_cache()

def main():
    print("=" * 60)
    print("MRI模拟智能代理系统 (ReAct框架)")
    print("=" * 60)
    print()

    agent = ReActAgent()

    print("可用功能：")
    print("1. 列出体模数据库")
    print("2. 从数据库加载体模")
    print("3. 生成MRI体模")
    print("4. 运行MRI模拟")
    print("5. 重建并可视化图像")
    print()
    print("提示1：可以直接说'用gre_label序列，64*64大小的球体体模完成模拟并重建图像'")
    print("提示2：也可以说'列出数据库中的体模，然后用test体模运行模拟'")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 60)
    print()

    while True:
        user_input = input("用户: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        try:
            response = agent.chat(user_input)
            print(f"\n代理: {response}")
        except Exception as e:
            print(f"\n代理: 发生错误 - {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    main()

