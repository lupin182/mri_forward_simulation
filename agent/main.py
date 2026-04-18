
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import device_manager
device_manager.disable_cupy()

import requests
import json
import re
from agent.config import API_KEY, BASE_URL, MODEL
from agent.tools.phantom_tool import GeneratePhantomTool
from agent.tools.simulation_tool import RunSimulationTool
from agent.tools.recon_tool import ReconstructAndVisualizeTool

class SimpleMRIAgent:
    def __init__(self):
        self.conversation_history = []
        self.tools = {
            "generate_phantom": GeneratePhantomTool(),
            "run_simulation": RunSimulationTool(),
            "reconstruct_and_visualize": ReconstructAndVisualizeTool()
        }

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
        system_prompt = """你是一个专业的MRI模拟助手。你可以帮助用户完成以下任务：

1. 生成MRI体模 - 使用 generate_phantom 工具
2. 运行MRI模拟 - 使用 run_simulation 工具
3. 完整流程并可视化 - 使用 reconstruct_and_visualize 工具

当用户提出请求时，请判断需要使用哪个工具，并以JSON格式返回工具调用信息。
JSON格式如下：
{
    "tool": "工具名称",
    "params": {
        "参数1": "值1",
        "参数2": "值2"
    }
}

重要规则：
- phantom_type 只能是以下之一：asymmetric（非对称）、ring（圆环）、sphere（球体）
- sequence_type 只能是以下之一：gre、gre_label、se、epi、epi_se、epi_label
- 如果用户说"circle"或"圆形"，请使用 "sphere"
- 如果用户说"任意"或没有指定类型，请使用 "asymmetric"

可用工具及参数：
- generate_phantom: phantom_type, Nz, Nx, Ny, fov_x, fov_y, slice_thickness
- run_simulation: sequence_type, phantom_type, Nz, Nx, Ny, fov_x, fov_y, slice_thickness, tr, te, fine_dt
- reconstruct_and_visualize: sequence_type, phantom_type, Nz, Nx, Ny, fov_x, fov_y, slice_thickness, tr, te, fine_dt, output_path, show_plot

如果不需要调用工具，请直接回答用户的问题。
请确保返回的是纯JSON，不要有其他文字！"""

        messages = [
            {"role": "system", "content": system_prompt}
        ] + self.conversation_history + [
            {"role": "user", "content": user_input}
        ]
        
        response = self._call_api(messages)
        
        print("调试 - 原始响应:", response)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                print("调试 - 提取的JSON:", json_str)
                
                tool_call = json.loads(json_str)
                print("调试 - 解析后的工具调用:", tool_call)
                
                if 'tool' in tool_call and tool_call['tool'] in self.tools:
                    tool_name = tool_call['tool']
                    tool = self.tools[tool_name]
                    params = tool_call.get('params', {})
                    
                    if 'phantom_type' in params:
                        if params['phantom_type'] in ['circle', '圆形']:
                            params['phantom_type'] = 'sphere'
                        elif params['phantom_type'] not in ['asymmetric', 'ring', 'sphere']:
                            params['phantom_type'] = 'asymmetric'
                    
                    print(f"调试 - 执行工具: {tool_name}, 参数: {params}")
                    
                    tool_result = tool._run(json.dumps(params))
                    print(f"调试 - 工具执行结果: {tool_result}")
                    
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": response})
                    
                    summary_prompt = f"工具执行结果：\n{tool_result}\n\n请给用户一个友好的总结，告诉用户任务已完成。"
                    summary_messages = messages + [
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": summary_prompt}
                    ]
                    final_response = self._call_api(summary_messages)
                    self.conversation_history.append({"role": "assistant", "content": final_response})
                    return final_response
        except Exception as e:
            print(f"调试 - 解析错误: {str(e)}")
            import traceback
            traceback.print_exc()
        
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        self.conversation_history = []

def main():
    print("=" * 60)
    print("MRI模拟智能代理系统")
    print("=" * 60)
    print()

    agent = SimpleMRIAgent()

    print("可用功能：")
    print("1. 生成MRI体模")
    print("2. 运行MRI模拟")
    print("3. 重建并可视化图像")
    print()
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
            print(f"代理: {response}")
        except Exception as e:
            print(f"代理: 发生错误 - {str(e)}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    main()

