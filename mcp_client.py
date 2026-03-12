import asyncio
import json
import os  
from openai import OpenAI
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# 1. 链接本地 Qwen 大脑
llm_client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama-local')

async def run_agent(query):
    # 2. 核心机制：以子进程方式启动我们刚才写的 MCP 服务器
    mcp_env = os.environ.copy()
    mcp_env["HF_ENDPOINT"] = "https://hf-mirror.com"

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=mcp_env
    )

    print("🔌 正在启动并连接底层 MCP 知识库服务器...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP 协议握手成功！跨进程通信已建立。\n")

            # 3. 动态获取能力：向服务器询问“你有什么工具？”
            tools_response = await session.list_tools()
            
            # 将 MCP 标准工具转换为 OpenAI/Ollama 兼容的 Tool 格式
            llm_tools = []
            for tool in tools_response.tools:
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            # 4. 第一次对话：把用户问题和工具列表一起扔给大模型
            messages = [
                {"role": "system", "content": "你是顶尖的计算力学与 AI4S 科研 Agent。当用户询问 PINN 或偏微分方程相关论文细节时，你必须调用工具检索原文，绝不能凭空捏造。"},
                {"role": "user", "content": query}
            ]
            
            print(f"🤔 收到问题: '{query}'")
            print("🧠 Qwen2.5 正在进行逻辑推理，决定是否需要调用外部工具...\n")
            
            response = llm_client.chat.completions.create(
                model="qwen2.5:7b",
                messages=messages,
                tools=llm_tools,
                temperature=0.1
            )
            
            response_message = response.choices[0].message
            
            # 5. 核心高光时刻：判断大模型是否“决定”使用工具
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    # 大模型极其聪明，它会根据你的问题，自己提取出搜索关键词！
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🎯 [智能体决策] 触发工具调用: {tool_name}")
                    print(f"🔧 [自主生成参数]: {tool_args}\n")
                    
                    print("🚀 正在通过 MCP 协议向底层 ChromaDB 发送请求...")
                    # 6. 执行工具调用
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    tool_result_text = result.content[0].text
                    
                    print("📥 成功提取高维上下文，正在交还给大模型进行最终总结...\n")
                    
                    # 7. 第二次对话：带着工具返回的“真实数据”，让大模型做最后解答
                    messages.append(response_message) # 把之前的思考过程塞进历史记录
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result_text
                    })
                    
                    final_response = llm_client.chat.completions.create(
                        model="qwen2.5:7b",
                        messages=messages,
                        temperature=0.1
                    )
                    
                    print("================ 最终科研解答 ================\n")
                    print(final_response.choices[0].message.content)
            else:
                print("大模型认为不需要调用工具，直接给出回答：\n")
                print(response_message.content)

if __name__ == "__main__":
    # 我们再次使用那个曾让我们在传统 RAG 中翻车的简短问题
    # 看看拥有了自主意识的 Agent 会如何应对！
    test_q = "Raissi 2019年论文中的损失函数是如何构造的"
    asyncio.run(run_agent(test_q))