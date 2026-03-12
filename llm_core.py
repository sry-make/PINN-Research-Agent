from openai import OpenAI

# 初始化客户端，指向本地 Docker 中运行的 Ollama 服务
# 这样写的好处是，以后哪怕你把底层模型换成云端的闭源大模型，代码几乎不需要改动
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama-local' # 本地服务不需要真实的 API Key，占位即可
)

def ask_qwen(prompt):
    """向本地 Qwen 模型发送对话请求"""
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[
                # System Prompt：给大模型进行角色设定，这在 Agent 开发中极其重要
                {"role": "system", "content": "你是一个专业的 AI for Science 算法专家，精通物理信息神经网络 (PINN) 、偏微分方程 (PDE) 和计算力学。请用严谨、专业的中文回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # 针对科学问答，我们需要降低随机性，保证严谨度
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"模型调用失败，请检查 Docker 服务是否运行：{e}"

if __name__ == "__main__":
    # 测试一下它对 PINN 领域的理解
    test_question = "请简要解释什么是物理信息神经网络 (PINN)，它在求解偏微分方程时相比传统数值方法（如有限元法）有什么独特的优势？"
    
    print(f"👨‍💻 用户: {test_question}\n")
    print("🤖 Qwen2.5-7B (AI4S 专家模式) 思考中...\n")
    
    answer = ask_qwen(test_question)
    print(answer)