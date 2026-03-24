import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# 1. 挂载 Qwen 大脑 (大模型客户端)
client_llm = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama-local'
)

# 2. 挂载 ChromaDB 记忆 (向量数据库客户端)
CHROMA_DATA_PATH = "./chroma_db"
client_db = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# 使用与构建知识库时完全相同的 Embedding 模型，确保向量空间一致
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client_db.get_collection(name="pinn_papers", embedding_function=ef)

# 重写短文本、提高命中率。

def rewrite_query(short_query):
    """【黑科技：HyDE 查询重写】利用大模型自身的先验知识，将短查询扩写为包含丰富专业术语的段落"""
    print("🧠 触发查询重写引擎，正在扩写用户的短查询...")
    
    rewrite_prompt = f"""你是一个物理信息神经网络 (PINN) 领域的资深学者。
    用户提出了一个极其简短的搜索词。为了在专业的学术论文数据库中进行精准的向量检索，请你运用你的专业知识，将这个短搜索词扩写成一段包含丰富学术术语、核心概念（例如：正问题、逆问题、配点、残差等）的假设性陈述句或伪答案。
    
    注意：
    1. 不要输出多余的解释，直接输出扩写后的一段话。
    2. 扩写的长度控制在 100-200 字左右。
    
    【用户的短搜索词】：{short_query}
    【扩写后的专业文本】："""

    response = client_llm.chat.completions.create(
        model="pinn_qwen_expert",
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.5, # 扩写需要一定的发散性，稍微调高一点
        max_tokens=300
    )
    expanded_text = response.choices[0].message.content.strip()
    return expanded_text

def retrieve_context(query, top_k=3, use_reranker=False, reranker_instance=None, coarse_top_k=20):
    """【感知模块】从数据库中检索与用户问题最相关的 Top-K 个论文片段。

    Args:
        query: 查询文本
        top_k: 最终返回的文档数
        use_reranker: 是否启用重排
        reranker_instance: BGEReranker 实例
        coarse_top_k: 粗召回数量（仅 reranker 模式生效）

    Returns:
        (context_str, metadatas): 拼接后的上下文文本 + 元数据列表
    """
    if use_reranker and reranker_instance is not None:
        # 粗召回 Top-N → 精排 Top-K
        results = collection.query(query_texts=[query], n_results=coarse_top_k)
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        docs, metas = reranker_instance.rerank(query, docs, metas, top_k=top_k)
    else:
        results = collection.query(query_texts=[query], n_results=top_k)
        docs = results['documents'][0]
        metas = results['metadatas'][0]

    context_str = "\n\n---\n\n".join(docs)
    return context_str, metas

def ask_rag_agent(query, mode="hyde", reranker_instance=None):
    """【调度中心 2.0】支持多模式的 RAG 问答流程。

    Args:
        query: 用户问题
        mode: 检索模式 - "direct" / "hyde" / "hyde_reranker"
        reranker_instance: BGEReranker 实例（仅 hyde_reranker 模式需要）
    """
    print(f"\n🔍 [阶段 1] 收到原始问题: '{query}' (模式: {mode})")

    if mode == "direct":
        # 直接用原始 query 检索，不做 HyDE 重写
        search_query = query
        print("📌 [阶段 1] Direct 模式：跳过查询重写，直接检索。")
    else:
        # HyDE 或 HyDE+Reranker：先重写
        search_query = rewrite_query(query)
        print(f"✨ [阶段 1] 重写/扩写后的高维检索词:\n{search_query}\n")

    # 根据模式决定是否启用 reranker
    use_reranker = (mode == "hyde_reranker")
    print("🔍 [阶段 2] 正在从本地论文库中检索...")
    context, metadatas = retrieve_context(
        search_query,
        use_reranker=use_reranker,
        reranker_instance=reranker_instance,
    )
    print("✅ [阶段 2] 检索完成！成功提取到强相关上下文。\n")

    prompt = f"""你是一个专业的 AI for Science 和 PINN 算法专家。
    请严格基于以下提供的【参考论文片段】来回答用户的问题。
    如果参考片段中没有相关信息，请明确说明“根据提供的参考资料无法直接回答”，绝不要自行编造或产生幻觉。

    【参考论文片段】：
    {context}

    【用户真实问题】：
    {query}
    """

    print("🤖 [阶段 3] Qwen2.5-7B 正在进行终极推理...\n")
    response = client_llm.chat.completions.create(
        model="pinn_qwen_expert",
        messages=[
            {"role": "system", "content": """你是一个顶级的物理信息神经网络（PINN）学术专家。请基于检索到的上下文回答问题。

    【极其重要的严谨性指令】：
    1. 绝对禁止使用类似 MSE_u_u, MSE_u_f_f 这种非标准的晦涩符号！
    2. 描述损失函数时，必须使用 PINN 领域的标准符号：
     - 物理残差损失（PDE Residual Loss）必须表示为 $L_{f}$ 或 $L_{r}$。
     - 数据/边界/初始条件损失（Data/Boundary/Initial Loss）必须表示为 $L_{u}$ 或 $L_{bc}$ / $L_{ic}$。
    3. 所有数学公式必须使用极其标准的 LaTeX 语法渲染。
     例如：$$ \mathcal{L}(\theta) = \lambda_{u} \mathcal{L}_{u}(\theta) + \lambda_{f} \mathcal{L}_{f}(\theta) $$
    4. 解释概念时，请使用“数据项”与“物理项”，严禁使用“精确度”与“近似度”这种模糊词汇。
            """},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1, 
        max_tokens=1024
    )
    
    final_answer = response.choices[0].message.content
    
    # 【UI 修复小技巧】：将大模型输出的 LaTeX 符号转化为 Streamlit 原生完美支持的 $ 符号格式
    final_answer = final_answer.replace('\\[', '$$').replace('\\]', '$$').replace('\\(', '$').replace('\\)', '$')
    
    return final_answer

if __name__ == "__main__":
    # 我们用一个非常具体的硬核学术问题来测试它的能力
    # 这个问题只有真正“读”了 Raissi 2019 论文的模型才能准确回答
    test_q = "在 Raissi 等人2019年的论文中，解决 forward problems (正问题) 时，PINN 的损失函数 (loss function) 是如何构造的？它由哪两部分组成，各自代表什么物理意义？"
    
    print("================ 启动科研 Agent ================\n")
    answer = ask_rag_agent(test_q)
    print("================ 大模型深度回答 ================\n")
    print(answer)