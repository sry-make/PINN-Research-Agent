import sys
from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化一个 MCP 服务器，起个霸气的名字
mcp = FastMCP("PINN_Research_Tool_Server")

# 2. 连接你之前建好的“长期记忆” (ChromaDB)
CHROMA_DATA_PATH = "./chroma_db"
client_db = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client_db.get_collection(name="pinn_papers", embedding_function=ef)

# 3. 核心魔法：用 @mcp.tool() 装饰器，直接把一个普通的 Python 函数，变成大模型原生可调用的“高维工具”！
@mcp.tool()
def search_pinn_papers(query: str, top_k: int = 3) -> str:
    """
    专业学术检索引擎：当用户询问关于 PINN、物理信息神经网络、偏微分方程等学术细节时，
    必须调用此工具从本地文献库中检索原文片段。
    """
    # 这里的 print 只是为了让我们在后台看到大模型是否真的“自主”调用了工具
    print(f"\n[MCP 引擎] 📡 收到大模型自主调用请求！检索词: '{query}'", file=sys.stderr)
    
    results = collection.query(query_texts=[query], n_results=top_k)
    
    if not results['documents'][0]:
        return "本地文献库中未找到相关的原文片段。"
        
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i].get("source", "未知出处")
        context += f"【来源文献】: {source}\n【原文片段】: {doc}\n\n---\n\n"
    
    print("[MCP 引擎] ✅ 检索完成，已将高维数据返回给大模型大脑。", file=sys.stderr)
    return context

if __name__ == "__main__":
    print("🚀 PINN MCP 服务器初始化完成！", file=sys.stderr)
    print("🔌 正在通过标准 IO 协议暴露出工具箱，等待大模型客户端接入...", file=sys.stderr)
    # 启动 MCP 服务器（默认使用 stdio 标准输入输出协议进行极低延迟通信）
    mcp.run()