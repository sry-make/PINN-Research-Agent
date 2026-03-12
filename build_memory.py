import os
import chromadb
from pypdf import PdfReader
from chromadb.utils import embedding_functions

# 1. 初始化 ChromaDB 向量数据库 (数据会持久化保存在本地目录)
CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# 2. 核心：加载 Embedding 模型。
# 这里我们使用 sentence-transformers 的经典小模型，第一次运行会自动下载
# 它的作用是把大段的文本变成几百维的浮点数向量，用来计算语义相似度
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. 创建一张名为 "pinn_papers" 的数据表
collection = client.get_or_create_collection(
    name="pinn_papers",
    embedding_function=ef
)

def process_pdf(pdf_path):
    """读取 PDF 并进行简单的文本切分 (Chunking)"""
    print(f"📄 正在解析并切分: {pdf_path} ...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
            
    # 【工程考点】为什么要 Chunking？因为大模型上下文有限，且一次性塞入整本书会导致注意力分散。
    # 这里我们采用滑动窗口切分法：每段 500 字符，前后重叠 50 字符保证上下文不被硬切断。
    chunk_size = 500
    overlap = 50
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 50: # 过滤掉太短的无意义片段
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    papers_dir = "./papers"
    
    # 检查文件夹
    if not os.path.exists(papers_dir) or not os.listdir(papers_dir):
        print(f"⚠️ 请先将 PDF 论文放入 {papers_dir} 文件夹中！")
        exit()
        
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
    
    print(f"🔍 找到 {len(pdf_files)} 篇论文，开始构建知识库...")
    
    for doc_id, file_name in enumerate(pdf_files):
        file_path = os.path.join(papers_dir, file_name)
        chunks = process_pdf(file_path)
        
        # 准备入库需要的 ID 和 元数据 (用于溯源)
        ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name} for _ in range(len(chunks))]
        
        # 批量将切分好的文本、元数据和 ID 存入 ChromaDB
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ {file_name} 向量化入库完成！共生成 {len(chunks)} 个记忆碎片。")
        
    print("\n🎉 知识库构建成功！当前数据库中共有", collection.count(), "条记忆。")