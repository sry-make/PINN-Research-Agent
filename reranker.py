from sentence_transformers import CrossEncoder


class BGEReranker:
    """BGE 交叉编码器重排器，对粗召回结果进行精排。

    使用 BAAI/bge-reranker-base (~110M 参数)，运行在 CPU 上避免与 Ollama 争夺 VRAM。
    """

    def __init__(self, model_name="BAAI/bge-reranker-base", device="cpu"):
        print(f"🔄 正在加载 Reranker 模型: {model_name} (device={device})...")
        self.model = CrossEncoder(model_name, device=device)
        print("✅ Reranker 模型加载完成。")

    def rerank(self, query: str, documents: list, metadatas: list, top_k=3):
        """对 (query, doc) 对打分并返回精排后的 Top-K 结果。

        Args:
            query: 用户查询文本
            documents: 粗召回的文档列表
            metadatas: 与 documents 一一对应的元数据列表
            top_k: 精排后返回的文档数

        Returns:
            (ranked_documents, ranked_metadatas): 精排后的文档和元数据
        """
        if not documents:
            return [], []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        scored_results = sorted(
            zip(scores, documents, metadatas),
            key=lambda x: x[0],
            reverse=True,
        )

        top_results = scored_results[:top_k]
        ranked_documents = [doc for _, doc, _ in top_results]
        ranked_metadatas = [meta for _, _, meta in top_results]
        return ranked_documents, ranked_metadatas


if __name__ == "__main__":
    reranker = BGEReranker()

    test_query = "PINN 的损失函数由哪两部分组成？"
    test_docs = [
        "深度学习在图像分类中的应用非常广泛。",
        "PINN 的损失函数由数据项 L_u 和物理残差项 L_f 两部分组成，分别约束边界条件和 PDE 残差。",
        "强化学习可以用于机器人控制任务。",
        "物理信息神经网络通过在损失函数中加入 PDE 约束来实现物理一致性。",
    ]
    test_metas = [{"source": f"doc_{i}.pdf"} for i in range(len(test_docs))]

    ranked_docs, ranked_metas = reranker.rerank(test_query, test_docs, test_metas, top_k=2)

    print("\n📊 重排结果：")
    for i, (doc, meta) in enumerate(zip(ranked_docs, ranked_metas)):
        print(f"  Top-{i+1} [{meta['source']}]: {doc[:80]}...")
