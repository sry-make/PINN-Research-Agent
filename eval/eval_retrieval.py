"""RAG 检索质量评估模块。

支持指标：Recall@K, Hit Rate@K, MRR, NDCG@K
支持模式：Direct / HyDE / HyDE+Reranker
"""

import json
import math
import os
import sys

# 将项目根目录加入 path，以便导入 rag_agent 和 reranker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
TEST_DATASET_PATH = os.path.join(os.path.dirname(__file__), "test_dataset.json")


def load_test_dataset(path=TEST_DATASET_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_collection():
    client_db = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client_db.get_collection(name="pinn_papers", embedding_function=ef)


def is_relevant(metadata, ground_truth_sources, chunk_text=None, chunk_keywords=None):
    """判断一个检索结果是否与 ground truth 相关。

    主要标准：metadata["source"] 在 ground_truth_sources 中
    次要标准（可选）：chunk 文本包含至少一个 ground_truth_chunk_keywords
    """
    source = metadata.get("source", "")
    source_match = source in ground_truth_sources

    if chunk_keywords and chunk_text:
        keyword_match = any(kw.lower() in chunk_text.lower() for kw in chunk_keywords)
        return source_match or keyword_match

    return source_match


def recall_at_k(relevant_flags, total_relevant):
    """Recall@K = |检索到的相关文档| / |全部相关文档|"""
    if total_relevant == 0:
        return 0.0
    return sum(relevant_flags) / total_relevant


def hit_rate_at_k(relevant_flags):
    """Hit Rate@K = 1 if 至少一个相关文档在 Top-K 中, else 0"""
    return 1.0 if any(relevant_flags) else 0.0


def mrr(relevant_flags):
    """MRR = 1 / (第一个相关结果的排名位置)"""
    for i, flag in enumerate(relevant_flags):
        if flag:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant_flags):
    """NDCG@K（二元相关性）。"""
    # DCG
    dcg = 0.0
    for i, flag in enumerate(relevant_flags):
        if flag:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 rank 从 1 开始

    # Ideal DCG：假设所有相关文档排在最前
    num_relevant = sum(relevant_flags)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def retrieve_for_query(query, collection, use_hyde=True, reranker=None, top_k=10,
                       coarse_top_k=20, rewrite_fn=None):
    """对单条 query 执行检索，返回 (documents, metadatas)。"""
    search_query = query

    if use_hyde and rewrite_fn is not None:
        search_query = rewrite_fn(query)

    if reranker is not None:
        results = collection.query(query_texts=[search_query], n_results=coarse_top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        docs, metas = reranker.rerank(search_query, docs, metas, top_k=top_k)
    else:
        results = collection.query(query_texts=[search_query], n_results=top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]

    return docs, metas


def evaluate_retrieval(test_data, collection, reranker=None, use_hyde=True,
                       top_k_values=None, rewrite_fn=None):
    """对整个测试集执行评估，返回各 K 值下的指标。

    Args:
        test_data: 测试集列表
        collection: ChromaDB collection
        reranker: BGEReranker 实例或 None
        use_hyde: 是否使用 HyDE 重写
        top_k_values: 要评估的 K 值列表
        rewrite_fn: HyDE 查询重写函数

    Returns:
        {
            "per_query": [...],  # 逐条明细
            "aggregated": {k: {metric: value}}  # 聚合指标
        }
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    max_k = max(top_k_values)
    per_query_results = []

    for item in test_data:
        query = item["query"]
        gt_sources = item["ground_truth_sources"]
        gt_keywords = item.get("ground_truth_chunk_keywords", [])

        docs, metas = retrieve_for_query(
            query, collection,
            use_hyde=use_hyde,
            reranker=reranker,
            top_k=max_k,
            rewrite_fn=rewrite_fn,
        )

        # 计算每个位置是否相关
        relevant_flags = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            flag = is_relevant(meta, gt_sources, doc, gt_keywords)
            relevant_flags.append(flag)

        # 记录来源列表
        retrieved_sources = [m.get("source", "") for m in metas]

        per_query_results.append({
            "id": item["id"],
            "query": query,
            "ground_truth_sources": gt_sources,
            "retrieved_sources": retrieved_sources[:max_k],
            "relevant_flags": relevant_flags,
        })

    # 聚合各 K 值下的指标
    aggregated = {}
    for k in top_k_values:
        recalls, hits, mrrs, ndcgs = [], [], [], []
        for pq in per_query_results:
            flags_k = pq["relevant_flags"][:k]
            total_rel = len(pq["ground_truth_sources"])

            recalls.append(recall_at_k(flags_k, total_rel))
            hits.append(hit_rate_at_k(flags_k))
            mrrs.append(mrr(flags_k))
            ndcgs.append(ndcg_at_k(flags_k))

        n = len(per_query_results)
        aggregated[k] = {
            "Recall@K": sum(recalls) / n,
            "HitRate@K": sum(hits) / n,
            "MRR": sum(mrrs) / n,
            "NDCG@K": sum(ndcgs) / n,
        }

    return {"per_query": per_query_results, "aggregated": aggregated}


def print_results(results, config_name=""):
    """格式化打印评估结果。"""
    agg = results["aggregated"]
    print(f"\n{'=' * 60}")
    print(f"  配置: {config_name}")
    print(f"  评估样本数: {len(results['per_query'])}")
    print(f"{'=' * 60}")
    print(f"  {'K':>4}  {'Recall@K':>10}  {'HitRate@K':>10}  {'MRR':>10}  {'NDCG@K':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for k, metrics in sorted(agg.items()):
        print(
            f"  {k:>4}  {metrics['Recall@K']:>10.4f}  {metrics['HitRate@K']:>10.4f}"
            f"  {metrics['MRR']:>10.4f}  {metrics['NDCG@K']:>10.4f}"
        )
