"""消融实验入口：运行 Direct / HyDE / HyDE+Reranker 三组对比并输出结果。

使用方式：
    python eval/run_ablation.py
    python eval/run_ablation.py --top-k 1 3 5    # 自定义 K 值
    python eval/run_ablation.py --no-hyde-reranker # 跳过 reranker 组（无需下载模型）
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.eval_retrieval import (
    evaluate_retrieval,
    get_collection,
    load_test_dataset,
    print_results,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def get_rewrite_fn():
    """延迟导入 rewrite_query 以避免在不需要时连接 Ollama。"""
    from rag_agent import rewrite_query
    return rewrite_query


def get_reranker():
    from reranker import BGEReranker
    return BGEReranker()


def save_markdown_table(all_results, top_k_values, output_path):
    """将三组结果保存为 Markdown 对比表。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = ["# RAG 检索消融实验结果\n"]
    lines.append(f"评估样本数: {len(list(all_results.values())[0]['per_query'])}\n")

    for k in top_k_values:
        lines.append(f"\n## K = {k}\n")
        lines.append("| 配置 | Recall@{k} | HitRate@{k} | MRR | NDCG@{k} |".format(k=k))
        lines.append("|------|-----------|------------|-----|---------|")
        for config_name, results in all_results.items():
            m = results["aggregated"][k]
            lines.append(
                f"| {config_name} | {m['Recall@K']:.4f} | {m['HitRate@K']:.4f} "
                f"| {m['MRR']:.4f} | {m['NDCG@K']:.4f} |"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n📊 Markdown 表格已保存: {output_path}")


def save_per_query_csv(all_results, output_path):
    """将逐条明细保存为 CSV。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for config_name, results in all_results.items():
        for pq in results["per_query"]:
            rows.append({
                "config": config_name,
                "id": pq["id"],
                "query": pq["query"],
                "ground_truth_sources": "; ".join(pq["ground_truth_sources"]),
                "retrieved_sources": "; ".join(pq["retrieved_sources"][:5]),
                "relevant_flags_top5": str(pq["relevant_flags"][:5]),
            })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"📋 逐条明细已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAG 检索消融实验")
    parser.add_argument(
        "--top-k", nargs="+", type=int, default=[1, 3, 5, 10],
        help="要评估的 K 值列表 (默认: 1 3 5 10)"
    )
    parser.add_argument(
        "--no-hyde-reranker", action="store_true",
        help="跳过 HyDE+Reranker 组（无需下载 reranker 模型）"
    )
    args = parser.parse_args()

    top_k_values = args.top_k
    print("🚀 RAG 检索消融实验")
    print(f"   K 值: {top_k_values}")
    print(f"   配置组: Direct, HyDE" + ("" if args.no_hyde_reranker else ", HyDE+Reranker"))

    # 加载数据
    test_data = load_test_dataset()
    collection = get_collection()
    print(f"📚 测试集: {len(test_data)} 条 | ChromaDB: {collection.count()} 条记忆")

    all_results = {}

    # ---- 配置 1: Direct ----
    print("\n" + "=" * 60)
    print("🔬 [1/3] 运行 Direct 配置（原始 query → ChromaDB）...")
    t0 = time.time()
    results_direct = evaluate_retrieval(
        test_data, collection,
        use_hyde=False, reranker=None,
        top_k_values=top_k_values,
    )
    print(f"   耗时: {time.time() - t0:.1f}s")
    print_results(results_direct, "Direct")
    all_results["Direct"] = results_direct

    # ---- 配置 2: HyDE ----
    print("\n" + "=" * 60)
    print("🔬 [2/3] 运行 HyDE 配置（HyDE 重写 → ChromaDB）...")
    rewrite_fn = get_rewrite_fn()
    t0 = time.time()
    results_hyde = evaluate_retrieval(
        test_data, collection,
        use_hyde=True, reranker=None,
        top_k_values=top_k_values,
        rewrite_fn=rewrite_fn,
    )
    print(f"   耗时: {time.time() - t0:.1f}s")
    print_results(results_hyde, "HyDE")
    all_results["HyDE"] = results_hyde

    # ---- 配置 3: HyDE + Reranker ----
    if not args.no_hyde_reranker:
        print("\n" + "=" * 60)
        print("🔬 [3/3] 运行 HyDE+Reranker 配置（HyDE → Top-20 → BGE 精排）...")
        reranker = get_reranker()
        t0 = time.time()
        results_reranker = evaluate_retrieval(
            test_data, collection,
            use_hyde=True, reranker=reranker,
            top_k_values=top_k_values,
            rewrite_fn=rewrite_fn,
        )
        print(f"   耗时: {time.time() - t0:.1f}s")
        print_results(results_reranker, "HyDE+Reranker")
        all_results["HyDE+Reranker"] = results_reranker

    # ---- 保存结果 ----
    md_path = os.path.join(RESULTS_DIR, "ablation_results.md")
    csv_path = os.path.join(RESULTS_DIR, "per_query_detail.csv")
    save_markdown_table(all_results, top_k_values, md_path)
    save_per_query_csv(all_results, csv_path)

    print("\n✅ 消融实验完成！")


if __name__ == "__main__":
    main()
