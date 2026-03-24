"""LLM 辅助生成 PINN 领域 RAG 检索评测集草稿。

遍历 papers/ 目录下的 PDF 论文，对每篇论文提取代表性 chunk，
调用本地 LLM 生成 1-2 个硬核学术问题，输出到 eval/test_dataset.json。

使用方式：
    python eval/build_test_dataset.py          # 生成草稿
    python eval/build_test_dataset.py --dry-run # 仅打印 chunk，不调用 LLM
"""

import json
import os
import sys
import argparse

from pypdf import PdfReader
from openai import OpenAI

# ---- 配置 ----
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "papers")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "test_dataset.json")

client_llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local")
MODEL_NAME = "pinn_qwen_expert"

CHUNK_SIZE = 500
OVERLAP = 50


def extract_representative_chunks(pdf_path, max_chunks=2):
    """从 PDF 中提取代表性 chunk（取正文前 1/3 部分的最长 chunk）。"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    if len(text.strip()) < 100:
        return []

    # 切分
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
        chunk = text[i : i + CHUNK_SIZE]
        if len(chunk.strip()) > 100:
            chunks.append(chunk)

    if not chunks:
        return []

    # 取正文前 1/3 中最长的 chunk（通常包含核心方法描述）
    front_third = chunks[: max(len(chunks) // 3, 1)]
    front_third.sort(key=lambda c: len(c), reverse=True)
    return front_third[:max_chunks]


def generate_questions(chunk_text, source_filename):
    """调用 LLM 为给定 chunk 生成 1-2 个检索评测问题。"""
    prompt = f"""你是一个 PINN 领域学术专家。下面是一篇论文的某个片段：

---
{chunk_text}
---

请基于上述片段，生成 1-2 个高质量的学术检索问题。要求：
1. 问题必须紧密关联片段内容，只有检索到该片段才能回答
2. 问题应包含具体的技术细节（如方法名、公式名、作者等）
3. 同时为每个问题提取 2-3 个关键词（出现在片段中的核心术语）

请严格按以下 JSON 格式输出（不要输出其他内容）：
[
  {{
    "query": "问题文本",
    "chunk_keywords": ["关键词1", "关键词2"]
  }}
]"""

    try:
        response = client_llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        # 尝试提取 JSON 部分
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception as e:
        print(f"  ⚠️ LLM 生成失败: {e}")
    return []


def build_dataset(dry_run=False):
    """主函数：遍历论文并生成测试集草稿。"""
    pdf_files = sorted(
        [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")],
    )

    print(f"📚 发现 {len(pdf_files)} 篇论文，开始生成评测集...")

    dataset = []
    q_id = 1

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PAPERS_DIR, pdf_file)
        print(f"\n📄 处理: {pdf_file}")

        chunks = extract_representative_chunks(pdf_path, max_chunks=2)
        if not chunks:
            print("  ⏭️  无法提取有效 chunk，跳过。")
            continue

        for chunk in chunks:
            if dry_run:
                print(f"  [DRY RUN] chunk 长度={len(chunk)}: {chunk[:80]}...")
                continue

            questions = generate_questions(chunk, pdf_file)
            for q in questions:
                entry = {
                    "id": f"q_{q_id:03d}",
                    "query": q.get("query", ""),
                    "ground_truth_sources": [pdf_file],
                    "ground_truth_chunk_keywords": q.get("chunk_keywords", []),
                    "topic": "auto_generated",
                }
                dataset.append(entry)
                print(f"  ✅ q_{q_id:03d}: {entry['query'][:60]}...")
                q_id += 1

    if not dry_run and dataset:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 生成完成！共 {len(dataset)} 条，已保存到 {OUTPUT_PATH}")
        print("⚠️  请人工审核并修正测试集内容后再用于评估。")
    elif dry_run:
        print(f"\n[DRY RUN] 完成，未生成文件。")
    else:
        print("\n⚠️ 未生成任何问题。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 PINN RAG 检索评测集草稿")
    parser.add_argument("--dry-run", action="store_true", help="仅提取 chunk，不调用 LLM")
    args = parser.parse_args()
    build_dataset(dry_run=args.dry_run)
