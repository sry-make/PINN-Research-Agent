import os
import json
import re
from pypdf import PdfReader
from openai import OpenAI
from tqdm import tqdm  # 引入进度条，大厂处理数据标配

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama-local')

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        # 过滤掉太短的片段
        if len(chunk) > 300: 
            chunks.append(chunk)
    return chunks

def generate_qa_pair(chunk):
    """
    【升级版 Prompt】: 针对 PDF 公式乱码问题进行修复，并强制生成硬核内容
    """
    prompt = f"""你是一个顶尖的计算力学与 AI4S 专家。
请阅读下面的【学术论文片段】。由于该片段是从 PDF 机器提取的，里面的数学公式可能存在字符乱码或排版错乱。

【你的任务】：
从中提取最硬核、最底层的 1 个技术点（如：具体的偏微分方程形式、特定的 Loss 函数构造、边界条件惩罚项等），构建一对极其专业的“指令(instruction)”和“输出(output)”。

【严格要求】：
1. 如果该片段完全是水话（如致谢、泛泛的引言），请直接返回字符串 "None"。
2. 回答中必须包含数学推导或公式！请你利用你的物理和数学先验知识，自动修复片段中的乱码公式，并在 output 中使用标准的 LaTeX 语法（如 $$MSE = MSE_u + MSE_f$$）进行完美重构排版。
3. 必须输出合法的 JSON 格式。

【JSON 格式模板】：
{{
    "instruction": "探讨某个具体的 PINN 公式或机制",
    "input": "", 
    "output": "包含完美 LaTeX 公式排版的详细硬核解答"
}}

【学术论文片段】：
{chunk}
"""
    try:
        response = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, 
        )
        content = response.choices[0].message.content.strip()
        
        if "None" in content or content == "None":
            return None
            
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        return json.loads(content)
    except Exception:
        return None

if __name__ == "__main__":
    papers_dir = "./papers"
    output_file = "pinn_sft_dataset_v2.jsonl" 
    
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
    print(f"🚀 开始 V2 版本数据蒸馏，共发现 {len(pdf_files)} 篇论文，这将是一项计算密集型任务...\n")
    
    total_pairs = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in pdf_files:
            file_path = os.path.join(papers_dir, file_name)
            print(f"📄 正在解析: {file_name}")
            text = extract_text_from_pdf(file_path)
            
            # 【核心修改】去掉了 [:5] 限制，现在全量处理论文！
            chunks = chunk_text(text)
            
            # 使用 tqdm 显示进度条
            for chunk in tqdm(chunks, desc="🧠 提取 QA 数据中"):
                qa_pair = generate_qa_pair(chunk)
                
                if qa_pair and "instruction" in qa_pair and "output" in qa_pair:
                    json_line = json.dumps(qa_pair, ensure_ascii=False)
                    f.write(json_line + '\n')
                    total_pairs += 1

    print(f"\n🎉 V2 数据集构建完成！共提炼出 {total_pairs} 条硬核学术数据。")
    print("这才是大厂标准的 Self-Instruct 语料库！")