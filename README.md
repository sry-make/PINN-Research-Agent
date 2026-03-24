<div align="center">

# 🧠 PINN Research Agent

### Physics-Informed Neural Network 智能科研助手

**基于 RAG + LoRA 微调 + MCP Agent + Cross-Encoder Reranker 架构的 PINN 领域专业问答系统**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--7B-6F42C1?style=flat-square)](https://github.com/QwenLM/Qwen2.5)
[![Ollama](https://img.shields.io/badge/Inference-Ollama-000000?style=flat-square&logo=ollama)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MCP](https://img.shields.io/badge/Protocol-MCP-00B4D8?style=flat-square)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

*将 30+ 篇 PINN 顶刊论文融入本地大模型，打造你的私人 AI for Science 科研助手*

---

[快速开始](#-快速开始) · [系统架构](#-系统架构) · [工作流程](#-完整工作流) · [检索评估](#-检索质量评估与消融实验) · [技术栈](#-技术栈) · [更新计划](#-更新计划)

</div>

<br/>

## 📖 项目简介

**PINN Research Agent** 是一个全链路本地化部署的 AI 科研助手系统，专注于 **Physics-Informed Neural Networks (PINNs)** 和计算力学领域。

本项目将以下技术融合为一体化解决方案：

- **🔍 RAG (检索增强生成)** — 基于 30+ 篇 PINN 领域顶刊论文构建向量知识库，让模型回答有据可依
- **🎯 LoRA SFT 微调** — 从论文中自动蒸馏 QA 数据，对 Qwen2.5-7B 进行领域适配微调
- **🤖 MCP Agent** — 基于 Model Context Protocol 实现自主决策的工具调用 Agent
- **🔀 Cross-Encoder Reranker** — Bi-Encoder 粗召回 + BGE Cross-Encoder 精排的两级检索架构
- **📊 量化评估体系** — 35 条领域评测集 + Recall/MRR/HitRate/NDCG 四项指标 + 三组消融对比
- **🌐 Web UI** — 基于 Streamlit 的聊天式交互界面，原生支持 LaTeX 公式渲染

> **核心理念**：全链路本地化，从训练到推理不依赖任何云端 API，保护学术数据隐私。

<br/>

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户交互层                               │
│  ┌───────────────────┐         ┌───────────────────────────┐    │
│  │  Streamlit Web UI │         │  MCP Agent (终端模式)      │    │
│  │   (web_ui.py)     │         │  (mcp_client.py)          │    │
│  └────────┬──────────┘         └─────────┬─────────────────┘    │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
┌───────────┼──────────────────────────────┼──────────────────────┐
│           ▼          智能推理层           ▼                      │
│  ┌─────────────────────────┐   ┌──────────────────────┐         │
│  │   RAG Agent             │   │  MCP Server           │         │
│  │   (rag_agent.py)        │   │  (mcp_server.py)      │         │
│  │                         │   │                      │         │
│  │  ① HyDE 查询重写        │   │  search_pinn_papers  │         │
│  │  ② Bi-Encoder 粗召回    │   │  (stdio 协议)        │         │
│  │  ③ Cross-Encoder 精排   │   └──────────┬───────────┘         │
│  │  ④ LLM 终极推理         │              │                     │
│  └────────┬────────────────┘              │                     │
└───────────┼───────────────────────────────┼─────────────────────┘
            │                               │
┌───────────┼───────────────────────────────┼─────────────────────┐
│           ▼           数据 & 模型层        ▼                     │
│  ┌────────────────────┐    ┌──────────────────────────────┐     │
│  │  ChromaDB           │    │  Ollama (Docker Container)   │     │
│  │  向量数据库          │    │  ┌──────────────────────┐    │     │
│  │                    │    │  │  pinn_qwen_expert     │    │     │
│  │  all-MiniLM-L6-v2  │    │  │  Qwen2.5-7B + LoRA   │    │     │
│  │  (Embedding Model) │    │  │  Q4_K_M 量化          │    │     │
│  └────────────────────┘    │  └──────────────────────┘    │     │
│                            └──────────────────────────────┘     │
│           ▲                              ▲                      │
│  ┌────────┴────────────┐    ┌────────────┴───────────────┐     │
│  │  build_memory.py    │    │  模型训练流水线              │     │
│  │  PDF → Chunk → Vec  │    │  generate_sft_data.py      │     │
│  └────────┬────────────┘    │  clean_data.py             │     │
│           │                 │  train_lora.py             │     │
│  ┌────────┴────────────┐    │  export_model.py           │     │
│  │  papers/ (30+ PDFs) │    └────────────────────────────┘     │
│  │  PINN 领域顶刊论文   │                                       │
│  └─────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

## 🌊 完整工作流

本项目包含 **离线构建** 和 **在线推理** 两条核心流水线：

### Pipeline A：离线构建（知识库 + 模型训练）

```
📄 收集论文                    🧩 构建知识库                  🏭 模型训练
───────────                  ──────────                    ─────────
papers/ 目录                  build_memory.py               generate_sft_data.py
30+ 篇 PINN 论文     ──►     PDF 解析                ──►   Self-Instruct 数据蒸馏
(Raissi 2019,                 滑动窗口 Chunking              论文 → QA 数据对
 DeepXDE,                     500 字符/段, 50 重叠                  │
 HP-VPINN, ...)               all-MiniLM-L6-v2 向量化               ▼
                              存入 ChromaDB               clean_data.py
                                                          数据清洗 & 质量过滤
                                                                  │
                                                                  ▼
                                                          train_lora.py
                                                          Unsloth + LoRA SFT
                                                          r=16, alpha=32
                                                          4-bit QLoRA 训练
                                                                  │
                                                                  ▼
                                                          export_model.py
                                                          LoRA 融合 + Q4_K_M 量化
                                                          导出 GGUF 格式
                                                                  │
                                                                  ▼
                                                          Ollama Modelfile
                                                          注册为 pinn_qwen_expert
```

### Pipeline B：在线推理

#### 模式 1 — RAG + Web UI（主要模式）

```
用户提问 ──► Streamlit UI ──► RAG Agent
                                 │
                                 ├──► ① HyDE 查询重写 (LLM 扩写短查询为先验伪文档)
                                 │
                                 ├──► ② Bi-Encoder 粗召回 (ChromaDB Top-20 候选)
                                 │
                                 ├──► ③ Cross-Encoder 精排 (BGE-Reranker → Top-3)
                                 │
                                 └──► ④ LLM 终极推理 (上下文 + 问题 → 回答)
                                                   │
                                          ◄────────┘
                                     LaTeX 后处理 & 公式渲染
```

#### 模式 2 — MCP Agent（高级模式）

```
用户提问 ──► MCP Client ──► 启动 MCP Server (子进程, stdio)
                │                    │
                ├── 协议握手，获取工具清单
                │                    │
                ├── 第一轮对话：LLM 自主决策是否调用工具
                │       │
                │       ├── [需要工具] ──► MCP 调用 search_pinn_papers ──► 获取论文片段
                │       └── [无需工具] ──► 直接回答
                │                    │
                └── 第二轮对话：基于工具返回的数据生成最终回答
```

<br/>

## 🚀 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 + WSL2 |
| Docker | Docker Desktop (WSL2 backend) |
| GPU | NVIDIA GPU (≥ 16GB VRAM，训练阶段) |
| Python | 3.10+ (WSL 内 Conda 环境) |
| Ollama | Docker 容器运行 |

### 1. 启动 Ollama 服务

```bash
# 在 Docker 中启动 Ollama (GPU 支持)
docker run -d --gpus all -p 11434:11434 --name ollama ollama/ollama

# 导入微调后的模型 (在 pinn_qwen_gguf_gguf/ 目录下)
cd pinn_qwen_gguf_gguf
ollama create pinn_qwen_expert -f Modelfile
```

### 2. 安装 Python 依赖

```bash
# 激活你的 Conda 环境 (WSL 内)
conda activate your_env

# 安装推理所需依赖
pip install openai chromadb pypdf sentence-transformers streamlit mcp

# sentence-transformers 同时提供 Bi-Encoder (Embedding) 和 Cross-Encoder (Reranker)
```

### 3. 构建向量知识库

```bash
# 将 papers/ 目录下的论文向量化并存入 ChromaDB
python build_memory.py
```

### 4. 启动 Web UI

```bash
# 启动 Streamlit 聊天界面
streamlit run web_ui.py
```

访问 `http://localhost:8501` 即可开始使用。

### 5. (可选) MCP Agent 模式

```bash
# 以 Agent 模式运行，支持自主工具调用
python mcp_client.py
```

<br/>

## 📁 项目结构

```
PINN_AGENT_PROJECT/
│
├── 🔌 核心模块
│   ├── llm_core.py              # LLM 基础通信层 (Ollama API)
│   ├── rag_agent.py             # RAG Agent 核心 (HyDE + 检索 + Reranker + 推理)
│   ├── reranker.py              # BGE Cross-Encoder 重排器
│   ├── web_ui.py                # Streamlit Web 前端
│   ├── mcp_server.py            # MCP 工具服务器
│   └── mcp_client.py            # MCP Agent 客户端
│
├── 🏭 训练流水线
│   ├── generate_sft_data.py     # Self-Instruct 数据蒸馏
│   ├── clean_data.py            # 数据清洗脚本
│   ├── train_lora.py            # LoRA 微调训练
│   └── export_model.py          # 模型导出 (LoRA → GGUF)
│
├── 📊 数据文件
│   ├── pinn_sft_dataset.jsonl          # SFT 数据集 V1
│   ├── pinn_sft_dataset_v2.jsonl       # SFT 数据集 V2 (全量提取)
│   └── pinn_sft_dataset_v2_clean.jsonl # 清洗后纯净数据集
│
├── 📈 检索评估
│   ├── eval/
│   │   ├── test_dataset.json           # 35 条 PINN 领域评测集
│   │   ├── build_test_dataset.py       # LLM 辅助评测集生成
│   │   ├── eval_retrieval.py           # 核心评估 (Recall/MRR/HitRate/NDCG)
│   │   ├── run_ablation.py             # 消融实验入口
│   │   └── results/                    # 实验结果输出
│   │       ├── ablation_results.md     # 消融对比表格
│   │       └── per_query_detail.csv    # 逐条归因明细
│
├── 📚 知识库
│   ├── papers/                  # PINN 领域论文集 (30+ 篇 PDF)
│   └── chroma_db/               # ChromaDB 向量数据库 (持久化)
│
├── 🧪 模型产物
│   ├── lora_pinn_model/         # LoRA 适配器权重
│   ├── outputs/                 # 训练 Checkpoints
│   ├── pinn_qwen_gguf/          # 融合后的完整模型
│   └── pinn_qwen_gguf_gguf/     # GGUF 量化模型 + Modelfile
│
└── 📦 缓存
    └── unsloth_compiled_cache/  # Unsloth 编译缓存
```

<br/>

## ⚙️ 技术栈

<table>
<tr>
<td align="center" width="25%">

**大模型推理**

Ollama (Docker)<br/>
Qwen2.5-7B-Instruct<br/>
OpenAI 兼容 API

</td>
<td align="center" width="25%">

**模型微调**

Unsloth + TRL<br/>
LoRA (r=16, α=32)<br/>
4-bit QLoRA + GGUF Q4_K_M

</td>
<td align="center" width="25%">

**知识检索**

ChromaDB + BGE-Reranker<br/>
all-MiniLM-L6-v2 (Bi-Encoder)<br/>
bge-reranker-base (Cross-Encoder)<br/>
HyDE 查询重写

</td>
<td align="center" width="25%">

**Agent 框架**

MCP Protocol<br/>
FastMCP (stdio)<br/>
ReAct 两轮决策

</td>
</tr>
</table>

### 关键技术参数

| 参数 | 值 | 说明 |
|------|----|------|
| 基座模型 | `Qwen2.5-7B-Instruct` | 阿里通义千问 7B 参数模型 |
| 量化精度 | `Q4_K_M` | 4-bit 混合量化，平衡速度与精度 |
| LoRA Rank | `r=16, α=32` | 覆盖 Attention + FFN 全部 7 层 |
| 训练步数 | 60 steps (~1.67 epochs) | Loss: 1.953 → 0.648 |
| 训练数据 | 289 条 QA 对 | 从 30+ 篇论文 Self-Instruct 蒸馏 |
| Bi-Encoder | `all-MiniLM-L6-v2` | 384 维，粗召回阶段 |
| Cross-Encoder | `BAAI/bge-reranker-base` | ~110M 参数，精排阶段，CPU 推理 |
| Chunk 策略 | 500 字符 / 50 重叠 | 滑动窗口切分 |
| 粗召回 Top-K | 20 | Bi-Encoder 候选池 |
| 精排 Top-K | 3 | Cross-Encoder 最终返回 |
| 评测集规模 | 35 条 QA | 覆盖 30+ 篇论文，人工校验 |

<br/>

## 📚 论文知识库

涵盖 PINN 领域核心文献 **30+ 篇**，包括：

| 方向 | 代表论文 |
|------|---------|
| **奠基工作** | Raissi et al. 2019 — PINN 正/逆问题框架 |
| **开源框架** | Lu et al. 2021 — DeepXDE |
| **领域综述** | Cuomo et al. 2022 — *Where we are and What's Next* |
| **变分方法** | Kharazmi et al. 2021 — HP-VPINN + 域分解 |
| **多保真度** | Meng 2020, Howard 2023, Penwarden 2022 |
| **固体力学** | Haghighat 2021, Rao 2021, Li 2021 |
| **元学习** | Psaros 2022, Chen GPT-PINN 2024 |
| **架构搜索** | Wang 2024 — NAS-PINN |
| **新架构** | Wang 2024 — KAN-Informed Neural Network |

<br/>

## 🛠️ 运行环境详解

本项目采用 **WSL + Docker** 混合架构：

```
┌──────────────────────────────────────────────┐
│              Windows 10/11 Host              │
│                                              │
│   ┌──────────────────────────────────────┐   │
│   │              WSL2 (Linux)            │   │
│   │                                      │   │
│   │   ┌──────────────┐  ┌────────────┐   │   │
│   │   │  Conda Env   │  │  Docker    │   │   │
│   │   │  Python 3.10 │  │  ┌──────┐  │   │   │
│   │   │              │  │  │Ollama│  │   │   │
│   │   │  rag_agent   │◄─┼─►│:11434│  │   │   │
│   │   │  web_ui      │  │  │      │  │   │   │
│   │   │  mcp_client  │  │  │Model │  │   │   │
│   │   │  mcp_server  │  │  │Store │  │   │   │
│   │   │  train_lora  │  │  └──────┘  │   │   │
│   │   └──────────────┘  └────────────┘   │   │
│   │                                      │   │
│   │         localhost:11434 ◄──────►      │   │
│   └──────────────────────────────────────┘   │
│                                              │
│   浏览器访问 http://localhost:8501            │
└──────────────────────────────────────────────┘
```

- **Python 代码**运行在 WSL 内的 Conda 环境中
- **Ollama 推理引擎**运行在 Docker 容器中（GPU 直通）
- **微调后的模型**存储在 Ollama 容器的模型仓库中
- 两者通过 `localhost:11434` 通信

<br/>

## 📊 检索质量评估与消融实验

本项目构建了完整的 RAG 检索质量量化评估体系，通过三组消融对比实验验证各检索模块的独立贡献。

### 两级检索架构：Bi-Encoder + Cross-Encoder

传统 RAG 系统仅使用 Bi-Encoder（如 Sentence-BERT）进行单阶段检索，存在以下问题：

- **Bi-Encoder** 将 query 和 document 独立编码为向量，通过余弦相似度匹配，速度快但精度有上限
- **Cross-Encoder** 将 (query, document) 拼接后联合编码，通过 Transformer 注意力机制直接建模交互关系，精度更高但计算量大

本项目采用 **Coarse-to-Fine 两级架构**：

```
用户 Query
    │
    ▼ HyDE 重写（可选）
    │
    ▼ Stage 1: Bi-Encoder 粗召回
    │ all-MiniLM-L6-v2 (384维)
    │ ChromaDB ANN 检索 → Top-20 候选
    │
    ▼ Stage 2: Cross-Encoder 精排
    │ BAAI/bge-reranker-base (~110M)
    │ 逐对 (query, doc) 打分 → Top-3 精选
    │
    ▼ LLM 推理生成答案
```

**为什么不直接用 Cross-Encoder？** Cross-Encoder 需要对每个 (query, doc) 对做完整的 Transformer forward pass。如果知识库有 10000+ 个 chunk，逐个打分的延迟不可接受。因此先用 Bi-Encoder 从全量候选中快速缩小范围到 Top-20，再用 Cross-Encoder 对这 20 个候选精排，兼顾效率与精度。

### 评测集构建

构建了覆盖 **30+ 篇 PINN 领域论文**的 **35 条**标准评测集：

```json
{
  "id": "q_001",
  "query": "Raissi 2019 论文中 PINN 的损失函数由哪两部分组成？",
  "ground_truth_sources": ["Raissi 等 - 2019 - Physics-informed neural networks...pdf"],
  "ground_truth_chunk_keywords": ["MSE_u", "MSE_f", "loss function"],
  "topic": "loss_function"
}
```

**覆盖的主题分布**：loss function、forward/inverse problem、framework (DeepXDE)、variational method (hp-VPINNs)、conservation law、multi-fidelity、architecture search (NAS-PINN, KAN)、solid mechanics、boundary condition 等。

### 评估指标

| 指标 | 定义 | 含义 |
|------|------|------|
| **Recall@K** | \|检索相关 ∩ GT\| / \|GT\| | K 个结果中覆盖了多少真实相关文档 |
| **HitRate@K** | 至少 1 个相关文档在 Top-K 中的查询比例 | 检索"命中"的成功率 |
| **MRR** | 1 / (第一个相关结果的排名位置) 的均值 | 第一个正确结果排多靠前 |
| **NDCG@K** | 归一化折损累积增益（二元相关性） | 综合考虑相关性和排序质量 |

**相关性判定**：主要标准为检索 chunk 的 `metadata["source"]` 匹配 ground truth 论文源；辅助标准为 chunk 文本命中 ground truth 关键词。

### 消融实验结果

运行三组配置对比：

| 配置 | HyDE 重写 | Reranker 精排 | 说明 |
|------|:---------:|:------------:|------|
| **Direct** | ✗ | ✗ | 原始 query → Bi-Encoder Top-K |
| **HyDE** | ✓ | ✗ | HyDE 伪文档 → Bi-Encoder Top-K |
| **HyDE + Reranker** | ✓ | ✓ | HyDE → Bi-Encoder Top-20 → Cross-Encoder Top-K |

#### K = 1

| 配置 | Recall@1 | HitRate@1 | MRR | NDCG@1 |
|------|:--------:|:---------:|:---:|:------:|
| Direct | 0.4286 | 0.4286 | 0.4286 | 0.4286 |
| HyDE | 0.1714 | 0.1714 | 0.1714 | 0.1714 |
| HyDE+Reranker | 0.1143 | 0.1143 | 0.1143 | 0.1143 |

#### K = 3

| 配置 | Recall@3 | HitRate@3 | MRR | NDCG@3 |
|------|:--------:|:---------:|:---:|:------:|
| Direct | 1.3143 | 0.4857 | 0.4571 | 0.4659 |
| HyDE | 0.4286 | 0.2000 | 0.1857 | 0.1872 |
| HyDE+Reranker | 0.4571 | 0.2286 | 0.1667 | 0.1880 |

#### K = 5

| 配置 | Recall@5 | HitRate@5 | MRR | NDCG@5 |
|------|:--------:|:---------:|:---:|:------:|
| Direct | 2.1143 | 0.4857 | 0.4571 | 0.4622 |
| HyDE | 0.7429 | 0.2857 | 0.2043 | 0.2212 |
| HyDE+Reranker | 0.7714 | 0.2571 | 0.1724 | 0.2021 |

#### K = 10

| 配置 | Recall@10 | HitRate@10 | MRR | NDCG@10 |
|------|:---------:|:----------:|:---:|:-------:|
| Direct | 3.9714 | 0.5143 | 0.4612 | 0.4705 |
| HyDE | 1.3429 | 0.3429 | 0.2100 | 0.2348 |
| HyDE+Reranker | 1.4000 | 0.2571 | 0.1724 | 0.2100 |

### 结果分析与工程洞察

#### 1. Bi-Encoder 中英文语义空间偏置问题

实验中 Direct 模式的 HitRate@3 (48.6%) 显著优于 HyDE (20.0%)，这一反直觉的结果经逐条 query 归因分析（`per_query_detail.csv`）后定位到根本原因：

- **Embedding 模型语言偏置**：`all-MiniLM-L6-v2` 是英文预训练模型，其向量空间中中文文本分布过于密集
- **HyDE 中文伪文档的"语义黑洞"效应**：HyDE 由中文 LLM 生成中文伪文档，在英文 Embedding 空间中，这些中文伪文档的向量与知识库中唯一的中文综述论文（`AI+for+PDEs在固体力学领域的研究进展.pdf`）高度聚集，导致该论文吸走了大量 query 的检索结果
- **Direct 模式的优势来源**：测试集中的 query 本身包含英文关键术语（如 "DeepXDE"、"hp-VPINNs"、"NAS-PINN"），在英文 Embedding 空间中能更精准地匹配目标论文

#### 2. Cross-Encoder Reranker 的精排修正能力

对比 HyDE 与 HyDE+Reranker：

- **HitRate@3**：20.0% → 22.9%（**相对提升 ~14.3%**）
- **Recall@3**：0.4286 → 0.4571（**相对提升 ~6.7%**）
- **NDCG@3**：0.1872 → 0.1880

Cross-Encoder 通过 (query, doc) 联合编码，能部分修正 Bi-Encoder 的语义偏差。即使粗召回阶段被中文综述"污染"，精排阶段仍能将少量正确候选提升到更靠前的位置。

#### 3. 优化方向

基于消融实验的数据支撑，明确了后续优化路径：

| 优化方向 | 预期收益 | 技术方案 |
|---------|---------|---------|
| 替换中英双语 Embedding | 从根本上解决语义空间偏置 | `BAAI/bge-base-zh-v1.5` 或 `m3e-base` |
| 增大粗召回池 | 为 Reranker 提供更多正确候选 | Top-20 → Top-50 |
| 混合检索策略 | 结合稀疏检索的精确匹配能力 | BM25 + Dense Retrieval 融合 |

### 运行评估

```bash
# 安装依赖
pip install sentence-transformers

# 运行完整消融实验（需要 Ollama 运行中 + 首次下载 bge-reranker-base ~0.5GB）
python eval/run_ablation.py

# 跳过 Reranker 组（无需下载 reranker 模型）
python eval/run_ablation.py --no-hyde-reranker

# 自定义 K 值
python eval/run_ablation.py --top-k 1 3 5

# 用 LLM 生成更多测试数据（可选）
python eval/build_test_dataset.py
```

<br/>

## 🎯 设计亮点

| 特性 | 说明 |
|------|------|
| **Coarse-to-Fine 两级检索** | Bi-Encoder 粗召回 Top-20 + Cross-Encoder 精排 Top-3，兼顾效率与精度 |
| **HyDE 查询重写** | 用 LLM 将短查询扩写为假设性长文本，弥补短查询与长文献间的语义鸿沟 |
| **量化评估体系** | 35 条领域评测集 + 4 项标准 IR 指标 + 3 组消融对比 + 逐条归因分析 |
| **双模式推理** | 同时支持 RAG Web UI 模式和 MCP Agent 自主决策模式 |
| **Self-Instruct 蒸馏** | 零人工标注，用通用 LLM 从论文中自动生成领域训练数据 |
| **全链路本地化** | 训练、推理、知识库全部本地部署，不依赖云端 API |
| **LaTeX 原生支持** | 模型输出标准 LaTeX 公式，Streamlit 前端完美渲染 |
| **Ollama API 兼容** | 统一 OpenAI 格式接口，可无缝切换本地/云端模型 |

<br/>

## 🗺️ 更新计划

- [ ] **多轮对话记忆** — 引入对话历史管理，支持上下文连续追问
- [ ] **更多论文纳入** — 持续扩充 PINN 知识库，覆盖更多子方向
- [ ] **Embedding 模型升级** — 替换为 `bge-base-zh-v1.5` 等中英双语 Embedding（消融实验已验证必要性）
- [ ] **Agent 工具扩展** — 增加代码生成、公式推导、绘图等 MCP 工具
- [ ] **多模态支持** — 解析论文中的图表，支持图文混合问答
- [x] **评测体系** — ~~构建 PINN 领域 Benchmark，量化评估模型能力~~ ✅ 已完成（35 条评测集 + 4 指标 + 消融实验）
- [ ] **训练数据扩充** — 增加更多 SFT 数据，支持 DPO 偏好对齐
- [ ] **Docker Compose** — 一键部署整套服务（Ollama + ChromaDB + Web UI）
- [ ] **流式输出** — Web UI 支持 Streaming 逐字输出

<br/>

## 📄 License

本项目基于 [MIT License](LICENSE) 开源。

论文 PDF 文件版权归原作者所有，仅用于学术研究用途。

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

*Built with ❤️ for AI for Science*

</div>
