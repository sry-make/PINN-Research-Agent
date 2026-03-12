<div align="center">

# 🧠 PINN Research Agent

### Physics-Informed Neural Network 智能科研助手

**基于 RAG + LoRA 微调 + MCP Agent 架构的 PINN 领域专业问答系统**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--7B-6F42C1?style=flat-square)](https://github.com/QwenLM/Qwen2.5)
[![Ollama](https://img.shields.io/badge/Inference-Ollama-000000?style=flat-square&logo=ollama)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MCP](https://img.shields.io/badge/Protocol-MCP-00B4D8?style=flat-square)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

*将 30+ 篇 PINN 顶刊论文融入本地大模型，打造你的私人 AI for Science 科研助手*

---

[快速开始](#-快速开始) · [系统架构](#-系统架构) · [工作流程](#-完整工作流) · [技术栈](#-技术栈) · [更新计划](#-更新计划)

</div>

<br/>

## 📖 项目简介

**PINN Research Agent** 是一个全链路本地化部署的 AI 科研助手系统，专注于 **Physics-Informed Neural Networks (PINNs)** 和计算力学领域。

本项目将以下技术融合为一体化解决方案：

- **🔍 RAG (检索增强生成)** — 基于 30+ 篇 PINN 领域顶刊论文构建向量知识库，让模型回答有据可依
- **🎯 LoRA SFT 微调** — 从论文中自动蒸馏 QA 数据，对 Qwen2.5-7B 进行领域适配微调
- **🤖 MCP Agent** — 基于 Model Context Protocol 实现自主决策的工具调用 Agent
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
│  │  ② 向量检索 Top-K       │   │  (stdio 协议)        │         │
│  │  ③ LLM 终极推理         │   └──────────┬───────────┘         │
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
用户提问 ──► Streamlit UI ──► RAG Agent ──┬──► ① HyDE 查询重写 (LLM 扩写短查询)
                                          ├──► ② ChromaDB 向量检索 (Top-3 论文片段)
                                          └──► ③ LLM 终极推理 (上下文 + 问题 → 回答)
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
│   ├── rag_agent.py             # RAG Agent 核心 (HyDE + 检索 + 推理)
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

ChromaDB<br/>
all-MiniLM-L6-v2<br/>
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
| Embedding | `all-MiniLM-L6-v2` | 384 维，轻量高效 |
| Chunk 策略 | 500 字符 / 50 重叠 | 滑动窗口切分 |
| 检索 Top-K | 3 | 最相关的 3 个论文片段 |

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

## 🎯 设计亮点

| 特性 | 说明 |
|------|------|
| **HyDE 查询重写** | 用 LLM 将短查询扩写为假设性长文本，显著提升向量检索命中率 |
| **双模式推理** | 同时支持 RAG Web UI 模式和 MCP Agent 自主决策模式 |
| **Self-Instruct 蒸馏** | 零人工标注，用通用 LLM 从论文中自动生成领域训练数据 |
| **全链路本地化** | 训练、推理、知识库全部本地部署，不依赖云端 API |
| **LaTeX 原生支持** | 模型输出标准 LaTeX 公式，Streamlit 前端完美渲染 |
| **模型名称分层** | 通用 `qwen2.5:7b` 与专业 `pinn_qwen_expert` 清晰分层 |
| **Ollama API 兼容** | 统一 OpenAI 格式接口，可无缝切换本地/云端模型 |

<br/>

## 🗺️ 更新计划

- [ ] **多轮对话记忆** — 引入对话历史管理，支持上下文连续追问
- [ ] **更多论文纳入** — 持续扩充 PINN 知识库，覆盖更多子方向
- [ ] **Embedding 模型升级** — 替换为更强的中英双语 Embedding 模型
- [ ] **Agent 工具扩展** — 增加代码生成、公式推导、绘图等 MCP 工具
- [ ] **多模态支持** — 解析论文中的图表，支持图文混合问答
- [ ] **评测体系** — 构建 PINN 领域 Benchmark，量化评估模型能力
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
