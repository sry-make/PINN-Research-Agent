import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template

# ================= 1. 模型加载与 4-bit 量化 =================
print("🔥 正在初始化炼丹炉... 加载 4-bit 量化模型...")
max_seq_length = 2048 # 最大上下文长度，对于本地显卡 2048 是个安全的甜点位

# 【注意】我们不使用 Ollama 的模型文件，而是让 Unsloth 自动从 HuggingFace 下载专门优化过的权重
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", # 极其精简的 4-bit Qwen2.5 版本
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# ================= 2. 注入 LoRA 旁路适配器 =================
print("💉 正在向大模型注入 LoRA 适配器...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA 的秩（Rank），16 是一个在表达能力和参数量之间极其平衡的值
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, # LoRA 缩放因子
    lora_dropout = 0, # 配合 Unsloth 优化，必须设为 0
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 极限压榨显存的黑科技
    random_state = 3407,
)

# ================= 3. 数据集准备与格式化 =================
print("📚 正在加载并格式化 PINN SFT 语料库...")
# 使用 Qwen 官方的 ChatML 模板格式化你的问答数据
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

def format_dataset(examples):
    formatted_texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        # 将你的单轮问答包装成标准的大模型对话格式
        messages = [
            {"role": "system", "content": "你是一个严谨的计算力学与物理信息神经网络（PINN）专家。"},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": formatted_texts}

# 加载你刚刚跑出来的 V2 版本 jsonl 数据集
dataset = load_dataset("json", data_files="pinn_sft_dataset_v2_clean.jsonl", split="train")
dataset = dataset.map(format_dataset, batched=True)

# ================= 4. 配置 SFT 训练器并点火 =================
print("🚀 万事俱备，开始炼丹！")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    args = TrainingArguments(
        per_device_train_batch_size = 2, # 16G 显存通常只能塞下 2 的 batch_size
        gradient_accumulation_steps = 4, # 梯度累加：通过多跑几次变相实现 batch_size=8 的效果，稳定梯度
        warmup_steps = 10,
        max_steps = 60, # 因为数据量少（289条），我们先跑 60 步快速验证（大概需要十几分钟）
        learning_rate = 2e-4, # LoRA 推荐的经典学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit", # 8-bit 优化器，进一步省显存
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 开始训练！
trainer_stats = trainer.train()

# ================= 5. 保存你专属的 LoRA 权重 =================
print("🎉 训练完成！正在保存 PINN 专属 LoRA 权重...")
model.save_pretrained("lora_pinn_model") # 保存权重
tokenizer.save_pretrained("lora_pinn_model")
print("✅ 模型已成功保存至 lora_pinn_model 文件夹！")