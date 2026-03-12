from unsloth import FastLanguageModel

# 1. 加载你刚刚训练好的补丁模型
print("🔥 正在加载你的 PINN 专属微调模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_pinn_model", # 指向你刚保存的文件夹
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. 核心黑科技：将其与原模型融合，并量化打包为 GGUF 格式！
# q4_k_m 是目前兼顾推理速度、显存占用和回答精度的黄金量化比例
print("🚀 正在融合并导出 GGUF 格式，这可能需要几分钟，请耐心等待...")
model.save_pretrained_gguf("pinn_qwen_gguf", tokenizer, quantization_method = "q4_k_m")

print("✅ 大功告成！你的专属模型已打包完毕！")