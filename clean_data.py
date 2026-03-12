import json

input_file = "pinn_sft_dataset_v2.jsonl"
output_file = "pinn_sft_dataset_v2_clean.jsonl" # 清洗后的纯净数据集

valid_lines = 0
dropped_lines = 0

print("🧹 开始执行脏数据清洗流水线...")

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for i, line in enumerate(f_in):
        try:
            data = json.loads(line)
            # 核心规则：instruction, input, output 必须存在，且必须是纯字符串
            if isinstance(data.get("instruction"), str) and \
               isinstance(data.get("input"), str) and \
               isinstance(data.get("output"), str):
                f_out.write(line)
                valid_lines += 1
            else:
                print(f"⚠️ 发现格式异常数据，已剔除 (行号 {i+1})")
                dropped_lines += 1
        except json.JSONDecodeError:
            print(f"⚠️ 发现 JSON 解析错误，已剔除 (行号 {i+1})")
            dropped_lines += 1

print("-" * 30)
print(f"✅ 清洗完成！")
print(f"🗑️ 剔除坏数据: {dropped_lines} 条")
print(f"💎 保留健康数据: {valid_lines} 条")
print(f"📁 纯净数据集已保存为: {output_file}")