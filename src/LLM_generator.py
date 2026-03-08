import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
# 1. 指向你本地 Qwen 权重的路径
model_path = "/root/autodl-tmp/qwen/qwen/Qwen2.5-7B-Instruct" # 请根据实际路径修改

# 2. 文件路径配置
INPUT_JSON = "/root/autodl-tmp/anno/annotations.json"
TEST_LIST = "/root/autodl-tmp/REC_8K/z-newchange/name.txt"
OUTPUT_JSON = "/root/autodl-tmp/REC_8K/z-newchange/non_rec.json"

# 3. 推理设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def load_qwen():
    print(f"正在从 {model_path} 加载模型权重...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()
    return tokenizer, model

import random
import re

def generate_negative_attribute_only(tokenizer, model, class_name, attribute):
    """
    ECCV 生产级版：构建模拟真实世界语言分布的长尾负向算子集
    """
    # 1. 定义算子及其权重 (模拟 Zipf's Law)
    # 即使是剩下的 50%，我们也希望看到梯次分布
    operators = [
        "not",          # 绝对统治
        "without",      # 常用否定
        "except for",   # 常用排除
        "besides",      # 中频
        "other than",   # 中频
        "aside from",   # 低频
        "devoid of",    # 极低频
        "lack of"       # 极低频
    ]
    
    # 权重分配：确保 not 占约 50%，其余呈阶梯状下降
    # 总权重 200，not 占 100
    weights = [100, 40, 25, 15, 10, 5, 3, 2] 
    
    # 2. 根据权重选择本次的算子
    target_op = random.choices(operators, weights=weights, k=1)[0]
    is_simple = (target_op == "not")

    # 3. 构造强引导 Prompt
    if is_simple:
        system_prompt = (
            "You are a professional linguistic annotator.\n"
            "Task: Convert a positive attribute into its most direct 'not' form.\n"
            "Constraints: Output ONLY the result. Use ONLY 'not'. No class names."
        )
    else:
        system_prompt = (
            "You are a linguistic expert.\n"
            "Task: Negate the attribute using the EXACT operator provided.\n"
            "Constraints:\n"
            "1. USE ONLY the operator '{op}' to negate.\n"
            "2. DO NOT use 'not'.\n"
            "3. DO NOT include the class name '{class_name}'.\n"
            "4. Output ONLY the simplified attribute string."
        ).format(op=target_op, class_name=class_name)

    # 准备输入与推理 (保持原逻辑，但调低 Temperature 以增强指令遵循)
    user_input = f"Input Attribute: '{attribute}'"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=16,
        do_sample=True,      
        temperature=0.2, # 极致降温，强制模型必须使用指定的 target_op
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    # 4. 暴力后处理与保底逻辑 (确保 target_op 绝对出现)
    clean_res = response.strip().replace('"', '').replace("'", "").split('\n')[0].lower()
    
    # 核心：如果 LLM 没用指定的词，手动强制修正，解决“算子丢失”问题
    if target_op not in clean_res:
        # 特殊处理 'not'，防止重复
        final_attr = re.sub(r'\bnot\b', '', clean_res).strip()
        clean_res = f"{target_op} {final_attr if final_attr else attribute}"

    # 移除 class_name 的二次检查
    clean_res = re.sub(rf'\b{re.escape(class_name.lower())}\b', '', clean_res).strip()
    
    return clean_res
    
def main():
    tokenizer, model = load_qwen()

    if not os.path.exists(TEST_LIST):
        print(f"错误: 找不到 {TEST_LIST}")
        return
    with open(TEST_LIST, 'r', encoding='utf-8') as f:
        target_images = [line.strip() for line in f if line.strip()]

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    paired_data = {}

    print(f"开始成对构建数据集(修正Attribute)，共处理 {len(target_images)} 张图片...")
    
    with torch.no_grad():
        for img_name in target_images:
            if img_name in full_data:
                print(f"正在处理并推理: {img_name}")
                
                positive_dict = full_data[img_name] 
                # 创建副本用于存放最终结果
                combined_entries = positive_dict.copy()

                # 遍历原有正向条目
                for original_desc_key, entry in list(positive_dict.items()):
                    
                    # 1. 【核心修改】调用新的函数，只生成否定属性字符串（例如 "not red"）
                    neg_attribute_only = generate_negative_attribute_only(tokenizer, model, entry['class'], entry['attribute'])
                    
                    # 2. 【核心修改】在Python端拼接完整的否定描述作为 Key（例如 "not red pen"）
                    # 使用 f-string 确保中间有空格
                    full_neg_desc_key = f"{neg_attribute_only} {entry['class']}".strip()

                    # 3. 构造负向条目字典
                    neg_entry = {
                        "class": entry['class'],
                        # attribute 字段只存放否定修饰词
                        "attribute": neg_attribute_only, 
                        "points": [], 
                        "type": entry['type']
                    }
                    
                    # 4. 使用拼接好的完整描述作为 Key 插入
                    combined_entries[full_neg_desc_key] = neg_entry
                
                paired_data[img_name] = combined_entries
            else:
                print(f"跳过: {img_name} (未在 JSON 中找到)")

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, ensure_ascii=False, indent=4)
    
    print(f"修正完成！结果已保存至 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()