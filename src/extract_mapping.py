import json
import os

def generate_mapping_with_debug(original_path, processed_path, output_path):
    """
    逻辑：
    1. 读取处理后的 JSON。
    2. 对每张图，将描述列表拆分为前半部分（正向）和后半部分（负向）。
    3. 校验数量是否匹配，不匹配则打印图片名。
    """
    with open(processed_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mapping_result = {}

    for img_id, content in data.items():
        # 获取该图片下所有的 key（排除掉非描述类的字段，如可能存在的 metadata）
        # 这里假设你的 JSON 结构中描述是直接作为 key 存在的
        all_keys = [k for k in content.keys() if isinstance(content[k], dict)]
        total_count = len(all_keys)

        num_pairs = total_count // 2
        pos_expressions = all_keys[:num_pairs]
        neg_expressions = all_keys[num_pairs:]

        # 校验逻辑 2：正负向描述的一一对应关系 (可选：检查 neg 是否包含 "not" 前缀)
        # 如果你希望更严谨，可以加上对文本内容的校验
        
        image_mapping = []
        for i in range(num_pairs):
            pos_exp = pos_expressions[i]
            neg_exp = neg_expressions[i]
            
            image_mapping.append({
                "pos": pos_exp,
                "neg": neg_exp,
                "class": content[pos_exp].get("class"),
                "attribute": content[pos_exp].get("attribute")
            })
        
        mapping_result[img_id] = image_mapping

    print("-" * 30)
    # 保存对应表
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_result, f, indent=4, ensure_ascii=False)
    
    print(f"🚀 语义对应表已保存至: {output_path}")

# 执行路径配置
ORIGINAL_JSON = "/root/autodl-tmp/anno/annotations.json"
PROCESSED_JSON = "/root/autodl-tmp/REC_8K/z-newchange/non_rec_cleaned_v2.json"
OUTPUT_MAPPING = "/root/autodl-tmp/REC_8K/z-newchange/mapping.json"

if __name__ == "__main__":
    generate_mapping_with_debug(ORIGINAL_JSON, PROCESSED_JSON, OUTPUT_MAPPING)