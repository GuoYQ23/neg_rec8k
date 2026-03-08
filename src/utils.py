import os
import json
import re

# ==========================================
# 模块一：文件与目录操作 (File I/O Helpers)
# ==========================================

def make_dirs(path):
    """
    安全创建目录：如果目录不存在则创建。
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_json(file_path):
    """
    安全读取 JSON 文件。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """
    规范化保存 JSON 文件（支持中文，自动缩进）。
    """
    make_dirs(os.path.dirname(file_path)) # 确保父目录存在
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_points_from_txt(txt_path):
    """
    从 .txt 文件中读取 (x, y) 坐标点集。
    """
    if not os.path.exists(txt_path):
        return None
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: 
                continue
            try:
                # 默认提取最后两列作为 x, y (兼容多种空格分隔格式)
                points.append([float(parts[-2]), float(parts[-1])])
            except ValueError:
                continue
    return points

# ==========================================
# 模块二：文本清洗 (Text Processing Helpers)
# ==========================================

def clean_redundant_string(text):
    """
    清洗 LLM 生成的冗余文本（如下划线前缀、重复单词）。
    """
    if not isinstance(text, str):
        return text
    
    # 1. 剔除类似于 "other_than_" 这样的结构
    text = re.sub(r'[a-zA-Z0-9]+_[a-zA-Z0-9]+_', '', text)
    
    # 2. 剔除自然语言中连续重复的词组 (例如 "other than other than" -> "other than")
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    
    # 3. 清理多余空格
    text = text.replace("  ", " ").strip()
    return text

# ==========================================
# 模块三：几何与坐标计算 (Geometry Helpers)
# ==========================================

def box_to_center(xmin, ymin, xmax, ymax):
    """
    将 Bounding Box (边界框) 转化为中心点坐标。
    常用于 DETRAC, CARPK 等检测框数据集的点坐标转换。
    """
    x_center = (float(xmin) + float(xmax)) / 2.0
    y_center = (float(ymin) + float(ymax)) / 2.0
    return round(x_center, 2), round(y_center, 2)