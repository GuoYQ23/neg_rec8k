import os
import shutil
import xml.etree.ElementTree as ET
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from utils import (
    make_dirs, load_json, save_json, 
    load_points_from_txt, clean_redundant_string, box_to_center
)
from config import CONFIG, INTEGRATION_CONFIG, JSON_PROCESS_CONFIG

# =====================================================================
#                          阶段一 & 阶段二：格式转换与统一汇总
# =====================================================================

def convert_cs_data():
    print("\n[1/9] 开始处理 Crowd Surveillance 数据集...")
    make_dirs(CONFIG["cs"]["dst"])
    data = load_json(CONFIG["cs"]["json"]) 
    
    for entry in tqdm(data.get('annotations', []), desc="CS 提取中"):
        img_full_path = entry.get('name', '')
        if not img_full_path: continue
        txt_name = os.path.splitext(os.path.basename(img_full_path))[0] + ".txt"
        locs = entry.get('locations', [])
        with open(os.path.join(CONFIG["cs"]["dst"], txt_name), 'w') as f_out:
            for i in range(0, len(locs), 2):
                try: f_out.write(f"{locs[i]} {locs[i+1]}\n")
                except IndexError: break

def convert_detrac_data():
    print("\n[2/9] 开始处理 UA-DETRAC 数据集...")
    make_dirs(CONFIG["detrac"]["dst"])
    xml_files = [f for f in os.listdir(CONFIG["detrac"]["xml_dir"]) if f.endswith('.xml')]
    
    for xml_file in tqdm(xml_files, desc="DETRAC 提取中"):
        seq_name = os.path.splitext(xml_file)[0]
        try:
            tree = ET.parse(os.path.join(CONFIG["detrac"]["xml_dir"], xml_file))
            for frame in tree.getroot().findall('frame'):
                img_id = f"img{int(frame.get('num')):05d}"
                dot_coordinates = []
                target_list = frame.find('target_list')
                if target_list is None: continue
                
                for target in target_list.findall('target'):
                    box = target.find('box')
                    if box is not None:
                        left, top = float(box.get('left')), float(box.get('top'))
                        width, height = float(box.get('width')), float(box.get('height'))
                        cx, cy = box_to_center(left, top, left + width, top + height)
                        dot_coordinates.append(f"{cx:.2f} {cy:.2f}")
                        
                with open(os.path.join(CONFIG["detrac"]["dst"], f"{seq_name}__{img_id}.txt"), 'w') as f:
                    f.write('\n'.join(dot_coordinates))
        except Exception: pass

def convert_fsc147_data():
    print("\n[3/9] 开始处理 FSC147_384_V2 数据集...")
    make_dirs(CONFIG["fsc147"]["dst"])
    data = load_json(CONFIG["fsc147"]["json"]) 
    
    for img_name, info in tqdm(data.items(), desc="FSC147 提取中"):
        with open(os.path.join(CONFIG["fsc147"]["dst"], os.path.splitext(img_name)[0] + ".txt"), 'w') as f_out:
            for p in info.get('points', []):
                if len(p) == 2: f_out.write(f"{p[0]:.2f} {p[1]:.2f}\n")

def convert_jhu_data():
    print("\n[4/9] 开始处理 JHU Crowd 数据集...")
    make_dirs(CONFIG["jhu"]["dst"])
    files = [f for f in os.listdir(CONFIG["jhu"]["src"]) if f.endswith('.txt')]
    for file_name in tqdm(files, desc="JHU 提取中"):
        with open(os.path.join(CONFIG["jhu"]["src"], file_name), 'r') as f_in, \
             open(os.path.join(CONFIG["jhu"]["dst"], file_name), 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 2: f_out.write(f"{parts[0]} {parts[1]}\n")

def convert_mall_data():
    print("\n[5/9] 开始处理 Mall 数据集...")
    make_dirs(CONFIG["mall"]["dst"])
    try:
        frames = sio.loadmat(CONFIG["mall"]["mat"])['frame'][0]
        for i in tqdm(range(len(frames)), desc="Mall 提取中"):
            try:
                points = frames[i][0][0][0] 
                with open(os.path.join(CONFIG["mall"]["dst"], f"seq_{i+1:06d}.txt"), 'w') as f_out:
                    for p in points:
                        if len(p) >= 2: f_out.write(f"{p[0]:.2f} {p[1]:.2f}\n")
            except: pass
    except Exception as e: print(f"⚠️ Mall 解析失败: {e}")

def convert_nwpu_data():
    print("\n[6/9] 开始处理 NWPU 数据集...")
    make_dirs(CONFIG["nwpu"]["dst"])
    files = [f for f in os.listdir(CONFIG["nwpu"]["src"]) if f.endswith('.json')]
    for file_name in tqdm(files, desc="NWPU 提取中"):
        dst_path = os.path.join(CONFIG["nwpu"]["dst"], file_name.replace('.json', '.txt'))
        try:
            points = load_json(os.path.join(CONFIG["nwpu"]["src"], file_name)).get('points', [])
            with open(dst_path, 'w') as f_out:
                for p in points:
                    if len(p) >= 2: f_out.write(f"{p[0]} {p[1]}\n")
        except: pass

def convert_carpk_data():
    print("\n[7/9] 开始处理 CARPK 数据集...")
    make_dirs(CONFIG["carpk"]["dst"])
    files = [f for f in os.listdir(CONFIG["carpk"]["src"]) if f.endswith('.txt')]
    for file_name in tqdm(files, desc="CARPK 提取中"):
        with open(os.path.join(CONFIG["carpk"]["src"], file_name), 'r') as f_in, \
             open(os.path.join(CONFIG["carpk"]["dst"], file_name), 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 4:
                    cx, cy = box_to_center(parts[0], parts[1], parts[2], parts[3])
                    f_out.write(f"{cx:.2f} {cy:.2f}\n")

def extract_visdrone_points(source_path, frame_idx):
    points = []
    if not os.path.exists(source_path): return points
    with open(source_path, 'r') as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 3 and int(parts[0]) == frame_idx:
                points.append(f"{parts[1]} {parts[2]}")
    return points

def unify_all_annotations():
    print("\n[8/9] 开始智能整合所有真值文件 (Unified GT)...")
    make_dirs(INTEGRATION_CONFIG["unified_gt_dir"])
    data = load_json(INTEGRATION_CONFIG["input_json"]) 
    
    success_count = 0
    fail_list = []

    for rec_name in tqdm(data.keys(), desc="整合进度"):
        name_no_ext = os.path.splitext(rec_name)[0]
        parts = name_no_ext.split('-')
        if len(parts) < 3: 
            fail_list.append(f"跳过异常命名: {rec_name}")
            continue
        
        ds_label = parts[1]
        target_path = os.path.join(INTEGRATION_CONFIG["unified_gt_dir"], name_no_ext + ".txt")
        original_root = INTEGRATION_CONFIG["original_root"]

        if ds_label == "detrac" and len(parts) >= 4:
            source_path = os.path.join(original_root, "detrac", f"{parts[2]}__{parts[3]}.txt")
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
                success_count += 1
            else: fail_list.append(f"Detrac 缺失: {source_path}")
        elif ds_label == "visdrone" and len(parts) >= 4:
            source_path = os.path.join(original_root, "visdrone", f"{parts[2]}.txt")
            points_list = extract_visdrone_points(source_path, int(parts[3]))
            if points_list:
                with open(target_path, 'w') as f_out: f_out.write("\n".join(points_list))
                success_count += 1
            else: fail_list.append(f"VisDrone 缺失/无点: 帧 {parts[3]} in {source_path}")
        else:
            real_name = "-".join(parts[2:])
            source_path = os.path.join(original_root, ds_label, real_name + ".txt")
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
                success_count += 1
            else: fail_list.append(f"常规缺失: {source_path}")

    log_path = os.path.join(INTEGRATION_CONFIG["unified_gt_dir"], "missing_files.log")
    with open(log_path, 'w', encoding='utf-8') as f: f.write("\n".join(fail_list))
    print(f"✅ 智能整合完成！成功: {success_count} 个，失败/缺失: {len(fail_list)} 个 (详见 {log_path})")

# =====================================================================
#                          阶段三：并集挖掘与 JSON 清洗
# =====================================================================

class SSCMProcessor:
    def __init__(self, alpha=0.5, default_scale=50):
        self.alpha = alpha
        self.default_scale = default_scale

    def match_and_mine(self, total_pts, pos_pts, dist_thresh=15.0):
        total_pts = np.array(total_pts) if len(total_pts) > 0 else np.empty((0, 2))
        pos_pts = np.array(pos_pts) if len(pos_pts) > 0 else np.empty((0, 2))
    
        if len(total_pts) == 0: return pos_pts.tolist(), [], 0
        if len(pos_pts) == 0: return [], total_pts.tolist(), 0
    
        dist_matrix = cdist(total_pts, pos_pts)
        min_dist_to_pos = np.min(dist_matrix, axis=1)
        safe_negative_mask = min_dist_to_pos > dist_thresh
        
        final_pos = pos_pts.tolist()
        final_neg = total_pts[safe_negative_mask].tolist()
        match_count = len(total_pts) - np.sum(safe_negative_mask)
        return final_pos, final_neg, int(match_count)

def get_total_set(img_id, txt_dir):
    txt_path = os.path.join(txt_dir, f"{os.path.splitext(img_id)[0]}.txt")
    return load_points_from_txt(txt_path) 

def build_positive_index(anno_path):
    positive_indices = set()
    anno_data = load_json(anno_path) 
    for img_name, content in anno_data.items():
        if isinstance(content, dict):
            for expr in content.keys(): positive_indices.add((img_name, expr))
        elif isinstance(content, list):
            for expr in content: positive_indices.add((img_name, expr))
    return positive_indices

def run_integrated_pipeline():
    print("\n[9/9] 开始执行: 并集挖掘计算 + 结构清洗 + 文本去重 ...")
    cfg = JSON_PROCESS_CONFIG
    processor = SSCMProcessor()
    
    print("⏳ 加载正向索引...")
    positive_indices = build_positive_index(cfg["anno_json"])

    print("⏳ 加载待处理数据...")
    dataset = load_json(cfg["input_json"]) 

    final_data = {}
    for img_id, expressions_dict in tqdm(dataset.items(), desc="挖掘与清洗进度"):
        total_set = get_total_set(img_id, cfg["txt_dir"])
        if total_set is None: continue 
            
        if not isinstance(expressions_dict, dict):
            final_data[img_id] = expressions_dict
            continue

        processed_img_content = {}
        for original_expr_text, data_content in expressions_dict.items():
            pos_points = data_content.get('points', [])
            _, f_neg, _ = processor.match_and_mine(total_set, pos_points, dist_thresh=15.0)
            
            clean_expr_text = clean_redundant_string(original_expr_text)
            if 'attribute' in data_content:
                data_content['attribute'] = clean_redundant_string(data_content['attribute'])
                
            is_positive = (img_id, original_expr_text) in positive_indices
            if not is_positive:
                data_content['points'] = f_neg
            
            processed_img_content[clean_expr_text] = data_content
        
        final_data[img_id] = processed_img_content

    save_json(final_data, cfg["output_json"])
    print(f"✅ 净化版标注数据已保存至: {cfg['output_json']}")

# =====================================================================
#                          主入口执行区域
# =====================================================================
if __name__ == "__main__":
    print("🚀 启动 REC-8K 端到端数据工程流水线 (End-to-End Pipeline)...")
    print("=" * 60)
    
    convert_cs_data()
    convert_detrac_data()
    convert_fsc147_data()
    convert_jhu_data()
    convert_mall_data()
    convert_nwpu_data()
    convert_carpk_data()
    
    unify_all_annotations()
    run_integrated_pipeline()
    
    print("=" * 60)
    print("🎉 所有流程执行完毕！你的数据已经准备好投喂给模型了！")