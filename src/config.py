import os

# =====================================================================
#                          全局路径配置区域 (Global Configurations)
# =====================================================================

# 基础目录：建议根据你实际运行的服务器环境修改此路径
BASE_DIR = "/root/autodl-tmp/REC_8K"

# 1. 阶段一：各个子数据集的原始路径与点集输出配置
CONFIG = {
    "cs":     {"json": os.path.join(BASE_DIR, "Crowd_Surveillance/train.json"),            "dst": os.path.join(BASE_DIR, "Crowd_Surveillance/anno_dot")},
    "detrac": {"xml_dir": os.path.join(BASE_DIR, "DETRAC/Annotations"),                    "dst": os.path.join(BASE_DIR, "DETRAC/Anno_dot")},
    "fsc147": {"json": os.path.join(BASE_DIR, "FSC147_384_V2/annotation_FSC147_384.json"), "dst": os.path.join(BASE_DIR, "FSC147_384_V2/Anno_dot")},
    "jhu":    {"src": os.path.join(BASE_DIR, "jhu_crowd_v2.0/gt"),                         "dst": os.path.join(BASE_DIR, "jhu_crowd_v2.0/anno_dot")},
    "mall":   {"mat": os.path.join(BASE_DIR, "mall_dataset/mall_gt.mat"),                  "dst": os.path.join(BASE_DIR, "mall_dataset/anno_dot")},
    "nwpu":   {"src": os.path.join(BASE_DIR, "NWPU/jsons"),                                "dst": os.path.join(BASE_DIR, "NWPU/anno_dot")},
    "carpk":  {"src": os.path.join(BASE_DIR, "CARPK_devkit/data/Annotations"),             "dst": os.path.join(BASE_DIR, "CARPK_devkit/data/Anno_dot")}
}

# 2. 阶段二：真值统一汇总配置
INTEGRATION_CONFIG = {
    "original_root": os.path.join(BASE_DIR, "all_anno"),       
    "unified_gt_dir": os.path.join(BASE_DIR, "all_anno_rec"),  
    "input_json": "/root/autodl-tmp/anno/annotations.json"     
}

# 3. 阶段三：JSON 语义清洗与 SSCM 并集挖掘配置
JSON_PROCESS_CONFIG = {
    "txt_dir": INTEGRATION_CONFIG["unified_gt_dir"],                 
    "input_json": os.path.join(BASE_DIR, "z-newchange/non_rec.json"),
    "anno_json": INTEGRATION_CONFIG["input_json"],                   
    "output_json": os.path.join(BASE_DIR, "z-newchange/non_rec_cleaned_v2.json") 
}