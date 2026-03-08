# REC-8K-Negative

## 🌟 项目简介 (Overview)

本项目基于 [REC-8K](https://github.com/sydai/referring-expression-counting) (Referring Expression Counting) 数据集，构建了一个端到端（End-to-End）的数据清洗、格式统一与负向语义增强流水线。

在原始指代性计数任务中，模型往往只关注“正向”目标。本项目创新性地引入了 **负向语义指代 (Negative Referring Expressions)**，并设计了一套自动化的点集匹配挖掘算法，为开放词汇环境下的密集场景计数任务提供了更高质量的数据基座。

### 核心贡献 (Core Contributions)
* **负向语义生成与清洗**：结合 LLM 为每个正向指代生成 1:1 对应的负向描述（如 *“not red pen”*），并内置自动化文本清洗模块，消除冗余前缀与重复词汇。
* **多源异构标注统一**：自动解析并转换 VisDrone, DETRAC, FSC147, JHU, Mall, NWPU, CARPK, Crowd Surveillance 等 8 个不同来源的数据集，将其从复杂的 JSON, XML, MAT 格式统一转化为极简的 `.txt` 中心点坐标。
* **基于集合运算的负向挖掘算法 (SSCM)**：创新性地通过原始全集点与正向指代点的空间距离匹配，计算并提取纯净的负向样本点集。

---

## 📂 仓库结构 (Repository Structure)

```text
REC-8K-Negative-Pipeline/
├── data/
│   ├── samples/                 # 数据集标注示例，供快速了解数据结构
│   │   ├── sample_anno.json     # 包含 0045 / 5638 序列的精简版 JSON
│   │   └── 0000-fsc147-1106.jpg # 示例图片
│   └── metadata/                # 存放中间态元数据 (如 LLM 语义对照表)
├── src/
│   ├── config.py                # 全局路径与参数配置中心
│   ├── utils.py                 # 通用工具箱 (文件 I/O, 文本清洗, 框转点几何计算)
│   └── build_final_dataset.py   # 【主干流水线】一键执行转换、汇总与负向挖掘
├── .gitignore                   # Git 忽略配置
├── requirements.txt             # Python 依赖清单
└── README.md                    # 项目文档
```

---

## 📊 标注格式说明 (Annotation Format)

经过本项目流水线处理后，最终输出的 `non_rec_cleaned_v2.json` 具有高度结构化的特征。所有数据在图片层级下，按语义表达式作为 Key 进行索引。

### 数据结构示例
以下是本数据集的标准 JSON 结构示例：

```json
{
    "0000-fsc147-1106.jpg": {
        "red pen": {
            "class": "pen",
            "attribute": "red",
            "points": [
                [124.5, 330.2], 
                [450.1, 210.8]
            ], 
            "type": "color"
        },
        "not red pen": {
            "class": "pen",
            "attribute": "red",
            "points": [
                [98.3, 562.0], 
                [310.5, 115.4]
            ], 
            "type": "color"
        }
    }
}
```

### 负向点集挖掘逻辑 (Negative Points Mining)

对于任意一张图片，设其场景内所有目标的集合为 $P_{all}$，某正向指代语义（如 *“red pen”*）对应的目标集合为 $P_{pos}$。我们通过距离阈值过滤算法计算负向语义集合 $P_{neg}$：

$$P_{neg} = P_{all} \setminus P_{pos}$$

> **注意：** 我们的代码在内存中自动完成了上述判断与赋值。正向语义保留原始标注，负向语义的 `points` 则由 $P_{neg}$ 直接填充，确保了数据的绝对纯净且无冗余字段。

---

## 🚀 快速开始 (Getting Started)

### 1. 环境准备
```bash
git clone https://github.com/GuoYQ23/neg_rec8k.git
cd neg_rec8k
pip install -r requirements.txt
```

### 2. 准备原始数据
由于版权及存储限制，本仓库不提供原始图像文件。
1. 请前往 [REC-8K 官方仓库](https://github.com/sydai/referring-expression-counting) 下载完整数据集。
2. 将下载的数据集解压，并根据 `src/config.py` 中的 `BASE_DIR` 路径结构放置文件。

### 3. 一键运行流水线
确认好 `src/config.py` 中的路径无误后，直接执行主脚本：
```bash
python src/build_final_dataset.py
```

流水线将依次执行以下 3 个阶段：
1. **统一格式化 (Format Conversion)**：将 8 个子数据集转化为统一的 `.txt` 点坐标。
2. **智能汇总 (Integration)**：根据基准 JSON，将所需标注归档入统一目录。
3. **负向挖掘与清洗 (Mining & Cleaning)**：结合文本正则化与 SSCM 算法，输出最终纯净版 `non_rec_cleaned_v2.json`。

---

## 📝 致谢与引用 (Citation & Acknowledgements)

本项目深度依赖于 [sydai](https://github.com/sydai/referring-expression-counting) 团队提供的 REC-8K 数据集基准。如果你在研究中使用了本数据处理工具或生成的负向数据集，请务必同时引用本项目以及原始论文：

**引用本项目 (REC-8K-Negative):**
```bibtex
@misc{GuoYQ232026recnegative,
  author = {GuoYQ23},
  title = {REC-8K-Negative},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GuoYQ23/neg_rec8k}}
}
```

**引用原始 REC-8K 论文:**
```bibtex
@InProceedings{Dai_2024_CVPR,
    author    = {Dai, Siyang and Liu, Jun and Cheung, Ngai-Man},
    title     = {Referring Expression Counting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {16985-16995}
}
```
---
**License**: MIT License
