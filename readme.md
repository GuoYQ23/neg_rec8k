\# Negative-REC8K



本项目基于 \[REC-8K](https://github.com/sydai/referring-expression-counting) 原始数据集，通过引入 \*\*LLM\*\* 生成否定语义表达，并利用 \*\*点匹配算法\*\* 实现了从原始数据集点坐标与REC8K中正向指代性文本点坐标的匹配。



本项目解决的核心痛点：

1\. \*\*负向表达缺失\*\*：为每个正向指代生成对应的否定语义。

2\. \*\*标注不统一\*\*：将多源数据统一为 `.txt` 格式的点标注。

3\. \*\*样本清洗\*\*：自动剔除无原始真值样本。



\## 构建流程 

1\. \*\*Data Acquisition\*\*: 自动获取原始 REC-8K 图片及标注。

2\. \*\*LLM Generation\*\*: 调用大模型生成负向描述（如："besides red pen"）。

3\. \*\*Point Logic\*\*: 

&nbsp;  - 统一格式转点标注。

&nbsp;  - 负向真值计算：$P\_{neg} = P\_{all} - P\_{pos}$。

4\. \*\*Filtering\*\*: 通过点匹配算法进行样本筛选。



\## 快速开始



\### 1. 环境准备

```bash

git clone \[https://github.com/GuoYQ23/neg-rec8k.git](https://github.com/GuoYQ23/neg-rec8k.git)

cd neg-rec8k

pip install -r requirements.txt

```



\### 2. 数据准备

请先按照 \[REC-8K](https://github.com/sydai/referring-expression-counting) 官方仓库下载原始数据，并放入 `data/raw/`。



\### 3.运行

```bash

python main.py

```



