
---

# 🧠 Ar2Graph – “ArXiv to Knowledge Graph” — 自动抓取 ArXiv 最新论文输出知识图谱

一个轻量级 Python 工具，arXiv 多关键词爬虫 + OpenAI 智能体 → 知识图谱（全部匹配文章）

---

## ✨ 功能特点

* 🚀 爬取 arXiv 所有含关键词论文（articles_all.json）
* 📄 用 OpenAI 抽取三元组 (head, relation, tail)
* 🔍 生成交互式知识图谱 knowledge_graph.html
* 🧩 可选功能：


---

## 📦 安装依赖

```bash
pip install arxiv
```

或使用：

```bash
pip install -r requirements.txt
```

---

## 🚀 命令行使用

```bash

```

### 参数说明

| 参数           | 说明                                 | 示例                                                  |
| ------------ |------------------------------------|-----------------------------------------------------|
| SAVE_JSON    | 自定义 JSON 输出文件名（默认 `papers.json`）   | SAVE_JSON = "articles_all.json"                     |
| SAVE_MD | 自定义 Markdown 输出文件名（默认 `output.md`） | `--max-results 50`                                  |
| DEFAULT_KEY  | 搜索的关键词                             | DEFAULT_KEY = "Wind Turbine Blade Defect Detection" |
|DEFAULT_PAGES | 爬取几页数据                             | DEFAULT_PAGES = 1                                   |
| BAILIAN_KEY  | 百炼控制台api-key                       |                                                     |
|BAILIAN_BASE  | 固定兼容地址                             | 不用改                                                 |
| MODEL_NAME   | 模型                                 |      MODEL_NAME = "qwen-turbo"                                     |
| TRIPLES_FILE | 三元组存放链接                            | TRIPLES_FILE = "triples.json"                                    |
| GRAPH_FILE   | 生成知识图谱html名称                       |  GRAPH_FILE = "knowledge_graph.html"                                                                |

> 所有文件都会自动保存在 根文件夹下。

---

### 🕒 



---

## 🧩 Python 模块调用

可直接在 Python 中使用：

```python

```

---

## 📁 输出文件示例

### JSON (`articles_all.json`)

```json
{
    "title": "Non-contact Sensing for AnomalyDetectioninWindTurbineBlades: A focus-SVDD with Complex-Valued Auto-Encoder Approach",
    "abstract": "Abstract:The occurrence of manufacturingdefectsin…▽ MoreThe occurrence of manufacturingdefectsinwindturbineblade(WTB) production can result in significant increases in operation and maintenance costs and lead to severe and disastrous consequences. Therefore, inspection during the manufacturing process is crucial to ensure consistent fabrication of composite materials. Non-contact sensing techniques, such as Frequency Modulated Continuous Wave (FMCW) radar, are becoming increasingly popular as they offer a full view of these complex structures during curing. In this paper, we enhance the quality assurance of manufacturing utilizing FMCW radar as a non-destructive sensing modality. Additionally, a novel anomalydetectionpipeline is developed that offers the following advantages: (1) We use the analytic representation of the Intermediate Frequency signal of the FMCW radar as a feature to disentangle material-specific and round-trip delay information from the received wave. (2) We propose a novel anomalydetectionmethodology called focus Support Vector Data Description (focus-SVDD). This methodology involves defining the limit boundaries of the dataset after removing healthy data features, thereby focusing on the attributes of anomalies. (3) The proposed method employs a complex-valued autoencoder to remove healthy features and we introduces a new activation function called Exponential Amplitude Decay (EAD). EAD takes advantage of the Rayleigh distribution, which characterizes an instantaneous amplitude signal. The effectiveness of the proposed method is demonstrated through its application to collected data, where it shows superior performance compared to other state-of-the-art unsupervised anomalydetectionmethods. This method is expected to make a significant contribution not only to structural health monitoring but also to the field of deep complex-valued data processing and SVDD application.△ Less",
    "url": "https://arxiv.org/abs/2306.10808"
  }
```

### Markdown (`articles_all.md`)

```markdown
## Updated on 2025.11.12

## SLAM

| Rank | Title | Abstract | Link |
|------|-------|----------|------|
| 1 | Non-contact Sensing for AnomalyDetectioninWindTurbineBlades: A focus-SVDD with Complex-Valued Auto-Encoder Approach | Abstract:The occurrence of manufacturingdefectsin…▽ MoreThe occurrence of manufacturingdefectsinwindturbineblade(WTB) production can result in signifi... | [link](https://arxiv.org/abs/2306.10808) |
| 2 | Barely-Visible Surface CrackDetectionforWindTurbineSustainability | Abstract:The production ofwindenergy is a crucial part of sustainable development and reducing the reliance on fossil fuels. Maintaining the integrity... | [link](https://arxiv.org/abs/2407.07186) |
| 3 | Next-generation perception system for automateddefectsdetectionin composite laminates via polarized computational imaging | Abstract:Finishing operations on large-scale composite components likewind…▽ MoreFinishing operations on large-scale composite components likewindturb... | [link](https://arxiv.org/abs/2108.10819) |
| 4 | Identification of SurfaceDefectson Solar PV Panels andWindTurbineBladesusing Attention based Deep Learning Model | Abstract:…large plants remains challenging due to environmental factors that could result in reduced power generation, malfunctioning, and degradation... | [link](https://arxiv.org/abs/2211.15374) |
| 5 | A Novel Approach forDefectDetectionofWindTurbineBladeUsing Virtual Reality and Deep Learning | Abstract:Wind…▽ MoreWindturbinesare subjected to continuous rotational stresses and unusual external forces such as storms, lightning, strikes by flyi... | [link](https://arxiv.org/abs/2401.00237) |
| 6 | Thermal and RGB Images Work Better Together inWindTurbineDamageDetection | Abstract:The inspection ofwind…▽ MoreThe inspection ofwindturbineblades(WTBs) is crucial for ensuring their structural integrity and operational effic... | [link](https://arxiv.org/abs/2412.04114) |
| 7 | Seeing the Unseen: Towards Zero-Shot Inspection forWindTurbineBladesusing Knowledge-Augmented Vision Language Models | Abstract:Wind…▽ MoreWindturbinebladesoperate in harsh environments, making timely damagedetectionessential for preventing failures and optimizing main... | [link](https://arxiv.org/abs/2510.22868) |
```
### triples (`triples.json`)

```json
 {
    "head": "Non-contact Sensing for AnomalyDetectioninWindTurbineBlades",
    "relation": "focuses on",
    "tail": "Anomaly Detection"
  },
```
---

## ⚙️ 文件结构建议

```
arxiv-daily/
│
├── arxiv_daily.py          # 主程序
├── requirements.txt        # 依赖列表
├── articles_all.json       # 所有被爬取的文章json格式
├── triples.json            # 构建知识图谱后的三元组json格式   
├── articles_all.md         # 所有被爬取的文章json格式  
└── README.md               # 使用说明（本文件）
```

---

## 🧑‍💻 作者与许可

* 作者：[Peixin Wang](https://github.com/mark37-boom)
* License: **Unlicense**（公共领域，无任何使用限制）

> 你可以自由地使用、修改、分发本项目的全部或部分内容，无需署名或许可，欢迎点个star！！！

---
