# 🧠 Ar2Graph – "ArXiv to Knowledge Graph" — 自动抓取 ArXiv 最新论文输出知识图谱

一个轻量级 Python 工具，arXiv 多关键词爬虫 + OpenAI 智能体 → 知识图谱（全部匹配文章）

---

## ✨ 功能特点

* 🚀 爬取 arXiv 所有含关键词论文（articles_all.json）
* 📄 用 OpenAI 抽取三元组 (head, relation, tail)
* 🔍 生成交互式知识图谱 knowledge_graph.html
* 🎯 支持关键词或文章标题+摘要两种检索模式
* 🧩 智能匹配算法，自动优化检索结果
* 💡 知识图谱自动识别创新点，突出重要节点

---

## 📜 版本迭代历史

| 版本 | 日期 | 主要更新 |
| ---- | ---- | -------- |
| v1.2 | 2025-11-21 | 当前全部改动（包括初始版本、标题/摘要检索、匹配算法与知识图谱优化、爬虫基类重构）均归档在该版本；后续只有在实际提交（commit）时才记为新版本 |

> 约定：本地多次修改不另起版本号，真正提交一次才迭代版本。

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

### 使用关键词检索

```bash
python pythonProjectAr2Graph/mstest.py --key "Defect Detection Wind Turbine Blade" --pages 1
```

### 使用文章标题和摘要检索

```bash
python pythonProjectAr2Graph/mstest.py --title "Your Paper Title" --abstract "Your abstract content..." --pages 1
```

### 参数说明

| 参数           | 说明                                 | 示例                                                  |
| ------------ |------------------------------------|-----------------------------------------------------|
| `--key`      | 直接输入检索关键词（空格分隔），若留空则使用 `--title`+`--abstract` | `--key "slam vision"`                               |
| `--title`/`--abstract` | 文章标题与摘要，自动拆词构建检索关键词（可选） | `--title "Wind ..." --abstract "Inspection..."`     |
| `--pages` | 爬取页数（每页约50篇文章） | `--pages 2` |

### 配置文件说明

可在代码中修改以下配置：

| 配置项 | 说明 | 默认值 |
| ------ | ---- | ------ |
| `SAVE_JSON` | JSON 输出文件名 | `articles_all.json` |
| `SAVE_MD` | Markdown 输出文件名 | `articles_all.md` |
| `BAILIAN_KEY` | 百炼控制台 API Key | 需自行配置 |
| `BAILIAN_BASE` | 固定兼容地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MODEL_NAME` | 使用的模型 | `qwen-turbo` |
| `TRIPLES_FILE` | 三元组输出文件名 | `triples.json` |
| `GRAPH_FILE` | 知识图谱 HTML 文件名 | `knowledge_graph.html` |

> 所有文件都会自动保存在项目根目录下。

---

## 🧩 Python 模块调用

可直接在 Python 中使用：

```python
from pythonProjectAr2Graph.mstest import ArXivCrawler, build_graph, visualize
from openai import OpenAI

# 初始化爬虫
crawler = ArXivCrawler()
keywords = ["Defect Detection", "Wind Turbine"]
articles = crawler.crawl(keywords, pages=1)

# 构建知识图谱
client = OpenAI(api_key="your-api-key", base_url="your-base-url")
graph = build_graph(articles[:10], client=client)
visualize(graph)
```

---

## 📁 输出文件示例

### JSON (`articles_all.json`)

```json
{
    "title": "Non-contact Sensing for AnomalyDetectioninWindTurbineBlades: A focus-SVDD with Complex-Valued Auto-Encoder Approach",
    "abstract": "Abstract:The occurrence of manufacturingdefectsin…",
    "url": "https://arxiv.org/abs/2306.10808"
}
```

### Markdown (`articles_all.md`)

```markdown
| Rank | Title | Abstract | Link |
|------|-------|----------|------|
| 1 | Non-contact Sensing for AnomalyDetectioninWindTurbineBlades... | Abstract:The occurrence of manufacturingdefectsin… | [link](https://arxiv.org/abs/2306.10808) |
```

### 三元组 (`triples.json`)

```json
{
    "head": "Non-contact Sensing for AnomalyDetectioninWindTurbineBlades",
    "relation": "focuses on",
    "tail": "Anomaly Detection"
}
```

### 知识图谱 (`knowledge_graph.html`)

生成交互式知识图谱，包含：
- **红色节点**：论文（包含标题、摘要与链接）
- **蓝色节点**：从论文摘要中抽取的实体
- **绿色边**：论文提及的实体关系
- **蓝色边**：实体之间的语义关系
- **橙色节点**：具有较高创新度的论文

---

## ⚙️ 项目结构

```
Ar2Graph/
│
├── pythonProjectAr2Graph/
│   ├── mstest.py          # 主程序（包含爬虫基类和ArXiv爬虫实现）
│   ├── main.py            # 辅助脚本
│   ├── articles_all.json  # 爬取的文章（JSON格式）
│   ├── triples.json       # 知识图谱三元组（JSON格式）
│   ├── articles_all.md    # 爬取的文章（Markdown格式）
│   └── knowledge_graph.html # 交互式知识图谱
│
├── README.md              # 项目说明（本文件）
└── requirements.txt       # 依赖列表
```

---

## 🏗️ 代码架构

### 爬虫基类设计

项目采用面向对象设计，包含：

- **`BaseCrawler`**：爬虫基类，定义通用爬虫接口
  - `fetch_page()`: 获取页面内容
  - `compute_similarity()`: 计算相似度
  - `crawl()`: 爬取流程（通用逻辑）

- **`ArXivCrawler`**：arXiv 爬虫实现（继承 `BaseCrawler`）
  - `parse_page()`: 解析 arXiv 搜索结果页面
  - `build_search_url()`: 构建 arXiv 搜索URL

这种设计使得项目易于扩展，可以轻松添加其他数据源的爬虫（如 PubMed、IEEE 等）。

---

## 🧑‍💻 作者与许可

* 作者：[Peixin Wang](https://github.com/mark37-boom)
* License: **Unlicense**（公共领域，无任何使用限制）

> 你可以自由地使用、修改、分发本项目的全部或部分内容，无需署名或许可，欢迎点个star！！！

---
