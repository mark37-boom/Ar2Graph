# 🧠 Ar2Graph – “ArXiv to Knowledge Graph”

一个轻量级 Python 工具：通过 **arXiv 爬虫 + 大模型智能体**，自动抓取论文、抽取三元组并生成可交互的知识图谱，同时统计高频关键词并给出未来研究趋势分析。

---

## ✨ 功能特点

- **arXiv 多源检索**：支持通过 `关键词` 或 “论文标题 + 摘要” 两种方式组合检索相似文章，并可配置翻页数量。
- **智能匹配排序**：基于标题/摘要与检索词的相似度综合评分（词交集 + 序列相似度），自动过滤噪声并排序结果。
- **知识图谱构建**：调用大模型从摘要中抽取 `head–relation–tail` 三元组，构建有向多重图（`MultiDiGraph`）。
- **节点类型丰富**：
  - 红色：论文节点（包含标题、摘要片段、URL、创新度）
  - 蓝色：实体节点（从三元组中抽取的概念 / 实体）
  - 紫色：关键词节点（从论文中抽取的高频关键词）
  - 橙色：高创新度论文（摘要中出现大量“创新/新方法”等表征）
- **高频关键词分析**：自动提取并统计关键词频次，在图谱下方展示“高频关键词云”。
- **趋势讨论生成**：基于高频关键词，调用大模型自动生成该研究方向的未来趋势与发展建议（中文）。
- **爬虫基类封装**：通过 `BaseCrawler` 抽象通用爬虫流程，`ArXivCrawler` 继承实现 arXiv 适配，方便扩展其他数据源。

---

## 📜 版本迭代策略

| 版本 | 日期 | 说明 |
| ---- | ---- | ---- |
| v1.2 | 2025-11-21 | 当前所有改动（初版功能、标题/摘要检索、匹配算法优化、知识图谱可视化升级、关键词与趋势分析、爬虫基类重构）统一归档在此版本；仅在实际提交（commit）时更新版本号 |

> 约定：本地多次小改动不单独记版本，真正提交到远程仓库时再更新版本号。

---

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

如无 `requirements.txt`，至少需要：

```bash
pip install requests beautifulsoup4 lxml networkx pyvis openai
```

---

## 🚀 命令行使用

在项目根目录执行：

### 1. 使用关键词检索

```bash
python pythonProjectAr2Graph/mstest.py --key "Defect Detection Wind Turbine Blade" --pages 1
```

### 2. 使用“标题 + 摘要”检索

```bash
python pythonProjectAr2Graph/mstest.py \
  --title "Wind turbine blade defect detection via ..." \
  --abstract "Wind power generation is ..." \
  --pages 1
```

### 参数说明

| 参数               | 说明                                                           | 示例                                                   |
| ------------------ | -------------------------------------------------------------- | ------------------------------------------------------ |
| `--key`            | 直接输入检索关键词（空格分隔）。若留空，则使用 `--title`+`--abstract` | `--key "wind turbine defect detection"`                |
| `--title`          | 论文标题（可选，与摘要搭配使用）                              | `--title "Defect detection on a wind turbine blade"`   |
| `--abstract`       | 论文摘要（可选，与标题搭配使用）                              | `--abstract "Wind power generation is a widely ..."`   |
| `--pages`          | 爬取页数，每页约 50 篇论文                                     | `--pages 2`                                            |

### 配置项（在 `mstest.py` 中修改）

| 配置项          | 说明                        | 默认值                                               |
| --------------- | --------------------------- | ---------------------------------------------------- |
| `SAVE_JSON`     | JSON 输出文件名            | `articles_all.json`                                  |
| `SAVE_MD`       | Markdown 输出文件名        | `articles_all.md`                                    |
| `TRIPLES_FILE`  | 三元组输出文件名          | `triples.json`                                       |
| `GRAPH_FILE`    | 知识图谱 HTML 文件名       | `knowledge_graph.html`                               |
| `BAILIAN_KEY`   | 百炼 / Qwen API 密钥       | 环境变量或代码中配置                                 |
| `BAILIAN_BASE`  | API Base URL               | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MODEL_NAME`    | 大模型名称                 | `qwen-turbo`                                         |

> 所有输出文件默认保存在项目根目录或 `pythonProjectAr2Graph` 目录下。

---

## 🧩 作为 Python 模块使用

```python
from pythonProjectAr2Graph.mstest import ArXivCrawler, build_graph, visualize
from openai import OpenAI

# 1. 初始化爬虫
crawler = ArXivCrawler()
keywords = ["defect detection", "wind turbine blade"]
articles = crawler.crawl(keywords, pages=1)

# 2. 初始化大模型客户端
client = OpenAI(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 3. 构建知识图谱（返回图 + 关键词频次）
graph, keyword_freq = build_graph(articles[:10], client=client)

# 4. 可视化（生成 knowledge_graph.html）
visualize(graph)
```

---

## 📁 输出文件示例

### 1. 论文列表（`articles_all.json`）

```json
{
  "title": "Non-contact Sensing for Anomaly Detection in Wind Turbine Blades...",
  "abstract": "Abstract: The occurrence of manufacturing defects in wind turbine blades...",
  "url": "https://arxiv.org/abs/2306.10808"
}
```

### 2. Markdown 列表（`articles_all.md`）

```markdown
| Rank | Title                                                | Abstract                         | Link                         |
|------|------------------------------------------------------|----------------------------------|------------------------------|
| 1    | Non-contact Sensing for Anomaly Detection ...        | Abstract: The occurrence of ...  | https://arxiv.org/abs/2306… |
```

### 3. 三元组（`triples.json`）

```json
{
  "head": "Non-contact Sensing for Anomaly Detection in Wind Turbine Blades",
  "relation": "focuses on",
  "tail": "Anomaly Detection"
}
```

### 4. 知识图谱（`knowledge_graph.html`）

HTML 中集成了 PyVis 交互式网络图和一个说明面板，主要说明：

- **节点颜色**：
  - 红色：论文节点（包含标题、摘要、链接、创新度）
  - 蓝色：实体节点（从三元组中抽取的实体）
  - 紫色：关键词节点（从论文中提取的高频关键词）
  - 橙色：高创新度论文
- **边类型**：
  - 绿色 / 蓝色实线：实体间知识关系
  - 绿色虚线：论文与实体的“mention”关系
  - 紫色虚线：论文与关键词的“contains_keyword”关系
- 下方面板中还展示：
  - 重要论文节点、关键实体节点、创新亮点论文
  - 高频关键词 Tag 云
  - 基于高频关键词生成的“未来研究趋势分析”（中文长文本）

---

## ⚙️ 项目结构

```text
Ar2Graph/
│
├── pythonProjectAr2Graph/
│   ├── mstest.py             # 主程序：爬虫 + 知识图谱 + 可视化 + 趋势分析
│   ├── main.py               # 辅助脚本（如有）
│   ├── articles_all.json     # 爬取的论文列表（JSON）
│   ├── articles_all.md       # 爬取的论文列表（Markdown）
│   ├── triples.json          # 抽取出的知识三元组
│   └── knowledge_graph.html  # 交互式知识图谱页面
│
├── README.md                 # 使用说明（本文件）
└── requirements.txt          # 依赖列表
```

---

## 🏗️ 代码架构（爬虫部分）

项目采用面向对象方式封装爬虫逻辑：

- **`BaseCrawler`**：通用爬虫基类
  - `fetch_page()`：请求网页
  - `compute_similarity()`：计算文本与关键词的相似度
  - `crawl()`：通用爬取流程（翻页、解析、候选回退）

- **`ArXivCrawler`**（继承 `BaseCrawler`）
  - `build_search_url()`：构造 arXiv 搜索 URL
  - `parse_page()`：解析 arXiv 搜索结果列表，抽取标题、摘要、链接与相似度

后续如需接入其他数据源（如 IEEE、PubMed），只需新增对应子类实现这两个方法即可。

---

## 🧑‍💻 作者与许可

- 作者：[Peixin Wang](https://github.com/mark37-boom)
- License: **Unlicense**（公共领域，无任何使用限制）

> 你可以自由地使用、修改、分发本项目的全部或部分内容，无需署名或许可，欢迎点个 star！  

---


