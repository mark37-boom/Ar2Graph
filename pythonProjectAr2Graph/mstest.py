#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arXiv 多关键词爬虫 + OpenAI 智能体 → 知识图谱（全部匹配文章）
----------------------------------------------------------------
1. 爬取 arXiv 所有含关键词论文（articles_all.json）
2. 用 OpenAI 抽取三元组 (head, relation, tail)
3. 生成交互式知识图谱 knowledge_graph.html
"""
import argparse
import json
import os
import time
import datetime as dt
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, quote_plus
import difflib
import openai
import networkx as nx
from pyvis.network import Network
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import html

# ---------- 配置 ----------
BASE_URL = "https://arxiv.org"
SEARCH_TMP = "/search/?query={kw}&searchtype=all&start={start}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FuzzyBot/0.1)"}
SLEEP = 1
SAVE_JSON = "articles_all.json"
SAVE_MD = "articles_all.md"
DEFAULT_KEY = "Defect Detection on a Wind Turbine Blade"
DEFAULT_TITLE = " "
DEFAULT_ABSTRACT = ""
DEFAULT_PAGES = 1
MATCH_THRESHOLD = 0.12
FALLBACK_THRESHOLD = 0.02
FALLBACK_MAX_RESULTS = 30
MENTION_RELATION = "mentions"
INNOVATION_KEYWORDS = [
    "novel", "innovative", "proposed", "propose", "first", "new", "framework",
    "method", "approach", "architecture", "contribution", "breakthrough", "advance",
    "improve", "improved", "enhanced", "multi-modal", "multi modal", "协同", "创新",
    "提出", "首个", "新型", "多模态", "框架", "方法", "系统"
]
GRAPH_DESCRIPTION = (
    "该知识图谱由三类节点组成：红色节点代表论文（包含标题、摘要与链接），"
    "蓝色节点代表从论文摘要中抽取的实体，紫色节点代表从论文中提取的高频关键词。"
    "绿色边表示论文提及的实体，蓝色边表示实体之间的语义关系，紫色边表示论文包含的关键词。"
    "可以拖拽节点查看局部结构，或悬停在节点/边上查看详细信息。"
    "橙色节点表示具有较高创新度的论文。"
)
BAILIAN_KEY = os.getenv("sk-acb35646d20348cea1fff58447e93430") or "sk-acb35646d20348cea1fff58447e93430"  # 百炼控制台获取
BAILIAN_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 固定兼容地址
MODEL_NAME = "qwen-turbo"                                            # 可选 qwen-
TRIPLES_FILE = "triples.json"
GRAPH_FILE = "knowledge_graph.html"
import re
# --------------------------


# ---------- 工具 ----------
def clean_json(raw: str) -> str:
    # 1. 去掉控制字符
    raw = re.sub(r'[\x00-\x1f]', ' ', raw)
    # 2. 把单个反斜杠替换成双反斜杠（防止无效转义）
    raw = raw.replace('\\', '\\\\')
    # 3. 再把 \\\\ 还原成合法 \\\\ 避免过度转义
    raw = raw.replace('\\\\\\\\', '\\\\')
    return raw

def save_json(path: str, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_md(path: str, data: List[Dict]) -> None:
    lines = ["| Rank | Title | Abstract | Link |",
             "|------|-------|----------|------|"]
    for rank, art in enumerate(data, 1):
        title = art["title"].replace("|", r"\|")
        abstract = art["abstract"][:150].replace("|", r"\|") + "..."
        link = f"[link]({art['url']})"
        lines.append(f"| {rank} | {title} | {abstract} | {link} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_keywords_from_text(title: str, abstract: str) -> List[str]:
    tokens = []
    for text in (title, abstract):
        if not text:
            continue
        tokens.extend([tok for tok in re.split(r"\s+", text.strip()) if tok])
    return tokens


def build_keywords_from_arg(key_arg: str) -> List[str]:
    if not key_arg:
        return []
    return [tok for tok in re.split(r"\s+", key_arg.strip()) if tok]


def compute_innovation_score(text: str) -> float:
    if not text:
        return 0.0
    tokens = normalize_tokens(text)
    if not tokens:
        return 0.0
    score = 0
    for kw in INNOVATION_KEYWORDS:
        normalized_kw = kw.lower()
        count = sum(1 for token in tokens if normalized_kw in token)
        score += count
    return score / len(tokens)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def normalize_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def compute_match_score(title: str, abstract: str, keywords: List[str]) -> float:
    key_tokens = [tok.lower() for tok in keywords if tok]
    text_tokens = normalize_tokens(f"{title} {abstract}")
    if not key_tokens or not text_tokens:
        return 0.0
    key_set = set(key_tokens)
    text_set = set(text_tokens)
    overlap = len(key_set & text_set) / len(key_set)
    seq_score = difflib.SequenceMatcher(
        None, " ".join(key_tokens), " ".join(text_tokens)
    ).ratio()
    return 0.7 * overlap + 0.3 * seq_score


# ---------- 多关键词 OR 匹配 ----------
def fuzzy_match(title: str, abstract: str, keywords: List[str], threshold: float = MATCH_THRESHOLD) -> bool:
    return compute_match_score(title, abstract, keywords) >= threshold


def similarity(title: str, abstract: str, keywords: List[str]) -> float:
    return compute_match_score(title, abstract, keywords)


# ---------- 爬虫基类 ----------
class BaseCrawler:
    """爬虫基类，定义通用爬虫接口和通用方法"""
    
    def __init__(self, base_url: str, headers: Dict[str, str] = None, sleep: float = 1.0):
        """
        初始化爬虫基类
        
        Args:
            base_url: 基础URL
            headers: HTTP请求头
            sleep: 请求间隔时间（秒）
        """
        self.base_url = base_url
        self.headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; FuzzyBot/0.1)"}
        self.sleep = sleep
    
    def fetch_page(self, url: str, timeout: int = 20) -> str:
        """
        获取页面内容（通用方法）
        
        Args:
            url: 目标URL
            timeout: 超时时间
            
        Returns:
            页面HTML内容
        """
        print(f"[+] 抓取 {url}")
        resp = requests.get(url, headers=self.headers, timeout=timeout)
        resp.raise_for_status()
        time.sleep(self.sleep)
        return resp.text
    
    def parse_page(self, html: str, keywords: List[str], threshold: float) -> Tuple[List[Dict], List[Dict]]:
        """
        解析页面内容（子类需实现）
        
        Args:
            html: 页面HTML内容
            keywords: 关键词列表
            threshold: 匹配阈值
            
        Returns:
            (匹配的文章列表, 候选文章列表)
        """
        raise NotImplementedError("子类必须实现 parse_page 方法")
    
    def build_search_url(self, keywords: List[str], page: int) -> str:
        """
        构建搜索URL（子类需实现）
        
        Args:
            keywords: 关键词列表
            page: 页码
            
        Returns:
            搜索URL
        """
        raise NotImplementedError("子类必须实现 build_search_url 方法")
    
    def compute_similarity(self, title: str, abstract: str, keywords: List[str]) -> float:
        """
        计算相似度（通用方法）
        
        Args:
            title: 文章标题
            abstract: 文章摘要
            keywords: 关键词列表
            
        Returns:
            相似度分数
        """
        return compute_match_score(title, abstract, keywords)
    
    def crawl(self, keywords: List[str], pages: int, threshold: float = MATCH_THRESHOLD) -> List[Dict]:
        """
        爬取文章（通用方法）
        
        Args:
            keywords: 关键词列表
            pages: 爬取页数
            threshold: 匹配阈值
            
        Returns:
            匹配的文章列表
        """
        matched_articles = []
        fallback_candidates = []
        
        for p in range(pages):
            url = self.build_search_url(keywords, p)
            html = self.fetch_page(url)
            page_matches, page_candidates = self.parse_page(html, keywords, threshold)
            matched_articles.extend(page_matches)
            fallback_candidates.extend(page_candidates)
        
        matched_articles.sort(key=lambda x: x["score"], reverse=True)
        if not matched_articles:
            fallback_candidates.sort(key=lambda x: x["score"], reverse=True)
            matched_articles = fallback_candidates[:FALLBACK_MAX_RESULTS]
        
        return matched_articles


# ---------- arXiv 爬虫实现 ----------
class ArXivCrawler(BaseCrawler):
    """arXiv 爬虫，继承自 BaseCrawler"""
    
    def __init__(self, base_url: str = BASE_URL, search_template: str = SEARCH_TMP, 
                 headers: Dict[str, str] = None, sleep: float = SLEEP):
        """
        初始化 arXiv 爬虫
        
        Args:
            base_url: arXiv 基础URL
            search_template: 搜索URL模板
            headers: HTTP请求头
            sleep: 请求间隔时间
        """
        super().__init__(base_url, headers, sleep)
        self.search_template = search_template
    
    def build_search_url(self, keywords: List[str], page: int) -> str:
        """构建 arXiv 搜索URL"""
        kw_str = " ".join(keywords)
        start = page * 50
        query = quote_plus(kw_str)
        path = self.search_template.format(kw=query, start=start)
        return urljoin(self.base_url, path)
    
    def parse_page(self, html: str, keywords: List[str], threshold: float) -> Tuple[List[Dict], List[Dict]]:
        """解析 arXiv 搜索结果页面"""
        soup = BeautifulSoup(html, "lxml")
        articles = []
        candidates = []
        
        for entry in soup.select("li.arxiv-result"):
            title_tag = entry.select_one("p.title")
            abs_tag = entry.select_one("p.abstract")
            link_tag = entry.select_one("a[href*='/abs/']")
            
            if not all([title_tag, abs_tag, link_tag]):
                continue
            
            title = title_tag.get_text(strip=True)
            abstract = abs_tag.get_text(strip=True)
            link = urljoin(self.base_url, link_tag["href"])
            score = self.compute_similarity(title, abstract, keywords)
            
            candidate = {
                "title": title,
                "abstract": abstract,
                "url": link,
                "score": score
            }
            candidates.append(candidate)
            
            if score >= threshold:
                articles.append(candidate)
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return articles, candidates


# ---------- 智能体抽取 ----------
def extract_triples(text: str, title: str = "", client: OpenAI = None) -> List[Dict]:
    if client is None:
        client = OpenAI(api_key=BAILIAN_KEY, base_url=BAILIAN_BASE, timeout=60)
    prompt = f"""
You are a knowledge graph extractor.  
Given a paper title and abstract, output **only** a JSON list of triples:  
[{{"head": "...", "relation": "...", "tail": "..."}}]

Title: {title}
Abstract: {text}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = clean_json(response.choices[0].message.content.strip())
    try:
        return json.loads(content)
    except Exception as e:
        print("解析失败，返回空列表", e)
        return []


def extract_keywords(text: str, title: str = "", client: OpenAI = None) -> List[str]:
    """
    从文章标题和摘要中提取关键词
    
    Args:
        text: 文章摘要
        title: 文章标题
        client: OpenAI客户端
        
    Returns:
        关键词列表
    """
    if client is None:
        client = OpenAI(api_key=BAILIAN_KEY, base_url=BAILIAN_BASE, timeout=60)
    prompt = f"""
You are a keyword extractor.  
Given a paper title and abstract, extract 3-5 most important keywords that represent the core research topics.
Output **only** a JSON list of keywords: ["keyword1", "keyword2", "keyword3"]

Title: {title}
Abstract: {text}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = clean_json(response.choices[0].message.content.strip())
        keywords = json.loads(content)
        if isinstance(keywords, list):
            return [str(kw).strip() for kw in keywords if kw]
        return []
    except Exception as e:
        print(f"关键词提取失败: {e}")
        return []


def generate_research_trends(hot_keywords: List[Dict], client: OpenAI = None) -> str:
    """
    根据高频关键词生成未来研究趋势讨论
    
    Args:
        hot_keywords: 高频关键词列表，格式为 [{"keyword": "...", "frequency": 10}, ...]
        client: OpenAI客户端
        
    Returns:
        研究趋势讨论文本
    """
    if client is None:
        client = OpenAI(api_key=BAILIAN_KEY, base_url=BAILIAN_BASE, timeout=60)
    
    keywords_str = ", ".join([f"{kw['keyword']} (出现{kw['frequency']}次)" for kw in hot_keywords[:10]])
    prompt = f"""
You are a research trend analyst. Based on the following hot keywords extracted from recent papers, 
analyze and discuss the future research trends and directions in this field.

Hot Keywords:
{keywords_str}

Please provide a comprehensive analysis in Chinese, covering:
1. Current research focus areas
2. Emerging trends
3. Potential future directions
4. Interdisciplinary opportunities

Output format: A well-structured discussion text (no JSON, just plain text).
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"研究趋势生成失败: {e}")
        return "研究趋势分析生成失败，请稍后重试。"

# ---------- 构建图谱 ----------
def build_graph(articles: List[Dict], client: OpenAI = None) -> Tuple[nx.MultiDiGraph, Dict[str, int]]:
    """
    构建知识图谱，包含论文、实体和关键词节点
    
    Returns:
        (知识图谱, 关键词频率字典)
    """
    G = nx.MultiDiGraph()
    triples = []
    entity_frequency = {}
    keyword_frequency = {}  # 统计关键词频率

    def add_or_update_entity(name: str):
        if not name:
            return
        current = entity_frequency.get(name, 0) + 1
        entity_frequency[name] = current
        node_data = G.nodes.get(name, {})
        G.add_node(
            name,
            type="entity",
            weight=current,
            description=node_data.get("description", "")
        )

    def add_or_update_keyword(keyword: str):
        """添加或更新关键词节点"""
        if not keyword or len(keyword.strip()) < 2:
            return
        keyword = keyword.strip()
        current = keyword_frequency.get(keyword, 0) + 1
        keyword_frequency[keyword] = current
        node_data = G.nodes.get(keyword, {})
        G.add_node(
            keyword,
            type="keyword",
            weight=current,
            frequency=current
        )

    def add_weighted_edge(src: str, dst: str, relation: str, edge_type: str, source_article: Optional[str] = None):
        if not src or not dst:
            return
        edge_data = G.get_edge_data(src, dst, default={})
        for key, data in edge_data.items():
            if data.get("relation") == relation and data.get("edge_type") == edge_type:
                data["weight"] = data.get("weight", 1) + 1
                if source_article:
                    sources = data.setdefault("sources", [])
                    if source_article not in sources:
                        sources.append(source_article)
                return
        G.add_edge(
            src,
            dst,
            relation=relation,
            edge_type=edge_type,
            weight=1,
            sources=[source_article] if source_article else []
        )

    print("[*] 开始提取关键词和构建知识图谱...")
    for idx, art in enumerate(articles, 1):
        print(f"[*] 处理文章 {idx}/{len(articles)}: {art['title'][:50]}...")
        article_node = art["title"][:80]
        summary = art["abstract"][:200].replace("\n", " ")
        innovation_score = compute_innovation_score(art["abstract"])
        G.add_node(
            article_node,
            type="article",
            url=art["url"],
            summary=summary,
            full_title=art["title"],
            innovation_score=innovation_score
        )
        
        # 提取关键词
        keywords = extract_keywords(art["abstract"], art["title"], client=client)
        for keyword in keywords:
            add_or_update_keyword(keyword)
            # 连接论文和关键词
            add_weighted_edge(article_node, keyword, "contains_keyword", "keyword", source_article=article_node)
        
        # 提取三元组
        extracted = extract_triples(art["abstract"], art["title"], client=client)
        triples.extend(extracted)
        for triple in extracted:
            if "head" not in triple or "relation" not in triple or "tail" not in triple:
                print("跳过不完整三元组：", triple)
                continue
            head = triple["head"]
            relation = triple["relation"]
            tail = triple["tail"]
            add_or_update_entity(head)
            add_or_update_entity(tail)
            add_weighted_edge(head, tail, relation, "knowledge", source_article=article_node)
            add_weighted_edge(article_node, head, MENTION_RELATION, "mention")
            add_weighted_edge(article_node, tail, MENTION_RELATION, "mention")

    json.dump(triples, open(TRIPLES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return G, keyword_frequency


# ---------- 图谱洞察 ----------
def generate_graph_insights(graph: nx.MultiDiGraph, keyword_frequency: Dict[str, int] = None) -> Dict[str, List[Dict]]:
    summary = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges()
    }
    article_nodes = []
    entity_nodes = []
    keyword_nodes = []
    
    for node, data in graph.nodes(data=True):
        deg = graph.degree(node, weight="weight")
        info = {
            "name": node,
            "type": data.get("type"),
            "degree": deg,
            "weight": data.get("weight", 1),
            "url": data.get("url"),
            "summary": data.get("summary", ""),
            "innovation_score": data.get("innovation_score", 0),
            "frequency": data.get("frequency", 0)
        }
        node_type = data.get("type")
        if node_type == "article":
            article_nodes.append(info)
        elif node_type == "keyword":
            keyword_nodes.append(info)
        else:
            entity_nodes.append(info)

    article_nodes_by_degree = sorted(
        article_nodes,
        key=lambda x: (x["degree"], x.get("innovation_score", 0)),
        reverse=True
    )
    article_nodes_by_innovation = sorted(
        article_nodes,
        key=lambda x: x.get("innovation_score", 0),
        reverse=True
    )
    entity_nodes.sort(key=lambda x: x["degree"], reverse=True)
    keyword_nodes.sort(key=lambda x: (x.get("frequency", 0), x["degree"]), reverse=True)

    knowledge_edges = [
        {
            "src": src,
            "dst": dst,
            "relation": data.get("relation"),
            "weight": data.get("weight", 1)
        }
        for src, dst, data in graph.edges(data=True)
        if data.get("edge_type") == "knowledge"
    ]
    innovation = None
    if knowledge_edges:
        innovation = max(knowledge_edges, key=lambda x: x["weight"])

    # 准备高频关键词列表（用于生成研究趋势）
    hot_keywords = [
        {"keyword": kw["name"], "frequency": kw.get("frequency", 0)}
        for kw in keyword_nodes[:15]
    ]

    return {
        "summary": summary,
        "top_articles": article_nodes_by_degree[:3],
        "top_entities": entity_nodes[:3],
        "top_keywords": keyword_nodes[:10],
        "hot_keywords": hot_keywords,
        "innovation_edge": innovation,
        "top_innovations": article_nodes_by_innovation[:3]
    }


# ---------- HTML 注入 ----------
def inject_graph_description(graph_file: str, description: str, insights: Dict[str, List[Dict]], 
                             research_trends: str = "") -> None:
    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return
    if "kg-description" in content:
        return
    summary = insights.get("summary", {})
    top_articles = insights.get("top_articles", [])
    top_entities = insights.get("top_entities", [])
    top_keywords = insights.get("top_keywords", [])
    innovation = insights.get("innovation_edge")
    top_innovations = insights.get("top_innovations", [])

    def render_list(items: List[Dict], empty_text: str) -> str:
        if not items:
            return f"<li>{empty_text}</li>"
        rows = []
        for item in items:
            name = html.escape(item["name"])
            degree = item.get("degree", 0)
            extra = ""
            if item.get("type") == "article" and item.get("url"):
                extra = f'<a href="{html.escape(item["url"])}" target="_blank">原文链接</a>'
            innovation_badge = ""
            score = item.get("innovation_score", 0)
            if score and score > 0:
                innovation_badge = f"<span style='color:#f97316;'>· 创新度 {score:.2f}</span>"
            frequency_badge = ""
            if item.get("type") == "keyword":
                freq = item.get("frequency", 0)
                frequency_badge = f"<span style='color:#9333ea;'>· 出现{freq}次</span>"
            rows.append(f"<li><strong>{name}</strong>（连接度 {degree}） {innovation_badge} {frequency_badge} {extra}</li>")
        return "".join(rows)
    
    def render_keywords(keywords: List[Dict]) -> str:
        if not keywords:
            return "<p>暂无关键词数据</p>"
        rows = []
        for kw in keywords[:10]:
            name = html.escape(kw["name"])
            freq = kw.get("frequency", 0)
            rows.append(f"<span style='display:inline-block;margin:4px 8px;padding:6px 12px;background:#f3e8ff;border-radius:6px;color:#7c3aed;'>{name} ({freq})</span>")
        return "<div style='margin-top:8px;'>" + "".join(rows) + "</div>"

    innovation_text = ""
    if innovation:
        innovation_text = (
            f"<p><strong>图谱创新亮点：</strong>实体 <em>{html.escape(innovation['src'])}</em> "
            f"与 <em>{html.escape(innovation['dst'])}</em> 之间的关系 "
            f"<em>{html.escape(innovation['relation'] or '关联')}</em> 出现频次最高"
            f"（权重 {innovation['weight']}），代表当前主题下最重要的语义联结。</p>"
        )

    desc_html = f"""
    <section id="kg-description" style="padding:20px 32px;margin:20px auto;max-width:1200px;
    background:#ffffff;border-radius:16px;box-shadow:0 12px 32px rgba(15,23,42,0.08);font-family:'Inter',Arial,sans-serif;">
        <h2 style="margin-top:0;font-size:24px;color:#111827;">知识图谱说明</h2>
        <p style="font-size:15px;line-height:1.7;color:#374151;">{description}</p>
        <div style="display:flex;gap:24px;flex-wrap:wrap;margin:12px 0;">
            <div style="flex:1;min-width:220px;">
                <h3 style="font-size:18px;color:#111827;margin-bottom:4px;">概览</h3>
                <p style="margin:4px 0;color:#4b5563;">节点：{summary.get("node_count", 0)} 个</p>
                <p style="margin:4px 0;color:#4b5563;">关系：{summary.get("edge_count", 0)} 条</p>
            </div>
            <div style="flex:1;min-width:220px;">
                <h3 style="font-size:18px;color:#111827;margin-bottom:4px;">重要论文节点</h3>
                <ul style="padding-left:18px;margin:0;color:#4b5563;">
                    {render_list(top_articles, "暂无数据")}
                </ul>
            </div>
            <div style="flex:1;min-width:220px;">
                <h3 style="font-size:18px;color:#111827;margin-bottom:4px;">关键实体节点</h3>
                <ul style="padding-left:18px;margin:0;color:#4b5563;">
                    {render_list(top_entities, "暂无数据")}
                </ul>
            </div>
            <div style="flex:1;min-width:220px;">
                <h3 style="font-size:18px;color:#111827;margin-bottom:4px;">创新亮点论文</h3>
                <ul style="padding-left:18px;margin:0;color:#4b5563;">
                    {render_list(top_innovations, "暂无数据")}
                </ul>
            </div>
        </div>
        <div style="margin-top:20px;padding:16px;background:#f9fafb;border-radius:8px;">
            <h3 style="font-size:18px;color:#111827;margin-bottom:8px;">🔑 高频关键词</h3>
            {render_keywords(top_keywords)}
            <p style="margin-top:12px;font-size:14px;color:#6b7280;">紫色节点表示从论文中提取的关键词，节点大小反映关键词出现频率。</p>
        </div>
        {innovation_text}
        {f'<div style="margin-top:24px;padding:20px;background:#fef3c7;border-left:4px solid #f59e0b;border-radius:8px;"><h3 style="font-size:18px;color:#111827;margin-top:0;margin-bottom:12px;">🔮 未来研究趋势分析</h3><div style="font-size:15px;line-height:1.8;color:#374151;white-space:pre-wrap;">{html.escape(research_trends)}</div></div>' if research_trends else ''}
        <ul style="padding-left:20px;font-size:14px;line-height:1.6;color:#4b5563;margin-top:20px;">
            <li>节点大小与出现频次相关，越大代表被引用或提及越多。</li>
            <li><strong>节点颜色说明：</strong>红色=论文，蓝色=实体，紫色=关键词，橙色=高创新度论文</li>
            <li>悬停可查看详细信息；点击节点可固定位置，方便分析局部结构。</li>
            <li>若图谱过于密集，可使用左上角的导航按钮或鼠标滚轮放大缩小。</li>
        </ul>
    </section>
    """
    updated = content.replace("</body>", f"{desc_html}\n</body>", 1)
    with open(graph_file, "w", encoding="utf-8") as f:
        f.write(updated)


# ---------- 可视化 ----------
def visualize(graph: nx.MultiDiGraph, graph_file: str = "knowledge_graph.html"):
    net = Network(height="850px", width="100%", bgcolor="#f7f8fb", font_color="#1f1f1f")
    color_map = {"entity": "#6c6cff", "article": "#cc6666", "keyword": "#9333ea"}  # 关键词用紫色
    edge_color_map = {"knowledge": "#1f78b4", "mention": "#33a02c", "keyword": "#a855f7"}  # 关键词边用紫色
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "scaling": {
          "min": 10,
          "max": 45
        },
        "font": {
          "size": 16,
          "face": "Inter, Arial"
        },
        "borderWidth": 1
      },
      "edges": {
        "smooth": {
          "type": "dynamic",
          "roundness": 0.4
        },
        "color": {
          "inherit": false
        },
        "width": 1.5,
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.7
          }
        }
      },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "navigationButtons": true,
        "tooltipDelay": 120
      },
      "physics": {
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -22000,
          "springLength": 180,
          "springConstant": 0.045,
          "damping": 0.12
        },
        "stabilization": {
          "iterations": 250
        }
      }
    }
    """)
    for node, data in graph.nodes(data=True):
        print(f"添加节点 {node} 类型 {data.get('type', '未知类型')}")
        weight = data.get("weight", 1)
        size = 15 + min(weight, 10)
        node_type = data.get("type", "unknown")
        tooltip_lines = [
            f"类型: {node_type}",
            f"出现次数: {weight}"
        ]
        if node_type == "article":
            tooltip_lines.append(f"原始标题: {html.escape(data.get('full_title', node))}")
            summary = html.escape(data.get("summary", ""))
            if summary:
                tooltip_lines.append(f"摘要: {summary}...")
            url = data.get("url")
            if url:
                tooltip_lines.append(f"链接: {url}")
            innovation_score = data.get("innovation_score", 0)
            if innovation_score:
                tooltip_lines.append(f"创新度: {innovation_score:.2f}")
        elif node_type == "keyword":
            frequency = data.get("frequency", weight)
            tooltip_lines.append(f"关键词频率: {frequency}")
            tooltip_lines.append("该关键词在论文中出现次数")
        else:
            description = html.escape(data.get("description", ""))
            if description:
                tooltip_lines.append(f"描述: {description}")
        node_color = color_map.get(node_type, "gray")
        if node_type == "article" and data.get("innovation_score", 0) > 0:
            node_color = "#ff914d"
        net.add_node(
            node,
            label=node,
            color=node_color,
            value=size,
            title="<br>".join(tooltip_lines)
        )
    for src, dst, data in graph.edges(data=True):
        relation = data.get("relation", "未知关系")
        edge_type = data.get("edge_type", "knowledge")
        weight = data.get("weight", 1)
        print(f"添加边 {src} -> {dst} 关系 {relation}")
        sources = data.get("sources") or []
        tooltip = [
            f"关系: {html.escape(relation)}",
            f"类型: {edge_type}",
            f"权重: {weight}"
        ]
        if sources:
            tooltip.append("来源文章: " + ", ".join(html.escape(s) for s in sources[:5]))
        # 关键词边使用虚线
        is_dashed = edge_type in ["mention", "keyword"]
        net.add_edge(
            src,
            dst,
            title="<br>".join(tooltip),
            color=edge_color_map.get(edge_type, "#555555"),
            value=weight,
            width=1 + min(weight, 5),
            dashes=is_dashed,
            smooth=is_dashed
        )
    net.write_html(graph_file)


# ---------- 入口 ---------
def main():

    client = OpenAI(
        api_key=BAILIAN_KEY,
        base_url=BAILIAN_BASE,
        timeout=60)
    parser = argparse.ArgumentParser(description="爬虫 + 智能体知识图谱")
    parser.add_argument("--key", default=DEFAULT_KEY,
                        help="直接输入检索关键词（空格分隔）；默认 %(default)s")
    parser.add_argument("--title", default=DEFAULT_TITLE,
                        help="用于检索的论文标题（可选，与摘要搭配）")
    parser.add_argument("--abstract", default=DEFAULT_ABSTRACT,
                        help="用于检索的论文摘要内容（可选，与标题搭配）")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES,
                        help="翻页次数；默认 %(default)s")
    args = parser.parse_args()

    keywords = build_keywords_from_arg(args.key)
    if not keywords:
        keywords = build_keywords_from_text(args.title, args.abstract)
    if not keywords:
        parser.error("请提供关键词，或提供论文标题与摘要用于检索")

    # 1. 爬取（使用 ArXivCrawler）
    crawler = ArXivCrawler()
    articles = crawler.crawl(keywords, args.pages, MATCH_THRESHOLD)
    if not articles:
        print(f"[!] 未匹配到文章，尝试将阈值降低到 {FALLBACK_THRESHOLD} 重新搜索")
        articles = crawler.crawl(keywords, args.pages, FALLBACK_THRESHOLD)
    for art in articles:
        art.pop("score", None)
    save_json(SAVE_JSON, articles)
    save_md(SAVE_MD, articles)
    print(f"[√] 共 {len(articles)} 篇文章已保存")

    # 2. 构建图谱
    print("开始构建图谱（:")
    top10 = articles[:10]
    graph, keyword_frequency = build_graph(top10, client)
    print("graph制作中")
    visualize(graph)
    
    # 3. 生成图谱洞察和研究趋势
    print("[*] 生成图谱洞察...")
    insights = generate_graph_insights(graph, keyword_frequency)
    
    # 4. 根据高频关键词生成未来研究趋势讨论
    hot_keywords = insights.get("hot_keywords", [])
    research_trends = ""
    if hot_keywords:
        print("[*] 根据高频关键词生成未来研究趋势分析...")
        research_trends = generate_research_trends(hot_keywords, client)
    
    # 5. 注入描述和研究趋势到HTML
    inject_graph_description(GRAPH_FILE, GRAPH_DESCRIPTION, insights, research_trends)
    
    print(f"[√] 知识图谱已生成：{GRAPH_FILE}")
    print(f"[√] 三元组已保存：{TRIPLES_FILE}")
    if hot_keywords:
        print(f"[√] 高频关键词统计完成，共 {len(hot_keywords)} 个关键词")
        print(f"[√] 研究趋势分析已生成")


if __name__ == "__main__":
    main()