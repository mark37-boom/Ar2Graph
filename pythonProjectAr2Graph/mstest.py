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
from typing import List, Dict
from urllib.parse import urljoin, quote_plus
import difflib
import openai
import networkx as nx
from pyvis.network import Network
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# ---------- 配置 ----------
BASE_URL = "https://arxiv.org"
SEARCH_TMP = "/search/?query={kw}&searchtype=all&start={start}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FuzzyBot/0.1)"}
SLEEP = 1
SAVE_JSON = "articles_all.json"
SAVE_MD = "articles_all.md"
DEFAULT_KEY = "Wind Turbine Blade Defect Detection"
DEFAULT_PAGES = 1
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


# ---------- 多关键词 OR 匹配 ----------
def fuzzy_match(text: str, keywords: List[str]) -> bool:
    return any(k.lower() in text.lower() for k in keywords)


def similarity(text: str, keywords: List[str]) -> float:
    key_text = " ".join(keywords).lower()
    return difflib.SequenceMatcher(None, key_text, text.lower()).ratio()


# ---------- 爬虫 ----------
def parse_one_page(html: str, keywords: List[str]) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    articles = []
    for entry in soup.select("li.arxiv-result"):
        title_tag = entry.select_one("p.title")
        abs_tag = entry.select_one("p.abstract")
        link_tag = entry.select_one("a[href*='/abs/']")
        if not all([title_tag, abs_tag, link_tag]):
            continue
        title = title_tag.get_text(strip=True)
        abstract = abs_tag.get_text(strip=True)
        if not fuzzy_match(title + " " + abstract, keywords):
            continue
        link = urljoin(BASE_URL, link_tag["href"])
        score = similarity(title + " " + abstract, keywords)
        articles.append({"title": title, "abstract": abstract,
                         "url": link, "score": score})
    return articles


def crawl(keywords: List[str], pages: int) -> List[Dict]:
    all_articles = []
    kw_str = " ".join(keywords)
    for p in range(pages):
        start = p * 50
        url = BASE_URL + SEARCH_TMP.format(kw=quote_plus(kw_str), start=start)
        print(f"[+] 抓取 {url}")
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        all_articles.extend(parse_one_page(resp.text, keywords))
        time.sleep(SLEEP)
    all_articles.sort(key=lambda x: x["score"], reverse=True)
    return all_articles


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

# ---------- 构建图谱 ----------
def build_graph(articles: List[Dict],client: OpenAI = None) -> nx.DiGraph:
    G = nx.DiGraph()
    triples = []
    for art in articles:
        node = art["title"][:50] + "..."
        G.add_node(node, type="article", url=art["url"])
        triples += extract_triples(art["abstract"], art["title"],client=client)

    for t in triples:
        if "head" not in t or "relation" not in t or "tail" not in t:
            print("跳过不完整三元组：", t)
            continue
        h, r, ta = t["head"], t["relation"], t["tail"]
        G.add_node(h, type="entity")
        G.add_node(ta, type="entity")
        G.add_edge(h, ta, relation=r)

    json.dump(triples, open(TRIPLES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return G


# ---------- 可视化 ----------
def visualize(graph: nx.DiGraph, graph_file: str = "knowledge_graph.html"):
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
    color_map = {"entity": "#666ccff", "article": "#cc666"}
    for node, data in graph.nodes(data=True):
        print(f"添加节点 {node} 类型 {data.get('type', '未知类型')}")
        net.add_node(node, label=node, color=color_map.get(data.get("type"), "gray"))
    for src, dst, data in graph.edges(data=True):
        print(f"添加边 {src} -> {dst} 关系 {data.get('relation', '未知关系')}")
        net.add_edge(src, dst, title=data.get("relation", ""))
    net.write_html(graph_file)


# ---------- 入口 ---------
def main():

    client = OpenAI(
        api_key=BAILIAN_KEY,
        base_url=BAILIAN_BASE,
        timeout=60)
    parser = argparse.ArgumentParser(description="爬虫 + 智能体知识图谱")
    parser.add_argument("--key", nargs="?", default=DEFAULT_KEY,
                        help='多个关键词用空格分隔，例如 "slam vision"；默认 %(default)s')
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES,
                        help="翻页次数；默认 %(default)s")
    args = parser.parse_args()

    # 1. 爬取
    articles = crawl(args.key.split(), args.pages)
    for art in articles:
        art.pop("score", None)
    save_json(SAVE_JSON, articles)
    save_md(SAVE_MD, articles)
    print(f"[√] 共 {len(articles)} 篇文章已保存")

    # 2. 构建图谱
    print("开始构建图谱（:")
    top10 = articles[:10]
    graph = build_graph(top10,client)
    print("graph制作中")
    visualize(graph)
    print(f"[√] 知识图谱已生成：{GRAPH_FILE}")
    print(f"[√] 三元组已保存：{TRIPLES_FILE}")


if __name__ == "__main__":
    main()