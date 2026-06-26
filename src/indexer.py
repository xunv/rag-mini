"""索引构建脚本 — 扫描 docs/ 目录，支持 txt/pdf，增量入库"""

import os
import re
import hashlib
import logging
import argparse

# 兼容直接运行 (python src/indexer.py) 和模块运行 (python -m src.indexer)
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

import fitz  # pymupdf
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    ES_INDEX_NAME,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    VECTOR_DIMS,
)
from .embedding import get_embeddings_batch
from .match import get_es_client

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


# ==================== 索引管理 ====================


def ensure_index(client: Elasticsearch, rebuild: bool = False) -> None:
    """确保向量索引存在，rebuild 时先删除旧索引"""
    if rebuild and client.indices.exists(index=ES_INDEX_NAME):
        client.indices.delete(index=ES_INDEX_NAME)
        logger.info(f"已删除旧索引 '{ES_INDEX_NAME}'")

    if client.indices.exists(index=ES_INDEX_NAME):
        logger.info(f"索引 '{ES_INDEX_NAME}' 已存在，跳过创建")
        return

    index_mapping = {
        "mappings": {
            "properties": {
                "text_content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart",
                },
                "chapter": {"type": "keyword"},
                "chapter_title": {"type": "keyword"},
                "text_vector": {
                    "type": "dense_vector",
                    "dims": VECTOR_DIMS,
                    "index": True,
                    "similarity": "cosine",
                },
                "source_file": {"type": "keyword"},
                "file_hash": {"type": "keyword"},
            }
        }
    }
    client.indices.create(index=ES_INDEX_NAME, body=index_mapping)
    logger.info(f"索引 '{ES_INDEX_NAME}' 创建成功")


# ==================== 文件扫描 ====================


def scan_files(data_dir: str) -> list[str]:
    """扫描目录下所有支持的文件"""
    files = []
    for entry in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(entry)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(data_dir, entry))
    logger.info(f"扫描到 {len(files)} 个文件: {[os.path.basename(f) for f in files]}")
    return files


def compute_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希，用于增量判断"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_indexed_file_hashes(client: Elasticsearch) -> dict[str, str]:
    """查询 ES 中已索引文件的哈希值，返回 {source_file: file_hash}"""
    if not client.indices.exists(index=ES_INDEX_NAME):
        return {}

    # 聚合查询每个 source_file 的 file_hash
    resp = client.search(
        index=ES_INDEX_NAME,
        body={
            "size": 0,
            "aggs": {
                "files": {
                    "terms": {"field": "source_file", "size": 1000},
                    "aggs": {
                        "hash": {"terms": {"field": "file_hash", "size": 1}}
                    },
                }
            },
        },
    )
    result = {}
    for bucket in resp["aggregations"]["files"]["buckets"]:
        filename = bucket["key"]
        hash_buckets = bucket["hash"]["buckets"]
        if hash_buckets:
            result[filename] = hash_buckets[0]["key"]
    return result


def delete_file_docs(client: Elasticsearch, source_file: str) -> int:
    """删除某个文件的所有文档"""
    resp = client.delete_by_query(
        index=ES_INDEX_NAME,
        body={"query": {"term": {"source_file": source_file}}},
        refresh=True,
    )
    deleted = resp.get("deleted", 0)
    logger.info(f"已删除文件 '{source_file}' 的 {deleted} 条旧文档")
    return deleted


# ==================== 文档加载 ====================


def extract_chapter_info(line: str) -> tuple[str, str] | None:
    """从行文本中提取章节号和标题，如 '第1章 甄士隐梦幻识通灵 ...'"""
    match = re.match(r"第(\d+)章\s+(.+?)(?:\r|\n|$)", line.strip())
    if match:
        return f"第{match.group(1)}章", match.group(2).strip()
    return None


def load_txt(file_path: str) -> list[dict]:
    """读取 txt 文件，按章节分段再切片"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 按章节标题分段
    chapter_blocks: list[tuple[str, str, str]] = []
    current_chapter = ""
    current_title = ""
    current_lines: list[str] = []

    for line in raw_text.split("\n"):
        info = extract_chapter_info(line)
        if info:
            if current_chapter and current_lines:
                chapter_blocks.append((current_chapter, current_title, "\n".join(current_lines)))
            current_chapter, current_title = info
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_chapter and current_lines:
        chapter_blocks.append((current_chapter, current_title, "\n".join(current_lines)))

    # 如果没有章节结构，整体作为一个块
    if not chapter_blocks:
        chapter_blocks = [("", "", raw_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    results = []
    for chapter, title, content in chapter_blocks:
        chunks = splitter.split_text(content)
        for chunk in chunks:
            if not chunk.strip():
                continue
            results.append({"text": chunk, "chapter": chapter, "chapter_title": title})

    return results


def load_pdf(file_path: str) -> list[dict]:
    """读取 PDF 文件，按页提取文本再切片"""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            full_text += text + "\n"
    doc.close()

    if not full_text.strip():
        logger.warning(f"PDF 文件 '{file_path}' 未提取到文本内容")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )

    chunks = splitter.split_text(full_text)
    results = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        results.append({"text": chunk, "chapter": "", "chapter_title": ""})

    return results


def load_file(file_path: str) -> list[dict]:
    """根据文件扩展名选择加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


# ==================== 索引写入 ====================


def index_file(client: Elasticsearch, file_path: str, file_hash: str, batch_size: int = 100) -> int:
    """对单个文件进行切片、向量化、写入 ES"""
    source_file = os.path.basename(file_path)
    logger.info(f"正在处理文件: {source_file}")

    chunks = load_file(file_path)
    if not chunks:
        logger.warning(f"文件 '{source_file}' 无有效内容，跳过")
        return 0

    logger.info(f"  切分为 {len(chunks)} 个片段，开始向量化...")

    texts = [c["text"] for c in chunks]
    vectors = get_embeddings_batch(texts, batch_size=32)

    logger.info(f"  向量化完成，写入 ES...")

    # 用 source_file + 序号作为文档 ID，方便增量更新
    def gen_bulk_actions():
        for i, (chunk_data, vector) in enumerate(zip(chunks, vectors)):
            yield {
                "_index": ES_INDEX_NAME,
                "_id": f"{source_file}_{i}",
                "_source": {
                    "text_content": chunk_data["text"],
                    "text_vector": vector,
                    "chapter": chunk_data["chapter"],
                    "chapter_title": chunk_data["chapter_title"],
                    "source_file": source_file,
                    "file_hash": file_hash,
                },
            }

    success, failed = bulk(client, gen_bulk_actions(), chunk_size=batch_size)
    if failed:
        for item in failed[:5]:
            logger.error(f"  写入失败: {item}")

    logger.info(f"  写入完成: {success} 条成功")
    return success


# ==================== 主流程 ====================


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="RAG 索引构建工具（支持增量入库）")
    parser.add_argument("--rebuild", action="store_true", help="删除旧索引并全量重建")
    parser.add_argument("--dir", default=DATA_DIR, help=f"文档目录（默认: {DATA_DIR}）")
    args = parser.parse_args()

    client = get_es_client()
    ensure_index(client, rebuild=args.rebuild)

    # 扫描文件
    files = scan_files(args.dir)
    if not files:
        logger.warning("未找到任何支持的文件")
        return

    # 获取已索引文件的哈希
    indexed_hashes = get_indexed_file_hashes(client) if not args.rebuild else {}

    total_indexed = 0
    skipped = 0

    for file_path in files:
        source_file = os.path.basename(file_path)
        file_hash = compute_file_hash(file_path)

        # 增量判断：哈希一致则跳过
        if source_file in indexed_hashes and indexed_hashes[source_file] == file_hash:
            logger.info(f"文件 '{source_file}' 未变更，跳过")
            skipped += 1
            continue

        # 文件有变更，先删除旧文档再重新入库
        if source_file in indexed_hashes:
            delete_file_docs(client, source_file)

        count = index_file(client, file_path, file_hash)
        total_indexed += count

    # 刷新索引使文档可搜索
    client.indices.refresh(index=ES_INDEX_NAME)

    logger.info(f"\n入库完成: 新增/更新 {total_indexed} 条，跳过 {skipped} 个未变更文件")


if __name__ == "__main__":
    main()