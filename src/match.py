"""检索模块 — 混合检索（sqlite-vec 向量 + FTS5 BM25），RRF 融合 + Reranker 精排"""

import logging
import sqlite3
import struct
import threading

import sqlite_vec
import jieba

from .config import DB_PATH, SEARCH_TOP_K, RECALL_TOP_K, RRF_RANK_CONSTANT, KNN_WEIGHT, BM25_WEIGHT, RERANKER_TOP_K
from .models import get_embedding, rerank

logger = logging.getLogger(__name__)

# SQLite 连接单例（线程安全）
_local = threading.local()


class RetrievalError(RuntimeError):
    """检索链路失败时抛出，避免把系统错误伪装成无结果"""


def get_db() -> sqlite3.Connection:
    """获取当前线程的 SQLite 连接"""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(DB_PATH)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def _serialize_vector(vec: list[float]) -> bytes:
    """将 float 列表序列化为 sqlite-vec 需要的 bytes 格式"""
    return struct.pack(f"{len(vec)}f", *vec)


def _rrf_score(rank: int, k: int = RRF_RANK_CONSTANT) -> float:
    """计算 RRF 分数: 1 / (k + rank)"""
    return 1.0 / (k + rank)


def _knn_search(query_vector: list[float], top_k: int) -> list[dict]:
    """sqlite-vec 向量检索"""
    db = get_db()
    query_bytes = _serialize_vector(query_vector)
    rows = db.execute(
        """
        SELECT d.rowid, d.text_content, d.chapter, d.chapter_title, d.source_file,
               v.distance
        FROM vec_chunks v
        JOIN chunks d ON d.rowid = v.rowid
        WHERE v.embedding MATCH ?
          AND v.k = ?
        ORDER BY v.distance
        """,
        (query_bytes, top_k),
    ).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row["rowid"],
            "content": row["text_content"],
            "chapter": row["chapter"] or "",
            "chapter_title": row["chapter_title"] or "",
            "source_file": row["source_file"] or "",
            "knn_score": 1.0 - row["distance"],  # cosine distance -> similarity
        })
    return results


def _bm25_search(query_text: str, top_k: int) -> list[dict]:
    """FTS5 BM25 全文检索 — jieba 分词后用 OR 连接构造查询"""
    db = get_db()
    # jieba 分词，过滤单字停用词，用 OR 连接以兼容 FTS5 unicode61 分词器
    words = [w for w in jieba.cut(query_text) if len(w) > 1]
    if not words:
        return []
    segmented = " OR ".join(words)
    rows = db.execute(
        """
        SELECT d.rowid, d.text_content, d.chapter, d.chapter_title, d.source_file,
               fts.rank AS bm25_score
        FROM chunks_fts fts
        JOIN chunks d ON d.rowid = fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY fts.rank
        LIMIT ?
        """,
        (segmented, top_k),
    ).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row["rowid"],
            "content": row["text_content"],
            "chapter": row["chapter"] or "",
            "chapter_title": row["chapter_title"] or "",
            "source_file": row["source_file"] or "",
            "bm25_score": -row["bm25_score"],  # FTS5 rank 是负数，越小越好
        })
    return results


def _merge_results(
    knn_results: list[dict],
    bm25_results: list[dict],
    knn_weight: float = KNN_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> list[dict]:
    """手动 RRF 融合：对每个文档按排名计算加权 RRF 分数"""
    doc_scores: dict[int, dict] = {}

    for rank, doc in enumerate(knn_results, 1):
        doc_id = doc["id"]
        score = knn_weight * _rrf_score(rank)
        doc_scores[doc_id] = {
            "id": doc_id,
            "content": doc["content"],
            "chapter": doc["chapter"],
            "chapter_title": doc["chapter_title"],
            "source_file": doc["source_file"],
            "final_score": score,
        }

    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc["id"]
        score = bm25_weight * _rrf_score(rank)
        if doc_id in doc_scores:
            doc_scores[doc_id]["final_score"] += score
        else:
            doc_scores[doc_id] = {
                "id": doc_id,
                "content": doc["content"],
                "chapter": doc["chapter"],
                "chapter_title": doc["chapter_title"],
                "source_file": doc["source_file"],
                "final_score": score,
            }

    return sorted(doc_scores.values(), key=lambda x: x["final_score"], reverse=True)


def search(query_text: str, top_k: int = SEARCH_TOP_K) -> list[dict]:
    """混合检索 + 精排：KNN → BM25 → RRF融合 → Reranker精排"""
    try:
        # 1. 向量化查询（使用 query prompt）
        query_vector = get_embedding(query_text, prompt_name="query")
        if not query_vector:
            return []

        # 2. 双路召回
        fetch_size = RECALL_TOP_K
        knn_results = _knn_search(query_vector, fetch_size)
        bm25_results = _bm25_search(query_text, fetch_size)

        logger.info(f"KNN 命中 {len(knn_results)} 条, BM25 命中 {len(bm25_results)} 条")

        # 3. RRF 融合
        merged = _merge_results(knn_results, bm25_results)

        # 4. Reranker 精排
        # 取 RRF 融合后的候选文档，送入精排模型
        recall_docs = merged[:RECALL_TOP_K]
        recall_texts = [doc["content"] for doc in recall_docs]
        ranked = rerank(query_text, recall_texts, top_k=RERANKER_TOP_K)

        # 5. 将精排结果与原始文档元数据关联
        # reranker 只返回文本，按文本队列保留重复片段对应的元数据
        content_to_meta: dict[str, list[dict]] = {}
        for doc in recall_docs:
            content_to_meta.setdefault(doc["content"], []).append(doc)

        results = []
        for item in ranked:
            text = item["text"]
            metas = content_to_meta.get(text, [])
            meta = metas.pop(0) if metas else {}
            results.append({
                "id": meta.get("id"),
                "score": item["score"],
                "content": text,
                "chapter": meta.get("chapter", ""),
                "chapter_title": meta.get("chapter_title", ""),
                "source_file": meta.get("source_file", ""),
            })

        return results[:top_k]
    except Exception as e:
        logger.exception("检索失败")
        raise RetrievalError(f"检索失败: {e}") from e
