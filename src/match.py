"""检索模块 — 混合检索（KNN 向量 + BM25 文本），手动 RRF 融合排序"""

import logging
import threading

from elasticsearch import Elasticsearch

from .config import ES_URL, ES_INDEX_NAME, SEARCH_TOP_K, SEARCH_NUM_CANDIDATES, RRF_RANK_CONSTANT, KNN_WEIGHT, BM25_WEIGHT
from .embedding import get_embedding

logger = logging.getLogger(__name__)

# ES 客户端单例：首次调用时创建，之后复用
_es_client: Elasticsearch | None = None
_es_client_lock = threading.Lock()


def get_es_client() -> Elasticsearch:
    """获取 ES 客户端单例"""
    global _es_client
    if _es_client is None:
        with _es_client_lock:
            if _es_client is None:
                _es_client = Elasticsearch(ES_URL)
                if not _es_client.ping():
                    raise ConnectionError(f"无法连接到 Elasticsearch: {ES_URL}")
    return _es_client


# 兼容旧接口
def create_es_client() -> Elasticsearch:
    """初始化 ES 客户端（已废弃，保留兼容性）"""
    return get_es_client()


def _rrf_score(rank: int, k: int = RRF_RANK_CONSTANT) -> float:
    """计算 RRF 分数: 1 / (k + rank)"""
    return 1.0 / (k + rank)


def _knn_search(
    client: Elasticsearch,
    query_vector: list[float],
    top_k: int,
    num_candidates: int,
    chapter_filter: str | None = None,
) -> list[dict]:
    """纯 KNN 向量检索"""
    knn_query = {
        "field": "text_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": num_candidates,
    }
    if chapter_filter:
        knn_query["filter"] = {"term": {"chapter": chapter_filter}}

    body = {
        "knn": knn_query,
        "_source": ["text_content", "chapter", "chapter_title", "source_file"],
        "size": top_k,
    }
    response = client.search(index=ES_INDEX_NAME, body=body)
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "id": hit["_id"],
            "content": source["text_content"],
            "chapter": source.get("chapter", ""),
            "chapter_title": source.get("chapter_title", ""),
            "source_file": source.get("source_file", ""),
            "knn_score": hit["_score"],
        })
    return results


def _bm25_search(
    client: Elasticsearch,
    query_text: str,
    top_k: int,
    chapter_filter: str | None = None,
) -> list[dict]:
    """纯 BM25 文本检索"""
    # ik_smart 分词后词条数较多，minimum_should_match 过高会导致大量相关文档被过滤
    # 设为 30% 在召回率�精度之间取得平衡
    must = {
        "match": {
            "text_content": {
                "query": query_text,
                "operator": "or",
                "minimum_should_match": "30%",
            }
        }
    }
    query = {"bool": {"must": must}}
    if chapter_filter:
        query["bool"]["filter"] = {"term": {"chapter": chapter_filter}}

    body = {
        "query": query,
        "_source": ["text_content", "chapter", "chapter_title", "source_file"],
        "size": top_k,
    }
    response = client.search(index=ES_INDEX_NAME, body=body)
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "id": hit["_id"],
            "content": source["text_content"],
            "chapter": source.get("chapter", ""),
            "chapter_title": source.get("chapter_title", ""),
            "source_file": source.get("source_file", ""),
            "bm25_score": hit["_score"],
        })
    return results


def _merge_results(
    knn_results: list[dict],
    bm25_results: list[dict],
    knn_weight: float = KNN_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> list[dict]:
    """手动 RRF 融合：对每个文档按排名计算加权 RRF 分数"""
    doc_scores: dict[str, dict] = {}

    # KNN 排名贡献
    for rank, doc in enumerate(knn_results, 1):
        doc_id = doc["id"]
        score = knn_weight * _rrf_score(rank)
        doc_scores[doc_id] = {
            "content": doc["content"],
            "chapter": doc["chapter"],
            "chapter_title": doc["chapter_title"],
            "source_file": doc.get("source_file", ""),
            "knn_score": doc.get("knn_score", 0),
            "bm25_score": 0,
            "final_score": score,
        }

    # BM25 排名贡献
    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc["id"]
        score = bm25_weight * _rrf_score(rank)
        if doc_id in doc_scores:
            doc_scores[doc_id]["final_score"] += score
            doc_scores[doc_id]["bm25_score"] = doc.get("bm25_score", 0)
        else:
            doc_scores[doc_id] = {
                "content": doc["content"],
                "chapter": doc["chapter"],
                "chapter_title": doc["chapter_title"],
                "source_file": doc.get("source_file", ""),
                "knn_score": 0,
                "bm25_score": doc.get("bm25_score", 0),
                "final_score": score,
            }

    # 按融合分数降序排列
    merged = sorted(doc_scores.values(), key=lambda x: x["final_score"], reverse=True)
    return merged


def search(
    query_text: str,
    top_k: int = SEARCH_TOP_K,
    num_candidates: int = SEARCH_NUM_CANDIDATES,
    chapter_filter: str | None = None,
) -> list[dict]:
    """混合检索：KNN 向量检索 + BM25 文本匹配，手动 RRF 融合排序

    Args:
        query_text: 用户查询文本
        top_k: 返回最相关的前 K 条结果
        num_candidates: KNN 粗筛候选数
        chapter_filter: 可选，按章节号过滤（如 "第1章"）

    Returns:
        匹配结果列表，每项包含 score / content / chapter / chapter_title
    """
    query_vector = get_embedding(query_text)
    if not query_vector:
        return []

    try:
        client = get_es_client()

        # 修复：使用固定的较大 fetch_size，不再用 top_k * 3
        # 原因：确保两路检索都有足够候选，让融合排序有更多机会
        fetch_size = 50
        knn_results = _knn_search(client, query_vector, fetch_size, num_candidates, chapter_filter)
        bm25_results = _bm25_search(client, query_text, fetch_size, chapter_filter)

        logger.info(f"KNN 命中 {len(knn_results)} 条, BM25 命中 {len(bm25_results)} 条")

        # 手动 RRF 融合
        merged = _merge_results(knn_results, bm25_results)

        # 返回 top_k 结果
        results = []
        for doc in merged[:top_k]:
            results.append({
                "score": doc["final_score"],
                "content": doc["content"],
                "chapter": doc["chapter"],
                "chapter_title": doc["chapter_title"],
                "source_file": doc.get("source_file", ""),
            })
        return results
    except Exception as e:
        logger.error(f"ES 检索失败: {e}")
        return []


if __name__ == "__main__":
    # 统一配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    user_question = "红尘中一二等富贵温柔之地是指哪里？"
    logger.info(f"用户提问: '{user_question}'")

    matched = search(user_question, top_k=3)

    print(f"\n检索完成，找到 {len(matched)} 个匹配片段:\n")
    for idx, doc in enumerate(matched):
        chapter_info = f" [{doc['chapter']} {doc['chapter_title']}]" if doc["chapter"] else ""
        print(f"【匹配段落 {idx+1}】(融合分数: {doc['score']:.4f}){chapter_info}")
        print(f"{doc['content'].strip()[:200]}...")
        print("-" * 50)