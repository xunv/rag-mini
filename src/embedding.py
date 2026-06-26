"""公共嵌入模块 — 通过 Ollama API 进行向量化"""

import logging

import requests

from .config import OLLAMA_BASE_URL, EMBED_MODEL_NAME

logger = logging.getLogger(__name__)

OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"


def get_embedding(text: str) -> list[float]:
    """获单条文本的向量"""
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL_NAME, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Ollama embedding 失败: {data['error']}")
    return data["embeddings"][0]


def get_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """批量获取向量，Ollama /api/embed 支持传入多条 input"""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL_NAME, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Ollama embedding 失败: {data['error']}")
        all_vectors.extend(data["embeddings"])

        if len(texts) > batch_size:
            logger.info(f"  向量化进度: {min(i + batch_size, len(texts))}/{len(texts)}")

    return all_vectors