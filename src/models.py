"""统一模型调用模块 — 支持 local / ollama 两种后端

所有模型（Embedding、Reranker、LLM）的调用集中在此文件管理，
通过环境变量切换后端，对外暴露一致的便捷函数。
"""

import logging
from abc import ABC, abstractmethod

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    EMBED_BACKEND,
    EMBED_MODEL_PATH,
    RERANKER_BACKEND,
    RERANKER_MODEL_PATH,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_URL,
    OLLAMA_EMBED_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_RERANKER_MODEL,
    CHAT_MODEL_NAME,
)

logger = logging.getLogger(__name__)


# ==================== Ollama HTTP 客户端 ====================


class OllamaClient:
    """Ollama HTTP API 封装"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embed"
        self.chat_url = f"{base_url}/api/chat"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, model: str, input_texts: list[str]) -> list[list[float]]:
        """调用 Ollama /api/embed 接口获取向量"""
        resp = requests.post(
            self.embed_url,
            json={"model": model, "input": input_texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(self, model: str, messages: list[dict], logprobs: int = 0, think: bool = True, **options) -> dict:
        """调用 Ollama /api/chat 接口，返回完整响应"""
        payload = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options
        if logprobs > 0:
            payload["logprobs"] = True
            payload["num_logprobs"] = logprobs
        if not think:
            payload["think"] = False
        resp = requests.post(self.chat_url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()


_ollama_client: OllamaClient | None = None


def _get_ollama_client() -> OllamaClient:
    """获取 Ollama 客户端单例"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


# ==================== Embedding 抽象与实现 ====================


class Embedder(ABC):
    """嵌入模型抽象基类"""

    @abstractmethod
    def embed(self, text: str, prompt_name: str = "query") -> list[float]:
        """获取单条文本的向量"""

    @abstractmethod
    def embed_batch(
        self, texts: list[str], prompt_name: str = "document", batch_size: int = 128
    ) -> list[list[float]]:
        """批量获取向量"""


class LocalEmbedder(Embedder):
    """本地 SentenceTransformer 嵌入模型"""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        logger.info(f"[Embedding] 本地加载模型: {EMBED_MODEL_PATH}")
        self._model = SentenceTransformer(EMBED_MODEL_PATH)
        logger.info("[Embedding] 本地模型加载完成")

    def embed(self, text: str, prompt_name: str = "query") -> list[float]:
        vec = self._model.encode(text, prompt_name=prompt_name, show_progress_bar=False)
        return vec.tolist()

    def embed_batch(
        self, texts: list[str], prompt_name: str = "document", batch_size: int = 128
    ) -> list[list[float]]:
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._model.encode(
                batch, prompt_name=prompt_name, show_progress_bar=False, batch_size=len(batch)
            )
            all_vectors.extend(vectors.tolist())
            if len(texts) > batch_size:
                logger.info(f"  向量化进度: {min(i + batch_size, len(texts))}/{len(texts)}")
        return all_vectors


class OllamaEmbedder(Embedder):
    """Ollama 嵌入模型"""

    def __init__(self):
        logger.info(f"[Embedding] 使用 Ollama 模型: {OLLAMA_EMBED_MODEL}")

    def embed(self, text: str, prompt_name: str = "query") -> list[float]:
        # Ollama embed API 不区分 query/document，直接返回向量
        embeddings = _get_ollama_client().embed(OLLAMA_EMBED_MODEL, [text])
        return embeddings[0]

    def embed_batch(
        self, texts: list[str], prompt_name: str = "document", batch_size: int = 128
    ) -> list[list[float]]:
        # Ollama 支持批量输入，batch_size 控制每次请求的文本数
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = _get_ollama_client().embed(OLLAMA_EMBED_MODEL, batch)
            all_vectors.extend(embeddings)
            if len(texts) > batch_size:
                logger.info(f"  向量化进度: {min(i + batch_size, len(texts))}/{len(texts)}")
        return all_vectors


# ==================== Reranker 抽象与实现 ====================


class Reranker(ABC):
    """精排模型抽象基类"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        """对候选文档精排，返回 [{"text": ..., "score": ...}] 按分数降序"""


class LocalReranker(Reranker):
    """本地 CrossEncoder 精排模型"""

    def __init__(self):
        from sentence_transformers import CrossEncoder

        logger.info(f"[Reranker] 本地加载模型: {RERANKER_MODEL_PATH}")
        self._model = CrossEncoder(RERANKER_MODEL_PATH)
        logger.info("[Reranker] 本地模型加载完成")

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        valid_pairs = [(query, doc) for doc in documents if doc and doc.strip()]
        if not valid_pairs:
            return []
        scores = self._model.predict(valid_pairs)
        ranked = sorted(
            [{"text": doc, "score": float(score)} for (_, doc), score in zip(valid_pairs, scores)],
            key=lambda x: x["score"],
            reverse=True,
        )
        logger.info(f"精排完成: {len(ranked)} 条候选, 取前 {top_k} 条")
        return ranked[:top_k]


class OllamaReranker(Reranker):
    """Ollama 精排模型 — 通过 chat 接口让 Reranker 判断 yes/no 相关性"""

    RERANKER_SYSTEM_PROMPT = (
        "Judge whether the Document is relevant to the Query. Answer only yes or no."
    )

    def __init__(self):
        logger.info(f"[Reranker] 使用 Ollama 模型: {OLLAMA_RERANKER_MODEL}")

    def _score_pair(self, query: str, document: str) -> float:
        """对单个 query-document 对打分，基于 yes/no token 的 logprobs 概率"""
        import math

        client = _get_ollama_client()
        user_content = f"/no_think\n<Query>{query}</Query>\n<Document>{document}</Document>"
        messages = [
            {"role": "system", "content": self.RERANKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        resp = client.chat(
            OLLAMA_RERANKER_MODEL, messages,
            num_predict=2, temperature=0,
            logprobs=10, think=False,
        )
        # 从 logprobs 中取第一个 yes/no token 的概率
        for lp in resp.get("logprobs", []):
            tok = lp["token"].strip().lower()
            prob = math.exp(lp["logprob"])
            if tok == "yes":
                return prob
            elif tok == "no":
                return 1.0 - prob

        # fallback: 文本匹配
        content = resp.get("message", {}).get("content", "").strip().lower()
        if "yes" in content:
            return 0.8
        elif "no" in content:
            return 0.2
        return 0.5

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        valid_docs = [doc for doc in documents if doc and doc.strip()]
        if not valid_docs:
            return []
        # 逐对打分（Reranker 是 cross-encoder，需要逐对计算）
        scored = []
        for doc in valid_docs:
            score = self._score_pair(query, doc)
            scored.append({"text": doc, "score": score})
        ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
        logger.info(f"精排完成: {len(ranked)} 条候选, 取前 {top_k} 条")
        return ranked[:top_k]


# ==================== LLM 抽象与实现 ====================


class LLM(ABC):
    """LLM 生成模型抽象基类"""

    @abstractmethod
    def chat(self, prompt: str, system_prompt: str = "") -> str:
        """调用 LLM 生成回答"""


class LocalLLM(LLM):
    """本地 LLM（预留，暂未实现本地 GGUF 加载）"""

    def __init__(self):
        logger.warning("[LLM] 本地 LLM 尚未实现，请使用 ollama 后端")
        raise NotImplementedError("本地 LLM 后端暂未实现，请设置 LLM_BACKEND=ollama")

    def chat(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError


class OllamaLLM(LLM):
    """Ollama LLM 生成模型"""

    def __init__(self):
        logger.info(f"[LLM] 使用 Ollama 模型: {CHAT_MODEL_NAME}")

    def chat(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = _get_ollama_client().chat(CHAT_MODEL_NAME, messages)
        return resp["message"]["content"]


# ==================== 工厂函数 ====================


_embedder: Embedder | None = None
_reranker: Reranker | None = None
_llm: LLM | None = None


def get_embedder() -> Embedder:
    """获取嵌入模型实例（按配置自动选择后端）"""
    global _embedder
    if _embedder is None:
        if EMBED_BACKEND == "ollama":
            _embedder = OllamaEmbedder()
        else:
            _embedder = LocalEmbedder()
    return _embedder


def get_reranker() -> Reranker:
    """获取精排模型实例（按配置自动选择后端）"""
    global _reranker
    if _reranker is None:
        if RERANKER_BACKEND == "ollama":
            _reranker = OllamaReranker()
        else:
            _reranker = LocalReranker()
    return _reranker


def get_llm() -> LLM:
    """获取 LLM 实例（按配置自动选择后端）"""
    global _llm
    if _llm is None:
        if LLM_BACKEND == "ollama":
            _llm = OllamaLLM()
        else:
            _llm = LocalLLM()
    return _llm


# ==================== 兼容便捷函数（保持原接口不变） ====================


def get_embedding(text: str, prompt_name: str = "query") -> list[float]:
    """获取单条文本的向量"""
    return get_embedder().embed(text, prompt_name=prompt_name)


def get_embeddings_batch(
    texts: list[str], prompt_name: str = "document", batch_size: int = 128
) -> list[list[float]]:
    """批量获取向量"""
    return get_embedder().embed_batch(texts, prompt_name=prompt_name, batch_size=batch_size)


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    """对候选文档精排"""
    return get_reranker().rerank(query, documents, top_k=top_k)


def chat_with_ollama(prompt: str, system_prompt: str = "") -> str:
    """调用 LLM 生成回答（保留原函数名兼容）"""
    return get_llm().chat(prompt, system_prompt=system_prompt)
