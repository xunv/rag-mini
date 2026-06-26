"""LLM 生成模块 — Ollama chat 接口"""

import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import OLLAMA_CHAT_URL, CHAT_MODEL_NAME

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def chat_with_ollama(prompt: str, system_prompt: str = "") -> str:
    """调用 Ollama chat 模型生成回答

    Raises:
        Exception: 当请求失败时抛出异常，由 tenacity 接管重试
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": CHAT_MODEL_NAME, "messages": messages, "stream": False}
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["message"]["content"]