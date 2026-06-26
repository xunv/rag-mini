"""RAG 生成模块 — 检索增强生成完整流程"""

import logging
import argparse

# 兼容直接运行 (python src/main.py) 和模块运行 (python -m src.main)
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .match import search
from .llm import chat_with_ollama
from .config import SEARCH_TOP_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是一个基于知识库的问答助手。请根据下方提供的参考资料来回答用户的问题。
要求：
1. 只基于参考资料中的内容作答，不要使用自身知识补充
2. 如果参考资料中没有直接提到问题所问的人物、事件或概念，回复"根据现有资料，无法回答该问题。"
3. 不要从无关段落中拼凑答案
4. 回答要简洁准确"""

RAG_USER_TEMPLATE = """参考资料：
{context}

---
用户问题：{question}

请基于参考资料回答。如果资料中没有相关内容则回复"根据现有资料，无法回答该问题。"，不要用自身知识补充。"""


def build_context(matched_docs: list[dict]) -> str:
    """将检索结果拼接为上下文文本"""
    context_parts = []
    for i, doc in enumerate(matched_docs, 1):
        source_info = ""
        if doc.get("chapter") and doc.get("chapter_title"):
            source_info = f"【{doc['chapter']} {doc['chapter_title']}】"
        elif doc.get("source_file"):
            source_info = f"【{doc['source_file']}】"
        context_parts.append(f"[参考资料{i}] {source_info}\n{doc['content'].strip()}")
    return "\n\n".join(context_parts)


def rag_query(question: str, top_k: int = SEARCH_TOP_K) -> dict:
    """完整的 RAG 流程：检索 -> 拼接上下文 -> LLM 生成"""
    # 1. 检索
    logger.info(f"正在检索: '{question}'")
    matched = search(question, top_k=top_k)

    if not matched:
        return {
            "answer": "抱歉，未找到相关的参考资料，无法回答该问题。",
            "sources": [],
            "context": "",
        }

    # 2. 拼接上下文
    context = build_context(matched)
    prompt = RAG_USER_TEMPLATE.format(context=context, question=question)

    # 3. LLM 生成
    logger.info("正在调用 LLM 生成回答...")
    try:
        answer = chat_with_ollama(prompt, system_prompt=RAG_SYSTEM_PROMPT)
    except Exception as e:
        logger.error(f"LLM 生成失败: {e}")
        answer = "抱歉，LLM 生成失败，请稍后重试。"

    return {
        "answer": answer,
        "sources": [
            {
                "chapter": doc.get("chapter", ""),
                "chapter_title": doc.get("chapter_title", ""),
                "source_file": doc.get("source_file", ""),
                "score": doc.get("score", 0),
            }
            for doc in matched
        ],
        "context": context,
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 问答")
    parser.add_argument("--top-k", type=int, default=SEARCH_TOP_K, help="检索片段数")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 知识库问答系统（输入 q 或 quit 退出）")
    print("=" * 60)

    while True:
        try:
            question = input("\n> 请输入问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        result = rag_query(question, top_k=args.top_k)

        print(f"\n回答:\n{result['answer']}")
        print(f"\n参考来源:")
        for s in result["sources"]:
            if s.get("chapter"):
                source_info = f"{s['chapter']} {s['chapter_title']}"
            else:
                source_info = s.get("source_file", "未知来源")
            print(f"  - {source_info} (相关度: {s['score']:.4f})")


if __name__ == "__main__":
    main()