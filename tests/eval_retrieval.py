"""检索质量评测脚本 — 计算 Recall@K、MRR、章节命中率"""

import json
import logging
import argparse
from pathlib import Path
from collections.abc import Callable

from src.match import search
from src.config import SEARCH_TOP_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_TESTS_DIR = Path(__file__).resolve().parent
_DEFAULT_DATASET = str(_TESTS_DIR / "eval_dataset.json")


def load_dataset(file_path: str = _DEFAULT_DATASET) -> list[dict]:
    """加载评测数据集"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"评测数据集不存在: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    top_k: int = SEARCH_TOP_K,
    dataset_path: str = _DEFAULT_DATASET,
    search_fn: Callable[[str, int], list[dict]] = search,
) -> dict:
    """评测检索质量

    Args:
        top_k: 检索返回的片段数
        dataset_path: 评测数据集路径

    Returns:
        dict: 包含各项指标的评测结果
    """
    dataset = load_dataset(dataset_path)
    logger.info(f"加载评测数据集: {len(dataset)} 条问题")

    results = []
    chapter_hits = 0  # 章节命中计数（仅章节正确）
    content_hits = 0  # 答案片段命中计数（章节+关键词同时命中）
    reciprocal_ranks = []  # 用于计算 MRR

    for item in dataset:
        question = item["question"]
        expected_chapters = set(item["expected_chapters"])
        keywords = item.get("keywords", [])
        expected_source = item.get("source", "")

        # 执行检索
        matched = search_fn(question, top_k=top_k)

        # 提取检索到的章节
        retrieved_chapters = set()
        for doc in matched:
            if doc.get("chapter"):
                retrieved_chapters.add(doc["chapter"])

        # 章节命中率：有期望章节时按章节判断，否则按 source_file 判断
        if expected_chapters:
            chapter_hit = bool(expected_chapters & retrieved_chapters)
        else:
            # 无章节结构的文档，检查是否检索到了对应文件的片段
            chapter_hit = any(
                doc.get("source_file", "") == expected_source for doc in matched
            )
        if chapter_hit:
            chapter_hits += 1

        # 答案片段命中：片段中是否包含关键词
        content_hit = False
        content_hit_rank = 0
        for rank, doc in enumerate(matched, 1):
            content = doc.get("content", "")
            if expected_chapters:
                # 有章节结构：必须章节匹配且包含关键词
                if doc.get("chapter") in expected_chapters:
                    if any(kw in content for kw in keywords):
                        content_hit = True
                        content_hit_rank = rank
                        break
            else:
                # 无章节结构：只需来源文件匹配且包含关键词
                if doc.get("source_file", "") == expected_source:
                    if any(kw in content for kw in keywords):
                        content_hit = True
                        content_hit_rank = rank
                        break
        if content_hit:
            content_hits += 1

        # 计算 Reciprocal Rank（基于答案片段命中，而非章节命中）
        rr = 1.0 / content_hit_rank if content_hit_rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        # 记录单条结果
        result = {
            "id": item["id"],
            "question": question,
            "expected_chapters": list(expected_chapters),
            "retrieved_chapters": list(retrieved_chapters),
            "chapter_hit": chapter_hit,
            "content_hit": content_hit,
            "reciprocal_rank": rr,
            "top_source": matched[0].get("source_file", "") if matched else "",
            "top_chapter": matched[0].get("chapter", "") if matched else "",
            "top_content_preview": matched[0].get("content", "")[:100] if matched else "",
        }
        results.append(result)

        expected_label = list(expected_chapters) if expected_chapters else [expected_source]
        logger.info(
            f"[{item['id']}] 来源{'✓' if chapter_hit else '✗'} 答案片段{'✓' if content_hit else '✗'} "
            f"RR={rr:.3f} | 期望={expected_label}"
        )

    # 汇总指标
    total = len(dataset)
    metrics = {
        "total_questions": total,
        "top_k": top_k,
        "chapter_hit_rate": chapter_hits / total if total else 0.0,  # 章节命中率（宽松）
        "content_hit_rate": content_hits / total if total else 0.0,  # 答案片段命中率（严格）
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,  # Mean Reciprocal Rank
    }

    return {"metrics": metrics, "details": results}


def print_report(eval_result: dict) -> None:
    """打印评测报告"""
    metrics = eval_result["metrics"]
    details = eval_result["details"]

    print("\n" + "=" * 60)
    print("检索质量评测报告")
    print("=" * 60)

    print(f"\n📊 总体指标 (Top-{metrics['top_k']}):")
    print(f"  • 来源命中率: {metrics['chapter_hit_rate']:.1%} ({int(metrics['chapter_hit_rate'] * metrics['total_questions'])}/{metrics['total_questions']})")
    print(f"  • 答案片段命中率: {metrics['content_hit_rate']:.1%} ({int(metrics['content_hit_rate'] * metrics['total_questions'])}/{metrics['total_questions']}) ← 更严格")
    print(f"  • MRR (Mean Reciprocal Rank): {metrics['mrr']:.3f}")

    print(f"\n📋 详细结果:")
    print("-" * 60)
    for d in details:
        source_status = "✓" if d["chapter_hit"] else "✗"
        content_status = "✓" if d["content_hit"] else "✗"
        print(f"[{d['id']:3d}] 来源{source_status} 片段{content_status} RR={d['reciprocal_rank']:.3f} | {d['question'][:25]}...")
        top_label = d["top_chapter"] if d["top_chapter"] else d.get("top_source", "")
        print(f"      Top-1: {top_label} | 预览: {d['top_content_preview'][:50]}...")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 检索质量评测")
    parser.add_argument("--top-k", type=int, default=SEARCH_TOP_K, help="检索片段数")
    parser.add_argument("--dataset", type=str, default=_DEFAULT_DATASET, help="评测数据集路径")
    parser.add_argument("--json", type=str, help="输出 JSON 结果到文件")
    args = parser.parse_args()

    eval_result = evaluate_retrieval(top_k=args.top_k, dataset_path=args.dataset)
    print_report(eval_result)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        logger.info(f"评测结果已保存到: {args.json}")
