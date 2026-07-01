import unittest
from unittest.mock import patch
import logging

from src import match


class MatchTest(unittest.TestCase):
    def test_merge_results_keeps_ids_and_adds_rrf_scores(self):
        knn = [
            {"id": 1, "content": "A", "chapter": "第1章", "chapter_title": "一", "source_file": "a.txt"},
            {"id": 2, "content": "B", "chapter": "第2章", "chapter_title": "二", "source_file": "b.txt"},
        ]
        bm25 = [
            {"id": 2, "content": "B", "chapter": "第2章", "chapter_title": "二", "source_file": "b.txt"},
            {"id": 3, "content": "C", "chapter": "第3章", "chapter_title": "三", "source_file": "c.txt"},
        ]

        merged = match._merge_results(knn, bm25, knn_weight=0.5, bm25_weight=0.5)

        self.assertEqual([doc["id"] for doc in merged], [2, 1, 3])
        self.assertGreater(merged[0]["final_score"], merged[1]["final_score"])

    def test_search_preserves_duplicate_content_metadata_order_after_rerank(self):
        recall_doc_a = {
            "id": 1,
            "content": "重复文本",
            "chapter": "第1章",
            "chapter_title": "一",
            "source_file": "a.txt",
            "final_score": 1.0,
        }
        recall_doc_b = {
            "id": 2,
            "content": "重复文本",
            "chapter": "第2章",
            "chapter_title": "二",
            "source_file": "b.txt",
            "final_score": 0.9,
        }

        with (
            patch.object(match, "get_embedding", return_value=[0.1, 0.2]),
            patch.object(match, "_knn_search", return_value=[]),
            patch.object(match, "_bm25_search", return_value=[]),
            patch.object(match, "_merge_results", return_value=[recall_doc_a, recall_doc_b]),
            patch.object(
                match,
                "rerank",
                return_value=[
                    {"text": "重复文本", "score": 0.8},
                    {"text": "重复文本", "score": 0.7},
                ],
            ),
        ):
            results = match.search("问题", top_k=2)

        self.assertEqual([r["id"] for r in results], [1, 2])
        self.assertEqual([r["source_file"] for r in results], ["a.txt", "b.txt"])

    def test_search_wraps_unexpected_errors(self):
        with patch.object(match, "get_embedding", side_effect=RuntimeError("boom")):
            with self.assertLogs(match.logger, level=logging.ERROR):
                with self.assertRaises(match.RetrievalError) as ctx:
                    match.search("问题")

        self.assertIn("boom", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
