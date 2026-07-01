import json
import tempfile
import unittest
from pathlib import Path

from tests.eval_retrieval import evaluate_retrieval, load_dataset


class EvalMetricsTest(unittest.TestCase):
    def test_load_dataset_reports_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset("/tmp/not-a-real-rag-eval-dataset.json")

    def test_evaluate_retrieval_with_injected_search_function(self):
        dataset = [
            {
                "id": 1,
                "question": "章节问题",
                "expected_chapters": ["第1章"],
                "keywords": ["关键句"],
            },
            {
                "id": 2,
                "question": "文件问题",
                "expected_chapters": [],
                "source": "paper.pdf",
                "keywords": ["术语"],
            },
        ]
        responses = {
            "章节问题": [
                {
                    "chapter": "第2章",
                    "source_file": "book.txt",
                    "content": "干扰内容",
                },
                {
                    "chapter": "第1章",
                    "source_file": "book.txt",
                    "content": "包含关键句",
                },
            ],
            "文件问题": [
                {
                    "chapter": "",
                    "source_file": "paper.pdf",
                    "content": "包含术语",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "dataset.json"
            dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

            result = evaluate_retrieval(
                top_k=3,
                dataset_path=str(dataset_path),
                search_fn=lambda question, top_k: responses[question],
            )

        self.assertEqual(result["metrics"]["total_questions"], 2)
        self.assertEqual(result["metrics"]["chapter_hit_rate"], 1.0)
        self.assertEqual(result["metrics"]["content_hit_rate"], 1.0)
        self.assertEqual(result["metrics"]["mrr"], 0.75)
        self.assertEqual(result["details"][0]["reciprocal_rank"], 0.5)
        self.assertEqual(result["details"][1]["reciprocal_rank"], 1.0)

    def test_evaluate_empty_dataset_returns_zero_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "empty.json"
            dataset_path.write_text("[]", encoding="utf-8")

            result = evaluate_retrieval(
                top_k=3,
                dataset_path=str(dataset_path),
                search_fn=lambda question, top_k: [],
            )

        self.assertEqual(result["metrics"]["total_questions"], 0)
        self.assertEqual(result["metrics"]["chapter_hit_rate"], 0.0)
        self.assertEqual(result["metrics"]["content_hit_rate"], 0.0)
        self.assertEqual(result["metrics"]["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
