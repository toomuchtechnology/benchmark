import os
import sys
import unittest


CURRENT_DIR = os.path.dirname(__file__)
BENCHMARK_DIR = os.path.dirname(CURRENT_DIR)
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

import benchmark


class BenchmarkMetricsTests(unittest.TestCase):
    def test_retrieval_metrics_values_are_correct(self):
        relevant = ["doc_a.md", "doc_b.md"]
        retrieved = ["doc_x.md", "doc_b.md", "doc_a.md"]

        metrics = benchmark.calculate_retrieval_metrics(relevant, retrieved)

        self.assertEqual(metrics["HitRate@k"], 1.0)
        self.assertEqual(metrics["Precision@k"], round(2 / 3, 4))
        self.assertEqual(metrics["Recall@k"], 1.0)
        self.assertEqual(metrics["MRR"], 0.5)

    def test_retrieval_metrics_normalize_source_and_deduplicate(self):
        relevant = ["bsuir.by_ru_obrazovanie.md"]
        retrieved = [
            "www.bsuir.by_ru_obrazovanie.md",
            "www.bsuir.by_ru_obrazovanie.md",
            "other.md",
        ]

        metrics = benchmark.calculate_retrieval_metrics(relevant, retrieved)

        self.assertEqual(metrics["HitRate@k"], 1.0)
        self.assertEqual(metrics["Precision@k"], 0.5)
        self.assertEqual(metrics["Recall@k"], 1.0)
        self.assertEqual(metrics["MRR"], 1.0)

    def test_answer_metrics_exact_match_and_token_f1(self):
        expected = "Канберра является столицей Австралии"
        generated = "Канберра является столицей Австралии"

        metrics = benchmark.calculate_answer_metrics(expected, generated)

        self.assertEqual(metrics["Exact_Match"], 1.0)
        self.assertEqual(metrics["Normalized_Exact_Match"], 1.0)
        self.assertEqual(metrics["Token_F1"], 1.0)
        self.assertIn("Semantic_Similarity", metrics)


if __name__ == "__main__":
    unittest.main()
