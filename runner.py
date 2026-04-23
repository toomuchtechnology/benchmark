import os
import time
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from clients import ask_rag_endpoint, generate_answer_openrouter
from config import MODEL_NAME
from io_utils import (
    load_dataset,
    load_dataset_from_api,
    post_results_to_api,
    save_results_to_file,
)
from metrics import (
    average_metrics,
    calculate_answer_metrics,
    calculate_retrieval_metrics,
    percentile,
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")


def run(
    dataset_path: str = DATASET_PATH,
    rag_endpoint: str | None = None,
    dataset_api_url: str | None = None,
    results_path: str = "benchmark_results.json",
    results_api_url: str | None = None,
) -> dict[str, Any]:
    dataset = load_dataset_from_api(dataset_api_url) if dataset_api_url else load_dataset(dataset_path)
    print(f"Запуск бенчмарка для модели: {MODEL_NAME}")
    if rag_endpoint:
        print(f"Режим: RAG endpoint {rag_endpoint}")
    if dataset_api_url:
        print(f"Dataset получен через API: {dataset_api_url}")
    print("---")

    answer_results: list[dict[str, float | None]] = []
    retrieval_results: list[dict[str, float]] = []
    latency_results_ms: list[float] = []
    per_test_results: list[dict[str, Any]] = []

    for index, item in enumerate(dataset, start=1):
        query = item["query"]
        expected = item.get("expected") or item.get("expected_answer") or ""
        relevant_sources = item.get("relevant_sources", [])
        print(f"\n[Тест {index}/{len(dataset)}]")
        print(f"Запрос: {query}")
        print(f"Ожидается: {expected}")

        started_at = time.perf_counter()
        if rag_endpoint:
            generated, retrieved_sources = ask_rag_endpoint(query, rag_endpoint)
        else:
            generated = generate_answer_openrouter(query)
            retrieved_sources = item.get("retrieved_sources", [])
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        latency_results_ms.append(elapsed_ms)

        print(f"Сгенерировано: {generated}")
        print(f"Время ответа (мс): {elapsed_ms}")
        if retrieved_sources:
            print(f"Полученные источники: {retrieved_sources}")

        if not generated:
            print("Ошибка генерации.")
            per_test_results.append(
                {
                    "test_id": index,
                    "query": query,
                    "expected": expected,
                    "generated": generated,
                    "retrieved_sources": retrieved_sources,
                    "relevant_sources": relevant_sources,
                    "latency_ms": elapsed_ms,
                    "answer_metrics": {},
                    "retrieval_metrics": {},
                    "status": "generation_error",
                }
            )
            continue

        answer_metrics = calculate_answer_metrics(expected, generated)
        print(f"Метрики ответа: {answer_metrics}")
        answer_results.append(answer_metrics)

        retrieval_metrics = calculate_retrieval_metrics(relevant_sources, retrieved_sources)
        if relevant_sources:
            print(f"Метрики retrieval: {retrieval_metrics}")
            retrieval_results.append(retrieval_metrics)

        per_test_results.append(
            {
                "test_id": index,
                "query": query,
                "expected": expected,
                "generated": generated,
                "retrieved_sources": retrieved_sources,
                "relevant_sources": relevant_sources,
                "latency_ms": elapsed_ms,
                "answer_metrics": answer_metrics,
                "retrieval_metrics": retrieval_metrics,
                "status": "ok",
            }
        )

    avg_answer_metrics = average_metrics(answer_results) if answer_results else {}
    avg_retrieval_metrics = average_metrics(retrieval_results) if retrieval_results else {}
    latency_summary = {
        "mean_ms": round(mean(latency_results_ms), 2) if latency_results_ms else 0.0,
        "min_ms": round(min(latency_results_ms), 2) if latency_results_ms else 0.0,
        "max_ms": round(max(latency_results_ms), 2) if latency_results_ms else 0.0,
        "p50_ms": round(percentile(latency_results_ms, 0.5), 2) if latency_results_ms else 0.0,
        "p95_ms": round(percentile(latency_results_ms, 0.95), 2) if latency_results_ms else 0.0,
    }

    if answer_results:
        print("\nИтоговые средние метрики ответа")
        for metric_name, avg_value in avg_answer_metrics.items():
            print(f"{metric_name}: {avg_value:.4f}")

    if retrieval_results:
        print("\nИтоговые средние метрики retrieval")
        for metric_name, avg_value in avg_retrieval_metrics.items():
            print(f"{metric_name}: {avg_value:.4f}")

    print("\nИтоги по latency")
    print(f"mean_ms: {latency_summary['mean_ms']:.2f}")
    print(f"p95_ms: {latency_summary['p95_ms']:.2f}")

    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
            "dataset_path": dataset_path if not dataset_api_url else None,
            "dataset_api_url": dataset_api_url,
            "rag_endpoint": rag_endpoint,
            "results_path": results_path,
            "tests_total": len(dataset),
            "tests_completed": len(per_test_results),
        },
        "summary": {
            "answer_metrics_avg": avg_answer_metrics,
            "retrieval_metrics_avg": avg_retrieval_metrics,
            "latency_ms": latency_summary,
        },
        "tests": per_test_results,
    }

    save_results_to_file(report, results_path)
    print(f"\nJSON-отчет записан в: {results_path}")

    if results_api_url:
        post_results_to_api(report, results_api_url)
        print(f"JSON-отчет отправлен в API: {results_api_url}")

    return report
