import argparse

from metrics import (
    calculate_answer_metrics,
    calculate_retrieval_metrics,
    ensure_nltk_tokenizer as _ensure_nltk_tokenizer,
    get_embedding_model as _get_embedding_model,
    tokenize as _tokenize,
)
from runner import DATASET_PATH, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк для RAG и генеративной модели.")
    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help="Путь к JSON dataset с query/expected и optional relevant_sources.",
    )
    parser.add_argument(
        "--rag-endpoint",
        default=None,
        help="URL RAG endpoint (например, http://localhost:8000/rag/ask).",
    )
    parser.add_argument(
        "--dataset-api-url",
        default=None,
        help="URL API, который возвращает dataset в JSON формате (list).",
    )
    parser.add_argument(
        "--results-path",
        default="benchmark_results.json",
        help="Путь для сохранения JSON-результатов бенчмарка.",
    )
    parser.add_argument(
        "--results-api-url",
        default=None,
        help="URL API для отправки JSON-результатов (POST).",
    )
    args = parser.parse_args()
    run(
        dataset_path=args.dataset,
        rag_endpoint=args.rag_endpoint,
        dataset_api_url=args.dataset_api_url,
        results_path=args.results_path,
        results_api_url=args.results_api_url,
    )


if __name__ == "__main__":
    main()