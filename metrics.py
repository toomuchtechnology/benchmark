import re
from collections import Counter
from functools import lru_cache
from statistics import mean

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def ensure_nltk_tokenizer() -> None:
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")
    except Exception:
        return


@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        print("Загрузка модели эмбеддингов")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Модель загружена.\n")
        return model
    except Exception as exc:
        print(f"Не удалось загрузить модель эмбеддингов: {exc}")
        return None


def tokenize(text: str) -> list[str]:
    normalized = (text or "").lower()
    try:
        import nltk

        return nltk.word_tokenize(normalized, language="russian")
    except Exception:
        return normalized.split()


def normalize_text(text: str) -> str:
    lowered = (text or "").strip().lower().replace("ё", "е")
    compact = re.sub(r"[^\w\s]+", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", compact).strip()


def normalize_source_name(source: str) -> str:
    normalized = (source or "").strip().lower()
    return normalized.replace("www.bsuir.by_", "bsuir.by_")


def deduplicate_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def calculate_retrieval_metrics(
    relevant_sources: list[str], retrieved_sources: list[str]
) -> dict[str, float]:
    metrics = {"HitRate@k": 0.0, "Precision@k": 0.0, "Recall@k": 0.0, "MRR": 0.0}
    if not relevant_sources or not retrieved_sources:
        return metrics

    relevant = {normalize_source_name(source) for source in relevant_sources}
    retrieved = deduplicate_keep_order([normalize_source_name(source) for source in retrieved_sources])
    retrieved_set = set(retrieved)
    hits = [source for source in retrieved if source in relevant]

    metrics["HitRate@k"] = 1.0 if hits else 0.0
    metrics["Precision@k"] = len(hits) / len(retrieved)
    metrics["Recall@k"] = len(relevant.intersection(retrieved_set)) / len(relevant)

    reciprocal_rank = 0.0
    for idx, source in enumerate(retrieved, start=1):
        if source in relevant:
            reciprocal_rank = 1.0 / idx
            break
    metrics["MRR"] = reciprocal_rank

    return {key: round(value, 4) for key, value in metrics.items()}


def calculate_answer_metrics(expected: str, generated: str) -> dict[str, float | None]:
    ensure_nltk_tokenizer()
    expected_tokens = tokenize(expected)
    generated_tokens = tokenize(generated)
    metrics: dict[str, float | None] = {}

    try:
        import nltk

        bleu = nltk.translate.bleu_score.sentence_bleu(
            [expected_tokens],
            generated_tokens,
            weights=(0.5, 0.5, 0, 0),
        )
        metrics["BLEU"] = round(float(bleu), 4)
    except Exception:
        overlap = Counter(expected_tokens) & Counter(generated_tokens)
        overlap_count = sum(overlap.values())
        bleu_fallback = overlap_count / len(generated_tokens) if generated_tokens else 0.0
        metrics["BLEU"] = round(float(bleu_fallback), 4)

    try:
        from rouge_score import rouge_scorer

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_scores = rouge.score((expected or "").lower(), (generated or "").lower())
        metrics["ROUGE-L_f1"] = round(rouge_scores["rougeL"].fmeasure, 4)
    except Exception:
        lcs = lcs_length(expected_tokens, generated_tokens)
        lcs_precision = lcs / len(generated_tokens) if generated_tokens else 0.0
        lcs_recall = lcs / len(expected_tokens) if expected_tokens else 0.0
        rouge_l_f1 = (
            2 * lcs_precision * lcs_recall / (lcs_precision + lcs_recall)
            if (lcs_precision + lcs_recall) > 0
            else 0.0
        )
        metrics["ROUGE-L_f1"] = round(rouge_l_f1, 4)

    overlap = Counter(expected_tokens) & Counter(generated_tokens)
    overlap_count = sum(overlap.values())
    precision = overlap_count / len(generated_tokens) if generated_tokens else 0.0
    recall = overlap_count / len(expected_tokens) if expected_tokens else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["Token_F1"] = round(token_f1, 4)
    metrics["Exact_Match"] = (
        1.0 if (expected or "").strip().lower() == (generated or "").strip().lower() else 0.0
    )
    metrics["Normalized_Exact_Match"] = (
        1.0 if normalize_text(expected) == normalize_text(generated) else 0.0
    )
    metrics["Answer_Length_Ratio"] = round(
        len(generated_tokens) / len(expected_tokens) if expected_tokens else 0.0, 4
    )

    emb_model = get_embedding_model()
    if emb_model is not None and expected and generated:
        emb_expected = emb_model.encode([expected])
        emb_generated = emb_model.encode([generated])
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            cos_sim = cosine_similarity(emb_expected, emb_generated)[0][0]
            metrics["Semantic_Similarity"] = round(float(cos_sim), 4)
        except Exception:
            metrics["Semantic_Similarity"] = None
    else:
        metrics["Semantic_Similarity"] = None
    return metrics


def average_metrics(results: list[dict[str, float | None]]) -> dict[str, float]:
    values_by_metric: dict[str, list[float]] = {}
    for item in results:
        for metric_name, metric_value in item.items():
            if metric_value is None:
                continue
            values_by_metric.setdefault(metric_name, []).append(float(metric_value))
    return {
        metric_name: round(float(mean(metric_values)), 4)
        for metric_name, metric_values in values_by_metric.items()
        if metric_values
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * pct))
    return float(sorted_values[index])
