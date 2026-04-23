import json
import urllib.error
import urllib.request
from typing import Any


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset_from_api(dataset_url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        dataset_url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
            if not isinstance(data, list):
                raise ValueError("Dataset API must return JSON list.")
            return data
    except (urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Не удалось получить dataset через API: {exc}") from exc


def save_results_to_file(results: dict[str, Any], results_path: str) -> None:
    with open(results_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)


def post_results_to_api(results: dict[str, Any], results_api_url: str) -> None:
    payload = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
    request = urllib.request.Request(
        results_api_url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as _:
            return
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось отправить результаты на API: {exc}") from exc
