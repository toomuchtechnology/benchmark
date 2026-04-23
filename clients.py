import json
import urllib.error
import urllib.request
from functools import lru_cache

from config import BASE_URL, MODEL_NAME, OPENROUTER_API_KEY


@lru_cache(maxsize=1)
def get_openai_client():
    from openai import OpenAI

    return OpenAI(base_url=BASE_URL, api_key=OPENROUTER_API_KEY)


def generate_answer_openrouter(query: str) -> str:
    try:
        response = get_openai_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Ты тестовая модель для бенчмарка. Отвечай умеренно кратко и по делу на русском языке.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"Ошибка при обращении к API: {exc}")
        return ""


def ask_rag_endpoint(query: str, endpoint_url: str) -> tuple[str, list[str]]:
    payload = json.dumps({"question": query}).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("answer", ""), data.get("sources", []) or []
    except urllib.error.URLError as exc:
        print(f"Ошибка запроса к RAG endpoint: {exc}")
        return "", []
