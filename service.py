from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from runner import DATASET_PATH, run

app = FastAPI(title="RAG Benchmark Service", version="1.0.0")


class RunRequest(BaseModel):
    dataset_path: str = DATASET_PATH
    rag_endpoint: str | None = None
    dataset_api_url: str | None = None
    results_path: str = "output/benchmark_results.json"
    results_api_url: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run_benchmark_job(payload: RunRequest) -> dict[str, Any]:
    try:
        report = run(
            dataset_path=payload.dataset_path,
            rag_endpoint=payload.rag_endpoint,
            dataset_api_url=payload.dataset_api_url,
            results_path=payload.results_path,
            results_api_url=payload.results_api_url,
        )
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
