FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN useradd --create-home --uid 10001 benchmark
COPY --chown=benchmark:benchmark . .
USER benchmark

EXPOSE 8080

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
