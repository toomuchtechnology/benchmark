import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")