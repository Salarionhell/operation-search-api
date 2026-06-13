"""
Эндпоинт семантического поиска по каталогу операций.

Запуск:
    pip install fastapi uvicorn sentence-transformers numpy --break-system-packages
    uvicorn search_api:app --host 0.0.0.0 --port 8000

Каталог operations-catalog-with-embeddings.json должен лежать рядом.
"""

import json
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

CATALOG_FILE = "operations-catalog-with-embeddings.json"
MODEL_NAME = "intfloat/multilingual-e5-small"

app = FastAPI()

# Настроить под свой домен фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружается один раз при старте процесса -- держится в памяти
print("Загружаю модель...")
model = SentenceTransformer(MODEL_NAME)

print("Загружаю каталог...")
with open(CATALOG_FILE, "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Матрица векторов для быстрого матричного умножения
catalog_embeddings = np.array([item["embedding"] for item in catalog])
print(f"Загружено {len(catalog)} операций, размерность {catalog_embeddings.shape[1]}")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/search")
def search(req: SearchRequest):
    # Префикс "query: " обязателен для e5-моделей
    query_text = f"query: {req.query}"
    query_embedding = model.encode(
        query_text, normalize_embeddings=True
    )

    # Косинусная близость = скалярное произведение нормализованных векторов
    similarities = catalog_embeddings @ query_embedding

    # Топ-K результатов
    top_indices = np.argsort(-similarities)[: req.top_k]

    results = [
        {
            "id": catalog[i]["id"],
            "name": catalog[i]["name"],
            "category": catalog[i]["category"],
            "similarity": float(similarities[i]),
        }
        for i in top_indices
    ]

    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok", "operations_count": len(catalog)}
