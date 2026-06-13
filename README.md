---
title: Operation Search API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Operation Search API

Семантический поиск по каталогу банковских операций (FastAPI + sentence-transformers).

## Эндпоинты

- `GET /health` — проверка статуса
- `POST /search` — поиск операций по запросу

```json
{
  "query": "поменялся паспорт",
  "top_k": 5
}
```
