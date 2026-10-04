# PrEng. RecSys — Movie Recommendation System

Реализация рекомендательной системы на датасете MovieLens.

## Реализованные подходы

1. **Popularity-based baseline** — популярные фильмы.
2. **Content-Based** — рекомендации на основе жанров и названий с TF-IDF и cosine similarity.
3. **Item-Based Collaborative Filtering** — рекомендации на основе пользовательских оценок.
4. **Hybrid** — объединение content-based и collaborative scores.

## Структура

```text
recsys_project/
├── data/
├── src/
│   ├── data.py
│   ├── recommenders.py
│   └── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```
