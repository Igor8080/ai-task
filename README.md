# Sentiment Analysis API

Небольшое API для определения тональности текста на русском языке.

Для анализа используется модель `blanchefort/rubert-base-cased-sentiment`, а API реализовано с помощью FastAPI.

## Установка

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать его в Windows:

```bash
.venv\Scripts\activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

## Запуск

```bash
uvicorn app.main:app --reload
```

Документация API:

`http://127.0.0.1:8000/docs`

## Пример

Запрос:

```json
{
  "text": "Мне очень понравился этот фильм!"
}
```

Ответ:

```json
{
  "label": "POSITIVE",
  "score": 0.99
}
```

## Тесты

Запуск тестов:

```bash
python -m pytest
```

Тесты также автоматически запускаются через GitHub Actions при отправке изменений в репозиторий.
