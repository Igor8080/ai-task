from transformers import pipeline


def main():
    classifier = pipeline(
        "text-classification",
        model="blanchefort/rubert-base-cased-sentiment",
    )

    texts = [
        "Мне нравится этот трек.",
        "Приложение вечно тупит и не работает.",
        "Сегодня был обычный день.",
    ]

    for text in texts:
        result = classifier(text)[0]

        print(f"Текст: {text}")
        print(f"Тональность: {result['label']}")
        print(f"Уверенность: {result['score']:.2%}")
        print()


if __name__ == "__main__":
    main()