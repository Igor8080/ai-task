from transformers import pipeline


MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
)


def predict_sentiment(text: str) -> dict:
    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": result["score"],
    }