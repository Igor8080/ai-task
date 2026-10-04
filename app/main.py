from fastapi import FastAPI
from pydantic import BaseModel

from app.model import predict_sentiment


app = FastAPI(
    title="Russian Sentiment Analysis API",
    description="API for sentiment analysis using RuBERT",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "Russian Sentiment Analysis API"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_sentiment(request.text)