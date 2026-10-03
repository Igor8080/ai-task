import torch
from transformers import pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
    )

    result = transcriber("sample.mp3")

    print("Распознанный текст:")
    print(result["text"])


if __name__ == "__main__":
    main()