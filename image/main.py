import torch
from PIL import Image
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


def main():
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    image = Image.open("sample.jpg").convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    values, indices = torch.topk(probabilities, 5)

    categories = weights.meta["categories"]

    print("Результаты:")

    for probability, index in zip(values, indices):
        print(
            f"{categories[index]}: "
            f"{probability.item():.2%}"
        )


if __name__ == "__main__":
    main()