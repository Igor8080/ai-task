import cv2
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

MODEL_NAME = "facebook/detr-resnet-50"
VIDEO_PATH = "D:\\ing\\video\\sample.mp4"
OUTPUT_PATH = "D:\\ing\\video\\output.mp4"


def main():
    print("1. Загружаю процессор...", flush=True)

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)

    print("2. Загружаю модель...", flush=True)

    model = DetrForObjectDetection.from_pretrained(MODEL_NAME)

    print("3. Модель загружена", flush=True)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"4. Видео открыто: {width}x{height}, {fps:.1f} FPS", flush=True)
    print(f"5. Всего кадров: {frame_count}", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height),
    )

    frame_number = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])

        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.7,
        )[0]

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"],
        ):
            score = score.item()
            label = label.item()

            x1, y1, x2, y2 = map(int, box.tolist())

            class_name = model.config.id2label[label]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{class_name} {score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out.write(frame)

        if frame_number % 10 == 0:
            elapsed = time.time() - start_time
            processed_fps = frame_number / elapsed

            print(
                f"Кадр {frame_number}/{frame_count} "
                f"({frame_number / frame_count:.0%}), "
                f"{processed_fps:.2f} кадр/сек",
                flush=True,
            )

    cap.release()
    out.release()

    print()
    print("6. Готово!")
    print(f"Результат: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()