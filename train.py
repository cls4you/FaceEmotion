from typing import Dict, Any
from ultralytics import YOLO
import pandas as pd
import os


def train_model(model_path: str, data_path: str, epochs: int = 100, img_size: int = 640,
                batch_size: int = 8, device: int = 0) -> Dict[str, Any]:
    """
    Обучает модель YOLOv11 на указанном датасете.

    Args:
        model_path (str): Путь к предобученной модели YOLOv11 (например, 'yolo11n.pt').
        data_path (str): Путь к файлу data.yaml с конфигурацией датасета.
        epochs (int): Количество эпох обучения. По умолчанию 50.
        img_size (int): Размер входных изображений. По умолчанию 640.
        batch_size (int): Размер батча. По умолчанию 16.
        device (int): Устройство для обучения (0 для GPU, None для CPU). По умолчанию 0.

    Returns:
        Dict[str, Any]: Словарь с результатами обучения, включая метрики.
    """
    # Инициализация модели YOLO
    model = YOLO(model_path)

    # Запуск обучения с указанными параметрами
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=20,  # Ранняя остановка после 20 эпох без улучшения
        project="bestModel",  # Папка для сохранения результатов
        name="exp",  # Имя эксперимента
        plots=True  # Включение создания графиков
    )
    return results


def save_metrics(results: Dict[str, Any], output_dir: str) -> None:
    """
    Сохраняет ключевые метрики обучения в текстовый файл.

    Args:
        results (Dict[str, Any]): Словарь с результатами обучения.
        output_dir (str): Путь к директории для сохранения файла метрик.
    """
    # Формирование словаря с ключевыми метриками
    output_metrics: Dict[str, float] = {
        "mAP@0.5": results.get("metrics/mAP50(B)", 0),
        "mAP@0.5:0.95": results.get("metrics/mAP50-95(B)", 0),
        "Precision": results.get("metrics/precision(B)", 0),
        "Recall": results.get("metrics/recall(B)", 0),
        "Box Loss (val)": results.get("val/box_loss", 0),
        "Cls Loss (val)": results.get("val/cls_loss", 0),
        "DFL Loss (val)": results.get("val/dfl_loss", 0)
    }

    # Путь к файлу для сохранения метрик
    metrics_file = os.path.join(output_dir, "exp", "metrics.txt")

    # Запись метрик в текстовый файл
    with open(metrics_file, "w") as f:
        for key, value in output_metrics.items():
            f.write(f"{key}: {value}\n")

    return output_metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Выводит метрики в консоль.

    Args:
        metrics (Dict[str, float]): Словарь с метриками для вывода.
    """
    print("Финальные метрики:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main() -> None:
    """
    Основная функция для обучения модели YOLOv11, сохранения метрик и графиков.

    Сохраняет график метрик в 'bestModel/exp/results.png' и метрики в
    'bestModel/exp/metrics.txt'. Выводит финальные метрики в консоль.
    """

    model_path = "yolo11n.pt"  # Путь к предобученной модели
    data_path = "./data.yaml"  # Путь к файлу data.yaml

    # Обучения модели
    results = train_model(model_path, data_path)

    # Сохранения метрик
    output_metrics = save_metrics(results.results_dict, "bestModel")

    # Вывод результатов
    print("Обучение завершено.")
    print("Графики сохранены в bestModel/exp/results.png")
    print("Метрики сохранены в bestModel/exp/metrics.txt")
    print_metrics(output_metrics)


if __name__ == "__main__":
    main()