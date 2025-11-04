from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal


# Конфигурационные переменные
MODEL_PATH: str = "bestModel/exp/weights/best.pt"
CLASS_NAMES: list[str] = ["Angry", "Fearful", "Happy", "Neutral", "Sad"]
CONF_THRESHOLD: float = 0.5
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
WINDOW_WIDTH: int = 720
WINDOW_HEIGHT: int = 700


# Глобальные переменные
app: Optional[QApplication] = None
main_menu: Optional["MainMenu"] = None
photo_window: Optional["PhotoWindow"] = None
camera_window: Optional["CameraWindow"] = None
model: Optional[YOLO] = None


# Вспомогательные функции
def create_info_label(text: str) -> QLabel:
    """
    Создаёт стилизованную метку с полупрозрачным фоном.

    Args:
        text (str): Текст метки.

    Returns:
        QLabel: Стилизованная метка.
    """
    label = QLabel(text)
    label.setStyleSheet("""
        QLabel {
            background-color: rgba(0, 0, 0, 180);
            color: white;
            padding: 8px 16px;
            border-radius: 12px;
            font: bold 15px 'Segoe UI';
            min-width: 290px;
            margin-top: 8px;
        }
    """)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def load_model(model_path: str) -> YOLO:
    """
    Загружает модель YOLO по указанному пути.

    Args:
        model_path (str): Путь к файлу модели (.pt).

    Returns:
        YOLO: Загруженная модель.

    Raises:
        FileNotFoundError: Если модель не найдена.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    return YOLO(model_path)


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """
    Преобразует кадр BGR (OpenCV) в QImage.

    Args:
        frame (np.ndarray): Кадр в формате BGR.

    Returns:
        QImage: Изображение в формате Qt.
    """
    h, w, ch = frame.shape
    return QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)


def scale_pixmap(pixmap: QPixmap, width: int, height: int) -> QPixmap:
    """
    Масштабирует QPixmap с сохранением пропорций.

    Args:
        pixmap (QPixmap): Исходный pixmap.
        width (int): Целевая ширина.
        height (int): Целевая высота.

    Returns:
        QPixmap: Масштабированный pixmap.
    """
    return pixmap.scaled(
        width, height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )


# InferenceWorker для обработки кадров
class InferenceWorker(QObject):
    """Потокобезопасный обработчик видео/фото с сигналами для GUI."""
    frame_ready = pyqtSignal(np.ndarray)
    info_ready = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.cap: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.last_time: float = time.time()

    def start_camera(self) -> bool:
        """
        Запускает захват с камеры.

        Returns:
            bool: True при успешном запуске.
        """
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.info_ready.emit("Камера недоступна")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.running = True
        return True

    def stop(self) -> None:
        """Останавливает захват и освобождает ресурсы."""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def process_camera(self) -> None:
        """Обрабатывает один кадр с веб-камеры."""
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
        self.last_time = current_time

        # Подготовка изображения
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Инференс
        results = model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)
        best_emotion, detected = self._process_results(results, frame_bgr)

        # Формирование текста
        emotion_text = "Лицо не распознано" if not detected else (best_emotion or "—")
        self.frame_ready.emit(frame_bgr)
        self.info_ready.emit(f"FPS: {fps:.1f} | Эмоция: {emotion_text}")

    def process_photo(self, path: str) -> None:
        """
        Обрабатывает загруженное изображение.

        Args:
            path (str): Путь к файлу изображения.
        """
        img = cv2.imread(path)
        if img is None:
            self.info_ready.emit("Ошибка загрузки")
            return

        # Масштабирование под рамку
        h, w = img.shape[:2]
        scale = min(FRAME_WIDTH / w, FRAME_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Создание холста
        canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        x = (FRAME_WIDTH - new_w) // 2
        y = (FRAME_HEIGHT - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = img

        # Инференс
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        input_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results = model(input_frame, conf=CONF_THRESHOLD, verbose=False)
        best_emotion, detected = self._process_results(results, canvas)

        # Результат
        emotion_text = ("Лицо не распознано, попробуйте другую фотографию"
                        if not detected else (best_emotion or "—"))
        self.frame_ready.emit(canvas)
        self.info_ready.emit(emotion_text)

    def _process_results(self, results: Any, frame: np.ndarray) -> Tuple[Optional[str], bool]:
        """
        Обрабатывает результаты YOLO и рисует рамки.

        Args:
            results: Результаты от model.predict().
            frame (np.ndarray): Кадр для рисования.

        Returns:
            Tuple[Optional[str], bool]: Лучшая эмоция и флаг обнаружения.
        """
        best_emotion: Optional[str] = None
        best_conf: float = 0.0
        detected: bool = False

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            detected = True
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                if conf > best_conf:
                    best_conf = conf
                    best_emotion = f"{CLASS_NAMES[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return best_emotion, detected


# Главное меню
class MainMenu(QMainWindow):
    """Главное меню приложения."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Эмоции")
        self.setFixedSize(400, 500)
        self.setStyleSheet(self._get_style())
        self._setup_ui()

    def _get_style(self) -> str:
        return """
        QMainWindow { background-color: #1a1a1a; }
        QPushButton {
            background-color: #2d2d2d; color: white; border: none;
            padding: 16px; border-radius: 16px; font: bold 14px 'Segoe UI';
            min-height: 50px;
        }
        QPushButton:hover { background-color: #3d3d3d; }
        QPushButton:pressed { background-color: #1a1a1a; }
        """

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 60, 40, 60)
        layout.setSpacing(20)

        title = QLabel("Выберите режим")
        title.setStyleSheet("color: #4CAF50; font: bold 20px; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        btn_photo = QPushButton("Загрузить фото")
        btn_photo.clicked.connect(self._open_photo_mode)
        layout.addWidget(btn_photo)

        btn_camera = QPushButton("Включить вебкамеру")
        btn_camera.clicked.connect(self._open_camera_mode)
        layout.addWidget(btn_camera)

        btn_exit = QPushButton("Закрыть программу")
        btn_exit.clicked.connect(QApplication.quit)
        layout.addWidget(btn_exit)

        layout.addStretch()

    def _open_photo_mode(self) -> None:
        global photo_window
        self.hide()
        photo_window = PhotoWindow()
        photo_window.show()

    def _open_camera_mode(self) -> None:
        global camera_window
        self.hide()
        camera_window = CameraWindow()
        camera_window.show()


# Окно загрузки фото
class PhotoWindow(QMainWindow):
    """Окно для загрузки и анализа фото."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Загрузка фото")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(self._get_style())
        self.worker = InferenceWorker()
        self._connect_worker()
        self._setup_ui()

    def _get_style(self) -> str:
        return """
        QMainWindow { background-color: #1e1e1e; }
        QLabel#frame {
            background-color: #000; 
            border: 2px solid #555; 
            color: #888; 
            font: 16px 'Segoe UI';
        }
        QLabel#frame[hasImage="true"] {
            border: 2px solid #333;
        }
        QPushButton {
            background-color: #2d2d2d; color: white; border: none;
            padding: 12px 20px; border-radius: 12px; font: bold 13px;
        }
        QPushButton:hover { background-color: #3d3d3d; }
        """

    def _connect_worker(self) -> None:
        self.worker.frame_ready.connect(self._show_image)
        self.worker.info_ready.connect(self._update_emotion_label)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("Детекция по фото")
        title.setStyleSheet("color: #4CAF50; font: bold 18px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Метка эмоции
        self.info_label = create_info_label("Эмоция: —")
        layout.addWidget(self.info_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Рамка изображения
        self.frame_label = QLabel("Нажмите, чтобы загрузить фото")
        self.frame_label.setObjectName("frame")
        self.frame_label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setScaledContents(False)
        self.frame_label.setProperty("hasImage", False)
        self.frame_label.mousePressEvent = self._load_photo
        layout.addWidget(self.frame_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Отступ перед кнопками
        layout.addStretch()

        # Кнопки внизу
        btns = QHBoxLayout()
        btn_webcam = QPushButton("Включить веб-камеру")
        btn_webcam.clicked.connect(self._open_camera)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self._close_and_back)
        btns.addWidget(btn_webcam)
        btns.addWidget(btn_close)
        layout.addLayout(btns)

    def _load_photo(self, event: Optional[Any] = None) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите фото", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.frame_label.clear()
            self.frame_label.setText("Обработка...")
            self.frame_label.setProperty("hasImage", False)
            self.frame_label.setStyleSheet(self.styleSheet())
            self.worker.process_photo(path)

    def _update_emotion_label(self, text: str) -> None:
        if not text.startswith("Эмоция:"):
            text = f"Эмоция: {text}"
        self.info_label.setText(text)

    def _show_image(self, frame_bgr: np.ndarray) -> None:
        qimg = bgr_to_qimage(frame_bgr)
        pixmap = QPixmap.fromImage(qimg)
        scaled = scale_pixmap(pixmap, FRAME_WIDTH, FRAME_HEIGHT)
        self.frame_label.setPixmap(scaled)
        self.frame_label.setProperty("hasImage", True)
        self.frame_label.setStyleSheet(self.styleSheet())

    def _open_camera(self) -> None:
        global camera_window
        self.close()
        camera_window = CameraWindow()
        camera_window.show()

    def _close_and_back(self) -> None:
        self.close()
        global main_menu
        main_menu = MainMenu()
        main_menu.show()

    def closeEvent(self, event: Any) -> None:
        self.worker.stop()
        event.accept()


# Окно веб-камеры
class CameraWindow(QMainWindow):
    """Окно для анализа видео с веб-камеры."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Детекция по веб-камере")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(self._get_style())
        self.worker = InferenceWorker()
        self.timer = QTimer()
        self._setup_ui()
        self._start_camera()

    def _get_style(self) -> str:
        return """
        QMainWindow { background-color: #1e1e1e; }
        QLabel#video {
            background-color: #000; border-radius: 16px; border: 2px solid #333;
        }
        QPushButton {
            background-color: #2d2d2d; color: white; border: none;
            padding: 10px 16px; border-radius: 10px; font: 13px;
        }
        QPushButton:hover { background-color: #3d3d3d; }
        """

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("Веб-камера")
        title.setStyleSheet("color: #4CAF50; font: bold 18px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Информация: FPS + Эмоция
        info_layout = QHBoxLayout()
        info_layout.setSpacing(20)
        self.fps_label = create_info_label("FPS: 0.0")
        self.emotion_label = create_info_label("Эмоция: —")
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.emotion_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(info_layout)

        # Видеопоток
        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Кнопки
        btns = QHBoxLayout()
        btn_photo = QPushButton("Загрузить фото")
        btn_photo.clicked.connect(self._open_photo)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self._close_and_back)
        btns.addWidget(btn_photo)
        btns.addWidget(btn_close)
        layout.addLayout(btns)

    def _start_camera(self) -> None:
        if self.worker.start_camera():
            self.timer.timeout.connect(self.worker.process_camera)
            self.timer.start(10)
            self.worker.frame_ready.connect(self._show_frame)
            self.worker.info_ready.connect(self._update_info)

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        qimg = bgr_to_qimage(frame_bgr)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def _update_info(self, text: str) -> None:
        parts = text.split(" | ")
        if len(parts) == 2:
            self.fps_label.setText(parts[0])
            self.emotion_label.setText(parts[1])
        else:
            self.emotion_label.setText(text)

    def _open_photo(self) -> None:
        global photo_window
        self.close()
        photo_window = PhotoWindow()
        photo_window.show()

    def _close_and_back(self) -> None:
        self.close()
        global main_menu
        main_menu = MainMenu()
        main_menu.show()

    def closeEvent(self, event: Any) -> None:
        self.worker.stop()
        self.timer.stop()
        event.accept()


# Запуск приложения
def main() -> None:
    """
    Точка входа в приложение.
    Инициализирует Qt, загружает модель и запускает главное меню.
    """
    global app, main_menu, model

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    model = load_model(MODEL_PATH)

    main_menu = MainMenu()
    main_menu.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()