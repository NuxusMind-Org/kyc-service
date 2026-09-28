from functools import lru_cache
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from .settings import settings


class FaceEngine:
    def __init__(self):
        model_dir = Path(settings.model_dir)
        self.detector = cv2.FaceDetectorYN.create(
            str(model_dir / "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            score_threshold=0.7, nms_threshold=0.3)
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(model_dir / "face_recognition_sface_2021dec.onnx"), "")
        self.lock = RLock()  # FaceDetectorYN.setInputSize mutates shared state.

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        height, width = image.shape[:2]
        with self.lock:
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(image)
        return [] if faces is None else list(faces)

    def similarity(self, image1: np.ndarray, face1: np.ndarray,
                   image2: np.ndarray, face2: np.ndarray) -> float:
        with self.lock:
            crop1 = self.recognizer.alignCrop(image1, face1)
            crop2 = self.recognizer.alignCrop(image2, face2)
            feature1 = self.recognizer.feature(crop1)
            feature2 = self.recognizer.feature(crop2)
            return float(self.recognizer.match(
                feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE))


@lru_cache(maxsize=1)
def face_engine() -> FaceEngine:
    return FaceEngine()
