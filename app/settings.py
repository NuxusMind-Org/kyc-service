import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_dir: str = os.getenv("MODEL_DIR", "/app/model-assets")
    max_upload_bytes: int = 5 * 1024 * 1024
    max_pixels: int = 12_000_000
    max_side: int = 640
    # Fractions of the decoded, upright photo. Match the React overlay to this box.
    guide_box: tuple[float, float, float, float] = (0.12, 0.05, 0.88, 0.95)
    min_face_height: float = 0.25
    max_eye_tilt_degrees: float = 15.0
    min_sharpness: float = 40.0  # Starting heuristic; tune with your samples.


settings = Settings()
