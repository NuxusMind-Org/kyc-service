import math

import cv2
import numpy as np

from .settings import settings


def inspect(image: np.ndarray, faces: list[np.ndarray]) -> dict:
    height, width = image.shape[:2]
    result = {"valid": False, "faceCount": len(faces), "issues": [], "face": None}
    if len(faces) != 1:
        result["issues"].append("no_face" if not faces else "multiple_faces")
        return result

    face = faces[0]
    x1, y1, face_width, face_height = map(float, face[:4])
    x2, y2 = x1 + face_width, y1 + face_height
    box = [round(x1 / width, 4), round(y1 / height, 4),
           round(x2 / width, 4), round(y2 / height, 4)]
    gx1, gy1, gx2, gy2 = settings.guide_box
    issues = result["issues"]
    if box[0] < gx1 or box[1] < gy1 or box[2] > gx2 or box[3] > gy2:
        issues.append("face_outside_guide")
    if x1 < 0.01 * width or y1 < 0.01 * height or x2 > 0.99 * width or y2 > 0.99 * height:
        issues.append("face_clipped")
    if face_height / height < settings.min_face_height:
        issues.append("face_too_small")

    # YuNet: x,y,w,h; eye landmarks at indices 4-7; score at index 14.
    eye_tilt = round(math.degrees(math.atan2(
        abs(float(face[5] - face[7])), abs(float(face[4] - face[6]))
    )), 1)
    if eye_tilt > settings.max_eye_tilt_degrees:
        issues.append("head_tilted")

    crop = image[max(0, int(y1)):min(height, int(y2)),
                 max(0, int(x1)):min(width, int(x2))]
    sharpness = None
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
        if sharpness < settings.min_sharpness:
            issues.append("face_blurry")
    else:
        issues.append("face_clipped")

    result["face"] = {"box": box, "eyeTiltDegrees": eye_tilt,
                      "sharpness": sharpness, "detectionConfidence": round(float(face[14]), 4)}
    result["valid"] = not issues
    return result
