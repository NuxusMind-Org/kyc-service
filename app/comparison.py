import math

import numpy as np
from fastapi import HTTPException

from .engine import face_engine
from .validation import inspect


def compare(image1: np.ndarray, faces1: list[np.ndarray],
            image2: np.ndarray, faces2: list[np.ndarray]) -> dict:
    # Never trust an earlier /validate response for these submitted bytes.
    first, second = inspect(image1, faces1), inspect(image2, faces2)
    if not first["valid"] or not second["valid"]:
        raise HTTPException(422, {"message": "Image validation failed",
                                  "image1": first, "image2": second})
    score = face_engine().similarity(image1, faces1[0], image2, faces2[0])
    if not math.isfinite(score):
        raise HTTPException(422, "Could not compute similarity")
    if score > 0.65:
        return {"cosineSimilarity": round(score, 6),
            "decision": "same_person"}
    return {"cosineSimilarity": round(score, 6),
            "decision": "not_same_person"}
