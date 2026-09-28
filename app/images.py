from io import BytesIO

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .settings import settings


async def read_image(upload: UploadFile) -> np.ndarray:
    raw = await upload.read(settings.max_upload_bytes + 1)
    await upload.close()
    if not raw or len(raw) > settings.max_upload_bytes:
        raise HTTPException(413, "Image is empty or exceeds 5 MiB")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Use JPEG, PNG, or WebP")
            if image.width * image.height > settings.max_pixels:
                raise HTTPException(413, "Image exceeds 12 megapixels")
            upright = ImageOps.exif_transpose(image).convert("RGB")
            upright.thumbnail((settings.max_side, settings.max_side))
            if min(upright.size) < 128:
                raise HTTPException(422, "Photo resolution is too low")
            rgb = np.asarray(upright)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(415, "Invalid or unsupported image") from None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
