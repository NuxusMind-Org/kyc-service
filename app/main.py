from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from starlette.concurrency import run_in_threadpool

from .comparison import compare
from .engine import face_engine
from .images import read_image
from .settings import settings
from .validation import inspect


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(face_engine)
    yield


app = FastAPI(title="Face comparison service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/faces/validate")
async def validate_face(image: UploadFile = File(...)):
    decoded = await read_image(image)
    faces = await run_in_threadpool(face_engine().detect, decoded)
    return {**inspect(decoded, faces), "guideBox": settings.guide_box}


@app.post("/v1/faces/compare")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    decoded1 = await read_image(image1)
    decoded2 = await read_image(image2)
    faces1 = await run_in_threadpool(face_engine().detect, decoded1)
    faces2 = await run_in_threadpool(face_engine().detect, decoded2)
    return await run_in_threadpool(compare, decoded1, faces1, decoded2, faces2)
