"""Fetch exact OpenCV Zoo model binaries and their license notices during build."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


ASSETS = {
    "face_detection_yunet/face_detection_yunet_2023mar.onnx":
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface/face_recognition_sface_2021dec.onnx":
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}

base = Path(__file__).resolve().parent
for source, expected in ASSETS.items():
    name = Path(source).name
    url = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/" + source
    with urlopen(url, timeout=120) as stream:
        content = stream.read()
    if sha256(content).hexdigest() != expected:
        raise RuntimeError(f"Unexpected model contents: {name}")
    (base / name).write_bytes(content)

for directory, name in (("face_detection_yunet", "YUNET-MIT.txt"),
                        ("face_recognition_sface", "SFACE-APACHE-2.0.txt")):
    url = ("https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
           f"{directory}/LICENSE")
    with urlopen(url, timeout=30) as stream:
        license_text = stream.read()
    if len(license_text) < 1000:
        raise RuntimeError(f"License notice missing: {name}")
    (base / name).write_bytes(license_text)
