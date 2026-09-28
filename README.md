# Free, self-hosted face comparison service

This is a Python FastAPI service for a Spring Boot + React application. React
uploads photos to Spring Boot; Spring forwards them to this service. The service
uses OpenCV Zoo's **YuNet** face detector and **SFace** face recognizer. YuNet's
model directory is MIT licensed and SFace's model directory says all its files
are Apache-2.0 licensed, including the model binaries. The Docker build downloads
the exact models, checks SHA-256 digests, and retains their license notices in
`/app/model-assets/`. No recognition API, model license purchase, API key, or
per-comparison fee is needed.

Sources: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
and https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface .
OpenCV's separate SFace training-data provenance has been questioned in an open
issue: https://github.com/opencv/opencv_zoo/issues/313 . Its published Apache-2.0
license is the basis of this implementation; nobody can guarantee zero legal
uncertainty in third-party pretrained weights. The models are free to use under
their published terms. Cloud Run can still charge for compute, builds, network,
and storage above applicable free allowances.

## API

`POST /v1/faces/validate` accepts multipart `image`. It detects faces and
reports issues with guide placement, size, tilt, and blur. A valid example:

```json
{
  "valid": true,
  "faceCount": 1,
  "issues": [],
  "face": {
    "box": [0.3, 0.2, 0.7, 0.8],
    "eyeTiltDegrees": 2.5,
    "sharpness": 153.4,
    "detectionConfidence": 0.99
  },
  "guideBox": [0.12, 0.05, 0.88, 0.95]
}
```

Boxes are `[left, top, right, bottom]` fractions of the upright photo, not
the browser CSS preview. An unsuitable photo returns HTTP 200 with
`valid: false` and one or more `issues` values: `no_face`, `multiple_faces`,
`face_outside_guide`, `face_clipped`, `face_too_small`, `head_tilted`,
`face_blurry`. These are starting heuristics, not a liveness test or complete
pose assessment. Match your React guide to the actual captured photo after
handling CSS mirroring and image orientation. Tune thresholds in
`app/settings.py` using your own camera samples.

`POST /v1/faces/compare` accepts multipart `image1` and `image2`. It detects
and validates **both current uploads again**, aligns each face using YuNet's
landmarks, and computes SFace cosine similarity. A successful result looks like:

```json
{"cosineSimilarity": 0.713291, "decision": "threshold_not_calibrated"}
```

Invalid photos return HTTP 422 with diagnostic objects at `detail.image1` and
`detail.image2`. A cosine score is **not** the probability of a match or a
percentage. Calibrate a threshold with representative same-person and
different-person pairs before adding a `samePerson` decision; don't use it alone
for consequential identity checks. The service stores neither photos nor
embeddings. `GET /health` is available for health checks.

## Run locally

From this folder:

```bash
docker build -t face-service .
docker run --rm -p 8080:8080 face-service
```

The build needs access to OpenCV's official Hugging Face model repository and
GitHub to fetch model binaries and notices. The binaries are SHA-256 checked.

In a second terminal:

```bash
curl -F "image=@first.jpg" http://localhost:8080/v1/faces/validate
curl -F "image1=@first.jpg" -F "image2=@second.jpg" http://localhost:8080/v1/faces/compare
```

Uploads may be JPEG, PNG, or WebP; each is limited to 5 MiB and 12 megapixels.
Processing is in memory. There are no external inference calls.

## Deploy to Google Cloud Run

After selecting a Google Cloud project and signing into `gcloud`, from this
folder:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy face-service \
  --source . \
  --region europe-west1 \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 1 \
  --timeout 120s \
  --no-allow-unauthenticated
```

Cloud Run builds the Dockerfile. Give the service account used by Spring Boot
the `roles/run.invoker` role on `face-service`, and have Spring send a Google
ID token with the Cloud Run base URL as audience in the outbound multipart
request. The browser calls Spring, not this private service. See
https://cloud.google.com/run/docs/authenticating/service-to-service .

For best latency and cost, Spring Boot can also host the same two OpenCV models
directly using OpenCV Java instead of a second Cloud Run service; this Python
project preserves the service split requested here.
