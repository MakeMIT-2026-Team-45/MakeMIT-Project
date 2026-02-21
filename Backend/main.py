import io

from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from ultralytics import YOLO

app = FastAPI()

device = "mps" if torch.backends.mps.is_available() else "cpu"
weights_path = hf_hub_download(
    repo_id="kendrickfff/waste-classification-yolov8-ken",
    filename="yolov8n-waste-12cls-best.pt",
)
model = YOLO(weights_path).to(device)

@app.get("/")
async def root():
    return {"message": "Backend is running"}


@app.post("/torch-test-video")
async def torch_test_video(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty request body")

    # Accept JPEG BLOB payloads over HTTP POST.
    if content_type not in {"image/jpeg", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

    # Basic JPEG signature validation (SOI/EOI markers).
    if len(raw) < 4 or raw[0] != 0xFF or raw[1] != 0xD8 or raw[-2] != 0xFF or raw[-1] != 0xD9:
        raise HTTPException(status_code=400, detail="Body is not a valid JPEG binary payload")

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    result = model(image, verbose=False)[0]

    names = result.names if isinstance(result.names, dict) else {}
    num_classes = len(names)
    if num_classes == 0:
        raise HTTPException(status_code=500, detail="Model did not provide class names")

    # This HF checkpoint is object detection. Convert detections into class scores
    # by taking max confidence per class.
    class_scores = torch.zeros(num_classes, dtype=torch.float32)
    if result.boxes is not None and len(result.boxes) > 0:
        cls_ids = result.boxes.cls.tolist()
        confs = result.boxes.conf.tolist()
        for cls_id, conf in zip(cls_ids, confs):
            idx = int(cls_id)
            class_scores[idx] = max(class_scores[idx], float(conf))

    top_idx = int(torch.argmax(class_scores).item())
    top_prob = float(class_scores[top_idx].item())
    most_likely_class = names.get(top_idx, str(top_idx)) if top_prob > 0 else None

    return {
        "content_type": content_type,
        "size_bytes": len(raw),
        "most_likely_class": most_likely_class,
        "most_likely_probability": top_prob,
    }