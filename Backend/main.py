import asyncio
import io
import json
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms
import paho.mqtt.client as mqtt_client

from train_taco_mobilenet_binary import MobileNetBinaryHead

app = FastAPI()

# --- MJPEG frame store (one latest frame per robot_id) ---
_latest_frame: dict[int, bytes] = {}
_frame_events: dict[int, asyncio.Event] = defaultdict(asyncio.Event)

# --- MQTT client (publishes Pi telemetry to the broker for the frontend) ---
mqtt = mqtt_client.Client()
_mqtt_available = False
try:
    mqtt.connect("localhost", 1880)
    mqtt.loop_start()
    _mqtt_available = True
except Exception as e:
    print(f"Warning: Could not connect to MQTT broker — telemetry endpoints will log-only ({e})")


class CapacityPayload(BaseModel):
    trashCapacity: float
    recycleCapacity: float
    batteryPercentage: float


class LocationPayload(BaseModel):
    lat: float
    lng: float


# --- Telemetry endpoints (called by the Raspberry Pi) ---

@app.post("/robot/{robot_id}/telemetry/capacity")
async def post_capacity(robot_id: int, payload: CapacityPayload):
    if _mqtt_available:
        mqtt.publish(
            f"robot/{robot_id}/telemetry/capacity",
            json.dumps(payload.model_dump()),
        )
    else:
        print(f"[no broker] capacity: {payload.model_dump()}")
    return {"ok": True}


@app.post("/robot/{robot_id}/telemetry/location")
async def post_location(robot_id: int, payload: LocationPayload):
    if _mqtt_available:
        mqtt.publish(
            f"robot/{robot_id}/telemetry/location",
            json.dumps(payload.model_dump()),
        )
    else:
        print(f"[no broker] location: {payload.model_dump()}")
    return {"ok": True}


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint_path = Path(__file__).with_name("checkpoints") / "mobilenetv3_taco_binary_best.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Trained checkpoint not found: {checkpoint_path}. Run training first."
    )

checkpoint = torch.load(checkpoint_path, map_location=device)
model = MobileNetBinaryHead().to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

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
    x = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    # Binary mapping from training:
    # 0 -> trash, 1 -> recycling
    prediction = int(top_idx.item())
    max_probability = float(top_prob.item())

    return {
        "content_type": content_type,
        "size_bytes": len(raw),
        "prediction": prediction,
        "max_probability": max_probability,
    }


# --- MJPEG video pipeline ---

@app.post("/video-frame/{robot_id}")
async def push_frame(robot_id: int, request: Request):
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty body")
    _latest_frame[robot_id] = raw
    _frame_events[robot_id].set()
    return {"ok": True}


@app.get("/video-feed/{robot_id}")
async def video_feed(robot_id: int):
    async def generate():
        while True:
            event = _frame_events[robot_id]
            await event.wait()
            event.clear()
            frame = _latest_frame.get(robot_id)
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )