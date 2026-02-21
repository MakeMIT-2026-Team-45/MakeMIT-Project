import io
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from PIL import Image
import torch
from torchvision import transforms

from train_taco_mobilenet_binary import MobileNetBinaryHead

app = FastAPI()

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