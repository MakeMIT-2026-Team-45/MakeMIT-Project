import argparse
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


@dataclass
class PiClientConfig:
    endpoint: str
    fps: float
    ai_every_n_frames: int
    image_path: Optional[Path]
    one_shot: bool
    robot_id: int


def parse_args() -> PiClientConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Basic Raspberry Pi client boilerplate: capture/load JPEG frames and "
            "send them to the backend inference endpoint."
        )
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://127.0.0.1:8000/torch-test-video",
        help="Backend endpoint URL.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target frames per second for the live video feed.",
    )
    parser.add_argument(
        "--ai-every-n-frames",
        type=int,
        default=30,
        help="Run AI inference once every N frames (default: every 30 = ~1/sec at 30fps).",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help=(
            "Optional path to a JPEG image. If omitted, the script tries to capture "
            "from picamera2."
        ),
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Send one frame and exit.",
    )
    parser.add_argument(
        "--robot-id",
        type=int,
        default=1,
        help="Robot ID used for the video-frame push endpoint.",
    )
    args = parser.parse_args()
    return PiClientConfig(
        endpoint=args.endpoint,
        fps=args.fps,
        ai_every_n_frames=args.ai_every_n_frames,
        image_path=args.image_path,
        one_shot=args.one_shot,
        robot_id=args.robot_id,
    )


def jpeg_bytes_from_path(image_path: Path) -> bytes:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def open_picamera2():
    from picamera2 import Picamera2  # type: ignore

    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (640, 480)})
    picam.configure(config)
    picam.start()
    time.sleep(0.5)  # let AEC/AWB settle once at startup
    return picam


def capture_jpeg_from_picamera2(picam) -> bytes:
    frame = picam.capture_array("main")
    img = Image.fromarray(frame).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def post_jpeg(endpoint: str, raw_jpeg: bytes) -> dict:
    resp = requests.post(
        url=endpoint,
        data=raw_jpeg,
        headers={"Content-Type": "image/jpeg"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def _base_url(endpoint: str) -> str:
    from urllib.parse import urlparse
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        endpoint = "https://" + endpoint
    p = urlparse(endpoint)
    return f"{p.scheme}://{p.netloc}"


def _run_ai_inference(endpoint: str, frame: bytes, frame_count: int) -> None:
    """Runs in a background thread so it never blocks the video loop."""
    try:
        result = post_jpeg(endpoint, frame)
        cls = result.get("prediction")
        prob = result.get("max_probability")
        label = "recycling" if cls == 1 else "trash"
        print(f"[frame {frame_count}] {label} ({prob:.1%}) bytes={len(frame)}")
    except requests.HTTPError as exc:
        print(f"[AI] HTTP error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        print(f"[AI] error: {exc}")


def run(config: PiClientConfig) -> None:
    base_url = _base_url(config.endpoint)
    stream_url = f"{base_url}/video-frame/{config.robot_id}"
    frame_interval = 1.0 / config.fps
    session = requests.Session()  # reuses TLS connection across frames

    print(f"Inference endpoint : {config.endpoint}")
    print(f"Stream push URL    : {stream_url}")
    print(f"Target FPS         : {config.fps}")
    print(f"AI every N frames  : {config.ai_every_n_frames}")
    print("Press Ctrl+C to stop.\n")

    picam = None
    if config.image_path is None:
        try:
            picam = open_picamera2()
        except ImportError:
            print(
                "picamera2 is not installed. Install it on Raspberry Pi or pass "
                "--image-path /path/to/sample.jpg"
            )
            return

    frame_count = 0
    try:
        while True:
            t_start = time.monotonic()

            try:
                if config.image_path is not None:
                    frame = jpeg_bytes_from_path(config.image_path)
                else:
                    frame = capture_jpeg_from_picamera2(picam)
            except FileNotFoundError as exc:
                print(exc)
                break

            frame_count += 1

            # Push every frame to the MJPEG stream (non-blocking best-effort)
            try:
                session.post(
                    stream_url,
                    data=frame,
                    headers={"Content-Type": "image/jpeg"},
                    timeout=(2, 2),  # (connect_timeout, read_timeout) in seconds
                )
            except requests.exceptions.Timeout:
                pass  # read timeout is expected on best-effort stream push
            except Exception as exc:
                print(f"[stream] {exc}")

            # Fire AI inference in a background thread — never blocks the video loop
            if frame_count % config.ai_every_n_frames == 0:
                threading.Thread(
                    target=_run_ai_inference,
                    args=(f"{base_url}/torch-test-video?robot_id={config.robot_id}", frame, frame_count),
                    daemon=True,
                ).start()

            if config.one_shot:
                break

            # Sleep only the remaining time in this frame period
            elapsed = time.monotonic() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if picam is not None:
            picam.stop()
            picam.close()


if __name__ == "__main__":
    run(parse_args())
