import argparse
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


@dataclass
class PiClientConfig:
    endpoint: str
    interval_sec: float
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
        "--interval-sec",
        type=float,
        default=1.0,
        help="Time between frame uploads in loop mode.",
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
        interval_sec=args.interval_sec,
        image_path=args.image_path,
        one_shot=args.one_shot,
        robot_id=args.robot_id,
    )


def jpeg_bytes_from_path(image_path: Path) -> bytes:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def jpeg_bytes_from_picamera2() -> bytes:
    from picamera2 import Picamera2  # type: ignore

    picam = Picamera2()
    try:
        config = picam.create_still_configuration()
        picam.configure(config)
        picam.start()
        time.sleep(0.3)
        frame = picam.capture_array("main")
        img = Image.fromarray(frame).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    finally:
        picam.stop()
        picam.close()


def post_jpeg(endpoint: str, raw_jpeg: bytes) -> dict:
    resp = requests.post(
        url=endpoint,
        data=raw_jpeg,
        headers={"Content-Type": "image/jpeg"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def get_frame_bytes(image_path: Optional[Path]) -> bytes:
    if image_path is not None:
        return jpeg_bytes_from_path(image_path)
    return jpeg_bytes_from_picamera2()


def _base_url(endpoint: str) -> str:
    from urllib.parse import urlparse
    p = urlparse(endpoint)
    return f"{p.scheme}://{p.netloc}"


def run(config: PiClientConfig) -> None:
    base_url = _base_url(config.endpoint)
    stream_url = f"{base_url}/video-frame/{config.robot_id}"
    print(f"Inference endpoint : {config.endpoint}")
    print(f"Stream push URL    : {stream_url}")
    print("Press Ctrl+C to stop.\n")

    frame_count = 0
    while True:
        try:
            frame = get_frame_bytes(config.image_path)
            frame_count += 1

            # Always push frame to the MJPEG stream
            requests.post(
                stream_url,
                data=frame,
                headers={"Content-Type": "image/jpeg"},
                timeout=5,
            )

            # Run AI inference every 10th frame
            if frame_count % 10 == 0:
                result = post_jpeg(config.endpoint, frame)
                cls = result.get("prediction")
                prob = result.get("max_probability")
                label = "recycling" if cls == 1 else "trash"
                print(f"[frame {frame_count}] {label} ({prob:.1%}) bytes={len(frame)}")

        except ImportError:
            print(
                "picamera2 is not installed. Install it on Raspberry Pi or pass "
                "--image-path /path/to/sample.jpg"
            )
            break
        except FileNotFoundError as exc:
            print(exc)
            break
        except requests.HTTPError as exc:
            print(f"HTTP error {exc.response.status_code}: {exc.response.text}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")

        if config.one_shot:
            break
        time.sleep(config.interval_sec)


if __name__ == "__main__":
    run(parse_args())
