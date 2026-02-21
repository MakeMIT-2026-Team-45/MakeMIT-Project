import argparse
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error, request

from PIL import Image


@dataclass
class PiClientConfig:
    endpoint: str
    interval_sec: float
    image_path: Optional[Path]
    one_shot: bool


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
    args = parser.parse_args()
    return PiClientConfig(
        endpoint=args.endpoint,
        interval_sec=args.interval_sec,
        image_path=args.image_path,
        one_shot=args.one_shot,
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
    req = request.Request(
        url=endpoint,
        data=raw_jpeg,
        method="POST",
        headers={"Content-Type": "image/jpeg"},
    )
    with request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)


def get_frame_bytes(image_path: Optional[Path]) -> bytes:
    if image_path is not None:
        return jpeg_bytes_from_path(image_path)
    return jpeg_bytes_from_picamera2()


def run(config: PiClientConfig) -> None:
    print(f"Sending frames to: {config.endpoint}")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            frame = get_frame_bytes(config.image_path)
            result = post_jpeg(config.endpoint, frame)
            cls = result.get("most_likely_class")
            prob = result.get("most_likely_probability")
            print(f"class={cls!r} prob={prob} bytes={len(frame)}")
        except ImportError:
            print(
                "picamera2 is not installed. Install it on Raspberry Pi or pass "
                "--image-path /path/to/sample.jpg"
            )
            break
        except FileNotFoundError as exc:
            print(exc)
            break
        except error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error {exc.code}: {msg}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")

        if config.one_shot:
            break
        time.sleep(config.interval_sec)


if __name__ == "__main__":
    run(parse_args())
