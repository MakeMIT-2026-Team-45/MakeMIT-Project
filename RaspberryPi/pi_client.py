import argparse
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Pickup arm servo
# ---------------------------------------------------------------------------
#
# Physical angle convention (from the diagrams):
#   0°  = arm pointing straight down / fully lowered  (zero.png)
#  20°  = resting position, slightly raised            (resting.png)
#  45°  = maximum raised position for pickup           (maximum.png)
#
# The servo duty cycle is REVERSED:
#   min_dc (12.5%) → 0°   (arm down)
#   max_dc  (2.5%) → 180° (arm fully up, unused)
# We map only the 0–45° working range within that reversed scale.
#
# BCM GPIO pin: 12  (physical pin 32, hardware PWM channel 0)
# Servo signal wired to this pin; VCC → external 5V; GND → common GND.

PICKUP_SERVO_PIN   = 12
PICKUP_FREQ_HZ     = 50.0
PICKUP_MIN_DC      = 12.5   # duty cycle at 0°  (arm down)
PICKUP_MAX_DC      = 2.5    # duty cycle at 180° (arm fully up)
PICKUP_MAX_DEG     = 180.0  # full servo range used for DC mapping

ANGLE_ZERO    =  0.0   # arm fully lowered
ANGLE_REST    = 20.0   # resting position
ANGLE_MAX     = 45.0   # maximum pickup height

PICKUP_CONFIDENCE_THRESHOLD = 0.70   # trigger when recycling prob > 70%


class PickupServo:
    """
    Controls the pickup arm servo.

    All angles are in the physical arm convention:
      0° = arm down, 20° = rest, 45° = maximum.
    The reversed duty-cycle mapping is handled internally.
    """

    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
        self._lock = threading.Lock()
        self._busy = False

        if not mock:
            import RPi.GPIO as GPIO  # type: ignore
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(PICKUP_SERVO_PIN, GPIO.OUT)
            self._pwm = GPIO.PWM(PICKUP_SERVO_PIN, PICKUP_FREQ_HZ)
            self._pwm.start(0)
            time.sleep(0.1)

    def _angle_to_dc(self, angle_deg: float) -> float:
        """Map physical arm angle → PWM duty cycle (reversed mapping)."""
        angle_deg = max(ANGLE_ZERO, min(PICKUP_MAX_DEG, angle_deg))
        # Reversed: 0° → min_dc, 180° → max_dc, but min_dc > max_dc
        return PICKUP_MIN_DC + (angle_deg / PICKUP_MAX_DEG) * (PICKUP_MAX_DC - PICKUP_MIN_DC)

    def _move_to(self, angle_deg: float, settle_sec: float = 0.5) -> None:
        """Move to angle and wait for the servo to settle."""
        if self._mock:
            print(f"[arm] → {angle_deg:.0f}°")
            time.sleep(settle_sec)
            return
        dc = self._angle_to_dc(angle_deg)
        self._pwm.ChangeDutyCycle(dc)
        time.sleep(settle_sec)
        self._pwm.ChangeDutyCycle(0)  # release hold to stop jitter

    def move_to_rest(self) -> None:
        """Move arm to resting position (20°)."""
        self._move_to(ANGLE_REST)

    def pickup(self) -> None:
        """
        Run the full pickup sequence in the calling thread:
          rest (20°) → zero (0°) → max (45°) → rest (20°)
        Skips if a pickup is already in progress.
        """
        with self._lock:
            if self._busy:
                return
            self._busy = True
        try:
            print("[arm] pickup triggered")
            self._move_to(ANGLE_ZERO,  settle_sec=0.6)  # lower to zero
            self._move_to(ANGLE_MAX,   settle_sec=0.8)  # scoop up
            self._move_to(ANGLE_REST,  settle_sec=0.6)  # return to rest
            print("[arm] pickup complete")
        finally:
            with self._lock:
                self._busy = False

    def cleanup(self) -> None:
        if not self._mock:
            self._pwm.stop()
            self._gpio.cleanup(PICKUP_SERVO_PIN)


@dataclass
class PiClientConfig:
    endpoint: str
    fps: float
    ai_every_n_frames: int
    image_path: Optional[Path]
    one_shot: bool
    robot_id: int
    mock_servo: bool


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
        default=15,
        help="Run AI inference once every N frames (default: every 15 = ~1/sec at 15fps).",
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
    parser.add_argument(
        "--mock-servo",
        action="store_true",
        help="Simulate the pickup servo (prints angles instead of driving GPIO).",
    )
    args = parser.parse_args()
    return PiClientConfig(
        endpoint=args.endpoint,
        fps=args.fps,
        ai_every_n_frames=args.ai_every_n_frames,
        image_path=args.image_path,
        one_shot=args.one_shot,
        robot_id=args.robot_id,
        mock_servo=args.mock_servo,
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


def _run_ai_inference(
    endpoint: str,
    frame: bytes,
    frame_count: int,
    servo: "PickupServo",
) -> None:
    """
    Runs in a background thread so it never blocks the video loop.
    If recycling confidence exceeds PICKUP_CONFIDENCE_THRESHOLD, triggers
    the pickup arm sequence (also in this thread, so the arm moves once
    the inference result is known).
    """
    try:
        result = post_jpeg(endpoint, frame)
        cls = result.get("prediction")
        prob = result.get("max_probability", 0.0)
        label = "recycling" if cls == 1 else "trash"
        print(f"[frame {frame_count}] {label} ({prob:.1%}) bytes={len(frame)}")

        if cls == 1 and prob >= PICKUP_CONFIDENCE_THRESHOLD:
            print(f"[AI] recycling confidence {prob:.1%} ≥ {PICKUP_CONFIDENCE_THRESHOLD:.0%} — triggering pickup")
            servo.pickup()

    except requests.HTTPError as exc:
        print(f"[AI] HTTP error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        print(f"[AI] error: {exc}")


def run(config: PiClientConfig) -> None:
    base_url = _base_url(config.endpoint)
    stream_url = f"{base_url}/video-frame/{config.robot_id}"
    frame_interval = 1.0 / config.fps
    session = requests.Session()  # reuses TLS connection across frames

    # Initialise the pickup servo and move to resting position
    servo = PickupServo(mock=config.mock_servo)
    servo.move_to_rest()

    print(f"Inference endpoint : {config.endpoint}")
    print(f"Stream push URL    : {stream_url}")
    print(f"Target FPS         : {config.fps}")
    print(f"AI every N frames  : {config.ai_every_n_frames}")
    print(f"Pickup servo       : BCM {PICKUP_SERVO_PIN}  rest={ANGLE_REST}°  max={ANGLE_MAX}°  threshold={PICKUP_CONFIDENCE_THRESHOLD:.0%}")
    if config.mock_servo:
        print("Servo mode         : MOCK (no GPIO)")
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
            servo.cleanup()
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

            # Fire AI inference in a background thread — never blocks the video loop.
            # The servo is passed in; if confidence ≥ threshold the thread triggers pickup.
            if frame_count % config.ai_every_n_frames == 0:
                threading.Thread(
                    target=_run_ai_inference,
                    args=(
                        f"{base_url}/torch-test-video?robot_id={config.robot_id}",
                        frame,
                        frame_count,
                        servo,
                    ),
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
        servo.move_to_rest()
        servo.cleanup()


if __name__ == "__main__":
    run(parse_args())
