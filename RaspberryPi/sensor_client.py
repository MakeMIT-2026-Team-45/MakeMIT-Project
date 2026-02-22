import argparse
import json
import random
import time
from dataclasses import dataclass
from urllib import error, request


@dataclass
class SensorClientConfig:
    backend_base_url: str
    robot_id: int
    interval_sec: float
    mock: bool
    sonar_check: bool
    sonar_trigger_pin: int
    sonar_echo_pin: int


def parse_args() -> SensorClientConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Read Raspberry Pi sensors and send capacity/location telemetry to the backend."
        )
    )
    parser.add_argument(
        "--backend-base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL for the backend server (example: http://192.168.1.42:8000).",
    )
    parser.add_argument(
        "--robot-id",
        type=int,
        default=1,
        help="Robot ID used in backend telemetry routes.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="Seconds between telemetry sends.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock sensor values instead of hardware reads.",
    )
    parser.add_argument(
        "--sonar-check",
        action="store_true",
        help="Run a basic sonar alive check and exit.",
    )
    parser.add_argument(
        "--sonar-trigger-pin",
        type=int,
        default=23,
        help="BCM GPIO pin connected to sonar TRIG.",
    )
    parser.add_argument(
        "--sonar-echo-pin",
        type=int,
        default=24,
        help="BCM GPIO pin connected to sonar ECHO.",
    )
    args = parser.parse_args()
    return SensorClientConfig(
        backend_base_url=args.backend_base_url.rstrip("/"),
        robot_id=args.robot_id,
        interval_sec=args.interval_sec,
        mock=args.mock,
        sonar_check=args.sonar_check,
        sonar_trigger_pin=args.sonar_trigger_pin,
        sonar_echo_pin=args.sonar_echo_pin,
    )


def _clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def _measure_ultrasonic_distance_cm(trigger_pin: int, echo_pin: int) -> float | None:
    """
    Return measured distance in cm, or None if no echo was detected in time.
    """
    try:
        import RPi.GPIO as GPIO  # type: ignore
    except ImportError:
        raise RuntimeError("RPi.GPIO is not installed. Run this on a Raspberry Pi.")

    pulse_timeout_sec = 0.03
    speed_of_sound_cm_per_sec = 34300.0

    GPIO.setmode(GPIO.BCM)
    one_wire_mode = trigger_pin == echo_pin
    if one_wire_mode:
        # One-wire sonar: the same GPIO line is used for trigger and echo.
        GPIO.setup(trigger_pin, GPIO.OUT)
    else:
        GPIO.setup(trigger_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)

    try:
        if one_wire_mode:
            # Always force output mode before emitting a trigger pulse.
            GPIO.setup(trigger_pin, GPIO.OUT)
            GPIO.output(trigger_pin, False)
            time.sleep(0.01)

            # Send trigger pulse on the shared signal pin, then switch to input.
            GPIO.output(trigger_pin, True)
            time.sleep(0.00001)
            GPIO.output(trigger_pin, False)
            GPIO.setup(trigger_pin, GPIO.IN)

            wait_start = time.perf_counter()
            while GPIO.input(trigger_pin) == 0:
                if time.perf_counter() - wait_start > pulse_timeout_sec:
                    return None
            pulse_start = time.perf_counter()

            while GPIO.input(trigger_pin) == 1:
                if time.perf_counter() - pulse_start > pulse_timeout_sec:
                    return None
            pulse_end = time.perf_counter()
        else:
            # Always force output mode before emitting a trigger pulse.
            GPIO.setup(trigger_pin, GPIO.OUT)
            GPIO.output(trigger_pin, False)
            time.sleep(0.01)

            # Send a 10us trigger pulse.
            GPIO.output(trigger_pin, True)
            time.sleep(0.00001)
            GPIO.output(trigger_pin, False)

            wait_start = time.perf_counter()
            while GPIO.input(echo_pin) == 0:
                if time.perf_counter() - wait_start > pulse_timeout_sec:
                    return None
            pulse_start = time.perf_counter()

            while GPIO.input(echo_pin) == 1:
                if time.perf_counter() - pulse_start > pulse_timeout_sec:
                    return None
            pulse_end = time.perf_counter()

        pulse_duration = pulse_end - pulse_start
        return (pulse_duration * speed_of_sound_cm_per_sec) / 2.0
    finally:
        GPIO.cleanup((trigger_pin, echo_pin))


def sonar_is_alive(trigger_pin: int, echo_pin: int, attempts: int = 3) -> bool:
    """
    Basic health check: sonar is considered alive if any attempt gets an echo.
    """
    for _ in range(attempts):
        distance_cm = _measure_ultrasonic_distance_cm(trigger_pin, echo_pin)
        if distance_cm is not None:
            return True
        time.sleep(0.05)
    return False


# --- Sensor hooks ---
# Replace these functions with your exact sensor implementation.
def read_trash_capacity_percent() -> float:
    """
    Return trash bin fill percentage in [0, 100].
    """
    # Example: map ultrasonic distance to fill percentage.
    # distance_cm = get_ultrasonic_distance(trigger_pin=23, echo_pin=24)
    # return distance_to_fill_percent(distance_cm, empty_cm=35, full_cm=5)
    return random.uniform(10, 80)


def read_recycle_capacity_percent() -> float:
    """
    Return recycle bin fill percentage in [0, 100].
    """
    return random.uniform(5, 70)


def read_battery_percentage() -> float:
    """
    Return battery percentage in [0, 100].
    """
    return random.uniform(40, 100)


def read_gps_lat_lng() -> tuple[float, float]:
    """
    Return current GPS coordinates as (lat, lng).
    """
    # MIT approximate lat/lng as fallback.
    return (42.3601, -71.0942)


def read_capacity_payload(mock: bool) -> dict:
    if mock:
        return {
            "trashCapacity": _clamp_0_100(random.uniform(0, 100)),
            "recycleCapacity": _clamp_0_100(random.uniform(0, 100)),
            "batteryPercentage": _clamp_0_100(random.uniform(20, 100)),
        }
    return {
        "trashCapacity": _clamp_0_100(read_trash_capacity_percent()),
        "recycleCapacity": _clamp_0_100(read_recycle_capacity_percent()),
        "batteryPercentage": _clamp_0_100(read_battery_percentage()),
    }


def read_location_payload(mock: bool) -> dict:
    if mock:
        return {
            "lat": 42.3601 + random.uniform(-0.0004, 0.0004),
            "lng": -71.0942 + random.uniform(-0.0004, 0.0004),
        }
    lat, lng = read_gps_lat_lng()
    return {"lat": float(lat), "lng": float(lng)}


def run(config: SensorClientConfig) -> None:
    if config.sonar_check:
        try:
            alive = sonar_is_alive(
                trigger_pin=config.sonar_trigger_pin,
                echo_pin=config.sonar_echo_pin,
            )
            if alive:
                print(
                    f"SONAR OK (TRIG={config.sonar_trigger_pin}, ECHO={config.sonar_echo_pin})"
                )
            else:
                print(
                    f"SONAR NOT RESPONDING (TRIG={config.sonar_trigger_pin}, "
                    f"ECHO={config.sonar_echo_pin})"
                )
        except Exception as exc:
            print(f"Sonar check failed: {exc}")
        return

    capacity_url = (
        f"{config.backend_base_url}/robot/{config.robot_id}/telemetry/capacity"
    )
    location_url = (
        f"{config.backend_base_url}/robot/{config.robot_id}/telemetry/location"
    )

    print(f"Sending sensor telemetry to {config.backend_base_url} (robot {config.robot_id})")
    if config.mock:
        print("Mode: mock sensor values")
    else:
        print("Mode: hardware sensor values")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            capacity_payload = read_capacity_payload(mock=config.mock)
            location_payload = read_location_payload(mock=config.mock)

            cap_resp = _post_json(capacity_url, capacity_payload)
            loc_resp = _post_json(location_url, location_payload)

            print(
                "capacity="
                f"{capacity_payload} resp={cap_resp} | "
                "location="
                f"{location_payload} resp={loc_resp}"
            )
        except error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error {exc.code}: {msg}")
        except error.URLError as exc:
            print(f"Network error: {exc.reason}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")

        time.sleep(config.interval_sec)


if __name__ == "__main__":
    run(parse_args())
