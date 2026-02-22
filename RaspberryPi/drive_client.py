"""
drive_client.py — Pi-side autonomous drive client.

Continuously sweeps an ultrasonic sensor on a servo (BCM 12, 50Hz PWM, SG90)
from 0° to 180° and back, publishing each distance reading tagged with the servo
angle to MQTT. Simultaneously subscribes to motor commands and executes them via
four GPIO on/off pins.

Wiring (physical board pin numbers):
  Ultrasonic sensor (Grove single-wire):
    SIG/DATA → Pin 7  (BCM GPIO 4)
    VCC      → Pin 2  (5V)
    GND      → Pin 6  (GND)
  Servo (SG90):
    Signal   → Pin 32 (BCM GPIO 12, hardware PWM)
    VCC      → external 5V
    GND      → common GND

MQTT topics:
  Publish:   robot/{id}/sensor/ultrasonic  {"angle_deg": 45.0, "distance_cm": 82.3, "ts": …}
  Subscribe: robot/{id}/control/motors     {"left": 1, "right": 1}
"""

import argparse
import json
import math
import random
import threading
import time
from dataclasses import dataclass

import paho.mqtt.client as mqtt_client


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DriveClientConfig:
    broker_host: str
    broker_port: int
    robot_id: int
    servo_pin: int
    servo_min_dc: float
    servo_max_dc: float
    servo_freq_hz: float
    sweep_step_deg: float
    sweep_delay_sec: float
    sonar_trigger_pin: int
    sonar_echo_pin: int
    left_fwd_pin: int
    left_bwd_pin: int
    right_fwd_pin: int
    right_bwd_pin: int
    deadman_timeout_sec: float
    mock: bool


def parse_args() -> DriveClientConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Drive client: sweeps an ultrasonic sensor on a servo, publishes "
            "angle-tagged distance readings via MQTT, and executes motor commands."
        )
    )
    parser.add_argument("--broker-host", type=str, default="mit.ethanzhao.us")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--robot-id", type=int, default=1)

    # Servo
    parser.add_argument("--servo-pin", type=int, default=12,
                        help="BCM GPIO pin for servo signal (hardware PWM).")
    parser.add_argument("--servo-min-dc", type=float, default=2.5,
                        help="Duty cycle for 0° (SG90: 2.5%%).")
    parser.add_argument("--servo-max-dc", type=float, default=12.5,
                        help="Duty cycle for 180° (SG90: 12.5%%).")
    parser.add_argument("--servo-freq-hz", type=float, default=50.0,
                        help="Servo PWM frequency in Hz.")
    parser.add_argument("--sweep-step-deg", type=float, default=5.0,
                        help="Servo angle increment per step (degrees).")
    parser.add_argument("--sweep-delay-sec", type=float, default=0.08,
                        help="Delay between steps (seconds). Must exceed sonar pulse time (~30ms).")

    # Ultrasonic (Grove single-wire sensor)
    # Physical pin 7 = BCM GPIO 4. VCC→pin 2 (5V), GND→pin 6.
    parser.add_argument("--sonar-trigger-pin", type=int, default=4,
                        help="BCM GPIO pin for ultrasonic SIG (Grove single-wire). Physical pin 7.")
    parser.add_argument("--sonar-echo-pin", type=int, default=4,
                        help="BCM GPIO pin for ECHO (same as trigger in single-wire Grove mode).")

    # Motors
    parser.add_argument("--left-fwd-pin", type=int, default=17)
    parser.add_argument("--left-bwd-pin", type=int, default=18)
    parser.add_argument("--right-fwd-pin", type=int, default=27)
    parser.add_argument("--right-bwd-pin", type=int, default=22)

    parser.add_argument("--deadman-timeout-sec", type=float, default=0.5)
    parser.add_argument("--mock", action="store_true",
                        help="Simulate hardware (runs on any machine, no GPIO required).")

    args = parser.parse_args()
    return DriveClientConfig(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        robot_id=args.robot_id,
        servo_pin=args.servo_pin,
        servo_min_dc=args.servo_min_dc,
        servo_max_dc=args.servo_max_dc,
        servo_freq_hz=args.servo_freq_hz,
        sweep_step_deg=args.sweep_step_deg,
        sweep_delay_sec=args.sweep_delay_sec,
        sonar_trigger_pin=args.sonar_trigger_pin,
        sonar_echo_pin=args.sonar_echo_pin,
        left_fwd_pin=args.left_fwd_pin,
        left_bwd_pin=args.left_bwd_pin,
        right_fwd_pin=args.right_fwd_pin,
        right_bwd_pin=args.right_bwd_pin,
        deadman_timeout_sec=args.deadman_timeout_sec,
        mock=args.mock,
    )


# ---------------------------------------------------------------------------
# Servo controller
# ---------------------------------------------------------------------------

class ServoController:
    """Controls a standard PWM servo via a single GPIO pin."""

    def __init__(
        self,
        pin: int,
        freq_hz: float,
        min_dc: float,
        max_dc: float,
        mock: bool = False,
    ) -> None:
        self._pin = pin
        self._min_dc = min_dc
        self._max_dc = max_dc
        self._mock = mock
        self._pwm = None

        if not mock:
            import RPi.GPIO as GPIO  # type: ignore
            self._gpio = GPIO
            # GPIO.setmode(BCM) is called once in run() before this constructor
            GPIO.setup(pin, GPIO.OUT)
            self._pwm = GPIO.PWM(pin, freq_hz)
            self._pwm.start(min_dc)  # start at 0°

    def set_angle(self, angle_deg: float) -> None:
        """Move servo to angle_deg [-20, 180]. Maps -20°→min_dc, 180°→max_dc."""
        angle_deg = max(-20.0, min(180.0, angle_deg))
        dc = self._min_dc + ((angle_deg + 20.0) / 200.0) * (self._max_dc - self._min_dc)
        if self._mock:
            return  # no-op in mock mode
        self._pwm.ChangeDutyCycle(dc)

    def cleanup(self) -> None:
        if not self._mock and self._pwm is not None:
            self._pwm.stop()
            self._gpio.cleanup(self._pin)


# ---------------------------------------------------------------------------
# Motor controller
# ---------------------------------------------------------------------------

class MotorController:
    """Controls two motors via four GPIO on/off output pins."""

    def __init__(
        self,
        left_fwd: int,
        left_bwd: int,
        right_fwd: int,
        right_bwd: int,
        mock: bool = False,
    ) -> None:
        self._pins = {
            "left_fwd": left_fwd,
            "left_bwd": left_bwd,
            "right_fwd": right_fwd,
            "right_bwd": right_bwd,
        }
        self._mock = mock
        self._lock = threading.Lock()

        if not mock:
            import RPi.GPIO as GPIO  # type: ignore
            self._gpio = GPIO
            # GPIO.setmode(BCM) already called in run()
            for pin in self._pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, False)

    def set(self, left: int, right: int) -> None:
        left = max(-1, min(1, int(left)))
        right = max(-1, min(1, int(right)))
        with self._lock:
            if self._mock:
                print(f"[MOCK MOTORS] left={left}  right={right}")
                return
            GPIO = self._gpio
            GPIO.output(self._pins["left_fwd"],  left == 1)
            GPIO.output(self._pins["left_bwd"],  left == -1)
            GPIO.output(self._pins["right_fwd"], right == 1)
            GPIO.output(self._pins["right_bwd"], right == -1)

    def stop(self) -> None:
        self.set(0, 0)

    def cleanup(self) -> None:
        if not self._mock:
            for pin in self._pins.values():
                self._gpio.cleanup(pin)


# ---------------------------------------------------------------------------
# Deadman switch
# ---------------------------------------------------------------------------

class DeadmanSwitch:
    """Stops the motors if no motor command is received within timeout_sec."""

    def __init__(self, motors: MotorController, timeout_sec: float) -> None:
        self._motors = motors
        self._timeout = timeout_sec
        self._last_cmd_ts = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def feed(self) -> None:
        self._last_cmd_ts = time.monotonic()

    def _loop(self) -> None:
        while True:
            time.sleep(0.05)
            if time.monotonic() - self._last_cmd_ts > self._timeout:
                self._motors.stop()


# ---------------------------------------------------------------------------
# Ultrasonic sensor — single-wire Grove mode
#
# Wiring: SIG → BCM 4 (physical pin 7), VCC → pin 2 (5V), GND → pin 6.
#
# Protocol: the Grove single-wire sensor uses one pin for both trigger and
# echo.  Pull it LOW briefly, pulse HIGH for 10µs to trigger, then switch
# the pin to INPUT and time the HIGH pulse that comes back.
#
# CRITICAL: do NOT call GPIO.cleanup() globally here — that would destroy
# the servo PWM object on BCM 12.  The sonar pin is set up once in
# setup_ultrasonic_pin() and reused across every reading.
# ---------------------------------------------------------------------------

def setup_ultrasonic_pin(trigger_pin: int) -> None:
    """
    One-time GPIO setup for the sonar pin.
    Call this after GPIO.setmode(BCM) and before the sweep loop starts.
    The pin starts in OUTPUT mode (idle LOW).
    """
    import RPi.GPIO as GPIO  # type: ignore
    GPIO.setup(trigger_pin, GPIO.OUT)
    GPIO.output(trigger_pin, False)
    time.sleep(0.01)  # let pin settle


def _measure_ultrasonic_distance_cm(trigger_pin: int) -> float | None:
    """
    Measure distance using single-wire Grove ultrasonic sensor on trigger_pin.
    Returns distance in cm, or None if no echo received within timeout.

    Does NOT call GPIO.cleanup() — the pin is reused across readings.
    The caller is responsible for setting up the pin once via setup_ultrasonic_pin().
    """
    import RPi.GPIO as GPIO  # type: ignore

    pulse_timeout_sec = 0.03
    speed_of_sound_cm_per_sec = 34300.0

    # Trigger: drive LOW→HIGH→LOW
    GPIO.setup(trigger_pin, GPIO.OUT)
    GPIO.output(trigger_pin, False)
    time.sleep(0.002)           # 2ms pre-settle

    GPIO.output(trigger_pin, True)
    time.sleep(0.00001)         # 10µs trigger pulse
    GPIO.output(trigger_pin, False)

    # Switch to input and time the echo pulse
    GPIO.setup(trigger_pin, GPIO.IN)

    wait_start = time.perf_counter()
    while GPIO.input(trigger_pin) == 0:
        if time.perf_counter() - wait_start > pulse_timeout_sec:
            return None         # no echo start
    pulse_start = time.perf_counter()

    while GPIO.input(trigger_pin) == 1:
        if time.perf_counter() - pulse_start > pulse_timeout_sec:
            return None         # echo pulse too long (out of range)
    pulse_end = time.perf_counter()

    return ((pulse_end - pulse_start) * speed_of_sound_cm_per_sec) / 2.0


def _mock_ultrasonic_distance_cm(angle_deg: float) -> float:
    """
    Simulate a non-uniform room: closer walls to the sides, open forward.
    angle_deg: -20=right, 80=forward, 180=left.
    """
    # Convert to radians centered at forward (80° = 0 rad offset)
    rad = math.radians(angle_deg - 80.0)
    # Ellipse-ish room: 3m forward, 1m to sides
    a = 3.0  # meters forward/backward
    b = 1.0  # meters left/right
    # Parametric ray distance to ellipse wall
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    if abs(cos_a) < 1e-9 and abs(sin_a) < 1e-9:
        return 200.0
    # Distance to ellipse: r = ab / sqrt((b*cos)^2 + (a*sin)^2)
    denom = math.sqrt((b * cos_a) ** 2 + (a * sin_a) ** 2)
    dist_m = (a * b) / denom if denom > 0 else 4.0
    dist_cm = dist_m * 100.0
    # Add small noise
    return round(max(5.0, dist_cm + random.uniform(-3.0, 3.0)), 1)


# ---------------------------------------------------------------------------
# Servo sweep loop (runs in main thread)
# ---------------------------------------------------------------------------

def servo_sweep_loop(
    config: DriveClientConfig,
    servo: ServoController,
    client: mqtt_client.Client,
    sensor_topic: str,
) -> None:
    """
    Continuously oscillates the servo from -20° to 180° and back.
    At each step: positions servo, waits for it to settle, measures distance,
    publishes the angle-tagged reading to MQTT.
    """
    current_angle = -20.0
    direction = 1  # +1 = sweeping toward 180°, -1 = sweeping toward -20°

    while True:
        servo.set_angle(current_angle)
        time.sleep(config.sweep_delay_sec)

        if config.mock:
            dist = _mock_ultrasonic_distance_cm(current_angle)
        else:
            dist = _measure_ultrasonic_distance_cm(config.sonar_trigger_pin)

        if dist is not None:
            payload = json.dumps({
                "angle_deg": round(current_angle, 1),
                "distance_cm": round(dist, 1),
                "ts": time.time(),
            })
            client.publish(sensor_topic, payload)
            if config.mock:
                print(f"[sensor] angle={current_angle:.0f}°  dist={dist:.1f}cm")

        # Advance angle
        current_angle += direction * config.sweep_step_deg
        if current_angle >= 180.0:
            current_angle = 180.0
            direction = -1
        elif current_angle <= -20.0:
            current_angle = -20.0
            direction = 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: DriveClientConfig) -> None:
    # GPIO.setmode must be called exactly once before any GPIO setup
    if not config.mock:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setmode(GPIO.BCM)
        setup_ultrasonic_pin(config.sonar_trigger_pin)  # one-time pin init

    servo = ServoController(
        pin=config.servo_pin,
        freq_hz=config.servo_freq_hz,
        min_dc=config.servo_min_dc,
        max_dc=config.servo_max_dc,
        mock=config.mock,
    )
    motors = MotorController(
        left_fwd=config.left_fwd_pin,
        left_bwd=config.left_bwd_pin,
        right_fwd=config.right_fwd_pin,
        right_bwd=config.right_bwd_pin,
        mock=config.mock,
    )
    deadman = DeadmanSwitch(motors, config.deadman_timeout_sec)

    motor_topic  = f"robot/{config.robot_id}/control/motors"
    sensor_topic = f"robot/{config.robot_id}/sensor/ultrasonic"

    client = mqtt_client.Client()

    def on_connect(c, userdata, flags, rc):
        print(f"[MQTT] connected to {config.broker_host}:{config.broker_port} (rc={rc})")
        c.subscribe(motor_topic)
        print(f"[MQTT] subscribed to {motor_topic}")

    def on_message(c, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            left = int(data["left"])
            right = int(data["right"])
            motors.set(left, right)
            deadman.feed()
        except Exception as exc:
            print(f"[MQTT] bad motor command: {exc}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(config.broker_host, config.broker_port)
    client.loop_start()
    deadman.start()

    print(f"Sensor topic  : {sensor_topic}")
    print(f"Motor topic   : {motor_topic}")
    print(f"Servo pin     : BCM {config.servo_pin} (physical pin 32)  sweep {config.sweep_step_deg}°/step  {config.sweep_delay_sec*1000:.0f}ms/step")
    print(f"Sonar pin     : BCM {config.sonar_trigger_pin} (physical pin 7)  VCC→pin2  GND→pin6")
    print(f"Deadman       : {config.deadman_timeout_sec}s")
    if config.mock:
        print("Mode: MOCK (simulating servo + sensor, no GPIO)")
    print("Press Ctrl+C to stop.\n")

    try:
        servo_sweep_loop(config, servo, client, sensor_topic)
    except KeyboardInterrupt:
        print("\n[drive_client] shutting down...")
    finally:
        motors.stop()
        servo.cleanup()
        motors.cleanup()
        if not config.mock:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.cleanup(config.sonar_trigger_pin)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    run(parse_args())
