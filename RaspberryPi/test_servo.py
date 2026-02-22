"""
test_servo.py — Standalone servo wiring test.

Sweeps the servo through 0° → 90° → 180° → 90° → 0° so you can visually
confirm the servo is wired correctly before running the full drive_client.

Direction is reversed (min_dc=12.5, max_dc=2.5) so 0° physically moves CW
and 180° moves CCW, opposite of the standard SG90 convention.

Wiring (SG90 / MG996R):
  Signal → Pin 32 (BCM GPIO 12, hardware PWM channel 0)
  VCC    → external 5V (NOT the Pi's 3.3V or 5V pin — servo draws too much)
  GND    → common GND with Pi

Usage:
  python3 test_servo.py
  python3 test_servo.py --pin 18 --angles 0 45 90 135 180
"""

import argparse
import time

import RPi.GPIO as GPIO  # type: ignore


def set_angle(pwm: GPIO.PWM, angle: float, min_dc: float = 2.5, max_dc: float = 12.5) -> None:
    """Set servo to angle_deg [0, 180] and hold for settle time."""
    angle = max(0.0, min(180.0, angle))
    dc = min_dc + (angle / 180.0) * (max_dc - min_dc)
    pwm.ChangeDutyCycle(dc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Servo wiring test — sweeps servo through a sequence of angles.")
    parser.add_argument("--pin", type=int, default=12,
                        help="BCM GPIO pin for servo signal (default: 12, physical pin 32).")
    parser.add_argument("--freq", type=float, default=50.0,
                        help="PWM frequency in Hz (default: 50).")
    parser.add_argument("--min-dc", type=float, default=12.5,
                        help="Duty cycle for 0° (reversed: 12.5%% = CW start).")
    parser.add_argument("--max-dc", type=float, default=2.5,
                        help="Duty cycle for 180° (reversed: 2.5%% = CCW end).")
    parser.add_argument("--settle", type=float, default=0.6,
                        help="Seconds to wait after each move (default: 0.6).")
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 90, 180, 90, 0],
                        help="Sequence of angles to sweep (default: 0 90 180 90 0).")
    args = parser.parse_args()

    print(f"Servo test  BCM GPIO {args.pin}  (physical pin {'32' if args.pin == 12 else '?'})")
    print(f"Frequency   {args.freq} Hz   duty cycle {args.min_dc}%% (0°) – {args.max_dc}%% (180°)")
    print(f"Sequence    {args.angles}")
    print()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(args.pin, GPIO.OUT)
    pwm = GPIO.PWM(args.pin, args.freq)
    pwm.start(0)
    time.sleep(0.2)  # let PWM stabilize before first move

    try:
        for angle in args.angles:
            print(f"  Moving to {angle:5.1f}° ...")
            set_angle(pwm, angle, args.min_dc, args.max_dc)
            time.sleep(args.settle)
            pwm.ChangeDutyCycle(0)   # release hold to reduce jitter/heat
            time.sleep(0.1)

        print()
        print("Done. If the servo moved through the sequence, wiring is correct.")
        print("If nothing moved, check: signal wire on BCM pin, VCC to external 5V, common GND.")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        pwm.stop()
        GPIO.cleanup(args.pin)


if __name__ == "__main__":
    main()
