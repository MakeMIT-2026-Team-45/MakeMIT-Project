"""
test_motors.py — Standalone DC motor wiring test.

Runs each motor forward, stops, then backward so you can visually confirm
both motors and their direction wiring before running drive_client.

Wiring (L298N / L293D motor driver):
  Motor A (left):
    IN1 → BCM 17 (physical pin 11)  — forward
    IN2 → BCM 18 (physical pin 12)  — backward
  Motor B (right):
    IN3 → BCM 27 (physical pin 13)  — forward
    IN4 → BCM 22 (physical pin 15)  — backward

  ENA / ENB → 3.3V or 5V (always-on enable) or PWM pin for speed control
  Motor power → external supply (6–12V depending on motors)
  GND → common GND with Pi

Usage:
  python3 test_motors.py
  python3 test_motors.py --in1 17 --in2 18 --in3 27 --in4 22 --duration 2
"""

import argparse
import time

import RPi.GPIO as GPIO  # type: ignore


def stop_all(pins: list[int]) -> None:
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)


def main() -> None:
    parser = argparse.ArgumentParser(description="DC motor wiring test — runs each motor forward and backward.")
    parser.add_argument("--in1", type=int, default=17, help="BCM pin for Motor A forward (default: 17).")
    parser.add_argument("--in2", type=int, default=18, help="BCM pin for Motor A backward (default: 18).")
    parser.add_argument("--in3", type=int, default=27, help="BCM pin for Motor B forward (default: 27).")
    parser.add_argument("--in4", type=int, default=22, help="BCM pin for Motor B backward (default: 22).")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds to run each direction (default: 2).")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to pause between steps (default: 1).")
    args = parser.parse_args()

    pins = [args.in1, args.in2, args.in3, args.in4]

    print(f"Motor test  BCM pins: A={args.in1}/{args.in2}  B={args.in3}/{args.in4}")
    print(f"Duration    {args.duration}s per direction  {args.pause}s pause between steps")
    print()

    GPIO.setmode(GPIO.BCM)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    try:
        # --- Motor A ---
        print("Motor A (left) forward...")
        GPIO.output(args.in1, GPIO.HIGH)
        GPIO.output(args.in2, GPIO.LOW)
        time.sleep(args.duration)

        print("Motor A stop.")
        stop_all(pins)
        time.sleep(args.pause)

        print("Motor A backward...")
        GPIO.output(args.in1, GPIO.LOW)
        GPIO.output(args.in2, GPIO.HIGH)
        time.sleep(args.duration)

        print("Motor A stop.")
        stop_all(pins)
        time.sleep(args.pause)

        # --- Motor B ---
        print("Motor B (right) forward...")
        GPIO.output(args.in3, GPIO.HIGH)
        GPIO.output(args.in4, GPIO.LOW)
        time.sleep(args.duration)

        print("Motor B stop.")
        stop_all(pins)
        time.sleep(args.pause)

        print("Motor B backward...")
        GPIO.output(args.in3, GPIO.LOW)
        GPIO.output(args.in4, GPIO.HIGH)
        time.sleep(args.duration)

        print("Motor B stop.")
        stop_all(pins)
        time.sleep(args.pause)

        # --- Both forward (drive test) ---
        print("Both motors forward (drive forward)...")
        GPIO.output(args.in1, GPIO.HIGH)
        GPIO.output(args.in2, GPIO.LOW)
        GPIO.output(args.in3, GPIO.HIGH)
        GPIO.output(args.in4, GPIO.LOW)
        time.sleep(args.duration)

        print("Stop.")
        stop_all(pins)

        print()
        print("Done. If both motors spun in both directions, wiring is correct.")
        print("If a motor spun the wrong direction, swap its two wires at the motor driver.")
        print("If nothing moved, check the motor driver power supply and enable pins.")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        stop_all(pins)
        GPIO.cleanup()


if __name__ == "__main__":
    main()
