"""
test_gpio_output.py — Minimal GPIO output test.

Drives one or more BCM pins HIGH/LOW and holds until Ctrl+C.
Useful for checking a single pin with a multimeter or LED before
wiring up the full motor driver.

Usage:
  python3 test_gpio_output.py                  # default: 17 HIGH, 27 LOW
  python3 test_gpio_output.py --high 17 --low 27
  python3 test_gpio_output.py --high 17 18 --low 27 22
"""

import argparse
import time

import RPi.GPIO as GPIO  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive GPIO pins HIGH/LOW and hold.")
    parser.add_argument("--high", type=int, nargs="+", default=[17],
                        help="BCM pins to drive HIGH (default: 17).")
    parser.add_argument("--low", type=int, nargs="+", default=[27],
                        help="BCM pins to drive LOW (default: 27).")
    args = parser.parse_args()

    all_pins = args.high + args.low

    print(f"GPIO output test")
    print(f"  HIGH: BCM {args.high}")
    print(f"  LOW:  BCM {args.low}")
    print("Press Ctrl+C to stop and clean up.")
    print()

    GPIO.setmode(GPIO.BCM)
    for pin in all_pins:
        GPIO.setup(pin, GPIO.OUT)

    for pin in args.high:
        GPIO.output(pin, GPIO.HIGH)
    for pin in args.low:
        GPIO.output(pin, GPIO.LOW)

    print("Pins set. Holding...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
