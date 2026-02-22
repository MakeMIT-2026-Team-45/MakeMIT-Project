import time

import RPi.GPIO as GPIO  # type: ignore


PIN = 18  # BCM 18 is physical pin 12.

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

print("Attempting to force pulses on GPIO 18 (physical pin 12)...")
print("Listen closely to the sensor for a faint clicking sound.")

try:
    while True:
        GPIO.output(PIN, True)
        time.sleep(0.00001)
        GPIO.output(PIN, False)
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup(PIN)
