#!/bin/bash

echo "---===== MAKEMIT 2026: INSTALLATION SCRIPT =====---"

echo "[?] First time setup? [y/n]"
read first_time

if [ "$first_time" == "y" ]
then
    echo "[!] Installing required packages..."
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-picamera2 \
        libcamera-apps \
        ffmpeg \
        git \
        curl \
        wget \


    echo "[!] Activating virtual environment..."
    python3 -m venv ~/venv --system-site-packages
    source ~/venv/bin/activate
    echo 'source ~/venv/bin/activate' >> ~/.bashrc

    echo "[!] Installing required python packages..."
    ~/venv/bin/pip install pillow requests RPi.GPIO
fi

echo "[!] Testing the camera..."
OUTPUT=$(python3 - <<'EOF'
from picamera2 import Picamera2
import time

picam = Picamera2()
config = picam.create_still_configuration()
picam.configure(config)
picam.start()
time.sleep(0.3)
frame = picam.capture_array("main")
picam.stop()
picam.close()
print(f"Captured frame: shape={frame.shape}, dtype={frame.dtype}")
EOF
)
EXPECTED="Captured frame: shape=(2464, 3280, 3), dtype=uint8"

if [ "$OUTPUT" = "$EXPECTED" ]; then
  echo "[+] Camera test PASSED"
else
  echo "[!] Camera test FAILED — got: $OUTPUT"
  echo "[!] Expected: $EXPECTED"
fi

echo "---===== MAKEMIT 2026: INSTALLATION SCRIPT DONE =====---"
