# Raspberry Pi Setup Guide

This guide covers everything needed to set up a Raspberry Pi as a robot node in this system. The Pi is responsible for three things:

1. **Image classification** — capturing camera frames and sending them to the backend AI endpoint
2. **Telemetry reporting** — sending bin capacity and GPS location data to the backend
3. **Live video streaming** — broadcasting a real-time video feed via mediamtx so the frontend dashboard can display it

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [OS Installation](#2-os-installation)
3. [Initial Pi Configuration](#3-initial-pi-configuration)
4. [Enable the Camera](#4-enable-the-camera)
5. [Install System Dependencies](#5-install-system-dependencies)
6. [Install Python Dependencies](#6-install-python-dependencies)
7. [Clone the Repository](#7-clone-the-repository)
8. [Test the Camera](#8-test-the-camera)
9. [Configure the Backend URL](#9-configure-the-backend-url)
10. [Run the AI Client (pi_client.py)](#10-run-the-ai-client-pi_clientpy)
11. [Send Telemetry Data](#11-send-telemetry-data)
12. [Set Up mediamtx for Live Video Streaming](#12-set-up-mediamtx-for-live-video-streaming)
13. [Run Everything on Boot](#13-run-everything-on-boot)
14. [Verifying the Full Pipeline](#14-verifying-the-full-pipeline)
15. [Troubleshooting](#15-troubleshooting)
16. [API Reference](#16-api-reference)

---

## 1. Hardware Requirements

| Component | Notes |
|-----------|-------|
| Raspberry Pi 4 (2GB+ RAM recommended) | Pi 3B+ may work but is slower for image encoding |
| Raspberry Pi Camera Module v2 or v3 | Must be the official CSI ribbon cable camera, **not** a USB webcam (unless you adapt) |
| MicroSD card (16GB+, Class 10 or better) | Faster cards improve camera I/O |
| Power supply (5V/3A USB-C for Pi 4) | Underpowered supplies cause random crashes |
| Network connection (Ethernet or WiFi) | Ethernet is more reliable for telemetry |

> **USB webcam alternative:** If you use a USB camera instead of the CSI camera module, `picamera2` will not work. You will need to capture frames with OpenCV (`cv2.VideoCapture(0)`) and replace the `jpeg_bytes_from_picamera2()` function in `pi_client.py`.

---

## 2. OS Installation

Use **Raspberry Pi OS Lite (64-bit)** for a headless server setup, or **Raspberry Pi OS (64-bit)** if you want a desktop.

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your laptop.
2. Insert the microSD card into your laptop.
3. Open Raspberry Pi Imager and select:
   - **Device:** Raspberry Pi 4
   - **OS:** Raspberry Pi OS (64-bit) — the full version includes `picamera2` pre-installed; Lite requires manual install
   - **Storage:** your microSD card
4. Click the **gear icon** (Advanced Options) before writing:
   - Enable SSH (use password authentication)
   - Set a username (e.g. `pi`) and a strong password
   - Configure WiFi SSID and password if using wireless
   - Set hostname (e.g. `robot-1`)
5. Click **Write** and wait for it to finish.
6. Eject the card and insert it into the Pi.

---

## 3. Initial Pi Configuration

Power on the Pi, wait ~60 seconds, then SSH in from your laptop:

```bash
ssh pi@robot-1.local
# or use the Pi's IP address if mDNS doesn't work:
ssh pi@192.168.x.x
```

Once logged in, update all packages:

```bash
sudo apt update && sudo apt upgrade -y
```

Set the correct timezone (important for log timestamps):

```bash
sudo raspi-config
# Navigate to: Localisation Options → Timezone
```

Expand the filesystem if it wasn't done automatically:

```bash
sudo raspi-config
# Navigate to: Advanced Options → Expand Filesystem
# Then reboot:
sudo reboot
```

---

## 4. Enable the Camera

The CSI camera must be enabled before `picamera2` can use it.

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
sudo reboot
```

After reboot, verify the camera is detected:

```bash
libcamera-hello --list-cameras
```

You should see output like:
```
Available cameras
-----------------
0 : imx219 [3280x2464] (/base/soc/i2c0mux/i2c@1/imx219@10)
```

If the list is empty, double-check that the ribbon cable is fully seated (the connector locks by pulling up the tab, inserting the cable silver-side down, then pushing the tab back down).

---

## 5. Install System Dependencies

```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    libcamera-apps \
    ffmpeg \
    git \
    curl \
    wget
```

> `python3-picamera2` installs `picamera2` as a system package. This is the recommended approach on Raspberry Pi OS — installing it via pip into a venv can have issues with the underlying libcamera bindings.

---

## 6. Install Python Dependencies

The Pi client only needs a small set of packages. Create a virtual environment that can access system packages (needed for `picamera2`):

```bash
python3 -m venv ~/venv --system-site-packages
source ~/venv/bin/activate
```

Add this line to `~/.bashrc` so the venv activates automatically on login:

```bash
echo 'source ~/venv/bin/activate' >> ~/.bashrc
```

Install the Python dependencies for the client:

```bash
pip install pillow requests
```

> `picamera2` comes from the system package installed in Step 5. `pillow` is used to encode frames as JPEG. `requests` is used in any custom telemetry scripts (the built-in `pi_client.py` uses Python's standard `urllib` so it has no extra dependencies).

---

## 7. Clone the Repository

```bash
cd ~
git clone https://github.com/MakeMIT-2026-Team-45/MakeMIT-Project.git
cd MakeMIT-Project
```

If the repo is private, authenticate with a personal access token:

```bash
git clone https://YOUR_GITHUB_TOKEN@github.com/MakeMIT-2026-Team-45/MakeMIT-Project.git
```

---

## 8. Test the Camera

Before running any client code, confirm that `picamera2` can capture a frame:

```bash
python3 - <<'EOF'
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
```

Expected output:
```
Captured frame: shape=(2464, 3280, 3), dtype=uint8
```

If you get an error about the camera not being found, revisit Step 4.

---

## 9. Configure the Backend URL

The Pi needs to know the IP address of the machine running the backend. The backend must be running and accessible on the network (see the [Backend README](../Backend/README.md) — run it with Docker).

Find your backend machine's local IP:

```bash
# On the machine running Docker:
ifconfig | grep "inet " | grep -v 127.0.0.1
# or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

You'll use this IP throughout the rest of the guide. Replace `<BACKEND_IP>` with the actual IP (e.g. `192.168.1.42`) in every command below.

Verify the backend is reachable from the Pi:

```bash
curl http://<BACKEND_IP>:8000/
# Expected: {"message":"Backend is running"}
```

---

## 10. Run the AI Client (pi_client.py)

`pi_client.py` captures JPEG frames from the camera and POSTs them to the backend's `/torch-test-video` endpoint, which returns a waste classification (trash or recycling).

### One-shot test with a static image (no camera required)

This is the fastest way to verify the backend connection works:

```bash
cd ~/MakeMIT-Project
python3 RaspberryPi/pi_client.py \
    --endpoint http://<BACKEND_IP>:8000/torch-test-video \
    --image-path Backend/data/IMG_7397.JPG \
    --one-shot
```

Expected output:
```
Sending frames to: http://192.168.1.42:8000/torch-test-video
Press Ctrl+C to stop.

class='recycling' prob=0.9213 bytes=84321
```

### Continuous loop with a static image (stress test)

```bash
python3 RaspberryPi/pi_client.py \
    --endpoint http://<BACKEND_IP>:8000/torch-test-video \
    --image-path Backend/data/IMG_7397.JPG \
    --interval-sec 1.0
```

### Live camera — one shot

```bash
python3 RaspberryPi/pi_client.py \
    --endpoint http://<BACKEND_IP>:8000/torch-test-video \
    --one-shot
```

### Live camera — continuous loop every 2 seconds

```bash
python3 RaspberryPi/pi_client.py \
    --endpoint http://<BACKEND_IP>:8000/torch-test-video \
    --interval-sec 2.0
```

### All available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--endpoint` | `http://127.0.0.1:8000/torch-test-video` | Full URL of the backend inference endpoint |
| `--interval-sec` | `1.0` | Seconds between frames in loop mode |
| `--image-path` | *(none)* | Path to a JPEG file; omit to use live camera |
| `--one-shot` | *(off)* | Send a single frame and exit immediately |

---

## 11. Send Telemetry Data

The Pi is also responsible for sending bin capacity and GPS location to the backend. These are standard HTTP POST requests with JSON bodies.

The backend expects a **robot ID** (integer) in the URL path — this ID must match one of the robot IDs in the frontend's hardcoded list (currently 1–6).

### Capacity telemetry

Fields: `trashCapacity` (0–100), `recycleCapacity` (0–100), `batteryPercentage` (0–100).

```bash
curl -X POST http://<BACKEND_IP>:8000/robot/1/telemetry/capacity \
  -H "Content-Type: application/json" \
  -d '{"trashCapacity": 45.0, "recycleCapacity": 72.5, "batteryPercentage": 88.0}'
```

Expected response:
```json
{"ok": true}
```

### Location telemetry

Fields: `lat` (latitude), `lng` (longitude).

```bash
curl -X POST http://<BACKEND_IP>:8000/robot/1/telemetry/location \
  -H "Content-Type: application/json" \
  -d '{"lat": 42.3601, "lng": -71.0942}'
```

### Sending telemetry from Python

Use this pattern inside your own scripts on the Pi to send telemetry programmatically:

```python
import requests

BACKEND = "http://<BACKEND_IP>:8000"
ROBOT_ID = 1

# Capacity
requests.post(f"{BACKEND}/robot/{ROBOT_ID}/telemetry/capacity", json={
    "trashCapacity": 45.0,
    "recycleCapacity": 72.5,
    "batteryPercentage": 88.0,
})

# Location
requests.post(f"{BACKEND}/robot/{ROBOT_ID}/telemetry/location", json={
    "lat": 42.3601,
    "lng": -71.0942,
})
```

The backend bridges these over MQTT to the frontend dashboard. As long as the backend Docker container is running with ports 1883 and 9001 exposed, the frontend will receive updates in real time.

---

## 12. Set Up mediamtx for Live Video Streaming

The frontend displays a live WebRTC video feed from the robot. This is handled by [mediamtx](https://github.com/bluenviron/mediamtx), a media server that runs on the Pi and re-streams the camera over WebRTC.

### Download mediamtx

Find the latest release for `linux_arm64` (Pi 4 / 64-bit OS):

```bash
cd ~
# Check https://github.com/bluenviron/mediamtx/releases for the latest version
wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_v1.12.3_linux_arm64v8.tar.gz
tar -xzf mediamtx_v1.12.3_linux_arm64v8.tar.gz
# You should now have: mediamtx  mediamtx.yml
```

> Replace the version in the URL with the actual latest version from the [releases page](https://github.com/bluenviron/mediamtx/releases).

### Configure mediamtx

Open `mediamtx.yml` and confirm (or set) the following. The defaults usually work:

```yaml
# These are the default values — verify they match:
rtmpAddress: :1935
webrtcAddress: :8889
api: yes
apiAddress: :9997
```

You can leave the rest of the file at defaults for basic operation.

### Publish the Pi camera stream to mediamtx

mediamtx accepts RTSP/RTMP input. Use `libcamera-vid` with `ffmpeg` to pipe the camera into mediamtx over RTMP:

```bash
libcamera-vid -t 0 --inline --width 1280 --height 720 --framerate 30 -o - | \
  ffmpeg -re -i pipe:0 -c:v copy -f flv rtmp://localhost:1935/live/camera
```

- `-t 0` — run forever
- `--inline` — embed SPS/PPS in every IDR frame (required for streaming)
- `pipe:0` — pipe raw H.264 to ffmpeg via stdin
- The stream will be available at `rtmp://localhost:1935/live/camera`

Leave this running in one terminal window (or use a systemd service — see Step 13).

### Access the WebRTC stream

mediamtx automatically re-publishes the RTMP stream as WebRTC. From a browser on the same network:

```
http://<PI_IP>:8889/live/camera
```

The frontend's `useWebRTC` hook connects to the backend's `/offer` signaling endpoint, which in turn talks to mediamtx. Ensure the backend knows the Pi's IP and stream path — you may need to update the signaling endpoint URL in the backend if it's not hardcoded to localhost.

---

## 13. Run Everything on Boot

Use systemd to start the camera stream and pi_client automatically when the Pi boots.

### systemd service for the camera stream

Create the service file:

```bash
sudo nano /etc/systemd/system/camera-stream.service
```

Paste:

```ini
[Unit]
Description=Pi Camera → mediamtx RTMP stream
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi
ExecStart=/bin/bash -c 'libcamera-vid -t 0 --inline --width 1280 --height 720 --framerate 30 -o - | ffmpeg -re -i pipe:0 -c:v copy -f flv rtmp://localhost:1935/live/camera'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### systemd service for mediamtx

```bash
sudo nano /etc/systemd/system/mediamtx.service
```

Paste:

```ini
[Unit]
Description=mediamtx media server
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi
ExecStart=/home/pi/mediamtx /home/pi/mediamtx.yml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### systemd service for pi_client.py

```bash
sudo nano /etc/systemd/system/pi-client.service
```

Paste (replace `<BACKEND_IP>`):

```ini
[Unit]
Description=MakeMIT Pi AI client
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/MakeMIT-Project
Environment=PATH=/home/pi/venv/bin:/usr/bin:/bin
ExecStart=/home/pi/venv/bin/python3 RaspberryPi/pi_client.py \
    --endpoint http://<BACKEND_IP>:8000/torch-test-video \
    --interval-sec 2.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable and start all services

```bash
sudo systemctl daemon-reload

sudo systemctl enable mediamtx
sudo systemctl enable camera-stream
sudo systemctl enable pi-client

sudo systemctl start mediamtx
sudo systemctl start camera-stream
sudo systemctl start pi-client
```

### Check service status

```bash
sudo systemctl status mediamtx
sudo systemctl status camera-stream
sudo systemctl status pi-client

# View live logs:
sudo journalctl -fu pi-client
sudo journalctl -fu camera-stream
```

---

## 14. Verifying the Full Pipeline

Work through this checklist in order. Each step validates one layer of the stack.

### Step 1 — Backend is reachable

```bash
curl http://<BACKEND_IP>:8000/
# Expected: {"message":"Backend is running"}
```

### Step 2 — Image classification works

```bash
python3 RaspberryPi/pi_client.py \
    --endpoint http://<BACKEND_IP>:8000/torch-test-video \
    --image-path Backend/data/IMG_7397.JPG \
    --one-shot
# Expected: class='trash' or class='recycling' with a probability
```

### Step 3 — Telemetry reaches the backend

```bash
curl -X POST http://<BACKEND_IP>:8000/robot/1/telemetry/capacity \
  -H "Content-Type: application/json" \
  -d '{"trashCapacity": 50.0, "recycleCapacity": 50.0, "batteryPercentage": 75.0}'
# Expected: {"ok": true}
```

Open the frontend dashboard and confirm that Robot 1's capacity gauges update.

### Step 4 — Location updates on the map

```bash
curl -X POST http://<BACKEND_IP>:8000/robot/1/telemetry/location \
  -H "Content-Type: application/json" \
  -d '{"lat": 42.3601, "lng": -71.0942}'
# Expected: {"ok": true}
```

Select Robot 1 in the dashboard and confirm the map marker moves.

### Step 5 — Live video is streaming

```bash
# On the Pi, check mediamtx is running:
sudo systemctl status mediamtx

# Check the stream is being published:
curl http://localhost:9997/v3/paths/list
# Should show "live/camera" in the paths list

# From a browser on the same network:
# Navigate to http://<PI_IP>:8889/live/camera
```

---

## 15. Troubleshooting

### `picamera2 is not installed`

You installed `picamera2` into a virtualenv without `--system-site-packages`. Delete the venv and recreate it:

```bash
deactivate
rm -rf ~/venv
python3 -m venv ~/venv --system-site-packages
source ~/venv/bin/activate
pip install pillow requests
```

### `HTTP error 400: Body is not a valid JPEG binary payload`

The JPEG being sent is corrupt or truncated. Check:
- The image file exists and is a valid JPEG (open it with an image viewer)
- Disk space is not full: `df -h`
- No partial writes — the Pi has stable power

### `HTTP error 404` or `Connection refused`

The backend is not running or the IP/port is wrong.
- Confirm the Docker container is up on the backend machine: `docker ps`
- Confirm the correct port is exposed: `docker ps` should show `0.0.0.0:8000->8000/tcp`
- Confirm the Pi can reach it: `ping <BACKEND_IP>` and `curl http://<BACKEND_IP>:8000/`

### Camera not found by `libcamera-hello`

- Re-seat the ribbon cable (common cause)
- Check `sudo raspi-config` → Interface Options → Camera is enabled
- Check kernel messages: `dmesg | grep -i camera`
- Confirm you are using a CSI camera module, not a USB webcam

### mediamtx stream not appearing in browser

- Check the RTMP publisher is actually running: `sudo journalctl -fu camera-stream`
- Confirm mediamtx is listening: `ss -tlnp | grep 8889`
- Firewall: ensure ports 8889 (WebRTC) and 1935 (RTMP) are open: `sudo ufw allow 8889 && sudo ufw allow 1935`

### Telemetry not updating in the frontend

- Confirm the backend MQTT broker is running (it runs inside the Docker container)
- Confirm port 9001 (WebSocket MQTT) is exposed: `docker ps`
- Open browser DevTools → Network → WS to see if the frontend has an open WebSocket connection
- Check the backend logs for MQTT warnings: `docker logs <container_name>`

### Services not starting on boot

```bash
sudo systemctl daemon-reload
sudo systemctl enable <service-name>
sudo reboot
# After reboot:
sudo systemctl status <service-name>
sudo journalctl -b -u <service-name>   # logs since last boot
```

---

## 16. API Reference

All requests go to the machine running the backend Docker container.

### `GET /`
Health check.
- **Response:** `{"message": "Backend is running"}`

### `POST /torch-test-video`
Classify a JPEG frame as trash or recycling.
- **Header:** `Content-Type: image/jpeg`
- **Body:** Raw JPEG bytes (not multipart, not base64)
- **Response:**
```json
{
  "content_type": "image/jpeg",
  "size_bytes": 84321,
  "prediction": 1,
  "max_probability": 0.9213
}
```
- `prediction`: `0` = trash, `1` = recycling

### `POST /robot/{robot_id}/telemetry/capacity`
Report bin fill levels and battery.
- **Body:**
```json
{
  "trashCapacity": 45.0,
  "recycleCapacity": 72.5,
  "batteryPercentage": 88.0
}
```
- **Response:** `{"ok": true}`
- Internally published to MQTT topic `robot/{robot_id}/telemetry/capacity`

### `POST /robot/{robot_id}/telemetry/location`
Report GPS coordinates.
- **Body:**
```json
{
  "lat": 42.3601,
  "lng": -71.0942
}
```
- **Response:** `{"ok": true}`
- Internally published to MQTT topic `robot/{robot_id}/telemetry/location`
