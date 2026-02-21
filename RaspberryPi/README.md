# Raspberry Pi Layer (Boilerplate)

This folder contains a minimal Raspberry Pi client that sends JPEG frames to the backend endpoint:

- Backend endpoint: `POST /torch-test-video`
- Default URL: `http://127.0.0.1:8000/torch-test-video`

## Run

From repo root:

```bash
python RaspberryPi/pi_client.py --image-path Backend/data/IMG_7397.JPG --one-shot
```

Loop every second:

```bash
python RaspberryPi/pi_client.py --image-path Backend/data/IMG_7397.JPG --interval-sec 1.0
```

Use Pi camera (requires `picamera2`):

```bash
python RaspberryPi/pi_client.py
```

## Notes

- `--image-path` is great for first bring-up and backend testing.
- Without `--image-path`, the script tries to read from the Pi camera.
- This is intentionally basic boilerplate so you can customize capture, retry logic, and queueing.
