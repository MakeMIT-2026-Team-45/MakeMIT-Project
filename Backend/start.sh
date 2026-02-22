#!/usr/bin/env bash
set -euo pipefail

echo "[start.sh] Starting Mosquitto..."
mosquitto -c /etc/mosquitto/mosquitto.conf &

echo "[start.sh] Waiting for Mosquitto on port 1883..."
for i in $(seq 1 20); do
    if (echo > /dev/tcp/localhost/1883) 2>/dev/null; then
        echo "[start.sh] Mosquitto ready."
        break
    fi
    [ "$i" -eq 20 ] && { echo "ERROR: Mosquitto did not start." >&2; exit 1; }
    sleep 0.5
done

echo "[start.sh] Starting uvicorn on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
