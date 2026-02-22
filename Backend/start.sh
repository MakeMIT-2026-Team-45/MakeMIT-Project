#!/usr/bin/env bash
set -euo pipefail

echo "[start.sh] Starting Mosquitto..."
mosquitto -c /etc/mosquitto/mosquitto.conf > /tmp/mosquitto.log 2>&1 &
MOSQ_PID=$!

echo "[start.sh] Waiting for Mosquitto on port 1883..."
for i in $(seq 1 20); do
    if (echo > /dev/tcp/localhost/1883) 2>/dev/null; then
        echo "[start.sh] Mosquitto ready."
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "ERROR: Mosquitto did not start. Log:" >&2
        cat /tmp/mosquitto.log >&2
        exit 1
    fi
    sleep 0.5
done

echo "[start.sh] Starting uvicorn on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
