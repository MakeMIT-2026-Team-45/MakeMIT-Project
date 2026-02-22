FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev mosquitto curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch
RUN pip3 install --no-cache-dir --timeout 120 torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir \
        fastapi "uvicorn[standard]" pillow openai paho-mqtt

COPY main.py train_taco_mobilenet_binary.py openai_label_mapping.py label_cache.json .
COPY checkpoints/ ./checkpoints/
COPY mosquitto.conf /etc/mosquitto/mosquitto.conf
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000 1883 9001
ENTRYPOINT ["/start.sh"]
