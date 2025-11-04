FROM python:3.9-slim-bullseye
WORKDIR /app
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ python3-dev \
    libopenblas-dev libatlas-base-dev \
    libhdf5-dev libjpeg-dev libtiff5-dev \
    libgl1 libglib2.0-0 \
    git \
    espeak espeak-data \  
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --timeout=300 \
    numpy==1.24.3 \  
    Pillow==8.3.2 \
    pyttsx3==2.90 \
    opencv-python-headless==4.5.5.64

# Install PyTorch
RUN pip install --no-cache-dir --timeout=300 \
    torch==1.13.1 \
    torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install ultralytics
RUN pip install --no-cache-dir --timeout=300 --disable-pip-version-check ultralytics==8.0.0

COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen  
ENV DISPLAY=

CMD ["python", "main.py"]