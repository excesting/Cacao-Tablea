# 1. Use the highly-stable Python 3.11 server environment
FROM python:3.11-slim

# 2. Set the working directory inside the server
WORKDIR /app

# 3. Install the core Linux graphics libraries OpenCV is crying for
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. The Ultimate YOLO Hack: Forcefully remove the crashing OpenCV and replace it
RUN pip uninstall -y opencv-python || true
RUN pip install --no-cache-dir opencv-python-headless

# 6. Copy the rest of your app's files into the server
COPY . .

# 7. Start Gunicorn using Railway's dynamic port system
CMD gunicorn app:app --bind 0.0.0.0:$PORT
