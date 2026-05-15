FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY helmet_plate_detection.py .
CMD ["python", "helmet_plate_detection.py"]