import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import torch

# --- Configuration ---
# IMPORTANT: You need a YOLO model that can detect 'number_plate'.
# You can train one using your dataset, or use a placeholder.
# For this example, let's assume we have a model file named 'best.pt' that can detect plates.
# If you don't have one, this script will show you how to use a standard model for detection first.

# Load the YOLO model (Replace 'best.pt' with your trained model's path if you have it)
# If you don't have a custom model, the standard model will still find people/motorcycles.
model = YOLO('yolov8n.pt') # Standard model for person/bike detection. For plates, you need a custom one.

# Initialize the PaddleOCR engine
# 'en' is for English. The 'lang' parameter can be changed.
ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)

# --- Main Processing ---
video_path = r"C:\Users\avina\Downloads\12.mp4"
output_path = "output_with_plates.mp4"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing video for number plates... Output will be saved to {output_path}")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Run YOLO detection for general objects (like people, bikes)
    results = model(frame)
    result = results[0]

    # 2. Look for number plates
    # For a real implementation, you would have a separate YOLO model or a specific class for 'number_plate'.
    # Since our standard model doesn't have that, we'll simulate the process.
    # We'll run PaddleOCR on the *entire* frame to find any text. This is less efficient but good for testing.

    # A more advanced method would first detect the plate region using another YOLO model.
    # Let's search the whole frame for text as a demonstration.
    ocr_result = ocr.ocr(frame)

    plate_text = ""
    if ocr_result and ocr_result[0]:
        # The result format is a list of lists: [[[box], (text, confidence)], ...]
        for line in ocr_result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5: # Only consider high-confidence text
                plate_text = text
                # Draw a bounding box around the detected text
                box = line[0]
                # Convert box points to integer tuple
                box = [tuple(map(int, point)) for point in box]
                cv2.polylines(frame, [np.array(box)], True, (0, 0, 255), 2)
                cv2.putText(frame, f"Plate: {text}", (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 3. Draw the standard YOLO detections (people, bikes)
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            # Different color for potential violators (people without a helmet logic could go here)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"✅ Done! Output saved to: {output_path}")