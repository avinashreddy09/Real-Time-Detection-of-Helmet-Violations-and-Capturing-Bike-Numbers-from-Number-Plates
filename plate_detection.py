import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Initialize EasyOCR (more compatible than PaddleOCR)
reader = easyocr.Reader(['en'])

# Load YOLO model
model = YOLO('yolov8n.pt')

video_path = r"C:\Users\avina\Downloads\12.mp4"
output_path = "output_with_plates_easyocr.mp4"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing video for number plates... Output will be saved to {output_path}")
print("Press 'q' to stop early")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    result = results[0]
    
    # Look for text (number plates) using EasyOCR on the whole frame
    # For better performance, you'd first detect plate region, but this works
    if frame_count % 5 == 0:  # Check for text every 5 frames to save processing
        ocr_results = reader.readtext(frame)
        
        for (bbox, text, confidence) in ocr_results:
            if confidence > 0.5 and len(text) > 3:  # Likely a number plate
                # Draw bounding box
                bbox = np.array(bbox, dtype=np.int32)
                cv2.polylines(frame, [bbox], True, (0, 0, 255), 2)
                # Put text
                x, y = bbox[0]
                cv2.putText(frame, f"Plate: {text}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Found plate: {text} (confidence: {confidence:.2f})")
    
    # Draw YOLO detections (people, bikes)
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out.write(frame)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"✅ Done! Output saved to: {output_path}")