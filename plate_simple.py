import cv2
from ultralytics import YOLO
import re

# Load YOLO model
model = YOLO('yolov8n.pt')

video_path = r"C:\Users\avina\Downloads\12.mp4"
output_path = "output_with_plates_fixed.mp4"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing video... Output will be saved to {output_path}")

# Simple function to check if text looks like a number plate
def is_number_plate(text):
    # Check if text contains letters and numbers (typical for plates)
    return len(text) >= 4 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    result = results[0]
    
    # Process each frame for potential number plates using basic image processing
    if frame_count % 10 == 0:  # Process every 10th frame for efficiency
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find potential text regions
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # Filter by area (number plate size range)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Number plates typically have aspect ratio between 2 and 5
                if 2 < aspect_ratio < 5:
                    # Draw rectangle around potential plate
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Potential Number Plate", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw YOLO detections (people, motorcycles)
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            
            # Color code based on what's detected
            if class_name in ['person']:
                color = (0, 255, 0)  # Green for people
                label = f"Person {conf:.2f}"
            elif class_name in ['motorcycle', 'bicycle']:
                color = (255, 165, 0)  # Orange for bikes
                label = f"Bike {conf:.2f}"
                # Mark as potential violation (no helmet detection)
                cv2.putText(frame, "Check Helmet!", (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                color = (255, 0, 0)  # Blue for others
                label = f"{class_name} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add info text
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"✅ Done! Output saved to: {output_path}")
print(f"Total frames processed: {frame_count}")