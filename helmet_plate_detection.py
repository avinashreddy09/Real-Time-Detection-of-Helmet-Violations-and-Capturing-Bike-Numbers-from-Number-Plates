import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Initialize EasyOCR for text reading
reader = easyocr.Reader(['en'])

# Load YOLO model
model = YOLO('yolov8n.pt')

# Video path (change this to your video)
video_path = r"C:\Users\avina\Downloads\12.mp4"  # or your video path
output_path = "helmet_plate_output.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 10
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing {video_path}...")
print("Looking for: Riders, Helmets, Number Plates")

frame_count = 0

def detect_number_plate_text(frame):
    """Detect and read number plate text using EasyOCR"""
    try:
        # Run OCR on the frame
        results = reader.readtext(frame)
        
        for (bbox, text, confidence) in results:
            # Filter for potential number plates (alphanumeric, length 6-12)
            clean_text = ''.join(c for c in text if c.isalnum())
            if len(clean_text) >= 6 and len(clean_text) <= 12 and confidence > 0.5:
                # Draw bounding box
                bbox = np.array(bbox, dtype=np.int32)
                cv2.polylines(frame, [bbox], True, (0, 0, 255), 2)
                
                # Put text with confidence
                x, y = bbox[0]
                cv2.putText(frame, f"NUMBER PLATE", (x, y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"{clean_text} {confidence*100:.2f}%", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return clean_text
    except:
        pass
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create a copy for annotations
    annotated_frame = frame.copy()
    
    # Run YOLO detection
    results = model(frame)
    result = results[0]
    
    rider_detected = False
    helmet_detected = False
    
    # Process YOLO detections
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            
            if class_name == 'person':
                rider_detected = True
                # Draw rider box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "RIDER", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check head region for helmet (top 20% of bounding box)
                head_y1 = y1
                head_y2 = y1 + int((y2 - y1) * 0.2)
                head_roi = frame[head_y1:head_y2, x1:x2]
                
                if head_roi.size > 0:
                    # Simple helmet detection - check for dark color on head
                    gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                    avg_intensity = np.mean(gray_head)
                    
                    if avg_intensity < 100:  # Dark region suggests helmet
                        helmet_detected = True
                        cv2.putText(annotated_frame, "HELMET ON", (x1, y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(annotated_frame, "WITHOUT HELMET", (x1, y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            elif class_name in ['motorcycle', 'bicycle']:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(annotated_frame, "BIKE", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    
    # Detect number plate text
    plate_text = detect_number_plate_text(frame)
    
    # Add overall status
    y_offset = 30
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if rider_detected:
        cv2.putText(annotated_frame, "STATUS: RIDER DETECTED", (10, y_offset + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if plate_text:
        cv2.putText(annotated_frame, f"PLATE: {plate_text}", (10, y_offset + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Write frame
    out.write(annotated_frame)
    frame_count += 1
    
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"\n✅ Done! Output saved to: {output_path}")
print(f"Total frames processed: {frame_count}")