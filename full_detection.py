import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Load models
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'])

# Video path - CHANGE THIS TO YOUR VIDEO
video_path = r"C:\Users\avina\Downloads\12.mp4"
output_path = "output_with_number_plates.mp4"

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing video for number plates...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    result = results[0]
    
    # Draw YOLO detections
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            
            if class_name == 'person':
                color = (0, 255, 0)
                label = f"RIDER {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            elif class_name == 'motorcycle':
                color = (0, 165, 255)
                label = f"BIKE {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, "CHECK HELMET!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # ============================================
    # NUMBER PLATE DETECTION USING CONTOURS
    # ============================================
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply edge detection
    edged = cv2.Canny(gray, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_detected = False
    
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        
        # If contour has 4 corners, it could be a number plate
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            
            # Filter by aspect ratio (number plates are usually wider than tall)
            if 2 < aspect_ratio < 5:
                # Draw rectangle around number plate
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, "NUMBER PLATE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Extract plate region for OCR
                plate_roi = frame[y:y+h, x:x+w]
                
                # Try to read text using EasyOCR
                try:
                    ocr_result = reader.readtext(plate_roi)
                    if ocr_result:
                        for (bbox, text, confidence) in ocr_result:
                            if confidence > 0.4:
                                clean_text = ''.join(c for c in text if c.isalnum())
                                if len(clean_text) >= 4:
                                    cv2.putText(frame, f"{clean_text} ({confidence*100:.1f}%)", 
                                               (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    print(f"Found plate: {clean_text} ({confidence*100:.1f}%)")
                except:
                    pass
                
                plate_detected = True
                break
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Write frame to output
    out.write(frame)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"✅ Done! Output saved to: {output_path}")
print(f"Total frames processed: {frame_count}")