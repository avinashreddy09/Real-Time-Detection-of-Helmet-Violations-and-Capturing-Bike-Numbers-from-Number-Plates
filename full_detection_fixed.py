import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Load models
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'])

# Video path
video_path = r"C:\Users\avina\Downloads\12.mp4"
output_path = "output_with_plates_visible.mp4"

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Processing video for Number Plates...")
print("-" * 50)

frame_count = 0
detected_plates = []  # Store all detected plate numbers

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    result = results[0]
    
    # Draw YOLO detections (People and Motorcycles)
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            
            if class_name == 'person':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"RIDER {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            elif class_name == 'motorcycle':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"BIKE {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(frame, "CHECK HELMET!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # ============================================
    # NUMBER PLATE DETECTION
    # ============================================
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edged = cv2.Canny(gray, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        
        # Look for rectangular shapes (4 corners)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Number plates typically have aspect ratio between 2 and 5
            if 2 < aspect_ratio < 5:
                # Draw RED rectangle around number plate
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, "NUMBER PLATE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Extract plate region
                plate_roi = frame[y:y+h, x:x+w]
                
                # Try to read text using EasyOCR
                try:
                    ocr_result = reader.readtext(plate_roi)
                    if ocr_result:
                        for (bbox, text, confidence) in ocr_result:
                            if confidence > 0.4:
                                # Clean the text (keep only alphanumeric)
                                clean_text = ''.join(c for c in text if c.isalnum()).upper()
                                if len(clean_text) >= 4:
                                    # Display text on frame
                                    cv2.putText(frame, f"PLATE: {clean_text}", (x, y+h+25), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    cv2.putText(frame, f"CONF: {confidence*100:.1f}%", (x, y+h+45), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                                    # Store detected plate if not already stored
                                    if clean_text not in detected_plates:
                                        detected_plates.append(clean_text)
                                        print(f"Frame {frame_count}: Detected Number Plate - {clean_text} ({confidence*100:.1f}%)")
                except:
                    pass
                
                break  # Exit after finding first plate in this frame
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Write frame to output video
    out.write(frame)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("-" * 50)
print("✅ PROCESSING COMPLETE!")
print(f"Total frames processed: {frame_count}")
print(f"Output video saved as: {output_path}")
print("-" * 50)
print("\n📋 DETECTED NUMBER PLATES SUMMARY:")
print("-" * 40)

if detected_plates:
    for i, plate in enumerate(detected_plates, 1):
        print(f"  {i}. {plate}")
    print("-" * 40)
    print(f"Total unique plates detected: {len(detected_plates)}")
else:
    print("  No number plates were detected in this video.")
    print("  Possible reasons:")
    print("  - Plates are not clearly visible")
    print("  - Lighting conditions are poor")
    print("  - Camera angle is not favorable")
print("-" * 40)