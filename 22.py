import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Load models
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'])

# Video path - CHANGE THIS
video_path = "bike.gif"
output_path = "22.mp4"

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Processing video with IMPROVED plate detection...")
print("-" * 50)

frame_count = 0
detected_plates = []
total_persons = 0
total_bikes = 0
violations = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame, conf=0.15)
    result = results[0]
    
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            
            if class_name == 'person':
                total_persons += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"RIDER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Helmet check
                head_roi = frame[y1:y1+int((y2-y1)*0.2), x1:x2]
                if head_roi.size > 0:
                    gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                    if np.mean(gray_head) > 100:
                        violations += 1
                        cv2.putText(frame, "CHECK HELMET!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            elif class_name == 'motorcycle':
                total_bikes += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"BIKE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Number plate detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 15000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 1.5 < aspect_ratio < 6:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, "NUMBER PLATE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                plate_roi = frame[y:y+h, x:x+w]
                if plate_roi.size > 0:
                    try:
                        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                        plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        plate_gray = cv2.medianBlur(plate_gray, 3)
                        _, plate_gray = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        ocr_result = reader.readtext(plate_gray)
                        if ocr_result:
                            for (bbox, text, confidence) in ocr_result:
                                clean_text = ''.join(c for c in text if c.isalnum()).upper()
                                if len(clean_text) >= 4 and confidence > 0.4:
                                    if clean_text not in detected_plates:
                                        detected_plates.append(clean_text)
                                        print(f"Frame {frame_count}: DETECTED PLATE - {clean_text} ({confidence*100:.1f}%)")
                                    cv2.putText(frame, f"{clean_text}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except:
                        pass
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out.write(frame)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()

print("-" * 50)
print("✅ PROCESSING COMPLETE!")
print(f"Total frames processed: {frame_count}")
print(f"Total Persons: {total_persons}")
print(f"Total Bikes: {total_bikes}")
print(f"Helmet Violations: {violations}")
print(f"Output video saved as: {output_path}")
print("-" * 50)
print("\n📋 DETECTED NUMBER PLATES SUMMARY:")
print("-" * 40)

if detected_plates:
    for i, plate in enumerate(detected_plates, 1):
        print(f"  {i}. {plate}")
else:
    print("  No number plates were detected.")
    print("  This could be because:")
    print("  - The video doesn't have clear license plates")
    print("  - The plates are too small or blurry")
    print("  - The camera angle is not showing plates properly")
print("-" * 40)