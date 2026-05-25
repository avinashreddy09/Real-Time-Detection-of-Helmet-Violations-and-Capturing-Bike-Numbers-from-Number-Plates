import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

print("Loading models...")
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'], gpu=False)
print("Ready!")

# Path to your 12.mp4
video_path = r"C:\Users\avina\Downloads\12.mp4"
output_path = "12mp4_output_with_plates.mp4"

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing {video_path}...")
print(f"Video size: {width}x{height}")
print("-" * 50)

frame_count = 0
plate_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Create a copy for zoomed plate detection
    frame_original = frame.copy()
    
    # METHOD 1: YOLO detection for people and bikes
    results = model(frame, conf=0.25, verbose=False)
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0:  # person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "RIDER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Helmet check
                    head = frame[y1:y1+int((y2-y1)*0.2), x1:x2]
                    if head.size > 0:
                        gray_head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
                        if np.mean(gray_head) > 100:
                            cv2.putText(frame, "CHECK HELMET!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                elif cls == 3:  # motorcycle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "BIKE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # METHOD 2: Enhanced Number Plate Detection for 12.mp4
    # Zoom into bottom half where plates usually are
    h, w = frame.shape[:2]
    bottom_half = frame[int(h*0.6):h, 0:w]  # Focus on bottom 40% of frame
    
    # Apply multiple preprocessing techniques
    gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
    
    # Try different thresholds
    for thresh_val in [100, 120, 150]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:  # Wider area range for 12.mp4
                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect = w_box / h_box if h_box > 0 else 0
                
                # Plates are usually wider than tall
                if 1.5 < aspect < 6:
                    # Adjust coordinates back to full frame
                    full_x = x
                    full_y = y + int(h*0.6)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (full_x, full_y), (full_x+w_box, full_y+h_box), (0, 0, 255), 2)
                    cv2.putText(frame, "POTENTIAL PLATE", (full_x, full_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Try OCR on the plate region
                    plate_roi = frame[full_y:full_y+h_box, full_x:full_x+w_box]
                    if plate_roi.size > 0:
                        try:
                            # Enlarge plate for better OCR
                            plate_roi_big = cv2.resize(plate_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            plate_gray = cv2.cvtColor(plate_roi_big, cv2.COLOR_BGR2GRAY)
                            plate_gray = cv2.medianBlur(plate_gray, 3)
                            
                            ocr_result = reader.readtext(plate_gray)
                            if ocr_result:
                                for (bbox, text, confidence) in ocr_result:
                                    clean = ''.join(c for c in text if c.isalnum()).upper()
                                    if len(clean) >= 4 and confidence > 0.3:
                                        cv2.putText(frame, f"PLATE: {clean}", (full_x, full_y+h_box+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        print(f"Frame {frame_count}: Detected plate - {clean} ({confidence:.1f}%)")
                                        plate_detected = True
                        except:
                            pass
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    out.write(frame)
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()

print("-" * 50)
print(f"✅ Done! Processed {frame_count} frames")
print(f"Output saved to: {output_path}")
if not plate_detected:
    print("\n⚠️ No number plates were detected in 12.mp4")
    print("   Possible reasons:")
    print("   - Plates are too small or blurry")
    print("   - Camera angle doesn't show plates clearly")
    print("   - Plates are moving too fast")
    print("\n   For better results, try a video with:")
    print("   - Clear, front-facing number plates")
    print("   - Good lighting conditions")
    print("   - Stationary or slow-moving vehicles")