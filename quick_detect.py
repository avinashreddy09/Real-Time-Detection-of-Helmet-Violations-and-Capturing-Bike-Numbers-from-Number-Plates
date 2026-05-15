import cv2
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

print("🎥 Helmet Detection System")
print("Options:")
print("1. Use webcam")
print("2. Use video file")

choice = input("Enter your choice (1/2): ")

if choice == "1":
    cap = cv2.VideoCapture(0)
    print("✅ Webcam started. Press 'q' to quit")
else:
    video_path = input("Enter video path: ").strip('"')
    cap = cv2.VideoCapture(video_path)
    print(f"✅ Playing video. Press 'q' to quit")

if not cap.isOpened():
    print("❌ Failed to open video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame)
    result = results[0]
    
    # Draw boxes manually (since .plot() doesn't work)
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Detection - Press q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Done!")