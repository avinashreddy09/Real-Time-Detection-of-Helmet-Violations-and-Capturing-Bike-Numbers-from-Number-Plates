from ultralytics import YOLO
import cv2

# Use a pretrained model for testing (not helmet-specific but can detect people)
model = YOLO('yolov8n.pt')  # Downloads pretrained COCO model

def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        
        # Visualize results
        annotated_frame = results[0].plot()
        cv2.imshow('Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = input("Enter path to your video file: ")
    detect_from_video(video_path)