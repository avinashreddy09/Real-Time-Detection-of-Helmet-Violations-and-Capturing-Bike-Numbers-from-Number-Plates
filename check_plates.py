import cv2

cap = cv2.VideoCapture(r"C:\Users\avina\Downloads\12.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % 20 == 0:  # Check every 20th frame
        cv2.imwrite(f"frame_{frame_count}.jpg", frame)
        print(f"Saved frame {frame_count}")
    
    frame_count += 1

cap.release()
print(f"Total frames: {frame_count}")
print("Check the saved frame_X.jpg images to see if plates are visible")