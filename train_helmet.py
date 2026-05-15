from ultralytics import YOLO
import os

# Check if dataset exists
if not os.path.exists("data.yaml"):
    print("❌ data.yaml not found!")
    exit()

# Check if images exist
train_images = "dataset/train/images"
if not os.path.exists(train_images):
    print(f"❌ Train images folder not found: {train_images}")
    exit()

print("✅ Dataset found! Starting training...")

# Load YOLO model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=50,
    batch=4,  # Reduced batch size for CPU
    imgsz=640,
    workers=0,
    patience=10,
    device='cpu'
)

print("✅ Training complete!")
print("📁 Model saved to: runs/detect/train/weights/best.pt")