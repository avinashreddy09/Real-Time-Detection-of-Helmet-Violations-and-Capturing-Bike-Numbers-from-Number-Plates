from ultralytics import YOLO
import os

# Set the correct path
current_dir = os.getcwd()
dataset_path = os.path.join(current_dir, "dataset")

print(f"Current directory: {current_dir}")
print(f"Dataset path: {dataset_path}")

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"❌ Dataset folder not found at: {dataset_path}")
    exit()

# Check for images
train_images = os.path.join(dataset_path, "train", "images")
if os.path.exists(train_images):
    num_images = len(os.listdir(train_images))
    print(f"✅ Found {num_images} training images")
else:
    print(f"❌ Train images not found at: {train_images}")
    exit()

# Create data.yaml content
yaml_content = f"""
path: {dataset_path}
train: train/images
val: val/images

nc: 4
names: ['rider', 'with_helmet', 'without_helmet', 'number_plate']
"""

# Save data.yaml
with open("data_correct.yaml", "w") as f:
    f.write(yaml_content)

print("✅ Created data_correct.yaml")

# Load and train
print("🚀 Starting training...")
model = YOLO('yolov8n.pt')

results = model.train(
    data='data_correct.yaml',
    epochs=30,
    batch=4,
    imgsz=640,
    workers=0,
    device='cpu'
)

print("✅ Training complete!")