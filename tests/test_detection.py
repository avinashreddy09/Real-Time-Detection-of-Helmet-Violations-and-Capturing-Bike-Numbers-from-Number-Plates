import pytest
from ultralytics import YOLO
import cv2
import os

def test_model_loads():
    """Test that YOLO model loads successfully"""
    try:
        model = YOLO('yolov8n.pt')
        assert model is not None
        print("✅ Model loaded successfully")
    except Exception as e:
        pytest.fail(f"Model failed to load: {e}")

def test_opencv_import():
    """Test OpenCV import"""
    try:
        import cv2
        assert cv2.__version__ is not None
        print(f"✅ OpenCV version: {cv2.__version__}")
    except Exception as e:
        pytest.fail(f"OpenCV import failed: {e}")

def test_easyocr_import():
    """Test EasyOCR import"""
    try:
        import easyocr
        assert easyocr.__version__ is not None
        print(f"✅ EasyOCR version: {easyocr.__version__}")
    except Exception as e:
        pytest.fail(f"EasyOCR import failed: {e}")

if __name__ == "__main__":
    test_model_loads()
    test_opencv_import()
    test_easyocr_import()
    print("✅ All tests passed!")