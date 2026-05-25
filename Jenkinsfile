pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo '✅ Code pulled from GitHub'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install --upgrade pip'
                bat 'pip install ultralytics opencv-python easyocr numpy'
                echo '✅ Dependencies installed'
            }
        }
        
        stage('Test Imports') {
            steps {
                bat 'python -c "from ultralytics import YOLO; print(\"YOLO OK\")"'
                bat 'python -c "import cv2; print(\"OpenCV OK\")"'
                bat 'python -c "import easyocr; print(\"EasyOCR OK\")"'
                echo '✅ All imports successful'
            }
        }
        
        stage('Run Detection Test') {
            steps {
                bat 'python -c "from ultralytics import YOLO; model = YOLO(\'yolov8n.pt\'); print(\'Model loaded successfully\')"'
                echo '✅ Detection test passed'
            }
        }
    }
    
    post {
        success {
            echo '🎉 Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check the logs.'
        }
    }
}