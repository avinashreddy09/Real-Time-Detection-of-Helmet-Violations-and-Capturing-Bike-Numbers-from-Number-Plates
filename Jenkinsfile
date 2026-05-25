pipeline {
    agent any
    stages {
        stage('Check Python') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe --version'
            }
        }
        stage('Install Dependencies') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe -m pip install ultralytics opencv-python easyocr numpy'
                echo '✅ Dependencies installed'
            }
        }
        stage('Test YOLO') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe -c "from ultralytics import YOLO; print(\"YOLO imported successfully\")"'
                echo '✅ YOLO test passed'
            }
        }
        stage('Test OpenCV') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe -c "import cv2; print(f\"OpenCV version: {cv2.__version__}\")"'
                echo '✅ OpenCV test passed'
            }
        }
        stage('Test EasyOCR') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe -c "import easyocr; print(\"EasyOCR imported successfully\")"'
                echo '✅ EasyOCR test passed'
            }
        }
        stage('All Tests Complete') {
            steps {
                echo '🎉 All pipeline tests passed!'
            }
        }
    }
    post {
        success {
            echo '✅ Jenkins pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check the logs.'
        }
    }
}