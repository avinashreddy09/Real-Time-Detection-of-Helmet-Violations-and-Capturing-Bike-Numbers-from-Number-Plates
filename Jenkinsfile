pipeline {
    agent any
    
    environment {
        APP_NAME = 'helmet-detection'
        DOCKER_IMAGE = 'helmet-detector:latest'
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', 
                    url: 'https://github.com/avinashreddy09/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates.git'
                echo '✅ Code pulled from GitHub'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh '''
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
                echo '✅ Dependencies installed'
            }
        }
        
        stage('Test Imports') {
            steps {
                sh '''
                    python -c "from ultralytics import YOLO; print('YOLO OK')"
                    python -c "import cv2; print('OpenCV OK')"
                    python -c "import easyocr; print('EasyOCR OK')"
                '''
                echo '✅ All imports successful'
            }
        }
        
        stage('Run Detection Test') {
            steps {
                sh '''
                    python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model loaded successfully')"
                '''
                echo '✅ Detection test passed'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKER_IMAGE .'
                echo '✅ Docker image built'
            }
        }
        
        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 5000:5000 --name $APP_NAME $DOCKER_IMAGE'
                echo '✅ Container running on port 5000'
            }
        }
    }
    
    post {
        success {
            echo '🎉 Pipeline completed successfully!'
            echo 'App running at http://localhost:5000'
        }
        failure {
            echo '❌ Pipeline failed. Check the logs.'
        }
    }
}