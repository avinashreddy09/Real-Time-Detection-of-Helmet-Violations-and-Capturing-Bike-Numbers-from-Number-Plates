pipeline {
    agent any
    stages {
        stage('Check Python') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe --version'
            }
        }
        stage('Install') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe -m pip install ultralytics opencv-python'
            }
        }
        stage('Test') {
            steps {
                bat 'C:\\Users\\avina\\AppData\\Local\\Programs\\Python\\Python39\\python.exe -c "print(\"Hello from Jenkins\")"'
            }
        }
    }
}