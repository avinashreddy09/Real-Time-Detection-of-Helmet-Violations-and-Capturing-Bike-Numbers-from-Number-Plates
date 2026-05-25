import os
import cv2
import numpy as np
import easyocr
import threading
import uuid
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Task store
tasks = {}

# Load Models
print("Loading YOLOv8n...")
yolo_model = YOLO('yolov8n.pt')
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

def process_video(task_id, input_path, output_path):
    try:
        tasks[task_id]['status'] = 'processing'
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error'] = 'Could not open video file.'
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0 or fps == 0:
            total_frames = 1
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics
        total_persons = 0
        total_bikes = 0
        total_violations = 0
        detected_plates = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize frame for better detection
            if width < 640:
                frame = cv2.resize(frame, (640, int(640 * height / width)))
            
            # Run YOLOv8
            results = yolo_model(frame, conf=0.25, verbose=False)
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = yolo_model.names[int(cls)]
                    
                    if class_name == 'person':
                        total_persons += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "RIDER", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Helmet check
                        head_h = int((y2 - y1) * 0.2)
                        head_y2 = y1 + head_h
                        head_roi = frame[max(0, y1):head_y2, max(0, x1):x2]
                        if head_roi.size > 0:
                            gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                            avg_intensity = np.mean(gray_head)
                            if avg_intensity > 100:
                                total_violations += 1
                                cv2.putText(frame, "CHECK HELMET!", (x1, max(0, y2+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    elif class_name == 'motorcycle':
                        total_bikes += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame, "BIKE", (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Number Plate Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(bfilter, 30, 200)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 300 < area < 8000:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = w / h if h > 0 else 0
                        if 1.5 < aspect_ratio < 6:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, "NUMBER PLATE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            plate_roi = frame[y:y+h, x:x+w]
                            if plate_roi.size > 0:
                                try:
                                    ocr_result = reader.readtext(plate_roi)
                                    if ocr_result:
                                        for (bbox, text, confidence) in ocr_result:
                                            clean_text = ''.join(c for c in text if c.isalnum()).upper()
                                            if len(clean_text) >= 4 and confidence > 0.4:
                                                if clean_text not in detected_plates:
                                                    detected_plates.append(clean_text)
                                                cv2.putText(frame, f"{clean_text}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                except:
                                    pass
                            break
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out.write(frame)
            
            tasks[task_id]['progress'] = int((frame_count / total_frames) * 100) if total_frames > 0 else 0

        cap.release()
        out.release()
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['output_file'] = os.path.basename(output_path)
        tasks[task_id]['stats'] = {
            'total_frames': frame_count,
            'total_persons': total_persons,
            'total_bikes': total_bikes,
            'violations': total_violations,
            'plates_found': len(detected_plates),
            'unique_plates': detected_plates
        }

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    output_filename = f"out_{task_id}_{filename}"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    file.save(input_path)
    
    tasks[task_id] = {
        'status': 'starting',
        'progress': 0,
        'output_file': output_filename
    }
    
    thread = threading.Thread(target=process_video, args=(task_id, input_path, output_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(tasks[task_id])

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['PROCESSED_FOLDER'], secure_filename(filename))
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)