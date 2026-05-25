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

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

tasks = {}

print("Loading YOLOv8n...")
model = YOLO('yolov8n.pt')
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
print("Ready!")

def process_video(task_id, input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_count = 0
        detected_plates = []
        
        # These will show the maximum in any single frame
        max_persons = 0
        max_bikes = 0
        max_violations = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection (EXACT same as improved_plate_detection.py)
            results = model(frame, conf=0.25)
            result = results[0]
            
            frame_persons = 0
            frame_bikes = 0
            frame_violations = 0
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = model.names[int(cls)]
                    
                    if class_name == 'person':
                        frame_persons += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "RIDER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Helmet check (EXACT same logic)
                        head_roi = frame[y1:y1+int((y2-y1)*0.2), x1:x2]
                        if head_roi.size > 0:
                            gray_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
                            if np.mean(gray_head) > 100:
                                frame_violations += 1
                                cv2.putText(frame, "CHECK HELMET!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    elif class_name == 'motorcycle':
                        frame_bikes += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame, "BIKE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Update max values (show the busiest frame)
            max_persons = max(max_persons, frame_persons)
            max_bikes = max(max_bikes, frame_bikes)
            max_violations = max(max_violations, frame_violations)
            
            # Number Plate Detection (EXACT same as improved_plate_detection.py)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 15000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 1.5 < aspect_ratio < 6:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        cv2.putText(frame, "NUMBER PLATE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        plate_roi = frame[y:y+h, x:x+w]
                        if plate_roi.size > 0:
                            try:
                                plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                                plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                                plate_gray = cv2.medianBlur(plate_gray, 3)
                                _, plate_gray = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                
                                ocr_result = reader.readtext(plate_gray)
                                if ocr_result:
                                    for (bbox, text, confidence) in ocr_result:
                                        clean_text = ''.join(c for c in text if c.isalnum()).upper()
                                        if len(clean_text) >= 4 and confidence > 0.4:
                                            if clean_text not in detected_plates:
                                                detected_plates.append(clean_text)
                                                print(f"Frame {frame_count}: PLATE - {clean_text} ({confidence*100:.1f}%)")
                                            cv2.putText(frame, f"{clean_text}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            except:
                                pass
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out.write(frame)
            
            # Update progress
            progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            tasks[task_id]['progress'] = progress
            tasks[task_id]['stats'] = {
                'frames': frame_count,
                'persons': max_persons,
                'bikes': max_bikes,
                'violations': max_violations,
                'plates': len(detected_plates),
                'plate_list': detected_plates
            }
        
        cap.release()
        out.release()
        tasks[task_id]['status'] = 'completed'
        
    except Exception as e:
        print(f"Error: {e}")
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    task_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{task_id}_output.mp4")
    
    file.save(input_path)
    
    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'output_file': f"{task_id}_output.mp4"
    }
    
    thread = threading.Thread(target=process_video, args=(task_id, input_path, output_path))
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def status(task_id):
    return jsonify(tasks.get(task_id, {'status': 'not found'}))

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)