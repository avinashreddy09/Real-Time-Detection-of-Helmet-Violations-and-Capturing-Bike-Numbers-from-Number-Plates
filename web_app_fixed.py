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

os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)

tasks = {}

print("Loading models...")
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'], gpu=False)
print("Models ready!")

def process_video(task_id, input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            
            results = model(frame, verbose=False)
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        
                        if cls == 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "RIDER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            head = frame[y1:y1+int((y2-y1)*0.2), x1:x2]
                            if head.size > 0:
                                if np.mean(cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)) > 100:
                                    cv2.putText(frame, "NO HELMET!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        elif cls == 3:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            cv2.putText(frame, "BIKE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Number plate detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(gray, 30, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours[:10]:
                area = cv2.contourArea(cnt)
                if 500 < area < 8000:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) == 4:
                        x, y, wb, hb = cv2.boundingRect(approx)
                        ar = wb / hb if hb > 0 else 0
                        if 2 < ar < 5:
                            cv2.rectangle(frame, (x, y), (x+wb, y+hb), (0, 0, 255), 2)
                            cv2.putText(frame, "PLATE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            break
            
            out.write(frame)
            tasks[task_id]['progress'] = int((count / total) * 100) if total > 0 else 0
        
        cap.release()
        out.release()
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['output'] = os.path.basename(output_path)
        
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Helmet Detection</title>
        <style>
            body { font-family: Arial; background: #1a1a2e; color: white; text-align: center; padding: 50px; }
            .container { max-width: 600px; margin: auto; background: #16213e; padding: 30px; border-radius: 20px; }
            input, button { padding: 12px; margin: 10px; font-size: 16px; border-radius: 8px; border: none; }
            button { background: #4CAF50; color: white; cursor: pointer; }
            .progress { width: 100%; background: #333; border-radius: 10px; margin: 20px 0; display: none; }
            .progress-bar { width: 0%; height: 30px; background: #4CAF50; border-radius: 10px; text-align: center; line-height: 30px; }
            .download-btn { background: #ff5722; display: inline-block; margin-top: 20px; padding: 12px 25px; border-radius: 8px; color: white; text-decoration: none; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Helmet & Plate Detection</h1>
            <input type="file" id="fileInput" accept="video/*">
            <button onclick="uploadVideo()">Upload & Process</button>
            <div class="progress" id="progressDiv"><div class="progress-bar" id="progressBar">0%</div></div>
            <div id="status"></div>
            <div id="downloadDiv" class="hidden"></div>
        </div>
        <script>
            let taskId = null;
            
            async function uploadVideo() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) return alert('Select a video');
                
                const formData = new FormData();
                formData.append('video', file);
                
                document.getElementById('progressDiv').style.display = 'block';
                document.getElementById('status').innerHTML = 'Uploading...';
                
                const uploadRes = await fetch('/upload', { method: 'POST', body: formData });
                const data = await uploadRes.json();
                taskId = data.task_id;
                
                checkStatus();
            }
            
            async function checkStatus() {
                const res = await fetch(`/status/${taskId}`);
                const data = await res.json();
                
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressBar').innerHTML = data.progress + '%';
                document.getElementById('status').innerHTML = 'Processing: ' + data.progress + '%';
                
                if (data.status === 'completed') {
                    document.getElementById('status').innerHTML = '✅ Complete!';
                    const downloadDiv = document.getElementById('downloadDiv');
                    downloadDiv.innerHTML = `<a href="/download/${data.output}" class="download-btn">📥 Download Processed Video</a>`;
                    downloadDiv.classList.remove('hidden');
                } else if (data.status === 'error') {
                    document.getElementById('status').innerHTML = '❌ Error: ' + data.error;
                } else {
                    setTimeout(checkStatus, 2000);
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if not file:
        return jsonify({'error': 'No file'}), 400
    
    task_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    input_path = os.path.join('uploads', f"{task_id}_{filename}")
    output_path = os.path.join('processed', f"{task_id}_output.mp4")
    
    file.save(input_path)
    
    tasks[task_id] = {'status': 'processing', 'progress': 0}
    
    thread = threading.Thread(target=process_video, args=(task_id, input_path, output_path))
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def get_status(task_id):
    return jsonify(tasks.get(task_id, {'status': 'not found'}))

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join('processed', filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return 'File not found', 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)