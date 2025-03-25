from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load the YOLOv5 model (since you prefer YOLOv5)
model = YOLO("weights/yolov5s.pt")

# Load class labels
class_names = model.names

# Global variable to control video feed
video_streaming = False

def generate_frames():
    global video_streaming
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height
    
    while video_streaming:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame, conf=0.45)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bb = box.xyxy.numpy()[0]  # Bounding box coordinates
                clsID = int(box.cls.numpy()[0])  # Class ID
                conf = box.conf.numpy()[0]  # Confidence score
                
                # Draw bounding box
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 3)
                
                # Display class label and confidence
                label = f"{class_names[clsID]} {round(conf * 100, 2)}%"
                cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_stream():
    global video_streaming
    video_streaming = True
    return "Streaming started"

@app.route('/stop')
def stop_stream():
    global video_streaming
    video_streaming = False
    return "Streaming stopped"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
