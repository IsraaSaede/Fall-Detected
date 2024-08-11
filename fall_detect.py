from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 is the default camera

# Load your YOLO model
model = YOLO('yolov8s.pt')
ASPECT_RATIO_THRESHOLD = 1.0

# Function to generate video frames
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            fall_detected = False
            # Apply your AI model for detection here
            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height
                        color = (0, 255, 0)
                        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
                            color = (0, 0, 255)
                            cv2.putText(frame, "Fall Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            fall_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            if fall_detected:
                socketio.emit('fall_detected')

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the output frame in byte format
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
