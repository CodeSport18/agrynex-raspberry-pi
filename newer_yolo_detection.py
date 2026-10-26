print(0)
import cv2
print(1)
import numpy as np
print(2)
import time
print(3)
import threading
import json
print(4)
# from playsound import playsound
import socket
print(5)
from flask import Flask, render_template, Response, request, jsonify
import os
from dotenv import load_dotenv
from controller import Controller
print(6)

# --- Performance Configuration ---
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame for speed
INPUT_WIDTH = 320  # Reduced from 416 for faster processing
INPUT_HEIGHT = 320  # Reduced from 416 for faster processing
ANIMAL_INPUT_SIZE = 224  # Reduced from 300 for faster processing

# --- YOLO Configuration ---
YOLO_CONFIG_PATH = "models/yolo/yolov3.cfg"
YOLO_WEIGHTS_PATH = "models/yolo/yolov3.weights"
YOLO_CLASSES_PATH = "models/yolo/coco.names"

# Model variables (will be initialized in load_models())
yolo_net = None
net = None
YOLO_CLASSES = None
output_layers = None

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# --- Animal Detection Model Setup ---
prototxt_path = "models/MobileNetSSD_deploy.prototxt.txt"
model_path = "models/MobileNetSSD_deploy.caffemodel"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]
ANIMAL_CLASSES = {"bird", "cat", "cow", "dog", "horse", "sheep"}

def detect_animals(frame):
    """Detect animals in the frame"""
    try:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (ANIMAL_INPUT_SIZE, ANIMAL_INPUT_SIZE)), 
                                     0.007843, (ANIMAL_INPUT_SIZE, ANIMAL_INPUT_SIZE), 127.5)
        net.setInput(blob)
        detections = net.forward()
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if confidence > 0.5 and label in ANIMAL_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                results.append({
                    'label': label,
                    'confidence': float(confidence),
                    'box': (startX, startY, endX, endY)
                })
        return results
    except Exception as e:
        print(f"Error in animal detection: {e}")
        return []


def detect_humans(frame):
    """Detect humans in the frame"""
    try:
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)
        
        # Lists to store detected objects
        class_ids = []
        confidences = []
        boxes = []
        
        # Process detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Only process if it's a person and confidence is high enough
                if YOLO_CLASSES[class_id] == 'person' and confidence > CONFIDENCE_THRESHOLD:
                    # Get box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    'box': (x, y, x + w, y + h),
                    'confidence': confidences[i]
                })
        
        return results
    except Exception as e:
        print(f"Error in human detection: {e}")
        return []


# --- Camera Class for Web Streaming ---
class Camera:
    def __init__(self, video_source=0, controller=None):
        self.source = video_source
        self.controller = controller
        self.cap = cv2.VideoCapture(self.source)
        self.camera_ok = self.cap.isOpened()
        self.frame = None
        self.frame_seq = 0
        self.lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.animal_detections = []
        self.human_detections = []
        self.animal_detection_enabled = True
        self.human_detection_enabled = True
        self.last_ding_time = 0
        self.ding_cooldown = 2  # seconds between sounds
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
    
    def _update(self):
        """Main camera update loop with detection"""
        while self.running:
            # System switched off from the app: pause the camera and detection.
            if self.controller and not self.controller.system_on:
                if self.cap.isOpened():
                    self.cap.release()
                    self.animal_detections = []
                    self.human_detections = []
                    with self.lock:
                        self.frame = None
                time.sleep(0.5)
                continue
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
                self.camera_ok = self.cap.isOpened()
                if not self.camera_ok:
                    time.sleep(2)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                # If video file, restart from beginning
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.camera_ok = False
                    time.sleep(0.1)
                    continue
            self.camera_ok = True

            self.frame_count += 1
            turn_on = False
            
            # Only run detection every N frames for performance
            if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Animal detection
                if self.animal_detection_enabled:
                    self.animal_detections = detect_animals(frame)
                    current_animal_count = len(self.animal_detections)
                    current_time = time.time()
                    if current_animal_count > 0 and current_time - self.last_ding_time > self.ding_cooldown:
                        self.last_ding_time = current_time
                        turn_on = True
                else:
                    self.animal_detections = []
                
                # Human detection
                if self.human_detection_enabled:
                    self.human_detections = detect_humans(frame)
                    current_human_count = len(self.human_detections)
                    current_time = time.time()
                    if current_human_count > 0 and current_time - self.last_ding_time > self.ding_cooldown:
                        self.last_ding_time = current_time
                        turn_on = True
                else:
                    self.human_detections = []

            # Draw animal detections
            for det in self.animal_detections:
                (sx, sy, ex, ey) = det['box']
                cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 0, 255), 2)
                label = f"{det['label']}: {det['confidence']:.2f}"
                cv2.putText(frame, label, (sx, sy - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw human detections
            for det in self.human_detections:
                (sx, sy, ex, ey) = det['box']
                cv2.rectangle(frame, (sx, sy), (ex, ey), (255, 0, 0), 2)
                label = f"Person: {det['confidence']:.2f}"
                cv2.putText(frame, label, (sx, sy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # The siren and sprinkler run for the time set in the app, and
            # the alert gets the frame with the boxes drawn on it.
            if self.controller:
                self.controller.add_frame(frame)
                if turn_on:
                    self.controller.on_detection(frame, self.animal_detections, self.human_detections)

            with self.lock:
                self.frame = frame
                self.frame_seq += 1
    
    def get_frame(self):
        """Get current frame as bytes"""
        with self.lock:
            if self.frame is None:
                return None
            ret, buffer = cv2.imencode('.jpg', self.frame)
            return buffer.tobytes()

    def get_live_jpeg(self, max_width, quality):
        """Smaller frame for the cloud live view. Returns (frame number, jpeg bytes)."""
        with self.lock:
            if self.frame is None:
                return None, None
            frame, seq = self.frame, self.frame_seq
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return (seq, buffer.tobytes()) if ret else (None, None)

    # def _play_ding(self):
    #     """Play the ding sound"""
    #     try:
    #         playsound('ding.mp3', block=False)
    #     except Exception as e:
    #         print(f"Error playing sound: {e}")
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=5)
        self.cap.release()


# Create camera instance
camera_instance = None

# --- Flask App Setup ---
app = Flask(__name__)


def generate_frames():
    """Generate frames for video streaming"""
    global camera_instance
    while True:
        if camera_instance is None:
            time.sleep(0.1)
            continue
        
        frame_bytes = camera_instance.get_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('detection_viewer.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/detection_settings', methods=['GET', 'POST'])
def detection_settings():
    """Get or update detection toggle state"""
    global camera_instance
    if request.method == 'POST':
        data = request.get_json() or {}
        if camera_instance is not None:
            if camera_instance.controller:
                # Also tells the cloud, so the app shows the change.
                camera_instance.controller.set_local_detection(data.get('animal'), data.get('human'))
            else:
                if 'animal' in data:
                    camera_instance.animal_detection_enabled = bool(data['animal'])
                if 'human' in data:
                    camera_instance.human_detection_enabled = bool(data['human'])
    if camera_instance is None:
        return jsonify({'animal': True, 'human': True})
    return jsonify({
        'animal': camera_instance.animal_detection_enabled,
        'human': camera_instance.human_detection_enabled
    })


@app.route('/api/detection_events')
def detection_events():
    """Server-Sent Events: pushes the detection switches to the page whenever they
    change, including changes made from the mobile app."""
    controller = camera_instance.controller if camera_instance else None
    if controller is None:
        return Response(status=404)

    def stream():
        version = None  # Sends the current switches straight away
        while True:
            new_version, animal, human = controller.wait_for_detection_change(version, timeout=15)
            if new_version != version:
                version = new_version
                yield f"data: {json.dumps({'animal': animal, 'human': human})}\n\n"
            else:
                # Keeps the connection open and notices closed pages.
                yield ": keep-alive\n\n"

    return Response(stream(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache'})


def find_free_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    return None


def load_models():
    """Load YOLO and MobileNetSSD models"""
    global yolo_net, net, YOLO_CLASSES, output_layers
    
    print("Loading YOLO model...")
    yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
    
    # Try to use GPU if available
    # try:
    #     if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    #         yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #         yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #         print("Using GPU acceleration for YOLO")
    #     else:
    #         print("CUDA not available, using CPU for YOLO")
    # except Exception as e:
    #     print(f"GPU acceleration failed for YOLO: {e}")
    #     print("Using CPU for YOLO")

    print("Let's load.")
    
    with open(YOLO_CLASSES_PATH, 'r') as f:
        YOLO_CLASSES = [line.strip() for line in f.readlines()]
    
    # Get output layer names
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    
    print("Loading MobileNetSSD model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Try to use GPU for animal detection too
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU acceleration for animal detection")
        else:
            print("CUDA not available, using CPU for animal detection")
    except Exception as e:
        print(f"GPU acceleration failed for animal detection: {e}")
        print("Using CPU for animal detection")
    
    print("Models loaded successfully!")


def main():
    """Main function to run the web detection system"""
    global camera_instance
    
    print("Starting Animal and Human Detection System (Web)...")
    # Cloud settings come from a .env file next to this script.
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    
    # Load models first
    load_models()
    
    # Video source - 0 for webcam, or set AGRYNEX_VIDEO_SOURCE to a video file path
    video_source = os.environ.get('AGRYNEX_VIDEO_SOURCE', '0')
    video_source = int(video_source) if video_source.isdigit() else video_source
    print(f"Video source: {video_source}")
    
    controller = Controller(os.environ.get('AGRYNEX_SERVER_URL'), os.environ.get('AGRYNEX_DEVICE_TOKEN'))
    
    # Initialize camera
    camera_instance = Camera(video_source, controller)
    
    if not camera_instance.camera_ok:
        print(f"Error: Could not open video source {video_source}")
        return
    controller.start(camera_instance)
    
    # Find an available port
    port = find_free_port(5000)
    if port is None:
        print("Error: Could not find an available port")
        return
    
    print("Starting Flask web server...")
    print(f"Open your browser and go to: http://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera_instance:
            camera_instance.stop()
        controller.relays.off()
        print("Detection system stopped.")


if __name__ == '__main__':
    main()
