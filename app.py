import cv2
import threading
import time
import os
import glob
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURATION VARIABLES ---
RECORDINGS_DIR = 'recordings'
MAX_FILE_AGE_DAYS = 5 
CLEANUP_INTERVAL_SECS = 3600 
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Load standard model to detect everything
model = YOLO('yolov8n.pt')

# --- GLOBAL SYSTEM STATES ---
show_boxes = True
is_recording = False

# --- AUTOMATIC STORAGE CLEANUP ENGINE ---
def storage_janitor_loop():
    """Background thread that continuously checks and removes expired footage."""
    while True:
        print("[STORAGE CLEANUP] Scanning recordings directory for expired files...")
        current_time = time.time()
        # Calculate the expiration threshold age in seconds (86400 seconds in a day)
        expiration_threshold = current_time - (MAX_FILE_AGE_DAYS * 86400)
        
        video_files = glob.glob(os.path.join(RECORDINGS_DIR, "*.mp4"))
        for file_path in video_files:
            try:
                file_modified_time = os.path.getmtime(file_path)
                if file_modified_time < expiration_threshold:
                    os.remove(file_path)
                    print(f"[STORAGE CLEANUP] Deleted expired file: {file_path}")
            except Exception as e:
                print(f"[STORAGE CLEANUP] Error processing file {file_path}: {e}")
        
        time.sleep(CLEANUP_INTERVAL_SECS)

janitor_thread = threading.Thread(target=storage_janitor_loop, daemon=True)
janitor_thread.start()


# --- CENTRALIZED CAMERA & WORKER ENGINE ---
class CameraStream:
    def __init__(self, camera_id, cam_label):
        self.camera_id = camera_id
        self.cam_label = cam_label
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.success = False
        self.processed_frame = None  # Ready-to-serve frame bytes
        self.running = True
        
        self.video_writer = None
        self._writer_lock = threading.Lock()
        
        # Dedicated thread for frame capture, AI inference, and video writing
        self.thread = threading.Thread(target=self._worker_loop, args=())
        self.thread.daemon = True
        self.thread.start()

    def _worker_loop(self):
        global show_boxes, is_recording
        
        while self.running:
            success, raw_frame = self.camera.read()
            if not success or raw_frame is None:
                time.sleep(0.01)
                continue

            # 1. DRAW TIMESTAMP OVERLAY
            current_now = time.time()
            time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_now))
            milliseconds = int((current_now - int(current_now)) * 1000)
            precise_timestamp = f"{time_string}.{milliseconds:03d}"

            cv2.rectangle(raw_frame, (10, 10), (340, 40), (0, 0, 0), -1)
            cv2.putText(
                raw_frame,
                text=f"{self.cam_label.upper()} | {precise_timestamp}",
                org=(15, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 118),
                thickness=1,
                lineType=cv2.LINE_AA
            )

            # 2. OPTIONAL AI INFERENCE (Happens exactly once per frame captured)
            if show_boxes:
                # stream=True optimizes generator processing inside Ultralytics
                results = model(raw_frame, stream=True, conf=0.25, verbose=False)
                for result in results:
                    raw_frame = result.plot()

            # 3. STABLE BACKGROUND RECORDING WRITER
            with self._writer_lock:
                if is_recording:
                    if self.video_writer is None:
                        timestamp_fn = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"{RECORDINGS_DIR}/{self.cam_label}_{timestamp_fn}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # Lock recording speed dynamically to ~20FPS via loop pacing
                        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
                    
                    self.video_writer.write(raw_frame)
                else:
                    if self.video_writer is not None:
                        self.video_writer.release()
                        self.video_writer = None

            # 4. COMPRESS & STORE TO MEMORY FOR WEB VIEWERS
            ret, buffer = cv2.imencode('.jpg', raw_frame)
            if ret:
                self.processed_frame = buffer.tobytes()
                self.success = True
            
            # Control pacing to avoid unthrottled high CPU loops (~30 FPS max processing speed)
            time.sleep(0.03)

    def get_encoded_frame(self):
        return self.success, self.processed_frame

    def stop_recording_explicitly(self):
        """Used by API endpoint to guarantee safe close down of file streams."""
        with self._writer_lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

    def stop(self):
        self.running = False
        self.stop_recording_explicitly()
        if self.camera.isOpened():
            self.camera.release()


# Initialize streams globally with assigned tags
cam1 = CameraStream(0, "cam1")
cam2 = CameraStream(1, "cam2")


def generate_frames(camera_instance):
    """Clean, high-performance generator loop optimized for multiple client views."""
    while True:
        success, frame_bytes = camera_instance.get_encoded_frame()
        if not success or frame_bytes is None:
            time.sleep(0.03)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- API CONTROL ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_1')
def video_feed_1():
    return Response(generate_frames(cam1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    return Response(generate_frames(cam2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_boxes', methods=['POST'])
def toggle_boxes():
    global show_boxes
    data = request.get_json() or {}
    show_boxes = data.get('enabled', True)
    return jsonify({"status": "success", "show_boxes": show_boxes})

@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    global is_recording
    data = request.get_json() or {}
    is_recording = data.get('enabled', False)
    
    # Safely spin down disk file systems explicitly if recording stops
    if not is_recording:
        cam1.stop_recording_explicitly()
        cam2.stop_recording_explicitly()
        
    return jsonify({"status": "success", "is_recording": is_recording})


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        print("[SYSTEM SHUTDOWN] Releasing all camera resources gracefully...")
        cam1.stop()
        cam2.stop()