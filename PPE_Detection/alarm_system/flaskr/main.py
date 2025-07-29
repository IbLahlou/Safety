# main.py
import threading
import argparse
import torch
from PPE_Detection.camera_stream.custom_detector import start_camera
from PPE_Detection.alarm_system.flaskr import create_app




def run_flask():
    app = create_app()
    socketio = app.socketio
    socketio.run(app, host='0.0.0.0', port=5000)

def run_camera():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt')
    parser.add_argument('--svo', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--conf_thres', type=float, default=0.4)
    opt = parser.parse_args()

    with torch.no_grad():
        start_camera(opt)

if __name__ == '__main__':
    # Start camera in a background thread
    camera_thread = threading.Thread(target=run_camera, daemon=True)
    camera_thread.start()
    # Start Flask app in main thread (or another thread if preferred)
    run_flask()
