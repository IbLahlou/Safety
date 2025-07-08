import pickle
from PPE_Detection.camera_stream.redis_queue import r  # The redis.Redis instance you use
from PPE_Detection.camera_stream.data_storage import dump_from_redis_to_mongo
from PPE_Detection.alarm_system.flaskr.sockets.socketio_setup import socketio
import cv2
import base64


confidence_threshold = 0.3

def process_detection_queue(class_risk, socketio):
    print("Starting to process detection queue...")
    processed_count = 0

    while True:
        payload = r.rpop('detection_queue')
        if not payload:
            break  

        timestamp, frame, det_cls, det_cnf, det_xywh = pickle.loads(payload)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = base64.b64encode(buffer).decode('utf-8')
        
        dump_from_redis_to_mongo(timestamp, frame, det_cls, det_cnf, det_xywh)

        print(f"[{timestamp}] Processing frame with {len(det_cls)} detections.")

        num_people = 0
        severity_raw = 0.0
        class_counts = {} 

        for cls, cnf in zip(det_cls, det_cnf):
            cls = int(cls)
            if cls == 9:
                num_people += 1
            if cnf > confidence_threshold and cls in class_risk:
                class_counts[cls] = class_counts.get(cls, 0) + 1
                severity_raw += class_risk[cls] * cnf

        frame_severity_normalized = severity_raw / max(num_people, 1)

        if frame_severity_normalized < 0.2:
            severity_index = "Low"
        elif frame_severity_normalized < 0.5:
            severity_index = "Moderate"
        else:
            severity_index = "High"
        payload = pickle.dumps((timestamp, frame_bytes, severity_index, class_counts))
        r.lpush('history_queue', payload)
        socketio.emit('high_severity_alert', {
            'timestamp': timestamp,
            'severity': frame_severity_normalized,
            'class_counts': class_counts
        })
        
        print(f"Number of people detected: {num_people}")
        print(f"Frame Severity: {frame_severity_normalized:.2f} ({severity_index})")
        print(f"Class counts: {class_counts}")

        processed_count += 1

    print(f"Finished processing {processed_count} items.")

