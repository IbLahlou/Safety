from flask import Flask, Response, request, jsonify, render_template
import json
import cv2
import pyzed.sl as sl
from flask_cors import cross_origin
import os, re
import supervision as sv
import time
import numpy as np

app = Flask(__name__)

# Global variable to store current detection data
current_detections = {}

# Init ZED
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Add this for consistency
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Change this for object detection
init_params.coordinate_units = sl.UNIT.METER
zed.open(init_params)

# Setup object detection
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = False
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
zed.enable_object_detection(obj_param)
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
objects = sl.Objects()

# Get camera info
camera_info = zed.get_camera_information()
image_width = camera_info.camera_configuration.resolution.width
image_height = camera_info.camera_configuration.resolution.height

# Initialize supervision components
byte_track = sv.ByteTrack(frame_rate=30)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=(image_width, image_height))
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(image_width, image_height))

box_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER,
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=30 * 2,
    position=sv.Position.BOTTOM_CENTER,
)


def gen_frames():
    image_left = sl.Mat()
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            frame = image_left.get_data()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def load_polygon_zones(file_path):
    """Load polygon zones from JSON file"""
    polygon_zones = {}
    corners = None
    corner_dict = {}
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for zone_id, zone_data in data.items():
                corner_list = zone_data.get('perspective1', [])
                corners = np.array([[corner["x"], corner["y"]] for corner in corner_list], dtype=np.int32)
                polygon_zone = sv.PolygonZone(polygon=corners)
                polygon_zones[zone_id] = polygon_zone
                corner_dict[zone_id] = corners.tolist()
                print(f"Created polygon zone: {zone_id}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    
    return corner_dict, polygon_zones

def zed_objects_to_sv_detections(objects):
    """Convert ZED Objects to Supervision Detections format"""
    if not objects.is_new or len(objects.object_list) == 0:
        return sv.Detections.empty()
    
    boxes = []
    confidences = []
    class_ids = []
    
    for obj in objects.object_list:
        if obj.label == sl.OBJECT_CLASS.PERSON:
            bbox_2d = obj.bounding_box_2d
            if len(bbox_2d) >= 4:
                x1 = int(bbox_2d[0][0])
                y1 = int(bbox_2d[0][1])
                x2 = int(bbox_2d[2][0])
                y2 = int(bbox_2d[2][1])
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(obj.confidence / 100.0)
                class_ids.append(0)
    
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int)
    )


corners, polygon_zones = load_polygon_zones('coordinates.json')            

def gen_frames_with_detection():
    image_left = sl.Mat()
    global current_detections
    current_detections = {}

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Get the image
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            frame = image_left.get_data()
            
            # Convert BGRA to BGR if needed
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Get object detections
            zed.retrieve_objects(objects, obj_runtime_param)
            detections = zed_objects_to_sv_detections(objects)
            
            # Apply polygon zone filtering if zones exist
            if polygon_zones and len(polygon_zones) > 0:
                all_zone_detections = []
                for zone_key, zone in polygon_zones.items():
                    zone_detections = detections[zone.trigger(detections)]
                    current_detections[zone_key] = len(zone_detections)
                    all_zone_detections.append(zone_detections)
                
                if all_zone_detections:
                    combined_detections = all_zone_detections[0]
                    for zone_det in all_zone_detections[1:]:
                        combined_detections = sv.Detections.merge([combined_detections, zone_det])
                    detections = combined_detections
                else:
                    detections = sv.Detections.empty()
            
            # Update tracker
            detections = byte_track.update_with_detections(detections=detections)
            
            # Create labels
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id] if detections.tracker_id is not None else []
            
            # Annotate frame
            annotated_frame = frame.copy()
            
            # Draw polygon zones
            if polygon_zones and len(polygon_zones) > 0:
                for i, (zone_key, zone) in enumerate(polygon_zones.items()):
                    zone_color = sv.Color.from_hex(f"#{(i * 1234567) % 0xFFFFFF:06x}")
                    if zone_key in corners:
                        polygon = np.array(corners[zone_key], dtype=np.int32)
                        annotated_frame = sv.draw_polygon(annotated_frame, polygon=polygon, color=zone_color)
            
            # Draw detections
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            if labels:
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Encode and yield frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/current_detections')
@cross_origin()
def get_current_detections():
    """Get current detection counts for each zone"""
    return jsonify({
        "timestamp": int(time.time()),
        "detections": current_detections,
        "total_count": sum(current_detections.values())
    })

#annotated video feed
@app.route('/video_feed_with_detection')
@cross_origin()
def video_feed_with_detection():
    return Response(gen_frames_with_detection(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

#button to reload zones
@app.route('/reload_zones', methods=['POST'])
@cross_origin()
def reload_zones():
    """Reload polygon zones from coordinates.json"""
    global corners, polygon_zones
    corners, polygon_zones = load_polygon_zones('coordinates.json')
    return jsonify({
        "status": "success", 
        "zones_loaded": len(polygon_zones),
        "zone_ids": list(polygon_zones.keys())
    })


@app.route('/video_feed')
@cross_origin()
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    data = request.get_json()

    area_data = {
        "perspective1": data.get("perspective1", []),
    }

    filepath = "coordinates.json"
    all_areas = {}

    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()
                if content:  # make sure file is not empty
                    all_areas = json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ Warning: coordinates.json is corrupted. Reinitializing.")
            all_areas = {}

    # Find next zone ID
    zone_numbers = [
        int(re.search(r"zone_(\d+)", key).group(1))
        for key in all_areas.keys()
        if isinstance(key, str) and re.match(r"zone_\d+", key)
    ]
    next_zone_number = max(zone_numbers, default=0) + 1
    new_zone_id = f"zone_{next_zone_number}"

    all_areas[new_zone_id] = area_data

    with open(filepath, "w") as f:
        json.dump(all_areas, f, indent=4)

    return jsonify({"status": "success", "zone_id": new_zone_id}), 200

@app.route("/detection_dashboard")
def dashboard():
    return render_template("detection_dashboard.html")


@app.route('/define_area')
def index():
    return render_template('area_definition.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
