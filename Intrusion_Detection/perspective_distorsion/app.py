from flask import Flask, Response, request, jsonify, render_template
import json
import cv2
import pyzed.sl as sl
from flask_cors import cross_origin
import os, re
import supervision as sv
import numpy as np

app = Flask(__name__)

# Init ZED
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.NONE
zed.open(init_params)

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


def create_polygon_filter(zones):
    """Create polygon zones from saved coordinates"""
    polygon_zones = {}
    
    for zone_id, zone in zones.items():
        if "perspective1" in zone:
            corners_list = zone["perspective1"]
            if len(corners_list) == 4:
                try:
                    # Convert list of dicts to a NumPy array [[x, y], [x, y], ...]
                    corners = np.array([[pt["x"], pt["y"]] for pt in corners_list], dtype=np.float32)
                    
                    # Create the polygon zone with your camera resolution
                    polygon_zone = sv.PolygonZone(
                        polygon=corners,
                        frame_resolution_wh=(1280, 720) 
                    )
                    polygon_zones[zone_id] = polygon_zone
                    print(f"✅ Created polygon zone: {zone_id}")
                    
                except Exception as e:
                    print(f"❌ Error creating zone {zone_id}: {e}")
                    continue
    return polygon_zones


@app.route("/detect_intrusion", methods=['GET'])
def detect_intrusion():
    filepath = "coordinates.json"
    
    # Check if coordinates file exists
    if not os.path.exists(filepath):
        return jsonify({"error": "No zones defined. Please create zones first."}), 400
    
    # Read coordinates from json file
    try:
        with open(filepath, "r") as f:
            zones = json.load(f)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid coordinates file format."}), 400
    
    # Create polygon zones from saved coordinates
    polygon_zones = create_polygon_filter(zones)
    
    if not polygon_zones:
        return jsonify({"error": "No valid polygon zones found."}), 400
    
    # Initialize model and tracking (adjust these parameters as needed)
    try:
        # Replace with your actual model initialization
        # model = get_roboflow_model(model_id=your_model_id, api_key=your_api_key)
        
        # Initialize ByteTrack for object tracking
        byte_track = sv.ByteTrack(
            frame_rate=30,  # Adjust based on your camera FPS
            track_activation_threshold=0.5  # Adjust confidence threshold
        )
        
        # Initialize annotators
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(1280, 720))
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(1280, 720))
        
        box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=60,  # 2 seconds at 30fps
            position=sv.Position.BOTTOM_CENTER,
        )
        
        return jsonify({
            "status": "success", 
            "message": "Intrusion detection activated",
            "zones_loaded": list(polygon_zones.keys()),
            "total_zones": len(polygon_zones)
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to initialize detection: {str(e)}"}), 500


def gen_frames_with_detection():
    """Generate frames with intrusion detection applied"""
    filepath = "coordinates.json"
    
    # Load polygon zones
    with open(filepath, "r") as f:
        zones = json.load(f)
    polygon_zones = create_polygon_filter(zones)
    
    # Initialize ZED camera
    image_left = sl.Mat()
    
    # Initialize detection components
    byte_track = sv.ByteTrack(frame_rate=30, track_activation_threshold=0.5)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(1280, 720))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(1280, 720))
    
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=60,
        position=sv.Position.BOTTOM_CENTER,
    )
    
    # Polygon zone annotator to visualize zones
    polygon_annotator = sv.PolygonZoneAnnotator(
        zone=list(polygon_zones.values())[0] if polygon_zones else None,
        color=sv.Color.RED,
        thickness=2
    )
    
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            frame = image_left.get_data()
            
            # Run object detection (replace with your actual model)
            # results = model.infer(frame)[0]
            # detections = sv.Detections.from_inference(results)
            
            # For demonstration, create empty detections
            # Replace this with your actual detection code
            detections = sv.Detections.empty()
            
            # Apply confidence threshold
            # detections = detections[detections.confidence > confidence_threshold]
            
            # Filter detections for each polygon zone
            zone_detections = {}
            for zone_id, polygon_zone in polygon_zones.items():
                # Filter detections within this polygon zone
                zone_filtered = detections[polygon_zone.trigger(detections)]
                
                # Apply NMS
                zone_filtered = zone_filtered.with_nms(threshold=0.5)
                
                # Update tracking
                zone_filtered = byte_track.update_with_detections(detections=zone_filtered)
                
                zone_detections[zone_id] = zone_filtered
                
                # Log intrusions
                if len(zone_filtered) > 0:
                    print(f"🚨 Intrusion detected in {zone_id}: {len(zone_filtered)} objects")
            
            # Combine all zone detections for annotation
            all_detections = sv.Detections.empty()
            for zone_detections_single in zone_detections.values():
                all_detections = sv.Detections.merge([all_detections, zone_detections_single])
            
            # Annotate frame
            annotated_frame = frame.copy()
            
            # Draw polygon zones
            for zone_id, polygon_zone in polygon_zones.items():
                # Draw polygon zone boundary
                polygon_points = polygon_zone.polygon.astype(np.int32)
                cv2.polylines(annotated_frame, [polygon_points], True, (0, 255, 0), 2)
                
                # Add zone label
                center_x = int(np.mean(polygon_points[:, 0]))
                center_y = int(np.mean(polygon_points[:, 1]))
                cv2.putText(annotated_frame, zone_id, (center_x, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw detection annotations
            if len(all_detections) > 0:
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, detections=all_detections
                )
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=all_detections
                )
                annotated_frame = trace_annotator.annotate(
                    scene=annotated_frame, detections=all_detections
                )
            
            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/video_feed_detection")
def video_feed_detection():
    """Video feed with intrusion detection"""
    return Response(gen_frames_with_detection(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route('/define_area')
def index():
    return render_template('area_definition.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
