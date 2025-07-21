from flask import Flask, Response, request, jsonify, render_template
import json
import cv2
import pyzed.sl as sl
from flask_cors import cross_origin
import os, re


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
        "perspective2": data.get("perspective2", [])
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

@app.route('/')
def index():
    return render_template('area_definition.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
