import supervision as sv 
import os
import json
import numpy as np 
import pyzed as sl


#read coordinates from json file
with open("coordinates.json", "r") as f:
    zones = json.load(f)

#create polygon filter using supervision library

def create_polygon_filter(zones):
    polygon_zones = {}
    for zone_id, zone in zones.items():
        if "perspective1" in zone:
            corners_list = zone["perspective1"]
            if len(corners_list) == 4:
                # Convert list of dicts to a NumPy array [[x, y], [x, y], ...]
                corners = np.array([[pt["x"], pt["y"]] for pt in corners_list], dtype=np.float32)
                
                # Create the polygon zone (adjust `frame_resolution_wh` to your context)
                polygon_zone = sv.PolygonZone(
                    polygon=corners,
                    frame_resolution_wh=(1280, 720)  
                )
                
                polygon_zones[zone_id] = polygon_zone
    return polygon_zones

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

video_info = sv.VideoInfo(#zed 2i 
    )

    