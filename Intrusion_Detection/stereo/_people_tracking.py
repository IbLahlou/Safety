import pyzed.sl as sl
import numpy as np
import cv2
import cv2.aruco as aruco
import json

# Marker and ArUco settings
marker_length = 0.17  # meters
ARUCO_DICT = aruco.DICT_6X6_100
JSON_FILE = "aruco_markers_pose.json"

# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED Open failed: {status}")
    exit()

# Load marker data
with open(JSON_FILE, "r") as f:
    marker_data = json.load(f)["areas"][-1]["markers"]  # use last zone
    marker_id = list(marker_data.keys())[0]
    marker_info = marker_data[marker_id]
    rvec = np.array(marker_info["rvec"])
    tvec = np.array(marker_info["tvec"])

# Get rotation matrix from rvec
R_marker_to_camera, _ = cv2.Rodrigues(rvec)
R_camera_to_marker = R_marker_to_camera.T
t_camera_to_marker = -R_camera_to_marker @ tvec

# Setup object detection
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = False
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_FAST
zed.enable_object_detection(obj_param)
obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
objects = sl.Objects()

try:
    print("Press Ctrl+C to stop")
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_objects(objects, obj_runtime_param)

            if objects.is_new:
                for obj in objects.object_list:
                    if obj.label == sl.OBJECT_CLASS.PERSON and np.all(obj.position):
                        # Get person position in camera frame (meters)
                        person_pos = np.array([coord / 1000.0 for coord in obj.position])

                        # Convert to marker frame
                        person_in_marker = R_camera_to_marker @ (person_pos - tvec)

                        print("Person Position (Camera frame):", person_pos)
                        print("Person Position (Marker frame):", person_in_marker)

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    zed.close()
