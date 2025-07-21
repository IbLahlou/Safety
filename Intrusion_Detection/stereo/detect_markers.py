import cv2
import cv2.aruco as aruco
import pyzed.sl as sl
import numpy as np
import json 
import os 

marker_length = 0.17
AREA_ID = "zone_1"  # or ask the user dynamically if needed
JSON_FILE = "aruco_markers_pose.json"

# Create a ZED camera object
zed = sl.Camera()

# Set configuration
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # or HD1080, VGA
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

# Open the camera
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED Open failed: {status}")
    exit()
    
# Camera matrix and distortion coefficients
cam_info = zed.get_camera_information().camera_configuration.calibration_parameters_raw
cam_params = cam_info.left_cam


#camera matrix 
fx = cam_params.fx
fy = cam_params.fy
cx = cam_params.cx
cy = cam_params.cy

camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])

# Distortion coefficients
dist_coeffs = np.array([
    cam_params.disto[0],  # k1
    cam_params.disto[1],  # k2
    cam_params.disto[2],  # p1
    cam_params.disto[3],  # p2
    cam_params.disto[4],  # k3
])

# Prepare image containers
image = sl.Mat()


# Load ZED left image (or any RGB image)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
parameters = aruco.DetectorParameters()
# Create detector
detector = aruco.ArucoDetector(dictionary, parameters)

try:
    print("Press Ctrl+C to stop the stream.")
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            img_np = image.get_data()

            # Convert to OpenCV format
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

            # Resize image for better performance (optional)
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(img_cv2)
            
            if ids is not None and len(ids) <= 4 and cv2.waitKey(1) & 0xFF == ord('i'):
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                markers_data = {}
                marker_id = int(ids[0][0])
                markers_data[marker_id] = {
                    "rvec": rvecs[0][0].tolist(), 
                    "tvec": tvecs[0][0].tolist()
                }
                if os.path.exists(JSON_FILE):
                    with open(JSON_FILE, "r") as f:
                        json_data = json.load(f)
                else:
                    json_data = {"areas": []}

                area_count = len(json_data["areas"])
                area_id = f"zone_{area_count + 1}"
                # Append new area
                json_data["areas"].append({
                    "area_id": area_id,
                    "markers": markers_data
                })

                # Save updated data
                with open(JSON_FILE, "w") as f:
                    json.dump(json_data, f, indent=4)

                print(f"Saved 1 marker poses to {JSON_FILE} under area_id: {area_id}")
                
            if ids is not None and len(ids) == 4 and cv2.waitKey(1) & 0xFF == ord('i'):

                # Estimate pose of markers
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

                markers_data = {}
                for i in range(4):
                    marker_id = int(ids[i][0])
                    markers_data[marker_id] = {
                        "rvec": rvecs[i][0].tolist(), 
                        "tvec": tvecs[i][0].tolist()
                    }
                if os.path.exists(JSON_FILE):
                    with open(JSON_FILE, "r") as f:
                        json_data = json.load(f)
                else:
                    json_data = {"areas": []}

                area_count = len(json_data["areas"])
                area_id = f"zone_{area_count + 1}"
                # Append new area
                json_data["areas"].append({
                    "area_id": area_id,
                    "markers": markers_data
                })

                # Save updated data
                with open(JSON_FILE, "w") as f:
                    json.dump(json_data, f, indent=4)

                print(f"Saved 4 marker poses to {JSON_FILE} under area_id: {area_id}")


            # Draw markers for visualization
            aruco.drawDetectedMarkers(img_cv2, corners, ids)

            cv2.imshow("Aruco Detection", img_cv2)
            print(f"Detected markers: {ids.flatten() if ids is not None else 'None'}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Clean up
    zed.close()
    cv2.destroyAllWindows()



