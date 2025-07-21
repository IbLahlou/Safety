import json
from shapely.geometry import Point, Polygon
import pyzed.sl as sl
import numpy as np
import cv2


with open("aruco_markers_pose.json", "r") as f:
    zones_data = json.load(f)["areas"]

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # or HD1080, VGA
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE



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

dist_coeffs = np.array(cam_params.disto[:5])

# Object detection parameters
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = False  
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.PERSON_HEAD_BOX_FAST
zed.enable_object_detection(obj_param)

obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
objects = sl.Objects()
image = sl.Mat()

"""
def draw_zones_on_image(img, zones_data, camera_matrix, dist_coeffs):
    for zone in zones_data:
        marker_tvecs = []
        for marker in sorted(zone["markers"].keys(), key=int):
            tvec = np.array(zone["markers"][marker]["tvec"]).reshape((3, 1))
            rvec = np.array(zone["markers"][marker]["rvec"]).reshape((3, 1))
            # Project the 3D point to 2D
            imgpts, _ = cv2.projectPoints(tvec, rvec, tvec, camera_matrix, dist_coeffs)
            marker_tvecs.append(imgpts[0][0])  # Extract (x, y)

        if len(marker_tvecs) == 4:
            pts = np.array(marker_tvecs, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(img, zone["area_id"], tuple(pts[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def is_inside_any_zone(person_pos_m, zones_data):
    person_2d = Point(person_pos_m[0], person_pos_m[2])  # only X and Z

    for zone in zones_data:
        markers = zone["markers"]
        # Ensure consistent order if needed, or sort by marker ID
        polygon_points = [
            (markers[k]["tvec"][0], markers[k]["tvec"][2])
            for k in sorted(markers.keys(), key=int)
        ]
        polygon = Polygon(polygon_points)

        if polygon.contains(person_2d):
            return zone["area_id"]  
    return None
"""
def is_point_inside_quad(point, quad_pts):
    """Check if a point is inside a convex quadrilateral using cross products."""
    def cross(a, b):
        return a[0]*b[1] - a[1]*b[0]

    def vector(p1, p2):
        return (p2[0]-p1[0], p2[1]-p1[1])

    for i in range(4):
        a = quad_pts[i]
        b = quad_pts[(i + 1) % 4]
        ab = vector(a, b)
        ap = vector(a, point)
        if cross(ab, ap) < 0:
            return False
    return True


def is_inside_any_zone(person_pos_m, zones_data):
    px, _, pz = person_pos_m
    person_2d = (px, pz)

    for zone in zones_data:
        markers = zone["markers"]
        if len(markers) != 4:
            continue  # skip malformed zones

        # Get 2D corners in X-Z plane
        quad = [
            (markers[k]["tvec"][0], markers[k]["tvec"][2])
            for k in sorted(markers.keys(), key=int)
        ]

        if is_point_inside_quad(person_2d, quad):
            return zone["area_id"]
    return None


try:
    print("Press Ctrl+C to stop")

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_objects(objects, obj_runtime_param)
            zed.retrieve_image(image, sl.VIEW.LEFT)
            img_np = image.get_data()
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            cv2.imshow("Intrusion Detection", img_cv2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if objects.is_new:
                for obj in objects.object_list:
                    if obj.label == sl.OBJECT_CLASS.PERSON and np.all(obj.position):
                        person_pos = obj.position
                        person_pos_m = [coord / 1000.0 for coord in person_pos]

                        print(f"Person detected at: x={person_pos_m[0]:.2f}, y={person_pos_m[1]:.2f}, z={person_pos_m[2]:.2f}")

                        # Check if inside any zone
                        zone_id = is_inside_any_zone(person_pos_m, zones_data)
                        if zone_id:
                            print(f"⚠️ Intrusion detected in {zone_id}!")
                        else:
                            print("✅ Person is outside all zones.")
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    zed.close()