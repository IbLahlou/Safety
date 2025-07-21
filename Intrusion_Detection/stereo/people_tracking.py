import pyzed.sl as sl
import numpy as np

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # or HD1080, VGA
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.MILLIMETER  

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED Open failed: {status}")
    exit()

# Object detection parameters
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
            # Retrieve detected objects
            zed.retrieve_objects(objects, obj_runtime_param)

            if objects.is_new:
                people = []  # List to hold (ID, position) of detected people
                for obj in objects.object_list:
                    if obj.label == sl.OBJECT_CLASS.PERSON and np.all(obj.position):
                        # Get position in meters
                        person_pos_m = [coord / 1000.0 for coord in obj.position]
                        people.append((obj.id, person_pos_m))

                if people:
                    print("Detected people and their positions (in meters):")
                    for person_id, pos in people:
                        print(f"  ID {person_id}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                else:
                    print("No people detected.")
except KeyboardInterrupt:
    print("Stopped by user")

finally:
    zed.close()
