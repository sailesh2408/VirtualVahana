import carla
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load the lightweight YOLO model
model = YOLO('yolov8n.pt') 

# Global variables to store the latest frames
latest_rgb = None
latest_depth_raw = None

def process_rgb_image(image):
    global latest_rgb
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    latest_rgb = i2[:, :, :3] # Extract RGB

def process_depth_image(image):
    global latest_depth_raw
    # We do NOT use the ColorConverter here. We need the raw data for math.
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    latest_depth_raw = i2[:, :, :3] # Extract RGB components of depth map

def get_distance(x, y, depth_frame):
    """ Converts CARLA raw depth pixel at (x,y) to meters """
    # Ensure coordinates are within image bounds
    x = max(0, min(x, depth_frame.shape[1] - 1))
    y = max(0, min(y, depth_frame.shape[0] - 1))
    
    # CARLA depth map uses B, G, R channels
    b, g, r = depth_frame[y, x]
    
    # CARLA depth decoding formula
    normalized_depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    distance_in_meters = normalized_depth * 1000.0
    return distance_in_meters

def main():
    global latest_rgb, latest_depth_raw
    actor_list = []
    
    try:
        print("Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # Spawn Vehicle
        vehicle_bp = bp_lib.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Spawn RGB Camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(rgb_camera)
        rgb_camera.listen(lambda image: process_rgb_image(image))

        # Spawn Depth Camera
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '640')
        depth_bp.set_attribute('image_size_y', '480')
        depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actor_list.append(depth_camera)
        depth_camera.listen(lambda image: process_depth_image(image))

        vehicle.set_autopilot(True)
        print("Running YOLO and Distance Estimation... Press Ctrl+C to stop.")

        # Main Processing Loop
        while True:
            if latest_rgb is not None and latest_depth_raw is not None:
                # 1. Copy the frame to avoid race conditions
                frame = latest_rgb.copy()
                depth_frame = latest_depth_raw.copy()

                # 2. Run YOLO inference
                results = model(frame, verbose=False, classes=[2, 5, 7]) # Only detect cars, buses, trucks

                # 3. Process detections
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Find the bottom-center of the bounding box (the vehicle's rear bumper)
                        center_x = int((x1 + x2) / 2)
                        bottom_y = y2
                        
                        # Calculate distance using our depth map
                        dist_meters = get_distance(center_x, bottom_y, depth_frame)
                        
                        # Draw everything on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, bottom_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"{dist_meters:.1f} m", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("ADAS Perception Output", frame)
                cv2.waitKey(1)
            else:
                time.sleep(0.01)

    finally:
        print("\nCleaning up actors...")
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == '__main__':
    main()