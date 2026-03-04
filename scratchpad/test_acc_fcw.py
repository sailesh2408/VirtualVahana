import carla
import cv2
import numpy as np
import time
import pygame
from ultralytics import YOLO

# Load the lightweight YOLO model
model = YOLO('yolov8n.pt')

# Global variables for sensor data
latest_rgb = None
latest_depth_raw = None

class ACC_FCW_Controller:
    def __init__(self, target_distance=15.0, kp=0.3, ki=0.01, kd=0.1):
        # PID Constants
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.target_distance = target_distance
        self.prev_error = 0.0
        self.integral = 0.0
        
        # Smoothing & TTC Variables
        self.prev_smoothed_dist = None
        self.smoothed_v_rel = 0.0

    def process(self, raw_distance, dt):
        throttle = 0.0
        brake = 0.0
        warning_state = "SAFE"
        warning_color = (0, 255, 0) # Green
        
        # 1. Smooth the distance to ignore YOLO bounding box jitter
        alpha_dist = 0.15 # Trust 15% of the new frame, 85% of the old smoothed data
        if self.prev_smoothed_dist is None:
            smoothed_dist = raw_distance
        else:
            smoothed_dist = (alpha_dist * raw_distance) + ((1.0 - alpha_dist) * self.prev_smoothed_dist)
        
        # 2. Calculate Relative Velocity and TTC (FCW Logic)
        if self.prev_smoothed_dist is not None and dt > 0:
            raw_v_rel = (self.prev_smoothed_dist - smoothed_dist) / dt
            
            # Smooth the velocity to prevent sudden spikes
            alpha_v = 0.2
            self.smoothed_v_rel = (alpha_v * raw_v_rel) + ((1.0 - alpha_v) * self.smoothed_v_rel)
            
            # If v_rel is positive, we are getting closer
            if self.smoothed_v_rel > 0.5: 
                ttc = smoothed_dist / self.smoothed_v_rel
                
                # Multi-stage AEB Logic
                if ttc < 1.5: 
                    warning_state = "CRITICAL: AEB FULL BRAKE"
                    warning_color = (0, 0, 255) # Red
                    self.prev_smoothed_dist = smoothed_dist
                    return 0.0, 1.0, warning_state, warning_color # Override with full brake
                elif ttc < 2.5:
                    warning_state = "WARNING: AEB PARTIAL BRAKE"
                    warning_color = (0, 165, 255) # Orange
                    self.prev_smoothed_dist = smoothed_dist
                    return 0.0, 0.5, warning_state, warning_color # Override with partial brake
                elif ttc < 3.5:
                    warning_state = "CAUTION: FCW ALERT"
                    warning_color = (0, 255, 255) # Yellow
                
        self.prev_smoothed_dist = smoothed_dist
        
        # 3. Calculate PID (ACC Logic) using smoothed distance
        error = smoothed_dist - self.target_distance
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        pid_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Map PID output to vehicle controls
        if pid_output > 0:
            throttle = min(pid_output, 0.6)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(pid_output), 1.0)
            
        return throttle, brake, warning_state, warning_color
    def __init__(self, target_distance=15.0, kp=0.1, ki=0.01, kd=0.05):
        # PID Constants
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.target_distance = target_distance
        self.prev_error = 0.0
        self.integral = 0.0
        
        # TTC Variables
        self.prev_distance = None
        self.prev_time = None

    def process(self, current_distance, dt):
        """Calculates TTC for FCW and PID for ACC"""
        throttle = 0.0
        brake = 0.0
        warning_state = "SAFE"
        warning_color = (0, 255, 0) # Green
        
        # 1. Calculate Relative Velocity and TTC (FCW Logic)
        if self.prev_distance is not None and dt > 0:
            v_rel = (self.prev_distance - current_distance) / dt
            
            # If v_rel is positive, we are getting closer to the car ahead
            if v_rel > 0.5: 
                ttc = current_distance / v_rel
                
                # Multi-stage AEB Logic per the rulebook
                if ttc < 0.8:
                    warning_state = "CRITICAL: AEB FULL BRAKE"
                    warning_color = (0, 0, 255) # Red
                    return 0.0, 1.0, warning_state, warning_color # Override with full brake
                elif ttc < 1.5:
                    warning_state = "WARNING: AEB PARTIAL BRAKE"
                    warning_color = (0, 165, 255) # Orange
                    return 0.0, 0.5, warning_state, warning_color # Override with partial brake
                elif ttc < 2.5:
                    warning_state = "CAUTION: FCW ALERT"
                    warning_color = (0, 255, 255) # Yellow
            else:
                ttc = float('inf')
                
        self.prev_distance = current_distance
        
        # 2. Calculate PID (ACC Logic)
        error = current_distance - self.target_distance
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        pid_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Map PID output to vehicle controls
        if pid_output > 0:
            throttle = min(pid_output, 0.6) # Cap throttle at 60% for safety
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(pid_output), 1.0)
            
        return throttle, brake, warning_state, warning_color

def process_rgb_image(image):
    global latest_rgb
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    latest_rgb = i2[:, :, :3]

def process_depth_image(image):
    global latest_depth_raw
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    latest_depth_raw = i2[:, :, :3]

def get_distance(x, y, depth_frame):
    x = max(0, min(x, depth_frame.shape[1] - 1))
    y = max(0, min(y, depth_frame.shape[0] - 1))
    b, g, r = depth_frame[y, x]
    normalized_depth = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth * 1000.0

def main():
    global latest_rgb, latest_depth_raw
    actor_list = []
    
    # Initialize Pygame for manual keyboard control
    pygame.init()
    pygame.display.set_mode((300, 200))
    pygame.display.set_caption("Manual Steering Focus Window")
    
    # Initialize our Custom Controller
    adas_controller = ACC_FCW_Controller(target_distance=12.0)
    acc_active = False
    
    try:
        print("Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # Spawn Ego Vehicle
        vehicle_bp = bp_lib.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Spawn Target Vehicle (to test following)
        target_bp = bp_lib.filter('lincoln')[0]
        target_spawn = spawn_point
        target_spawn.location.x += 20.0 # Spawn 20 meters ahead
        target_vehicle = world.spawn_actor(target_bp, target_spawn)
        actor_list.append(target_vehicle)
        target_vehicle.set_autopilot(True) # Let the target drive away

        # Spawn Sensors
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(rgb_camera)
        rgb_camera.listen(lambda image: process_rgb_image(image))

        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '640')
        depth_bp.set_attribute('image_size_y', '480')
        depth_camera = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actor_list.append(depth_camera)
        depth_camera.listen(lambda image: process_depth_image(image))

        print("System Ready.")
        print("Use W/A/S/D in the Pygame window to drive.")
        print("Press 'C' to toggle Adaptive Cruise Control (ACC).")
        
        last_time = time.time()

        while True:
            # 1. Handle Manual Inputs via Pygame
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            steer = 0.0
            man_throttle = 0.0
            man_brake = 0.0
            
            if keys[pygame.K_a]: steer = -0.5
            if keys[pygame.K_d]: steer = 0.5
            if keys[pygame.K_w]: man_throttle = 0.6
            if keys[pygame.K_s]: man_brake = 1.0
            
            # Toggle ACC
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    acc_active = not acc_active
                    print(f"ACC State: {'ON' if acc_active else 'OFF'}")

            if latest_rgb is not None and latest_depth_raw is not None:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                frame = latest_rgb.copy()
                depth_frame = latest_depth_raw.copy()

                # 2. Perception (YOLO)
                results = model(frame, verbose=False, classes=[2, 5, 7])
                car_detected = False
                
                # Default control variables (assume manual unless ADAS overrides)
                final_throttle = man_throttle
                final_brake = man_brake
                warning_text = "SAFE"
                warning_color = (0, 255, 0)

                for r in results:
                    boxes = r.boxes
                    if len(boxes) > 0:
                        car_detected = True
                        box = boxes[0] # Just track the first detected car for simplicity
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        center_x = int((x1 + x2) / 2)
                        
                        dist_meters = get_distance(center_x, y2, depth_frame)
                        
                        # 3. Decision & Control (ACC / FCW / AEB)
                        acc_throttle, acc_brake, warning_text, warning_color = adas_controller.process(dist_meters, dt)
                        
                        
                        
                        # 4. Feature Arbitration & Overrides
                        if "AEB" in warning_text:
                            # CRITICAL SAFETY: Override everything
                            final_throttle = 0.0
                            final_brake = acc_brake 
                        elif acc_active:
                            # CONVENIENCE: Use PID outputs, ignore manual throttle
                            final_throttle = acc_throttle
                            final_brake = max(acc_brake, man_brake) # Let driver brake harder if they want
                            
                        # HUD Drawing
                        cv2.rectangle(frame, (x1, y1), (x2, y2), warning_color, 2)
                        cv2.putText(frame, f"Dist: {dist_meters:.1f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warning_color, 2)
                        break # Only process the primary lead vehicle

                if not car_detected:
                    adas_controller.prev_distance = None # Reset TTC tracking if lane is clear

                # Display HUD Info
                acc_status = "ACC: ON" if acc_active else "ACC: OFF"
                cv2.putText(frame, acc_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warning_color, 2)
                
                cv2.imshow("ADAS Perception Output", frame)
                cv2.waitKey(1)
                
                # 5. Apply Vehicle Controls
                vehicle.apply_control(carla.VehicleControl(throttle=final_throttle, steer=steer, brake=final_brake))

            else:
                time.sleep(0.01)

    finally:
        print("\nCleaning up actors...")
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Done.")

if __name__ == '__main__':
    main()