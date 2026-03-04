import carla
import pygame
import cv2
import time
import sys
import numpy as np
import math
import traceback

from carla_utils.carla_setup import setup_carla_world, cleanup
from carla_utils.hud import ADASHUD
from modules.perception import PerceptionManager
from modules.lka_ldw import LKA_Controller
from modules.acc_fcw import ACC_FCW_Controller
from modules.aeb_traffic import AEB_Traffic_Controller

sensor_data = {'rgb': None, 'depth': None}

def rgb_callback(image):
    sensor_data['rgb'] = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]

def depth_callback(image):
    sensor_data['depth'] = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]

def main():
    pygame.init()
    pygame.display.set_mode((400, 300))
    pygame.display.set_caption("404botnotfound: Control")
    clock = pygame.time.Clock()
    actor_list = []

    try:
        world, ego_vehicle, rgb_cam, depth_cam, actor_list = setup_carla_world()
        rgb_cam.listen(rgb_callback)
        depth_cam.listen(depth_callback)

        perception = PerceptionManager()
        lka_system = LKA_Controller()
        acc_system = ACC_FCW_Controller(target_distance=15.0) # Increased target distance for safety
        traffic_system = AEB_Traffic_Controller()
        hud = ADASHUD()

        lka_active, acc_active = False, False
        last_time = time.time()

        while True:
            dt = time.time() - last_time
            last_time = time.time()

            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]: break

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_l: lka_active = not lka_active
                    if event.key == pygame.K_c: acc_active = not acc_active

            man_steer = 0.5 if keys[pygame.K_d] else (-0.5 if keys[pygame.K_a] else 0.0)
            man_throttle = 0.6 if keys[pygame.K_w] else 0.0
            man_brake = 1.0 if keys[pygame.K_s] else 0.0

            final_throttle, final_steer, final_brake = man_throttle, man_steer, man_brake
            speed = math.sqrt(sum(x**2 for x in [ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y, ego_vehicle.get_velocity().z]))

            if sensor_data['rgb'] is not None and sensor_data['depth'] is not None:
                frame, depth = sensor_data['rgb'].copy(), sensor_data['depth'].copy()
                
                detections, lane_data, adas_trust = perception.process_frames(frame, depth, dt)
                traffic_state, traffic_color, _, red_light_override = traffic_system.process(frame)

                # --- 1. LKA / Steering ---
                lka_steer, ldw_warn, ldw_color = lka_system.process(lane_data, speed)
                if lka_active and abs(man_steer) < 0.1:
                    final_steer = lka_steer

                # --- 2. ACC / AEB Processing (OPTIMIZED) ---
                # Default to cruise control (0.6 throttle) if no cars are detected!
                acc_throttle, acc_brake, fcw_warn, fcw_color, aeb_triggered = 0.6, 0.0, "CLEAR", (0,255,0), False
                
                if detections:
                    closest = min(detections, key=lambda x: x['distance'])
                    acc_throttle, acc_brake, fcw_warn, fcw_color, aeb_triggered = acc_system.process(closest['distance'], dt)

                # --- 3. Safety Arbitration ---
                if aeb_triggered or red_light_override:
                    final_throttle = 0.0 
                    final_brake = 1.0 # Slam brakes on red light or AEB
                elif acc_active:
                    final_throttle = acc_throttle
                    final_brake = max(acc_brake, man_brake)
                    if man_brake > 0.1: final_throttle = 0.0

                hud.draw_telemetry(frame, speed, lka_active, acc_active)
                hud.draw_warnings(frame, ldw_warn, ldw_color, fcw_warn, fcw_color, traffic_state, traffic_color)
                hud.draw_confidence_meter(frame, adas_trust)
                if detections: hud.render_bounding_boxes(frame, [closest], fcw_color)

                cv2.imshow("404botnotfound - Integrated ADAS", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

            ego_vehicle.apply_control(carla.VehicleControl(throttle=final_throttle, steer=final_steer, brake=final_brake))
            clock.tick(60)

    except Exception:
        traceback.print_exc()
    finally:
        cleanup(actor_list)
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == '__main__':
    main()