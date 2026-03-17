import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from carla_utils.hud import ADASHUD

def main():
    # 1. Create a colorful dummy "scene" instead of flat grey
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw Sky (Blue-ish) and Ground (Dark Green)
    cv2.rectangle(frame, (0, 0), (1280, 360), (235, 206, 135), -1) # Sky
    cv2.rectangle(frame, (0, 360), (1280, 720), (50, 100, 50), -1) # Ground
    
    # Draw a bright "Sun" exactly where the top-left telemetry goes
    cv2.circle(frame, (100, 80), 90, (0, 255, 255), -1) 
    
    # Draw a "Building" where the Trust score goes
    cv2.rectangle(frame, (1000, 10), (1200, 400), (150, 150, 200), -1)
    
    # Draw a "Road Line"
    cv2.line(frame, (640, 720), (640, 360), (255, 255, 255), 10)

    # 2. Initialize your HUD
    hud = ADASHUD()

    # 3. Feed it dummy data
    hud.draw_telemetry(frame, speed=22.5, lka_active=True, acc_active=False)
    
    hud.draw_warnings(
        frame, 
        ldw_warn="NONE", ldw_color=(0, 255, 0), 
        fcw_warn="VEHICLE AHEAD", fcw_color=(0, 165, 255), 
        traffic_state="RED LIGHT", traffic_color=(0, 0, 255)
    )
    
    hud.draw_confidence_meter(frame, trust_score=87.5)
    
    dummy_detections = [{'bbox': (550, 300, 750, 500), 'distance': 18.4}]
    hud.render_bounding_boxes(frame, dummy_detections, fcw_color=(0, 0, 255))

    # 4. Show the result
    cv2.imshow("Real Glass HUD Test", frame)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()