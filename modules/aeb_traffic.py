from ultralytics import YOLO

class AEB_Traffic_Controller:
    # UPGRADED to yolov8s.pt
    def __init__(self, weights_path='yolov8s.pt'):
        print("Loading Optimized Traffic Model on GPU...")
        self.model = YOLO(weights_path)
        self.target_classes = [9, 11]

    def process(self, rgb_frame):
        traffic_state, traffic_color, red_light_override = "CLEAR", (0, 255, 0), False
        
        # OPTIMIZED: conf=0.25 to catch tiny CARLA traffic lights
        results = self.model(rgb_frame, verbose=False, classes=self.target_classes, conf=0.25)

        for r in results:
            for box in r.boxes:
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                
                if cls_id == 9: # Traffic Light
                    traffic_state, traffic_color = "TRAFFIC LIGHT", (0, 255, 255)
                    # We flag it yellow for caution, but don't hard brake on all lights yet
                elif cls_id == 11 and conf > 0.4: # Stop Sign
                    traffic_state, traffic_color, red_light_override = "STOP SIGN", (0, 0, 255), True

        return traffic_state, traffic_color, None, red_light_override