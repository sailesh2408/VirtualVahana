import numpy as np
import cv2
import torch
import sys
import os

# --- THE PYTORCH 2.6 SECURITY BYPASS ---
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

# ================= UFLD IMPORT =================
current_dir = os.path.dirname(os.path.abspath(__file__))
ufld_path = os.path.abspath(os.path.join(current_dir, '..', 'UFLD'))
if ufld_path not in sys.path:
    sys.path.insert(0, ufld_path) 

try:
    from model.model_culane import parsingNet
    UFLD_AVAILABLE = True
except ImportError as e:
    print(f"CRITICAL UFLD ERROR: Could not import parsingNet. {e}")
    UFLD_AVAILABLE = False

class PerceptionManager:
    def __init__(self, yolo_weights='yolov8s.pt', ufld_weights='models/ufld_resnet18_culane.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.object_model = YOLO(yolo_weights)
        self.object_model.to(self.device)
        self.lane_model = None

        if UFLD_AVAILABLE and os.path.exists(ufld_weights):
            try:
                self.lane_model = parsingNet(
                    pretrained=False, backbone='18', num_grid_row=200, num_cls_row=72,
                    num_grid_col=100, num_cls_col=81, num_lane_on_row=4,
                    num_lane_on_col=4, use_aux=False, input_height=320, 
                    input_width=1600, fc_norm=True 
                ).to(self.device)
                
                state_dict = torch.load(ufld_weights, map_location=self.device)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                    
                compatible_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '')
                    if "cls.0" in new_key:
                        new_key = new_key.replace("cls.0", "cls")
                    compatible_state_dict[new_key] = v
                        
                self.lane_model.load_state_dict(compatible_state_dict, strict=False)
                self.lane_model.eval()
                print("SUCCESS: UFLDv2 Zero-Latency Calibrated Tracking initialized!")
            except Exception as e:
                print("CRITICAL UFLD ERROR: Weights failed to load:", e)

    def get_distance(self, x, y, depth_frame):
        x, y = max(0, min(x, depth_frame.shape[1] - 1)), max(0, min(y, depth_frame.shape[0] - 1))
        b, g, r = depth_frame[y, x]
        return ((r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)) * 1000.0

    def process_frames(self, rgb_frame, depth_frame, dt):
        detections = []
        conf_list = []
        
        # 1. YOLOv8 Object Detection
        results = self.object_model(rgb_frame, verbose=False, classes=[0, 1, 2, 3, 5, 7], conf=0.3)
        y_horizon, y_bottom = 220, 480
        left_top, right_top = 280, 360     
        left_bottom, right_bottom = 100, 540 
        cv2.line(rgb_frame, (left_top, y_horizon), (left_bottom, y_bottom), (255, 200, 0), 2)
        cv2.line(rgb_frame, (right_top, y_horizon), (right_bottom, y_bottom), (255, 200, 0), 2)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x = (x1 + x2) // 2
                if y2 > y_horizon:
                    prog = (y2 - y_horizon) / (y_bottom - y_horizon)
                    l_bound = left_top + (left_bottom - left_top) * prog
                    r_bound = right_top + (right_bottom - right_top) * prog
                    if l_bound < center_x < r_bound:
                        conf_list.append(float(box.conf[0]))
                        dist = self.get_distance(center_x, int(y2 * 0.95), depth_frame)
                        if dist < 80.0:
                            detections.append({'bbox': (x1, y1, x2, y2), 'distance': dist, 'confidence': float(box.conf[0])})
                            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

        # 2. UFLDv2 Zero-Latency Calibrated Tracking
        center_offset = 0.0
        lane_trust = 0.5 
        
        left_lane_pts = []
        right_lane_pts = []

        if self.lane_model is not None:
            crop_y = 160  
            road_img = rgb_frame[crop_y:, :, :]
            img = cv2.resize(road_img, (1600, 320))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
            img = (np.transpose(img, (2, 0, 1)) - mean) / std
            img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.lane_model(img)

            if isinstance(pred, dict) and 'loc_row' in pred:
                loc_row, exist_row = pred['loc_row'].cpu(), pred['exist_row'].cpu()
                max_indices, valid_row = loc_row.argmax(1), exist_row.argmax(1)
                
                screen_w = rgb_frame.shape[1]
                screen_h = rgb_frame.shape[0]
                center_x_screen = screen_w / 2.0
                
                # --- CALIBRATION CONTROLS ---
                fov_scale = 1.25  # Increase to fan lanes out wider, decrease to bring them inward
                x_offset = 0      # Tweak left/right
                
                for lane_id in range(4): 
                    # Lowered threshold to 2 to ensure we catch those faint outer lanes
                    if valid_row[0, :, lane_id].sum() > 2: 
                        for k in range(72):
                            if valid_row[0, k, lane_id]:
                                idx = max_indices[0, k, lane_id].item()
                                
                                soft_range = torch.arange(max(0, idx-3), min(199, idx+4))
                                prob = loc_row[0, soft_range, k, lane_id].softmax(0)
                                x_raw = (prob * soft_range.float()).sum() + 0.5
                                
                                # --- FOV X-SCALING FIX ---
                                px_raw = (x_raw / 199.0) * screen_w
                                px = int(center_x_screen + (px_raw - center_x_screen) * fov_scale) + x_offset
                                
                                # --- RESTORED HORIZON Y-MAPPING FIX ---
                                py = int(220 + (k / 71.0) * (screen_h - 220))
                                
                                # Ensure points stay on screen
                                if 0 <= px < screen_w and 0 <= py < screen_h:
                                    color = (0, 255, 0) if lane_id in [1, 2] else (255, 255, 0)
                                    cv2.circle(rgb_frame, (px, py), 4, color, -1)
                                    
                                    if lane_id == 1: left_lane_pts.append(px)
                                    elif lane_id == 2: right_lane_pts.append(px)

        # --- BULLETPROOF LDW CALCULATION ---
        # Look only at the bottom 5 dots closest to the car bumper
        b_left_x = left_lane_pts[-5:] if len(left_lane_pts) >= 5 else left_lane_pts
        b_right_x = right_lane_pts[-5:] if len(right_lane_pts) >= 5 else right_lane_pts

        if b_left_x and b_right_x:
            lane_center = (np.mean(b_left_x) + np.mean(b_right_x)) / 2.0
            center_offset = lane_center - (rgb_frame.shape[1] / 2.0)
            lane_trust = 0.98
        elif b_left_x:
            lane_center = np.mean(b_left_x) + 240 # Estimated half-lane width
            center_offset = lane_center - (rgb_frame.shape[1] / 2.0)
            lane_trust = 0.70
        elif b_right_x:
            lane_center = np.mean(b_right_x) - 240 # Estimated half-lane width
            center_offset = lane_center - (rgb_frame.shape[1] / 2.0)
            lane_trust = 0.70

        avg_yolo = np.mean(conf_list) if conf_list else 0.95
        sensor_health = 1.0 if dt < 0.1 else 0.6
        trust_score = ((avg_yolo * 0.5) + (lane_trust * 0.3) + (sensor_health * 0.2)) * 100

        return detections, {"center_offset": center_offset}, trust_score