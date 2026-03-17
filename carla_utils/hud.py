import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class ADASHUD:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def _draw_modern_text(self, frame, text, x, y, font_size, bgr_color):
        """Draws TrueType fonts, with absolute pathing to prevent missing file errors."""
        # This forces Python to look in the main VirtualVahana folder for the font
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        font_path = os.path.join(base_dir, "SF-Pro.ttf")
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"WARNING: Could not find font at {font_path}. Using tiny default font!")
            font = ImageFont.load_default()

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        
        draw.text((x, y - font_size), text, font=font, fill=(bgr_color[0], bgr_color[1], bgr_color[2]))
        frame[:] = np.array(img_pil)

    def _draw_glass_panel(self, frame, x, y, w, h, alpha=0.45, blur_kernel=(61, 61)):
        """Creates a 'smoked liquid glass' effect behind HUD elements."""
        h_frame, w_frame = frame.shape[:2]
        
        y1, y2 = max(0, y), min(h_frame, y + h)
        x1, x2 = max(0, x), min(w_frame, x + w)
        
        if y1 >= y2 or x1 >= x2:
            return

        # 1. Extract background
        roi = frame[y1:y2, x1:x2]
        
        # 2. Apply a much heavier blur (61, 61) to smooth out sharp edges like the sun
        blurred_roi = cv2.GaussianBlur(roi, blur_kernel, 0)
        
        # 3. Create a DARK overlay instead of white
        dark_overlay = np.zeros_like(roi, dtype=np.uint8)
        
        # 4. Blend the heavy blur with the dark overlay
        # alpha=0.45 means it will be 45% black, 55% blurred background
        glass_effect = cv2.addWeighted(blurred_roi, 1.0 - alpha, dark_overlay, alpha, 0)
        
        # 5. Apply the effect back to the frame
        frame[y1:y2, x1:x2] = glass_effect
        
        # 6. Draw a slightly dimmer, subtle border to look like the edge of the glass
        cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

    def _draw_rounded_glass_panel(self, frame, x, y, w, h, radius=15, alpha=0.2, blur_kernel=(61, 61)):
        """Creates a perfectly smooth 'white frosted glass' pill effect, safe for screen edges."""
        h_frame, w_frame = frame.shape[:2]
        
        # 1. Create the FULL mask in memory first (ignoring screen edges)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(full_mask, (radius, radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(full_mask, (w - radius, radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(full_mask, (radius, h - radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(full_mask, (w - radius, h - radius), radius, 255, -1, cv2.LINE_AA)
        cv2.rectangle(full_mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(full_mask, (0, radius), (w, h - radius), 255, -1)

        # 2. Calculate valid screen coordinates (prevent crashing)
        x1, x2 = max(0, x), min(w_frame, x + w)
        y1, y2 = max(0, y), min(h_frame, y + h)

        if x1 >= x2 or y1 >= y2:
            return # Entirely off-screen, don't draw anything

        # 3. Calculate how much of the mask is actually visible
        mask_x1 = x1 - x
        mask_x2 = mask_x1 + (x2 - x1)
        mask_y1 = y1 - y
        mask_y2 = mask_y1 + (y2 - y1)

        # 4. Extract only the visible portions of the screen and our mask
        roi = frame[y1:y2, x1:x2]
        visible_mask = full_mask[mask_y1:mask_y2, mask_x1:mask_x2]
        
        # 5. Blend using the perfectly cropped mask
        mask_3d = cv2.cvtColor(visible_mask, cv2.COLOR_GRAY2BGR) / 255.0
        blurred_roi = cv2.GaussianBlur(roi, blur_kernel, 0)
        light_overlay = np.full_like(roi, 255, dtype=np.uint8) 
        
        glass_effect = cv2.addWeighted(blurred_roi, 1.0 - alpha, light_overlay, alpha, 0)
        blended_roi = (glass_effect * mask_3d + roi * (1.0 - mask_3d)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended_roi
        
        # 6. Draw the border using the cropped mask outline
        contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shifted_contours = [c + [x1, y1] for c in contours]
        cv2.drawContours(frame, shifted_contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_telemetry(self, frame, x, y, speed, lka_active, acc_active):
        """Draws standard telemetry with balanced typography."""
        self._draw_rounded_glass_panel(frame, x, y, 325, 135, radius=67)

        speed_kmh = speed * 3.6
        # Dropped Speed to 28 (was 34). Baseline at y + 45.
        self._draw_modern_text(frame, f"Speed: {speed_kmh:.1f} km/h", x + 55, y + 45, 28, (40, 40, 40))
        
        # Dropped sub-text to 22 (was 28). Spacing remains 35-40px.
        lka_col = (255, 149, 0) if lka_active else (100, 100, 100) 
        self._draw_modern_text(frame, f"LKA: {'ACTIVE' if lka_active else 'STANDBY'}", x + 55, y + 80, 22, lka_col)
        
        acc_col = (255, 149, 0) if acc_active else (100, 100, 100)
        cv2.putText(frame, "", (0,0), 0, 0, (0,0,0)) # Dummy line for spacing
        self._draw_modern_text(frame, f"ACC: {'ACTIVE' if acc_active else 'STANDBY'}", x + 55, y + 115, 22, acc_col)

    def draw_warnings(self, frame, x, y, ldw_warn, ldw_color, fcw_warn, fcw_color, traffic_state, traffic_color):
        """Draws system alerts with balanced typography."""
        self._draw_rounded_glass_panel(frame, x, y, 325, 135, radius=67)

        # Uniform size 22 for warnings looks much cleaner
        self._draw_modern_text(frame, f"LDW: {ldw_warn}", x + 55, y + 45, 22, (0, 150, 0))
        self._draw_modern_text(frame, f"FCW: {fcw_warn}", x + 55, y + 80, 22, (0, 100, 255))
        self._draw_modern_text(frame, f"TRAFFIC: {traffic_state}", x + 55, y + 115, 22, (0, 0, 200))
        
    def draw_confidence_meter(self, frame, x, y, trust_score):
        """Draws the Confidence Score with balanced typography."""
        self._draw_rounded_glass_panel(frame, x, y, 300, 90, radius=45)

        if trust_score > 80: color = (0, 150, 0) 
        elif trust_score > 50: color = (0, 100, 255) 
        else: color = (0, 0, 200) 

        text = f"ADAS Trust: {trust_score:.1f}%"
        # Dropped to size 24 (was 30)
        self._draw_modern_text(frame, text, x + 35, y + 40, 24, color)
        
        bar_length = int((trust_score / 100.0) * 220)
        cv2.rectangle(frame, (x + 35, y + 55), (x + 255, y + 65), (200, 200, 200), -1) 
        cv2.rectangle(frame, (x + 35, y + 55), (x + 35 + bar_length, y + 65), color, -1)

    def render_bounding_boxes(self, frame, detections, fcw_color):
        """Draws object tracking boxes and distances."""
        # No glass background needed here, just the clean target boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            dist = det['distance']
            cv2.rectangle(frame, (x1, y1), (x2, y2), fcw_color, 2)
            
            # Optional: Add a tiny glass label behind the distance text for readability
            self._draw_glass_panel(frame, x1, y1 - 25, 70, 20, alpha=0.4, blur_kernel=(11, 11))
            cv2.putText(frame, f"{dist:.1f}m", (x1 + 5, y1 - 10), self.font, 0.5, fcw_color, 2)