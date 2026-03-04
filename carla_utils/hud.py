import cv2

class ADASHUD:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_telemetry(self, frame, speed, lka_active, acc_active):
        """Draws standard telemetry[cite: 209]."""
        cv2.putText(frame, f"Speed: {speed * 3.6:.1f} km/h", (10, 30), self.font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"LKA: {'ACTIVE' if lka_active else 'STANDBY'}", (10, 60), self.font, 0.6, (0, 255, 255) if lka_active else (150, 150, 150), 2)
        cv2.putText(frame, f"ACC: {'ACTIVE' if acc_active else 'STANDBY'}", (10, 90), self.font, 0.6, (0, 255, 255) if acc_active else (150, 150, 150), 2)

    def draw_warnings(self, frame, ldw_warn, ldw_color, fcw_warn, fcw_color, traffic_state, traffic_color):
        """Draws system alerts and feature states[cite: 94]."""
        cv2.putText(frame, f"LDW: {ldw_warn}", (10, 120), self.font, 0.6, ldw_color, 2)
        cv2.putText(frame, f"FCW: {fcw_warn}", (10, 150), self.font, 0.6, fcw_color, 2)
        cv2.putText(frame, f"TRAFFIC: {traffic_state}", (10, 180), self.font, 0.6, traffic_color, 2)

    def draw_confidence_meter(self, frame, trust_score):
        """Draws the Innovation Category Confidence Score[cite: 210]."""
        # Determine color based on trust
        if trust_score > 80: color = (0, 255, 0)
        elif trust_score > 50: color = (0, 165, 255)
        else: color = (0, 0, 255)

        text = f"ADAS Trust: {trust_score:.1f}%"
        
        # Get frame dimensions to place at top right
        h, w, _ = frame.shape
        cv2.putText(frame, text, (w - 250, 30), self.font, 0.7, color, 2)
        
        # Draw a visual trust bar
        bar_length = int((trust_score / 100.0) * 200)
        cv2.rectangle(frame, (w - 250, 45), (w - 50, 60), (100, 100, 100), -1) # Background
        cv2.rectangle(frame, (w - 250, 45), (w - 250 + bar_length, 60), color, -1) # Foreground

    def render_bounding_boxes(self, frame, detections, fcw_color):
        """Draws object tracking boxes and distances[cite: 210]."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            dist = det['distance']
            cv2.rectangle(frame, (x1, y1), (x2, y2), fcw_color, 2)
            cv2.putText(frame, f"{dist:.1f}m", (x1, y1 - 10), self.font, 0.5, fcw_color, 2)