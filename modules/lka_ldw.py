import math

class LKA_Controller:
    def __init__(self, k_gain=0.5, max_steer=0.8, ldw_threshold=150.0):
        """
        Initializes the Lane Keeping Assist and Lane Departure Warning system.
        
        :param k_gain: The control gain for the Stanley Controller.
        :param max_steer: Maximum allowed steering angle to prevent erratic jerks.
        :param ldw_threshold: Pixel offset threshold to trigger a Lane Departure Warning.
        """
        self.k = k_gain
        self.max_steer = max_steer
        self.ldw_threshold = ldw_threshold

    def process(self, lane_data, current_velocity):
        """
        Computes the steering angle and checks for lane departure.
        
        :param lane_data: Dictionary containing 'center_offset' from the perception module.
        :param current_velocity: Ego vehicle's current forward speed in m/s.
        :return: steering_command (float), ldw_warning (string), warning_color (tuple)
        """
        steer_cmd = 0.0
        warning_state = "LANE: SECURE"
        warning_color = (0, 255, 0) # Green

        # Extract the cross-track error (pixel offset from center)
        center_offset = lane_data.get('center_offset', 0.0)

        # 1. Lane Departure Warning (LDW) Logic
        # If the offset is too high, we are drifting out of the lane
        if abs(center_offset) > self.ldw_threshold:
            warning_state = "WARNING: LANE DEPARTURE!"
            warning_color = (0, 0, 255) # Red
        elif abs(center_offset) > (self.ldw_threshold * 0.7):
            warning_state = "CAUTION: DRIFTING"
            warning_color = (0, 165, 255) # Orange

        # 2. Lane Keeping Assist (LKA) Logic - Stanley Controller
        # If velocity is zero, avoid division by zero
        if current_velocity < 0.1:
            return 0.0, warning_state, warning_color

        # Approximate heading error (psi_e) to 0 for a simplified pixel-based approach
        # Cross-track error 'e' is our center_offset.
        # We normalize the offset by dividing by an arbitrary pixel width factor (e.g., 320 for half of 640 screen)
        normalized_error = center_offset / 320.0 

        # Calculate the Stanley steering correction
        # arctan(k * e / v)
        steer_correction = math.atan2((self.k * normalized_error), current_velocity)

        # Apply the correction (inverted because a positive pixel offset means we drifted right, so we steer left)
        steer_cmd = -steer_correction

        # Clamp the steering command to the maximum allowed limits
        steer_cmd = max(-self.max_steer, min(self.max_steer, steer_cmd))

        return steer_cmd, warning_state, warning_color

# Quick test execution if run standalone
if __name__ == "__main__":
    lka = LKA_Controller()
    dummy_lane_data = {"center_offset": 160.0} # Drifting right
    speed = 5.0 # m/s
    steer, warn, color = lka.process(dummy_lane_data, speed)
    print(f"Test -> Steer: {steer:.3f}, State: {warn}")