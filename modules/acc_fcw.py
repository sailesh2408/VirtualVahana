class ACC_FCW_Controller:
    def __init__(self, target_distance=15.0, kp=0.25, ki=0.002, kd=0.08):
        self.target_distance = target_distance
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_smoothed_dist = None
        self.smoothed_v_rel = 0.0

    def process(self, raw_distance, dt):
        throttle, brake = 0.0, 0.0
        warning_state, warning_color = "SAFE", (0, 255, 0)
        aeb_override = False 
        
        alpha_dist = 0.2 
        if self.prev_smoothed_dist is None:
            smoothed_dist = raw_distance
        else:
            smoothed_dist = (alpha_dist * raw_distance) + ((1.0 - alpha_dist) * self.prev_smoothed_dist)
        
        if self.prev_smoothed_dist is not None and dt > 0:
            raw_v_rel = (self.prev_smoothed_dist - smoothed_dist) / dt
            self.smoothed_v_rel = (0.2 * raw_v_rel) + (0.8 * self.smoothed_v_rel)
            
            # --- REALISTIC OPTIMIZATION: THE SAFETY BUBBLE ---
            # If the object is closer than 5 meters, lock brakes unless it is driving away from us fast
            if smoothed_dist < 5.0 and self.smoothed_v_rel > -0.5:
                self.prev_smoothed_dist = smoothed_dist
                return 0.0, 1.0, "CRITICAL: SAFETY BUBBLE LOCK", (0, 0, 255), True

            # Standard Time-To-Collision (TTC) Logic
            if self.smoothed_v_rel > 0.5: 
                ttc = smoothed_dist / self.smoothed_v_rel
                
                if ttc < 1.8: 
                    self.prev_smoothed_dist = smoothed_dist
                    return 0.0, 1.0, "CRITICAL: AEB FULL BRAKE", (0, 0, 255), True 
                elif ttc < 2.8:
                    self.prev_smoothed_dist = smoothed_dist
                    return 0.0, 0.5, "WARNING: AEB PARTIAL BRAKE", (0, 165, 255), True 
                elif ttc < 4.0:
                    warning_state, warning_color = "CAUTION: FCW ALERT", (0, 255, 255)
                
        self.prev_smoothed_dist = smoothed_dist
        
        # ACC Smooth Cruise Control
        error = smoothed_dist - self.target_distance
        if abs(error) < 1.5:  # Deadband to prevent jerky braking
            error = 0.0

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        pid_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        if pid_output > 0:
            throttle = min(pid_output, 0.55)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(pid_output), 0.35) 
            
        return throttle, brake, warning_state, warning_color, aeb_override