# posture_analyzer.py
import math
import cv2
import numpy as np
from keypoints import CocoKeypoints

class PostureAnalyzer:
    """Posture analysis and visualization with comprehensive validation"""

    #! The warnings depend on the average readings from the last 10 frames
    #! This "10" should be adjusted according to the sampling rate of the video
    
    def __init__(self, right_arm_angle_threshold=210, left_arm_angle_threshold=150, wrist_distance_threshold=170, history_length_to_average=10):
        self.history_length_to_average = history_length_to_average
        
        self.right_arm_angles = []
        self.left_arm_angles = []
        self.wrist_distances = []
        
        self.right_arm_angle_threshold = right_arm_angle_threshold
        self.left_arm_angle_threshold = left_arm_angle_threshold
        self.wrist_distance_threshold = wrist_distance_threshold
        
        self.warning_positions = {
            'arm_angles': (50, 50),
            'one_handed': (50, 100),
            'hands_not_on_chest': (50, 150)
        }
        self.warnings = []

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - 
                         math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang
        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0

    def check_bended_arms(self, keypoints):
        """Check for proper arm positioning (returns warnings)"""
        warnings = []
        try:
            # Right arm analysis
            shoulder = keypoints[CocoKeypoints.RIGHT_SHOULDER.value]
            elbow = keypoints[CocoKeypoints.RIGHT_ELBOW.value]
            wrist = keypoints[CocoKeypoints.RIGHT_WRIST.value]
            right_angle = self.calculate_angle(wrist, elbow, shoulder)
            self.right_arm_angles.append(right_angle)
            
            # Left arm analysis
            shoulder = keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            elbow = keypoints[CocoKeypoints.LEFT_ELBOW.value]
            wrist = keypoints[CocoKeypoints.LEFT_WRIST.value]
            left_angle = self.calculate_angle(wrist, elbow, shoulder)
            self.left_arm_angles.append(left_angle)
            
            # Analyze angles with moving average
            avg_right = np.mean(self.right_arm_angles[-self.history_length_to_average:] if self.right_arm_angles else 0)
            avg_left = np.mean(self.left_arm_angles[-self.history_length_to_average:] if self.left_arm_angles else 0)
            
            if avg_right > self.right_arm_angle_threshold:
                warnings.append("Right arm bent")
            if avg_left < self.left_arm_angle_threshold:
                warnings.append("Left arm bent")
                
        except Exception as e:
            print(f"Arm angle check error: {e}")
        
        return warnings

    def check_one_handed_cpr(self, keypoints):
        """Check for one-handed CPR pattern (returns warning)"""
        warnings = []
        try:
            # Calculate wrist distance
            left_wrist = keypoints[CocoKeypoints.LEFT_WRIST.value]
            right_wrist = keypoints[CocoKeypoints.RIGHT_WRIST.value]
            
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            self.wrist_distances.append(wrist_distance)
            
            # Analyze distance with moving average
            avg_distance = np.mean(self.wrist_distances[-self.history_length_to_average:] if self.wrist_distances else 0)
            
            if avg_distance > self.wrist_distance_threshold:
                warnings.append("One-handed CPR detected!")
                
        except Exception as e:
            print(f"One-handed CPR check error: {e}")
        
        return warnings

    def check_hands_on_chest(self, wrists_midpoint, chest_params):  # (cx, cy, cw, ch)
        """Check if hands are on the chest (returns warning)"""
        warnings = []
        try:
            # Check if hands are on the chest
            if wrists_midpoint is None or chest_params is None:
                return ["Hands not on chest"]
            
            # Unpack parameters
            wrist_x, wrist_y = wrists_midpoint
            cx, cy, cw, ch = chest_params
            
            if not ((cx - cw/2 < wrist_x < cx + cw/2) and (cy - ch/2 < wrist_y < cy + ch/2)):
                warnings.append("Hands not on chest")
                
        except Exception as e:
            print(f"Hands on chest check error: {e}")
        
        return warnings

    def validate_posture(self, keypoints):
        """Run all posture validations (returns aggregated warnings)"""
        warnings = []
        warnings += self.check_bended_arms(keypoints)
        warnings += self.check_one_handed_cpr(keypoints)
        return warnings

    def display_warnings(self, frame):
        """Display posture warnings on the frame"""

        if len(self.warnings) != 0:
            try:
                # Display arm angle warnings
                for i, warn in enumerate(w for w in self.warnings if w in ["Right arm bent", "Left arm bent"]):
                    pos = (self.warning_positions['arm_angles'][0],
                        self.warning_positions['arm_angles'][1] + i*30)
                    cv2.putText(frame, warn, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display one-handed CPR warning
                if any("One-handed" in w for w in self.warnings):
                    cv2.putText(frame, "One-handed CPR detected!", 
                            self.warning_positions['one_handed'],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                # Display hands not on chest warning
                if any("Hands not on chest" in w for w in self.warnings):
                    cv2.putText(frame, "Hands not on chest!", 
                            self.warning_positions['hands_not_on_chest'],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
            except Exception as e:
                print(f"Warning display error: {e}")
            
        return frame