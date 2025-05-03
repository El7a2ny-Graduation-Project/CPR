# posture_analyzer.py
import math
import cv2
import numpy as np
from keypoints import CocoKeypoints

class PostureAnalyzer:
    """Posture analysis and visualization with comprehensive validation"""
    
    def __init__(self, right_arm_angle_threshold=210, left_arm_angle_threshold=150, wrist_distance_threshold=170):
        self.right_arm_angles = []
        self.left_arm_angles = []
        self.wrist_distances = []
        self.right_arm_angle_threshold = right_arm_angle_threshold
        self.left_arm_angle_threshold = left_arm_angle_threshold
        self.wrist_distance_threshold = wrist_distance_threshold
        self.warning_positions = {
            'arm_angles': (50, 50),
            'one_handed': (50, 100)
        }

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
            avg_right = np.mean(self.right_arm_angles[-10:] if self.right_arm_angles else 0)
            avg_left = np.mean(self.left_arm_angles[-10:] if self.left_arm_angles else 0)
            
            if avg_right > self.right_arm_angle_threshold:
                warnings.append("Right arm bent")
            if avg_left < self.left_arm_angle_threshold:
                warnings.append("Left arm bent")
                
        except Exception as e:
            print(f"Arm angle check error: {e}")
        
        return warnings

    def check_missing_arms(self, keypoints):
        """Check for one-handed CPR pattern (returns warning)"""
        try:
            left_wrist = keypoints[CocoKeypoints.LEFT_WRIST.value]
            right_wrist = keypoints[CocoKeypoints.RIGHT_WRIST.value]
            
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            self.wrist_distances.append(wrist_distance)
            
            avg_distance = np.mean(self.wrist_distances[-10:] if self.wrist_distances else 0)
            
            if avg_distance > self.wrist_distance_threshold:
                return ["One-handed CPR detected!"]
        except Exception as e:
            print(f"One-handed check error: {e}")
        
        return []

    def validate_posture(self, keypoints, chest_point):
        """Run all posture validations (returns aggregated warnings)"""
        warnings = []
        warnings += self.check_bended_arms(keypoints)
        warnings += self.check_missing_arms(keypoints)
        return warnings

    def display_warnings(self, frame, warnings):
        """Display posture warnings on the frame"""
        try:
            # Display arm angle warnings
            for i, warn in enumerate(w for w in warnings if w in ["Right arm bent", "Left arm bent"]):
                pos = (self.warning_positions['arm_angles'][0],
                       self.warning_positions['arm_angles'][1] + i*30)
                cv2.putText(frame, warn, pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display one-handed CPR warning
            if any("One-handed" in w for w in warnings):
                cv2.putText(frame, "One-handed CPR detected!", 
                           self.warning_positions['one_handed'],
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        except Exception as e:
            print(f"Warning display error: {e}")
        
        return frame