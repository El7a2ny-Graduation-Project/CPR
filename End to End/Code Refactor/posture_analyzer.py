# posture_analyzer.py
import math
import cv2
import numpy as np
from keypoints import CocoKeypoints
from logging_config import cpr_logger

class PostureAnalyzer:
    """Posture analysis and visualization with comprehensive validation"""
    
    def __init__(self, right_arm_angle_threshold, left_arm_angle_threshold, wrist_distance_threshold, history_length_to_average):
        self.history_length_to_average = history_length_to_average
        
        self.right_arm_angles = []
        self.left_arm_angles = []
        self.wrist_distances = []
        
        self.right_arm_angle_threshold = right_arm_angle_threshold
        self.left_arm_angle_threshold = left_arm_angle_threshold
        self.wrist_distance_threshold = wrist_distance_threshold
        
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - 
                         math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang
        except Exception as e:
            cpr_logger.error(f"Angle calculation error: {e}")
            return 0
    
    def _check_bended_right_arm(self, keypoints):
        """Check for right arm bending (returns warning)"""
        warnings = []
        try:
            shoulder = keypoints[CocoKeypoints.RIGHT_SHOULDER.value]
            elbow = keypoints[CocoKeypoints.RIGHT_ELBOW.value]
            wrist = keypoints[CocoKeypoints.RIGHT_WRIST.value]
            
            right_angle = self._calculate_angle(wrist, elbow, shoulder)
            
            self.right_arm_angles.append(right_angle)

            avg_right = np.mean(self.right_arm_angles[-self.history_length_to_average:] if self.right_arm_angles else 0)

            if avg_right > self.right_arm_angle_threshold:
                warnings.append("Right arm bent!")

            return warnings
                
        except Exception as e:
            cpr_logger.error(f"Right arm check error: {e}")
        
        return warnings
    
    def _check_bended_left_arm(self, keypoints):
        """Check for left arm bending (returns warning)"""
        warnings = []
        try:
            shoulder = keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            elbow = keypoints[CocoKeypoints.LEFT_ELBOW.value]
            wrist = keypoints[CocoKeypoints.LEFT_WRIST.value]
            
            left_angle = self._calculate_angle(wrist, elbow, shoulder)
            
            self.left_arm_angles.append(left_angle)

            avg_left = np.mean(self.left_arm_angles[-self.history_length_to_average:] if self.left_arm_angles else 0)

            if avg_left < self.left_arm_angle_threshold:
                warnings.append("Left arm bent!")

            return warnings
                
        except Exception as e:
            cpr_logger.error(f"Left arm check error: {e}")
        
        return warnings

    def _check_one_handed_cpr(self, keypoints):
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
            cpr_logger.error(f"One-handed CPR check error: {e}")
        
        return warnings

    def _check_hands_on_chest(self, wrists_midpoint, chest_params):  # (cx, cy, cw, ch)
        """Check if hands are on the chest (returns warning)"""
        warnings = []
        try:
            #! Revisit this condition
            # Check if hands are on the chest
            if wrists_midpoint is None or chest_params is None:
                return ["Hands not on chest!"]
            
            # Unpack parameters
            wrist_x, wrist_y = wrists_midpoint
            cx, cy, cw, ch = chest_params
            
            if not ((cx - cw/2 < wrist_x < cx + cw/2) and (cy - ch/2 < wrist_y < cy + ch/2)):
                warnings.append("Hands not on chest!")
                
        except Exception as e:
            cpr_logger.error(f"Hands on chest check error: {e}")
        
        return warnings

    def validate_posture(self, keypoints, wrists_midpoint, chest_params):
        """Run all posture validations (returns aggregated warnings)"""
        warnings = []
        warnings += self._check_bended_right_arm(keypoints)
        warnings += self._check_bended_left_arm(keypoints)
        warnings += self._check_one_handed_cpr(keypoints)
        warnings += self._check_hands_on_chest(wrists_midpoint, chest_params)
        return warnings
