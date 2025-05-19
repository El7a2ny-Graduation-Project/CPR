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

    def _check_hands_on_chest(self, keypoints, chest_params):
        """Check individual hand positions and return specific warnings"""

        # Get the wrist keypoints
        left_wrist = keypoints[CocoKeypoints.LEFT_WRIST.value]
        right_wrist = keypoints[CocoKeypoints.RIGHT_WRIST.value]

        warnings = []
        try:
            if chest_params is None:
                return ["Both hands not on chest!"]  # Fallback warning
            
            cx, cy, cw, ch = chest_params
            left_in = right_in = False

            # Check left hand
            if left_wrist is not None:
                left_in = (cx - cw/2 < left_wrist[0] < cx + cw/2) and \
                          (cy - ch/2 < left_wrist[1] < cy + ch/2)
            
            # Check right hand
            if right_wrist is not None:
                right_in = (cx - cw/2 < right_wrist[0] < cx + cw/2) and \
                            (cy - ch/2 < right_wrist[1] < cy + ch/2)

            # Determine warnings
            if not left_in and not right_in:
                warnings.append("Both hands not on chest!")
            else:
                if not left_in:
                    warnings.append("Left hand not on chest!")
                if not right_in:
                    warnings.append("Right hand not on chest!")

        except Exception as e:
            cpr_logger.error(f"Hands check error: {e}")
        
        return warnings


    def validate_posture(self, keypoints, chest_params):
        """Run all posture validations (returns aggregated warnings)"""
        warnings = []

        warnings += self._check_hands_on_chest(keypoints, chest_params)

        if ("Right hand not on chest!" not in warnings) and ("Both hands not on chest!" not in warnings):
            warnings += self._check_bended_right_arm(keypoints)
        
        if ("Left hand not on chest!" not in warnings) and ("Both hands not on chest!" not in warnings):
            warnings += self._check_bended_left_arm(keypoints)
       
        return warnings
