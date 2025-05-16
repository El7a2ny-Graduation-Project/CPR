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
        
        self.warning_positions = {
            'right_arm_angle': (50, 50),
            'left_arm_angle': (50, 100),
            'one_handed': (50, 150),
            'hands_not_on_chest': (50, 200)
        }

        self.posture_errors_for_all_error_region = []

        self.posture_warnings_regions = []

        self.warnings = []

    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - 
                         math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang
        except Exception as e:
            cpr_logger.info(f"Angle calculation error: {e}")
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
                warnings.append("Right arm bent")

            return warnings
                
        except Exception as e:
            cpr_logger.info(f"Right arm check error: {e}")
        
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
                warnings.append("Left arm bent")

            return warnings
                
        except Exception as e:
            cpr_logger.info(f"Left arm check error: {e}")
        
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
            cpr_logger.info(f"One-handed CPR check error: {e}")
        
        return warnings

    def _check_hands_on_chest(self, wrists_midpoint, chest_params):  # (cx, cy, cw, ch)
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
            cpr_logger.info(f"Hands on chest check error: {e}")
        
        return warnings

    def validate_posture(self, keypoints, wrists_midpoint, chest_params):
        """Run all posture validations (returns aggregated warnings)"""
        warnings = []
        warnings += self._check_bended_right_arm(keypoints)
        warnings += self._check_bended_left_arm(keypoints)
        warnings += self._check_one_handed_cpr(keypoints)
        warnings += self._check_hands_on_chest(wrists_midpoint, chest_params)
        return warnings

    def display_warnings(self, frame):
        """Display posture warnings with colored background rectangles
        
        Args:
            frame: Input image frame to draw warnings on
            
        Returns:
            Frame with warnings and background rectangles drawn
        """
        if not self.warnings:
            return frame

        warning_config = {
            "Right arm bent": {
                "color": (52, 110, 235),  
                "position": self.warning_positions['right_arm_angle'],
                "text": "Right arm bent!"
            },
            "Left arm bent": {
                "color": (52, 110, 235),  
                "position": self.warning_positions['left_arm_angle'],
                "text": "Left arm bent!"
            },
            "One-handed": {
                "color": (27, 150, 70), 
                "position": self.warning_positions['one_handed'],
                "text": "One-handed CPR detected!"
            },
            "Hands not on chest": {
                "color": (161, 127, 18),  
                "position": self.warning_positions['hands_not_on_chest'],
                "text": "Hands not on chest!"
            }
        }

        try:
            for warning_text, config in warning_config.items():
                if any(warning_text in w for w in self.warnings):
                    self._draw_warning_banner(
                        frame=frame,
                        text=config['text'],
                        color=config['color'],
                        position=config['position']
                    )
            
        except Exception as e:
            cpr_logger.info(f"Warning display error: {e}")
        
        return frame

    def _draw_warning_banner(self, frame, text, color, position):
        """Helper function to draw a single warning banner"""
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        x, y = position
        # Calculate background rectangle coordinates
        x1 = x - 10
        y1 = y - text_height - 10
        x2 = x + text_width + 10
        y2 = y + 10
        
        # Draw background rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
        # Draw warning text
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
    
    def assign_posture_warnings_region_data(self, region_start_frame, region_end_frame, errors):
        """Capture error region data for later analysis
        
        Args:
            region_start_frame (int): Starting frame index of error region
            region_end_frame (int): Ending frame index of error region  
            errors (set): Posture errors detected in this region
        """
        self.posture_warnings_regions.append({
            'start_frame': region_start_frame,
            'end_frame': region_end_frame,
            'errors': errors.copy()
        })