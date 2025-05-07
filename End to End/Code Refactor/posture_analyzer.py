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

    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - 
                         math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang
        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0

    def _check_bended_arms(self, keypoints):
        """Check for proper arm positioning (returns warnings)"""
        warnings = []
        try:
            # Right arm analysis
            shoulder = keypoints[CocoKeypoints.RIGHT_SHOULDER.value]
            elbow = keypoints[CocoKeypoints.RIGHT_ELBOW.value]
            wrist = keypoints[CocoKeypoints.RIGHT_WRIST.value]
            right_angle = self._calculate_angle(wrist, elbow, shoulder)
            self.right_arm_angles.append(right_angle)
            
            # Left arm analysis
            shoulder = keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            elbow = keypoints[CocoKeypoints.LEFT_ELBOW.value]
            wrist = keypoints[CocoKeypoints.LEFT_WRIST.value]
            left_angle = self._calculate_angle(wrist, elbow, shoulder)
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
            print(f"One-handed CPR check error: {e}")
        
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
            print(f"Hands on chest check error: {e}")
        
        return warnings

    def validate_posture(self, keypoints, wrists_midpoint, chest_params):
        """Run all posture validations (returns aggregated warnings)"""
        warnings = []
        warnings += self._check_bended_arms(keypoints)
        warnings += self._check_one_handed_cpr(keypoints)
        warnings += self._check_hands_on_chest(wrists_midpoint, chest_params)
        return warnings

    def display_warnings(self, frame):
        """Display posture warnings on the frame with colored background rectangles
        
        Args:
            frame: Input image frame to draw warnings on
            
        Returns:
            Frame with warnings and background rectangles drawn
        """
        if not self.warnings:
            return frame

        try:
            # Define warning types and their properties
            warning_config = {
                "arm_angles": {
                    "texts": ["Right arm bent", "Left arm bent"],
                    "color": (0, 0, 255),  # Red
                    "position": self.warning_positions['arm_angles']
                },
                "one_handed": {
                    "text": "One-handed CPR detected!",
                    "color": (0, 255, 0),  # Green
                    "position": self.warning_positions['one_handed']
                },
                "hands_not_on_chest": {
                    "text": "Hands not on chest!",
                    "color": (255, 0, 0),  # Blue
                    "position": self.warning_positions['hands_not_on_chest']
                }
            }

            # Process arm angle warnings first (may have multiple)
            arm_warnings = [w for w in self.warnings if w in warning_config["arm_angles"]["texts"]]
            y_offset = 0
            for warn in arm_warnings:
                pos = (warning_config["arm_angles"]["position"][0],
                    warning_config["arm_angles"]["position"][1] + y_offset)
                
                # Calculate text size and rectangle
                (text_width, text_height), _ = cv2.getTextSize(
                    warn, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                # Draw rectangle
                cv2.rectangle(frame,
                            (pos[0] - 10, pos[1] - text_height - 10),
                            (pos[0] + text_width + 10, pos[1] + 10),
                            warning_config["arm_angles"]["color"], -1)
                
                # Draw text
                cv2.putText(frame, warn, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
                y_offset += 40

            # Process one-handed CPR warning
            if any("One-handed" in w for w in self.warnings):
                pos = warning_config["one_handed"]["position"]
                text = warning_config["one_handed"]["text"]
                
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                cv2.rectangle(frame,
                            (pos[0] - 10, pos[1] - text_height - 10),
                            (pos[0] + text_width + 10, pos[1] + 10),
                            warning_config["one_handed"]["color"], -1)
                
                cv2.putText(frame, text, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # Process hands not on chest warning
            if any("Hands not on chest" in w for w in self.warnings):
                pos = warning_config["hands_not_on_chest"]["position"]
                text = warning_config["hands_not_on_chest"]["text"]
                
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                cv2.rectangle(frame,
                            (pos[0] - 10, pos[1] - text_height - 10),
                            (pos[0] + text_width + 10, pos[1] + 10),
                            warning_config["hands_not_on_chest"]["color"], -1)
                
                cv2.putText(frame, text, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)

        except Exception as e:
            print(f"Warning display error: {e}")
        
        return frame