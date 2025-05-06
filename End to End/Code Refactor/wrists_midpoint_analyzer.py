import cv2
import numpy as np
from keypoints import CocoKeypoints

class WristsMidpointAnalyzer:
    """Analyzes and tracks wrist midpoints for rescuer"""
    
    def __init__(self, allowed_distance_between_wrists=170):
        self.allowed_distance_between_wrists = allowed_distance_between_wrists
        self.midpoint = None
        self.midpoint_history = []

    def detect_wrists_midpoint(self, rescuer_keypoints):
        """Calculate midpoint between wrists in pixel coordinates"""
        try:
            if rescuer_keypoints is None:
                return None
                
            # Get wrist coordinates
            lw = rescuer_keypoints[CocoKeypoints.LEFT_WRIST.value]
            rw = rescuer_keypoints[CocoKeypoints.RIGHT_WRIST.value]

            # If the distance between wrists is too large, return None
            distance = np.linalg.norm(np.array(lw) - np.array(rw))
            if distance > self.allowed_distance_between_wrists:
                return None
            
            # Calculate midpoint
            midpoint = (
                int((lw[0] + rw[0]) / 2),
                int((lw[1] + rw[1]) / 2)
            )
            
            return midpoint
            
        except Exception as e:
            print(f"Midpoint tracking error: {e}")
            return None

    def draw_midpoint(self, frame):
        """Visualize the midpoint on frame"""
        
        if self.midpoint is None:
            return frame
                    
        try:
            # Draw visualization
            cv2.circle(frame, self.midpoint, 8, (0, 255, 0), -1)
            cv2.putText(
                frame, "MIDPOINT", 
                (self.midpoint[0] + 5, self.midpoint[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            return frame
        except Exception as e:
            print(f"Midpoint drawing error: {e}")
            return frame
    
    def reset_midpoint_history(self):
        """Reset midpoint history"""
        self.midpoint_history = []