# analyzers.py
import cv2
import numpy as np
from keypoints import CocoKeypoints

class ShouldersAnalyzer:
    """Analyzes shoulder distances and posture"""
    
    def __init__(self):
        self.shoulder_distance = None
        self.shoulder_distance_history = []

    def calculate_shoulder_distance(self, rescuer_keypoints):
        """Calculate and store shoulder distance"""
        if rescuer_keypoints is None:
            return
            
        try:
            left = rescuer_keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            right = rescuer_keypoints[CocoKeypoints.RIGHT_SHOULDER.value]
            
            distance = np.linalg.norm(np.array(left) - np.array(right))
            
            return distance
        except Exception as e:
            print(f"Shoulder distance error: {e}")
            return
    
    def reset_shoulder_distances(self):
        """Reset shoulder distances"""
        self.shoulder_distance_history = []
