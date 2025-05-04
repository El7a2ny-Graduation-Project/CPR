# pose_estimation.py
import cv2
import numpy as np
from ultralytics import YOLO
from keypoints import CocoKeypoints

class PoseEstimator:
    """Human pose estimation using YOLO"""
    
    def __init__(self, model_path="yolo11n-pose.pt", min_confidence=0.2):
        self.model = YOLO(model_path)
        self.min_confidence = min_confidence

    def detect_poses(self, frame):
        """Detect human poses in a frame"""
        try:
            results = self.model(frame, verbose=False, conf=self.min_confidence, show=False)
            if not results or len(results[0].keypoints.xy) == 0:
                return None
            return results[0]
        except Exception as e:
            print(f"Pose detection error: {e}")
            return None

    def get_keypoints(self, results, person_idx=0):
        """Extract keypoints for a detected person"""
        try:
            if not results or len(results.keypoints.xy) <= person_idx:
                return None
            return results.keypoints.xy[person_idx].cpu().numpy()
        except Exception as e:
            print(f"Keypoint extraction error: {e}")
            return None

    def draw_keypoints(self, frame, results):
        """Draw detected keypoints on frame"""
        try:
            return results.plot()
        except Exception as e:
            print(f"Keypoint drawing error: {e}")
            return frame