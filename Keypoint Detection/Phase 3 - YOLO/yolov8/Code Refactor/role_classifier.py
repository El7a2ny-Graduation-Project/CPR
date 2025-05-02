# role_classifier.py
import cv2
import numpy as np
from keypoints import CocoKeypoints

class RoleClassifier:
    """Role classification and tracking using image processing"""
    
    def __init__(self, proximity_thresh=0.3):
        self.proximity_thresh = proximity_thresh
        self.rescuer_id = None
        self.midpoints = []
        self.shoulder_distances = []
        self.rescuer_keypoints = None

    def find_rescuer(self, pose_results, frame_shape):
        """Identify rescuer based on proximity to horizontal objects"""
        try:
            boxes = pose_results.boxes.xywh.cpu().numpy()
            horizontal_objects = [b for b in boxes if b[2] > b[3]*1.5]
            
            if not horizontal_objects:
                return None

            people = []
            for i, box in enumerate(boxes):
                if box[3] > box[2]*1.2 and len(pose_results.keypoints) > i:
                    people.append((i, box[0], box[1]))

            min_distance = float('inf')
            for (i, px, py) in people:
                for (hx, hy, hw, hh) in horizontal_objects:
                    distance = np.sqrt(((px-hx)/frame_shape[1])**2 + ((py-hy)/frame_shape[0])**2)
                    if distance < min_distance and distance < self.proximity_thresh:
                        min_distance = distance
                        self.rescuer_id = i
            return self.rescuer_id
        except Exception as e:
            print(f"Rescuer finding error: {e}")
            return None
        
    def track_rescuer_midpoints(self, keypoints, frame):
        """Track and draw rescuer's wrist midpoints (using absolute pixel coordinates)"""
        try:
            # Store keypoints for shoulder distance calculation
            self.rescuer_keypoints = keypoints
            
            # Get wrist coordinates directly in pixels
            lw = keypoints[CocoKeypoints.LEFT_WRIST.value]
            rw = keypoints[CocoKeypoints.RIGHT_WRIST.value]
            
            # Calculate midpoint in pixel space
            midpoint = (
                int((lw[0] + rw[0]) / 2),
                int((lw[1] + rw[1]) / 2)
            )
            self.midpoints.append(midpoint)
            
            # Draw tracking
            cv2.circle(frame, midpoint, 8, (0,255,0), -1)
            cv2.putText(frame, "MIDPOINT", (midpoint[0]+5, midpoint[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            return frame
        except Exception as e:
            print(f"Midpoint tracking error: {e}")
            return frame

    def update_shoulder_distance(self):
        """Helper function to calculate and store shoulder distance"""
        if self.rescuer_keypoints is None:
            return

        try:
            left_shoulder = self.rescuer_keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            right_shoulder = self.rescuer_keypoints[CocoKeypoints.RIGHT_SHOULDER.value]
            
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            self.shoulder_distances.append(shoulder_dist)
            print(f"Shoulder distance: {shoulder_dist:.2f} pixels")
        except Exception as e:
            print(f"Shoulder distance calculation error: {e}")