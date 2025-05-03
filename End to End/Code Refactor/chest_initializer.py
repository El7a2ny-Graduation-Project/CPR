# chest_initializer.py
import cv2
import numpy as np
from keypoints import CocoKeypoints

class ChestInitializer:
    """Handles chest point detection with separated debug visualization"""
    
    def __init__(self):
        self.chest_point = None

    def estimate_chest_region(self, keypoints, bounding_box):
        """Estimate chest region using eyes, shoulders, and bbox width.
        Returns (chest_center_x, chest_center_y, chest_delta_y, chest_delta_x)"""
        
        try:
            bbox_delta_y = bounding_box[3] - bounding_box[1]

            # Get available keypoints
            left_eye = keypoints[CocoKeypoints.LEFT_EYE.value]
            right_eye = keypoints[CocoKeypoints.RIGHT_EYE.value]
            left_shoulder = keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            right_shoulder = keypoints[CocoKeypoints.RIGHT_SHOULDER.value]

            # Calculate midpoints
            eye_center = np.array([(left_eye[0] + right_eye[0]) / 2,
                                (left_eye[1] + right_eye[1]) / 2])
            shoulder_center = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                    (left_shoulder[1] + right_shoulder[1]) / 2])
            
            # Direction from eyes to shoulders (torso orientation)
            eye_shoulder_direction = shoulder_center - eye_center
            eye_shoulder_distance = np.linalg.norm(eye_shoulder_direction)
            eye_shoulder_direction_normalized = eye_shoulder_direction / eye_shoulder_distance if eye_shoulder_distance > 0 else np.array([0, 0])

            # Apply separate X/Y adjustment ratios (0.8 for X, 0.6 for Y)
            chest_dx = 1.5 * eye_shoulder_direction_normalized[0] * eye_shoulder_distance
            chest_dy = -1 * eye_shoulder_direction_normalized[1] * eye_shoulder_distance
            chest_center = shoulder_center + np.array([chest_dx, chest_dy])
            self.chest_point = chest_center

            # Use bbox width as shoulder-to-shoulder distance
            shoulder_delta_y = bbox_delta_y
          
            # Chest width is 85% of shoulder width (since ribcage is slightly wider)
            chest_delta_y = shoulder_delta_y * 0.4

            return (*chest_center, chest_delta_y, chest_delta_y)

        except (IndexError, TypeError) as e:
            print(f"Chest estimation error: {e}")
            return None  # Skip if keypoints are missing

    def draw_chest_region(self, frame, chest_params, color=(255, 0, 0), thickness=5):
        """Draws the estimated chest region on the frame."""
        if chest_params is None:
            return frame

        try:
            cx, cy, cw, ch = chest_params
            
            # Calculate rectangle coordinates
            x1 = int(cx - cw/2)
            y1 = int(cy - ch/2)
            x2 = int(cx + cw/2)
            y2 = int(cy + ch/2)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point (red dot)
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            
            # Label with "Chest"
            cv2.putText(frame, "Chest", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except (ValueError, TypeError) as e:
            print(f"Drawing error: {e}")
        
        return frame