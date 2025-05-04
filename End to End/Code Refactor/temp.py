# chest_initializer.py
import cv2
import numpy as np
from keypoints import CocoKeypoints

class ChestInitializer:
    """Handles chest point detection with separated debug visualization"""
    
    def __init__(self):
        self.chest_params = None

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

            # Use bbox width as shoulder-to-shoulder distance
            shoulder_delta_y = bbox_delta_y
          
            # Chest width is 85% of shoulder width (since ribcage is slightly wider)
            chest_delta_y = shoulder_delta_y * 0.4

            return (*chest_center, chest_delta_y, chest_delta_y)

        except (IndexError, TypeError) as e:
            print(f"Chest estimation error: {e}")
            return None  # Skip if keypoints are missing

    def draw_chest_region(self, frame, color=(255, 0, 0), thickness=5):
        """Draws the estimated chest region on the frame with detailed boundary validation.
        
        Args:
            frame: Input image/frame (numpy array)
            color: Rectangle color (BGR tuple)
            thickness: Line thickness (int)
            
        Returns:
            The frame with chest region drawn (if parameters are valid and within bounds)
        """
        if self.chest_params is None:
            print("Chest drawing skipped: No chest parameters available")
            return frame

        try:
            # Get frame dimensions
            height, width = frame.shape[:2]
            cx, cy, cw, ch = self.chest_params
            
            # Initial coordinate calculation
            x1 = int(cx - cw/2)
            y1 = int(cy - ch/2)
            x2 = int(cx + cw/2)
            y2 = int(cy + ch/2)
            
            # Boundary validation with explicit messages
            validation_errors = []
            
            if x1 < 0:
                validation_errors.append(f"left edge (x1={x1}) extends beyond frame left boundary (0)")
            if y1 < 0:
                validation_errors.append(f"top edge (y1={y1}) extends beyond frame top boundary (0)")
            if x2 >= width:
                validation_errors.append(f"right edge (x2={x2}) extends beyond frame width ({width-1})")
            if y2 >= height:
                validation_errors.append(f"bottom edge (y2={y2}) extends beyond frame height ({height-1})")
            if x2 <= x1:
                validation_errors.append(f"right edge (x2={x2}) must be greater than left edge (x1={x1})")
            if y2 <= y1:
                validation_errors.append(f"bottom edge (y2={y2}) must be greater than top edge (y1={y1})")
                
            if validation_errors:
                error_msg = "Invalid chest region: " + ", ".join(validation_errors)
                print(error_msg)
                return frame
                
            # Clamp coordinates to frame boundaries
            safe_x1 = max(0, x1)
            safe_y1 = max(0, y1)
            safe_x2 = min(width - 1, x2)
            safe_y2 = min(height - 1, y2)
            
            # Draw rectangle
            cv2.rectangle(frame, (safe_x1, safe_y1), (safe_x2, safe_y2), color, thickness)
            
            # Calculate safe center point
            center_x = min(max(0, int(cx)), width - 1)
            center_y = min(max(0, int(cy)), height - 1)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Position text safely above rectangle (minimum 10px from top)
            text_y = max(10, safe_y1 - 5)
            cv2.putText(frame, "CHEST", (safe_x1, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except (ValueError, TypeError) as e:
            print(f"Chest drawing failed with calculation error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error during chest drawing: {str(e)}")
        
        return frame
    
    # chest_initializer.py