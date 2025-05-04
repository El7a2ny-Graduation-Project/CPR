import cv2
import numpy as np
from keypoints import CocoKeypoints

class ChestInitializer:
    """Handles chest point detection with validations in estimation."""
    
    def __init__(self):
        self.chest_params = None

    def estimate_chest_region(self, keypoints, bounding_box, frame_width, frame_height):
        """Estimate and validate chest region. Returns (cx, cy, cw, ch) or None."""
        try:
            # Unpack bounding box and calculate shoulder dimensions
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bounding_box
            bbox_delta_y = bbox_y2 - bbox_y1

            # Keypoints for eyes and shoulders
            left_eye = keypoints[CocoKeypoints.LEFT_EYE.value]
            right_eye = keypoints[CocoKeypoints.RIGHT_EYE.value]
            left_shoulder = keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            right_shoulder = keypoints[CocoKeypoints.RIGHT_SHOULDER.value]

            # Midpoints calculation
            eye_center = np.array([(left_eye[0] + right_eye[0]) / 2,
                                    (left_eye[1] + right_eye[1]) / 2])
            shoulder_center = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                        (left_shoulder[1] + right_shoulder[1]) / 2])
            
            # Direction from eyes to shoulders
            eye_shoulder_vec = shoulder_center - eye_center
            distance = np.linalg.norm(eye_shoulder_vec)
            if distance > 0:
                direction = eye_shoulder_vec / distance
            else:
                direction = np.array([0, 0])

            # Calculate chest center by applying directional adjustment separately for x and y
            chest_center_x = shoulder_center[0] + 3 * direction[0] * distance
            chest_center_y = shoulder_center[1] - 1 * direction[1] * distance
            chest_center = np.array([chest_center_x, chest_center_y])

            # Chest dimensions (85% of shoulder width, 40% height)
            chest_dx = bbox_delta_y * 0.8
            chest_dy = bbox_delta_y * 0.6

            # Calculate region coordinates
            x1 = chest_center[0] - chest_dx / 2
            y1 = chest_center[1] - chest_dy / 2
            x2 = chest_center[0] + chest_dx / 2
            y2 = chest_center[1] + chest_dy / 2

            # Clamp to frame boundaries
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            # Check validity
            if x2 <= x1 or y2 <= y1:
                return None

            # Adjusted parameters
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cw = x2 - x1
            ch = y2 - y1

            return (cx, cy, cw, ch)

        except (IndexError, TypeError, ValueError) as e:
            print(f"Chest estimation error: {e}")
            return None

    def draw_chest_region(self, frame):
        """Draws the chest region without validation."""
        if self.chest_params is None:
            return frame

        cx, cy, cw, ch = self.chest_params
        x1 = int(cx - cw / 2)
        y1 = int(cy - ch / 2)
        x2 = int(cx + cw / 2)
        y2 = int(cy + ch / 2)

        # Draw rectangle and center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

        cv2.circle(frame, (int(cx), int(cy)), 8, (255, 0, 0), -1)

        cv2.putText(frame, "CHEST", (x1, max(10, y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame