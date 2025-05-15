import cv2
import numpy as np
from keypoints import CocoKeypoints
from logging_config import cpr_logger

class ChestInitializer:
    """Handles chest point detection with validations in estimation."""
    
    def __init__(self):
        self.chest_params = None
        self.chest_params_history = []
        self.expected_chest_params = None

    def estimate_chest_region(self, keypoints, bounding_box, frame_width, frame_height):
        """Estimate and validate chest region. Returns (cx, cy, cw, ch) or None."""
        try:
            # Unpack bounding box and calculate shoulder dimensions
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bounding_box
            bbox_delta_y = abs(bbox_y2 - bbox_y1)

            # Keypoints for shoulders
            left_shoulder = keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            right_shoulder = keypoints[CocoKeypoints.RIGHT_SHOULDER.value]

            # Midpoints calculation
            shoulder_center = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                        (left_shoulder[1] + right_shoulder[1]) / 2])
            
            # Calculate chest center by applying directional adjustment separately for x and y
            chest_center_from_shoulder_x = shoulder_center[0] - 0.3 * bbox_delta_y
            chest_center_from_shoulder_y = shoulder_center[1] - 0.1 * bbox_delta_y
            chest_center_from_shoulder = np.array([chest_center_from_shoulder_x, chest_center_from_shoulder_y])

            # Chest dimensions (85% of shoulder width, 40% height)
            chest_dx = bbox_delta_y * 0.8
            chest_dy = bbox_delta_y * 1.75

            # Calculate region coordinates
            x1 = chest_center_from_shoulder[0] - chest_dx / 2
            y1 = chest_center_from_shoulder[1] - chest_dy / 2
            x2 = chest_center_from_shoulder[0] + chest_dx / 2
            y2 = chest_center_from_shoulder[1] + chest_dy / 2

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
            cpr_logger.info(f"Chest estimation error: {e}")
            return None

    def estimate_chest_region_weighted_avg(self, frame_width, frame_height, window_size=60, min_samples=3):
        """
        Calculate stabilized chest parameters using weighted averaging with boundary checks.
        
        Args:
            self.chest_params_history: List of recent chest parameters [(cx, cy, cw, ch), ...]
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            window_size: Number of recent frames to consider (default: 5)
            min_samples: Minimum valid samples required (default: 3)
            
        Returns:
            Tuple of (cx, cy, cw, ch) as integers within frame boundaries,
            or None if insufficient data or invalid rectangle
        """
        if not self.chest_params_history:
            return None
        
        # Filter out None values and get recent frames
        valid_history = [h for h in self.chest_params_history[-window_size:] if h is not None]
        
        if len(valid_history) < min_samples:
            return None
        
        # Convert to numpy array (preserve floating-point precision)
        history_array = np.array(valid_history, dtype=np.float32)
        
        # Exponential weights (stronger emphasis on recent frames)
        weights = np.exp(np.linspace(1, 3, len(history_array)))
        weights /= weights.sum()
        
        try:
            # Calculate weighted average in float space
            cx, cy, cw, ch = np.average(history_array, axis=0, weights=weights)
            
            # Convert to rectangle coordinates (still floating point)
            x1 = max(0.0, cx - cw/2)
            y1 = max(0.0, cy - ch/2)
            x2 = min(float(frame_width - 1), cx + cw/2)
            y2 = min(float(frame_height - 1), cy + ch/2)
            
            # Only round to integers after all calculations
            x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])
            
            # Validate rectangle
            if x2 <= x1 or y2 <= y1:
                return None
                
            return (
                (x1 + x2) // 2,  # cx
                (y1 + y2) // 2,  # cy
                x2 - x1,         # cw
                y2 - y1          # ch
            )
            
        except Exception as e:
            cpr_logger.info(f"Chest region estimation error: {e}")
            return None
    
    def draw_expected_chest_region(self, frame):
        """Draws the chest region without validation."""
        if self.expected_chest_params is None:
            return frame

        cx, cy, cw, ch = self.expected_chest_params
        x1 = int(cx - cw / 2)
        y1 = int(cy - ch / 2)
        x2 = int(cx + cw / 2)
        y2 = int(cy + ch / 2)

        # Draw rectangle and center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 0), 5)

        cv2.circle(frame, (int(cx), int(cy)), 8, (128, 128, 0), -1)

        cv2.putText(frame, "EXPECTED CHEST", (x1, max(10, y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 0), 2)
        
        return frame
