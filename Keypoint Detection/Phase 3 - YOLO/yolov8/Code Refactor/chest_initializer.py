# chest_initializer.py
import cv2
import numpy as np
from keypoints import CocoKeypoints

class ChestInitializer:
    """Handles chest point detection with separated debug visualization"""
    
    def __init__(self, num_init_frames=15, min_confidence=0.2, aspect_ratio_thresh=1.2):
        self.num_init_frames = num_init_frames
        self.min_confidence = min_confidence
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.chest_point = None
        self.shoulder_samples = []
        self.hip_samples = []
        self.debug_window = "Chest Detection Debug"

    def initialize(self, cap, pose_estimator, role_classifier):
        """Main initialization routine"""
        success = self._sample_initial_frames(cap, pose_estimator)
        self._calculate_chest_point(cap)
        cv2.destroyWindow(self.debug_window)
        return success

    def _sample_initial_frames(self, cap, pose_estimator):
        """Collect valid shoulder positions"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        valid_samples = 0

        for i in range(self.num_init_frames):
            ret, frame = cap.read()
            if not ret:
                break

            results = pose_estimator.detect_poses(frame) #!could remove this to prevent double detection
            if not results:
                continue

            debug_frame = results.plot()
            shoulders = self._detect_valid_shoulders(results, frame.shape)

            if shoulders:
                left_shoulder, right_shoulder, left_hip, right_hip = shoulders
                self.shoulder_samples.append((left_shoulder, right_shoulder))
                self.hip_samples.append((left_hip, right_hip))
                valid_samples += 1
                debug_frame = self._draw_debug_info(
                    debug_frame, 
                    left_shoulder, 
                    right_shoulder, 
                    frame.shape,
                    i+1
                )

            cv2.imshow(self.debug_window, debug_frame)
            if cv2.waitKey(400) == ord('q'):
                break

        return valid_samples > 0

    def _draw_debug_info(self, frame, left_shoulder, right_shoulder, frame_shape, frame_num):
        """Helper function for drawing debug information"""
        # Convert normalized coordinates to pixel values
        lx, ly = (left_shoulder * np.array([frame_shape[1], frame_shape[0]])).astype(int)
        rx, ry = (right_shoulder * np.array([frame_shape[1], frame_shape[0]])).astype(int)
        
        # Draw shoulder points
        cv2.circle(frame, (lx, ly), 5, (0, 0, 255), -1)
        cv2.circle(frame, (rx, ry), 5, (0, 0, 255), -1)
        
        # Add informational text
        cv2.putText(frame, f"CHEST pts - Frame {frame_num}", (lx, ly - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

    def _detect_valid_shoulders(self, results, frame_shape):
        """Validate detected shoulders using aspect ratio and confidence"""
        boxes = results.boxes.xywh.cpu().numpy()
        keypoints = results.keypoints.xyn.cpu().numpy()
        confs = results.keypoints.conf.cpu().numpy()

        for i, (box, kp) in enumerate(zip(boxes, keypoints)):
            x, y, w, h = box
            # Aspect ratio validation
            if h < w * self.aspect_ratio_thresh:
                continue

            # Confidence check
            if (confs[i][CocoKeypoints.LEFT_SHOULDER.value] < self.min_confidence or 
                confs[i][CocoKeypoints.RIGHT_SHOULDER.value] < self.min_confidence):
                continue

            return (kp[CocoKeypoints.LEFT_SHOULDER.value], 
                    kp[CocoKeypoints.RIGHT_SHOULDER.value],
                    kp[CocoKeypoints.LEFT_HIP.value],
                    kp[CocoKeypoints.RIGHT_HIP.value])

        return None

    def _calculate_chest_point(self, cap):
        """Calculate final chest point from valid samples"""
        if not self.shoulder_samples:
            return

        avg_left = np.median([s[0] for s in self.shoulder_samples], axis=0)
        avg_right = np.median([s[1] for s in self.shoulder_samples], axis=0)
        avg_left_hip = np.median([h[0] for h in self.hip_samples], axis=0)
        avg_right_hip = np.median([h[1] for h in self.hip_samples], axis=0)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        avg_left_px = avg_left * np.array([frame_width, frame_height])
        avg_right_px = avg_right * np.array([frame_width, frame_height])
        avg_left_hip_px = avg_left_hip * np.array([frame_width, frame_height])
        avg_right_hip_px = avg_right_hip * np.array([frame_width, frame_height])

        #midpoint = (avg_left_px + avg_right_px) / 2
        #shoulder_dist = np.linalg.norm(avg_left_px - avg_right_px)
        #downward_offset = 0.4 * shoulder_dist
        #self.chest_point = (int(midpoint[0]), int(midpoint[1] + downward_offset))

        if avg_left_px[1] < avg_right_px[1]:
            shoulder = np.array(avg_left_px)
            hip = np.array(avg_left_hip_px)
        else:
            shoulder = np.array(avg_right_px)
            hip = np.array(avg_right_hip_px)

        alpha = 0.412  # Relative chest position between shoulder and hip
        offset = 10  # move 10 pixels upward into the body
        self.chest_point = (
            int(shoulder[0] + alpha * (hip[0] - shoulder[0])),
            int(shoulder[1] + alpha * (hip[1] - shoulder[1])) - offset
        )

        # Visualize the chest point in the debug window for 2 seconds
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
        ret, frame = cap.read()
        if ret:
            frame_with_marker = self.draw_chest_marker(frame)
            cv2.imshow(self.debug_window, frame_with_marker)
            cv2.waitKey(2000)  # Wait for 2 seconds

    def draw_chest_marker(self, frame):
        """Draw chest point with visualization"""
        print(f"Chest point: {self.chest_point}")
        if self.chest_point:
            cv2.circle(frame, self.chest_point, 8, (0, 55, 120), -1)
            cv2.putText(frame, "Chest", (self.chest_point[0] + 5, self.chest_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame