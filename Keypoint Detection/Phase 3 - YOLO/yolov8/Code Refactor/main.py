# main.py
import cv2
import time
import tkinter as tk  # For screen size detection
from datetime import datetime

from pose_estimation import PoseEstimator
from role_classifier import RoleClassifier
from chest_initializer import ChestInitializer
from metrics_calculator import MetricsCalculator
from posture_analyzer import PostureAnalyzer

class CPRAnalyzer:
    """Main CPR analysis pipeline with execution tracing"""
    
    def __init__(self, video_path):
        print(f"\n[INIT] Initializing CPR Analyzer for: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("[ERROR] Failed to open video file")
            return

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Video properties: {self.frame_count} frames, "
              f"{self.cap.get(cv2.CAP_PROP_FPS):.1f} FPS")

        # Get screen dimensions
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        print(f"[DISPLAY] Detected screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.role_classifier = RoleClassifier()
        self.chest_initializer = ChestInitializer()
        self.metrics_calculator = MetricsCalculator(shoulder_width_cm=45)
        self.posture_analyzer = PostureAnalyzer()
        
        # Window configuration
        self.window_name = "CPR Analysis"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print("[INIT] System components initialized successfully")

    def run_analysis(self):
        """Main processing loop with execution tracing"""

        print("\n[PHASE] Starting initialization phase")
        start_time = time.time()
        
        if not self._initialize_system():
            return
        
        # Get the first frame to determine orientation
        ret, frame = self.cap.read()
        if ret:
            self._handle_chest_point_rotation(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        print(f"[INIT] Completed in {time.time()-start_time:.2f}s\n")
        print("[PHASE] Starting main processing loop")
        
        frame_counter = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("\n[INFO] End of video stream reached")
                break

            frame = self._handle_frame_rotation(frame)
            print(f"\n[FRAME {int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))}/{self.frame_count}] Processing")
            
            frame = self._process_frame(frame)
            
            self._display_frame(frame)

            if cv2.waitKey(1) == ord('q'):
                print("\n[USER] Analysis interrupted by user")
                break
                
        self._finalize_analysis()

    def _initialize_system(self):
        """Initialize system components with status tracking"""
        print("[INIT] Starting chest point initialization")
        init_start = time.time()
        
        if not self.chest_initializer.initialize(self.cap, self.pose_estimator, self.role_classifier):
            print("[ERROR] Chest initialization failed - check video content")
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"[INIT] Chest point initialized at {self.chest_initializer.chest_point}")
        print(f"[INIT] Sampled {len(self.chest_initializer.shoulder_samples)} valid shoulder positions")
        print(f"[INIT] Initialization completed in {time.time()-init_start:.2f}s")
        return True

    def _handle_chest_point_rotation(self, frame):
        """Handle chest point rotation once during initialization"""
        if frame.shape[1] > frame.shape[0]:  # Landscape orientation
            print(f"[ROTATE] Detected landscape video")
            
            if self.chest_initializer.chest_point:
                print(f"[ROTATE] Adjusting chest point coordinates")
                cx, cy = self.chest_initializer.chest_point
                # For 90° clockwise rotation:
                # New x = original height - original y
                # New y = original x
                new_cx = frame.shape[0] - cy
                new_cy = cx
                self.chest_initializer.chest_point = (int(new_cx), int(new_cy))
                print(f"[ROTATE] New chest point: {self.chest_initializer.chest_point}")

    def _handle_frame_rotation(self, frame):
        """Handle frame rotation without adjusting chest point"""
        if frame.shape[1] > frame.shape[0]:  # Width > Height
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def _process_frame(self, frame):
        """Process frame with execution tracing"""
        processing_start = time.time()
        
        # Pose estimation
        pose_results = self.pose_estimator.detect_poses(frame)
        if not pose_results:
            print("[PROCESS] No poses detected in frame")
            return frame
            
        print(f"[POSE] Detected {len(pose_results.boxes)} people")
        frame = self.pose_estimator.draw_keypoints(frame, pose_results)
        frame = self._analyze_rescuer(frame, pose_results)
        
        # Draw chest point
        frame = self.chest_initializer.draw_chest_marker(frame)
        print(f"[PROCESS] Frame processed in {(time.time()-processing_start)*1000:.1f}ms")
        return frame

    def _analyze_rescuer(self, frame, pose_results):
        """Analyze rescuer with detailed logging"""
        rescuer_id = self.role_classifier.find_rescuer(pose_results, frame.shape[:2])
        if rescuer_id is None:
            print("[RESCUER] No rescuer identified in frame")
            return frame
            
        print(f"[RESCUER] Identified at index {rescuer_id}")
        keypoints = self.pose_estimator.get_keypoints(pose_results, rescuer_id)
        
        # Draw rescuer bounding box
        frame = self._draw_rescuer_box(frame, pose_results, rescuer_id)
        
        # Posture analysis
        warnings = self.posture_analyzer.validate_posture(keypoints, self.chest_initializer.chest_point)
        frame = self.posture_analyzer.display_warnings(frame, warnings)

        if warnings:
            print(f"[WARNING] Posture issues: {', '.join(warnings)}")
        else:
            # Track midpoints
            frame = self.role_classifier.track_rescuer_midpoints(keypoints, frame)
            print(f"[TRACKING] Midpoint added at {self.role_classifier.midpoints[-1]}")
            self.role_classifier.update_shoulder_distance()
            
        return frame

    def _draw_rescuer_box(self, frame, results, rescuer_id):
        """Draw rescuer box with dimension logging"""
        boxes = results.boxes.xyxy.cpu().numpy()
        if rescuer_id < len(boxes):
            x1, y1, x2, y2 = boxes[rescuer_id].astype(int)
            w, h = x2-x1, y2-y1
            print(f"[BOX] Rescuer bounding box: {w}x{h} pixels")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, "RESCUER", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def _display_frame(self, frame):
        """Adaptive window sizing with aspect ratio preservation"""
        display_start = time.time()
        
        # Get original frame dimensions
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            return

        # Calculate maximum possible scale while maintaining aspect ratio
        scale_w = self.screen_width / w
        scale_h = self.screen_height / h
        scale = min(scale_w, scale_h) * 0.9  # 90% of max to leave some margin

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize and display
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center window
        pos_x = (self.screen_width - new_w) // 2
        pos_y = (self.screen_height - new_h) // 2
        cv2.moveWindow(self.window_name, pos_x, pos_y)
        
        cv2.imshow(self.window_name, resized)
        print(f"[DISPLAY] Resized to {new_w}x{new_h} (scale: {scale:.2f}) in {(time.time()-display_start)*1000:.1f}ms")

    def _finalize_analysis(self):
        """Final analysis with detailed reporting"""
        print("\n[PHASE] Starting final analysis")
        start_time = time.time()
        
        try:
            print("[METRICS] Smoothing midpoint data...")
            self.metrics_calculator.smooth_midpoints(self.role_classifier.midpoints)
            
            print("[METRICS] Detecting compression peaks...")
            self.metrics_calculator.detect_peaks()
            
            print("[METRICS] Calculating depth and rate...")
            depth, rate = self.metrics_calculator.calculate_metrics(
                self.role_classifier.shoulder_distances,
                self.cap.get(cv2.CAP_PROP_FPS))

            print(f"[RESULTS] Compression Depth: {depth:.1f} cm")
            print(f"[RESULTS] Compression Rate: {rate:.1f} cpm")
            
            print("[VISUAL] Generating motion curve plot...")
            self.metrics_calculator.plot_motion_curve()
            
        except Exception as e:
            print(f"[ERROR] Metric calculation failed: {str(e)}")
            
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"\n[ANALYSIS] Completed in {time.time()-start_time:.1f}s")
            print("[CLEANUP] Resources released")

if __name__ == "__main__":
    start_time = time.time()
    print("[START] CPR Analysis started")

    analyzer = CPRAnalyzer("video_5.mp4")
    analyzer.run_analysis()

    end_time = time.time()
    print(f"[END] Total execution time: {end_time - start_time:.2f}s")

