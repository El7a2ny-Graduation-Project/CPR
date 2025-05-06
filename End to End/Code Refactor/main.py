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
from wrists_midpoint_analyzer import WristsMidpointAnalyzer
from shoulders_analyzer import ShouldersAnalyzer

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
        self.pose_estimator = PoseEstimator(min_confidence=0.5)
        self.role_classifier = RoleClassifier()
        self.chest_initializer = ChestInitializer()
        self.metrics_calculator = MetricsCalculator(self.frame_count, shoulder_width_cm=45)
        self.posture_analyzer = PostureAnalyzer(right_arm_angle_threshold=250, left_arm_angle_threshold=150)
        self.wrists_midpoint_analyzer = WristsMidpointAnalyzer()
        self.shoulders_analyzer = ShouldersAnalyzer()
        
        # Window configuration
        self.window_name = "CPR Analysis"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print("[INIT] System components initialized successfully")

        # For filtering
        self.prev_rescuer_processed_results = None
        self.prev_patient_processed_results = None
        self.prev_chest_params = None
        self.prev_midpoint = None

    def run_analysis(self):
        """Main processing loop with execution tracing"""

        try:
            print("[PHASE] Starting main processing loop")

            
            first_time_to_have_a_proccessed_frame = True
            waiting_to_start_new_chunk = False

            while self.cap.isOpened():
                frame_counter = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                #& Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("\n[INFO] End of video stream reached")
                    break
                
                #& Rotate frame
                frame = self._handle_frame_rotation(frame)
                print(f"\n[FRAME {int(frame_counter)}/{self.frame_count}] Processing")
                
                #& Process frame
                processed_frame, is_complete_chunk = self._process_frame(frame)
                
                #& Display frame
                print(f"waiting_to_start_new_chunk: {waiting_to_start_new_chunk}, is_complete_chunk: {is_complete_chunk}")

                if processed_frame is not None:
                    self._display_frame(processed_frame)
                    if first_time_to_have_a_proccessed_frame:
                        print("[INFO] First processed frame displayed")
                        first_time_to_have_a_proccessed_frame = False
                        chunk_start_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                else:
                    self._display_frame(frame)
                
                if waiting_to_start_new_chunk and not is_complete_chunk:
                    print("[INFO] Waiting to start new chunk")
                    waiting_to_start_new_chunk = False
                    chunk_start_frame_index = frame_counter
                    print(f"[INFO] New chunk started at frame {chunk_start_frame_index}")

                #& Check for chunk completion and if so calculate metrics
                if (is_complete_chunk or frame_counter == self.frame_count - 1) and not waiting_to_start_new_chunk:                  
                    if is_complete_chunk:
                        chunk_end_frame_index = frame_counter - 1                
                    elif frame_counter == self.frame_count - 1:
                        chunk_end_frame_index = frame_counter

                    print(f"[CHUNK] Complete chunk detected from frame {chunk_start_frame_index} to {chunk_end_frame_index}")
                    
                    self._calculate_rate_and_depth(chunk_start_frame_index, chunk_end_frame_index)
                    # self._display_motion_plot(chunk_start_frame_index, chunk_end_frame_index)
                    #                 
                    waiting_to_start_new_chunk = True
                
                    self.shoulders_analyzer.reset_shoulder_distances()
                    self.wrists_midpoint_analyzer.reset_midpoint_history()
                                
                #& Check for user interrupt (close window)
                if cv2.waitKey(1) == ord('q'):
                    print("\n[USER] Analysis interrupted by user")
                    break
        except Exception as e:
            print(f"[ERROR] An error occurred during main execution loop: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[CLEANUP] Resources released")            

            print("[REPORT] Calculating final metrics")
            self._calculate_weighted_avg_rate_and_depth()
            self._plot_full_motion_curve()
            print("[REPORT] Final metrics calculated")

    def _handle_frame_rotation(self, frame):
        """Handle frame rotation without adjusting chest point"""
        if frame.shape[1] > frame.shape[0]:  # Width > Height
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def _process_frame(self, frame):
        """Process frame with execution tracing"""
        processing_start = time.time()

        #* Chunk Completion Check
        is_complete_chunk = False
        
        #& Pose Estimation
        print(f"[POSE ESTIMATION] Starting...")

        pose_results = self.pose_estimator.detect_poses(frame)
        
        if not pose_results:
            print("[POSE ESTIMATION] No poses detected in frame")
            return frame, is_complete_chunk
            
        print(f"[POSE ESTIMATION] Detected {len(pose_results.boxes)} people")

        print(f"[POSE ESTIMATION] Done")

        #& Rescuer and Patient Classification
        print(f"[ROLE CLASSIFICATION] Starting...")

        rescuer_processed_results, patient_processed_results = self.role_classifier.classify_roles(pose_results, self.prev_rescuer_processed_results, self.prev_patient_processed_results)

        #~ Handle Failed Classifications OR Update Previous Results
        if not rescuer_processed_results:
            rescuer_processed_results = self.prev_rescuer_processed_results
            print("[ROLE CLASSIFICATION] No rescuer detected, using previous results")
        else:
            self.prev_rescuer_processed_results = rescuer_processed_results

        if not patient_processed_results:
            patient_processed_results = self.prev_patient_processed_results
            print("[ROLE CLASSIFICATION] No patient detected, using previous results")
        else:
            self.prev_patient_processed_results = patient_processed_results
        
        if not rescuer_processed_results or not patient_processed_results:
            print("[ROLE CLASSIFICATION] Insufficient data for analysis")
            return None, is_complete_chunk
             
        #^ Set Params in Role Classifier (to draw later)
        self.role_classifier.rescuer_processed_results = rescuer_processed_results
        self.role_classifier.patient_processed_results = patient_processed_results

        print(f"[ROLE CLASSIFICATION] Done")
        
        #& Chest Estimation
        print(f"[CHEST ESTIMATION] Starting...")

        chest_params = self.chest_initializer.estimate_chest_region(patient_processed_results["keypoints"], patient_processed_results["bounding_box"], frame_width=frame.shape[1], frame_height=frame.shape[0])

        #~ Handle Failed Estimation or Update Previous Results
        if not chest_params:
            chest_params = self.prev_chest_params
            print("[PROCESS] No chest region detected, using previous results")
        else:
            self.prev_chest_params = chest_params

        if not chest_params:
            print("[PROCESS] Insufficient data for analysis")
            return None, is_complete_chunk

        #^ Set Params in Chest Initializer (to draw later)
        self.chest_initializer.chest_params = chest_params

        print(f"[CHEST ESTIMATION] Done")

        #& Posture Analysis (Phase 1: No Midpoint)
        print(f"[POSTURE ANALYSIS 1] Starting...")

        warnings = self.posture_analyzer.validate_posture(rescuer_processed_results["keypoints"])

        if warnings:
                print(f"[WARNING 1] Posture issues: {', '.join(warnings)}")

        print(f"[POSTURE ANALYSIS 1] Done")

        if len(warnings) == 0:
            print("[PROCESS] No warnings detected")

            #& Wrist Midpoint Detection
            print(f"[WRIST MIDPOINT ANALYSIS] Starting...")

            midpoint = self.wrists_midpoint_analyzer.detect_wrists_midpoint(rescuer_processed_results["keypoints"])

            #~ Handle Failed Detection or Update Previous Results
            if not midpoint:
                midpoint = self.prev_midpoint
                print("[PROCESS] No midpoint detected, using previous results")
            else:
                self.prev_midpoint = midpoint
            
            if not midpoint:
                print("[PROCESS] Insufficient data for analysis")
                return None, is_complete_chunk

            print(f"[WRIST MIDPOINT ANALYSIS] Done")

            #& Posture Analysis (Phase 2: With Midpoint)
            print(f"[POSTURE ANALYSIS 2] Starting...")
            warnings = self.posture_analyzer.check_hands_on_chest(midpoint, chest_params)    

            if warnings:
                print(f"[WARNING 2] Posture issues: {', '.join(warnings)}")

            print(f"[POSTURE ANALYSIS 2] Done")

        #^ Set Params in Posture Analyzer (to draw later)
        self.posture_analyzer.warnings = warnings  

        if len(warnings) == 0:
            #^ Set Params in Role Classifier (to draw later)
            self.wrists_midpoint_analyzer.midpoint = midpoint
            self.wrists_midpoint_analyzer.midpoint_history.append(midpoint)

            #& Shoulder Distance Calculation
            print(f"[SHOULDER DISTANCE ANALYSIS] Starting...")

            self.shoulders_analyzer.append_new_shoulder_distance(rescuer_processed_results["keypoints"])

            print(f"[SHOULDER DISTANCE ANALYSIS] Done")

        #& Bounding Boxes, Keypoints, Warnings, Wrists Midpoints, and Chest Region Drawing
        print(f"[VISUALIZATION] Starting...")

        # Bounding boxes and keypoints
        if frame is not None:
            frame = self.role_classifier.draw_rescuer_and_patient(frame)
            print(f"[VISUALIZATION] Drawn bounding boxes and keypoints")
      
        # Chest Region
        if frame is not None:
            frame = self.chest_initializer.draw_chest_region(frame)
            print(f"[VISUALIZATION] Drawn chest region")

        # Warning Messages
        if frame is not None:
            frame = self.posture_analyzer.display_warnings(frame)
            print(f"[VISUALIZATION] Drawn warnings")
        
        if frame is not None:
            if len(warnings) == 0:
                # Midpoint
                frame = self.wrists_midpoint_analyzer.draw_midpoint(frame)   
                print(f"[VISUALIZATION] Drawn midpoint")    

        print(f"[VISUALIZATION] Done") 

        if len(warnings) != 0:
            is_complete_chunk = True
    
        print(f"[PROCESS] Frame processed in {(time.time()-processing_start)*1000:.1f}ms")
        return frame, is_complete_chunk

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

    def _calculate_rate_and_depth(self, chunk_start_frame_index, chunk_end_frame_index):
        """Metric calcculation for a chunk of frames"""

        print("\n[PHASE] Starting metric calculation")
        start_time = time.time()
        
        try:
            print("[METRICS] Smoothing midpoint data...")
            self.metrics_calculator.smooth_midpoints(self.wrists_midpoint_analyzer.midpoint_history)
            
            print("[METRICS] Detecting compression peaks...")
            self.metrics_calculator.detect_peaks()
            
            print("[METRICS] Calculating depth and rate...")
            depth, rate = self.metrics_calculator.calculate_metrics(
                self.shoulders_analyzer.shoulder_distances,
                self.cap.get(cv2.CAP_PROP_FPS),
                chunk_start_frame_index, chunk_end_frame_index)

            if depth is None or rate is None:
                print("[ERROR] Depth and rate calculation failed; not enough data")
                return
            
            print(f"[RESULTS] Compression Depth: {depth:.1f} cm")
            print(f"[RESULTS] Compression Rate: {rate:.1f} cpm")
            
        except Exception as e:
            print(f"[ERROR] Metric calculation failed: {str(e)}")
        finally:
            print(f"\n[ANALYSIS] Completed in {time.time()-start_time:.1f}s")

    def _display_motion_plot(self, chunk_start_frame_index, chunk_end_frame_index):
        """Plotting motion curve with original and smoothed data"""
        self.metrics_calculator.plot_motion_curve(chunk_start_frame_index, chunk_end_frame_index)
        print("[PLOT] Motion curve plotted")

    def _calculate_weighted_avg_rate_and_depth(self):
        """Calculate weighted average rate and depth"""
        avg_rate, avg_depth = self.metrics_calculator.calculate_weighted_averages()
        
        return avg_rate, avg_depth
    
    def _plot_full_motion_curve(self):
        """Plot full motion curve"""
        self.metrics_calculator.plot_motion_curve_for_all_chunks()
        print("[PLOT] Full motion curve plotted")
    
if __name__ == "__main__":
    start_time = time.time()
    print("[START] CPR Analysis started")

    video_path = r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\Dataset\Tracking\video_3.mp4"

    analyzer = CPRAnalyzer(video_path)
    analyzer.run_analysis()

    end_time = time.time()
    print(f"[END] Total execution time: {end_time - start_time:.2f}s")

