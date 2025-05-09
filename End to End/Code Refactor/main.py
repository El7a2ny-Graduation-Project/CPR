# main.py
import cv2
import time
import tkinter as tk  # For screen size detection
from datetime import datetime
import math
import sys
import numpy as np

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
        
        #& Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("[ERROR] Failed to open video file")
            return
        print("[INIT] Video file opened successfully")

        #& Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INIT] Video has {self.frame_count} frames at {self.fps:.2f} FPS")

        #& Get screen dimensions
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        print(f"[INIT] Detected screen resolution: {self.screen_width}x{self.screen_height}")
        
        #& Initialize system components
        self.pose_estimator = PoseEstimator(min_confidence=0.5)
        self.role_classifier = RoleClassifier()
        self.chest_initializer = ChestInitializer()
        self.metrics_calculator = MetricsCalculator(self.frame_count, shoulder_width_cm=45*0.65)
        
        # if avg_right > self.right_arm_angle_threshold: error
        # if avg_left < self.left_arm_angle_threshold: error

        self.posture_analyzer = PostureAnalyzer(right_arm_angle_threshold=220, left_arm_angle_threshold=160, wrist_distance_threshold=170, history_length_to_average=10)
        self.wrists_midpoint_analyzer = WristsMidpointAnalyzer()
        self.shoulders_analyzer = ShouldersAnalyzer()
        print("[INIT] System components initialized")
        
        #& Configure display window
        self.window_name = "CPR Analysis"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        print(f"[INIT] Window '{self.window_name}' created")

        #& Keep track of previous results for continuity
        self.prev_rescuer_processed_results = None
        self.prev_patient_processed_results = None
        self.prev_chest_params = None
        self.prev_midpoint = None
        self.prev_pose_results = None
        print("[INIT] Previous results initialized")

        #& Workaround for minor glitches
        self.consecutive_frames_with_posture_errors = 0
        self.max_consecutive_frames_with_posture_errors = 1

        #& Initialize variables for reporting warnings
        self.posture_errors_for_current_error_region = set()

        #& Frequent depth and rate calculations
        self.reporting_interval_in_seconds = 5
        self.reporting_interval_in_frames = int(self.fps * self.reporting_interval_in_seconds)
        print(f"[INIT] Reporting interval set to {self.reporting_interval_in_seconds} seconds ({self.reporting_interval_in_frames} frames)")
        
		#& Sampling
        self.sampling_interval_in_seconds = 0.2
        self.sampling_interval_in_frames = int(self.fps * self.sampling_interval_in_seconds)
        print(f"[INIT] Sampling interval set to {self.sampling_interval_in_seconds} seconds ({self.sampling_interval_in_frames} frames)")

    def run_analysis(self):
        try:
            print("\n[RUN ANALYSIS] Starting analysis")

            main_loop_start_time = time.time()

            #& Initialize Variables
            # Handling chunks
            first_time_to_have_a_proccessed_frame = True
            waiting_to_start_new_chunk = False
            # Hndling mini chunks
            mini_chunk_start_frame_index = None

            print("[RUN ANALYSIS] Starting main execution loop")
            #& Main execution loop
            while self.cap.isOpened():
                #& Always advance to next frame first
                ret = self.cap.grab()  # Faster than read() for skipping
                if not ret: break
                
                #& Get frame number
                # Retrieve the current position of the video frame being processed in the video capture object (self.cap).
                frame_counter = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                print(f"\n[FRAME {int(frame_counter)}/{self.frame_count}]")
                
				#& Check if you want to skip the frame
                if frame_counter % self.sampling_interval_in_frames != 0:
                    print(f"[SKIP FRAME] Skipping frame {int(frame_counter)}")
                    continue
                
                #& Retrieve and process frame
                _, frame = self.cap.retrieve()
                print(f"[RUN ANALYSIS] Retrieved frame")
                
                #& Rotate frame
                frame = self._handle_frame_rotation(frame)
                print(f"[RUN ANALYSIS] Rotated frame")

                #& Process frame
                # Processing a frame means updating the values for the current and previous detections both in the CPR Analyzer and the system components it includes.
                # The returned flags are:
                # - is_complete_chunk: True if a "Posture Error" occurs in the frame, False otherwise.
                # - accept_frame: True if the frame is accepted for further processing, False otherwise.
                # Not that a frame containing an error could be accepted if the number of consecutive frames with errors is less than the threshold.
                is_complete_chunk, accept_frame = self._process_frame(frame)
                print(f"[RUN ANALYSIS] Processed frame")
                
                #& Compose frame
                # This function is responsible for drawing the data detected during the processing of the frame on it.
                # The frame would not be displayed yet, just composed.
                processed_frame = self._compose_frame(frame, accept_frame)
                print(f"[RUN ANALYSIS] Composed frame")
                
                #& Set the chunk start frame index for the first chunk
                # Along the video when a failure  in any step of the processing occurs, the variables are populated with the previous results to keep the analysis going.
                # The problem occurs when the first few frames have a failure in the processing, and the variables are not populated yet.
                # This is why the first chunk starts from the first frame that has been processed successfully.
                if (processed_frame is not None) and first_time_to_have_a_proccessed_frame:
                    first_time_to_have_a_proccessed_frame = False
                    chunk_start_frame_index = frame_counter
                    mini_chunk_start_frame_index = frame_counter
                    print(f"[RUN ANALYSIS] First processed frame detected")

                #& Set the chunk start frame index for the all chunks after the first one & append the errors detected in the error region before this chunk if any
                # When a "Posture Error" occurs, a chunk is considered complete, and the program becomes ready to start a new chunk.
                # is_complete_chunk is returned as true for every frame that has a "Posture Error" in it, and false for every other frame.
                # This is why we need to wait for a frame with a false is_complete_chunk to start a new chunk.
                if (waiting_to_start_new_chunk) and (not is_complete_chunk):
                    waiting_to_start_new_chunk = False
                    chunk_start_frame_index = frame_counter
                    mini_chunk_start_frame_index = frame_counter
                    print(f"[RUN ANALYSIS] A new chunk is starting")

                    if len(self.posture_errors_for_current_error_region) > 0:
                        self.posture_analyzer.posture_errors_for_all_error_region.append(self.posture_errors_for_current_error_region.copy())
                        self.posture_errors_for_current_error_region.clear()
                        print(f"[RUN ANALYSIS] Reset posture errors for current error region")

                #& Process the current chunk or mini chunk if the conditions are met
                process_chunk = (is_complete_chunk or frame_counter == self.frame_count - 1) and (not waiting_to_start_new_chunk)
                process_mini_chunk = (frame_counter % self.reporting_interval_in_frames == 0) and (frame_counter != 0) and (mini_chunk_start_frame_index is not None) and (not is_complete_chunk)                     

                if process_chunk: 
                    print(f"[RUN ANALYSIS] Chunk completion detected")

                    # The difference here results from the fact a first middle chunk is terminated by a "Posture Error" which is a frame not included in the chunk.
                    # While the last chunk is terminated by the end of the video, which is a frame included in the chunk.
                    if is_complete_chunk:
                        chunk_end_frame_index = frame_counter - 1                
                    elif frame_counter == self.frame_count - 1:
                        chunk_end_frame_index = frame_counter
                    print(f"[RUN ANALYSIS] Determined the last frame of the chunk")

                    self._calculate_rate_and_depth_for_chunk(chunk_start_frame_index, chunk_end_frame_index)
                    print(f"[RUN ANALYSIS] Calculated metrics for the chunk")

                elif process_mini_chunk:
                    print(f"[RUN ANALYSIS] Mini chunk completion detected")

                    mini_chunk_end_frame_index = frame_counter
                    print(f"[RUN ANALYSIS] Determined the last frame of the mini chunk")

                    self._calculate_rate_and_depth_for_chunk(mini_chunk_start_frame_index, mini_chunk_end_frame_index)
                    print(f"[RUN ANALYSIS] Calculated metrics for the mini chunk")                       
                        
                if process_chunk or process_mini_chunk:
                    waiting_to_start_new_chunk = True

                    self.shoulders_analyzer.reset_shoulder_distances()
                    self.wrists_midpoint_analyzer.reset_midpoint_history()
                    print(f"[RUN ANALYSIS] Reset shoulder distances and midpoint history")

                #& Display frame
                if processed_frame is not None:
                    self._display_frame(processed_frame)
                else:
                    self._display_frame(frame)
                print(f"[RUN ANALYSIS] Displayed frame")
                                
                #& Check if the user wants to quit
                if cv2.waitKey(1) == ord('q'):
                    print("\n[RUN ANALYSIS] Analysis interrupted by user")
                    break

            main_loop_end_time = time.time()
            elapsed_time = main_loop_end_time - main_loop_start_time
            print(f"[TIMING] Main loop elapsed time: {elapsed_time:.2f}s")

        except Exception as e:
            print(f"[ERROR] An error occurred during main execution loop: {str(e)}")

        finally:
            report_and_plot_start_time = time.time()

            #& Cleanup, calculate averages, and plot full motion curve
            self.cap.release()
            cv2.destroyAllWindows()
            print("[RUN ANALYSIS] Released video capture and destroyed all windows")         

            self._calculate_rate_and_depth_for_all_chunks()
            print("[RUN ANALYSIS] Calculated weighted averages of the metrics across all chunks")

            self._plot_full_motion_curve_for_all_chunks()
            print("[RUN ANALYSIS] Plotted full motion curve")

            report_and_plot_end_time = time.time()
            report_and_plot_elapsed_time = report_and_plot_end_time - report_and_plot_start_time
            print(f"[TIMING] Report and plot elapsed time: {report_and_plot_elapsed_time:.2f}s")

    def _handle_frame_rotation(self, frame):
        #! Till now, the code has only been testes on portrait videos.
        if frame.shape[1] > frame.shape[0]:  # Width > Height
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def _process_frame(self, frame):
        #* Chunk Completion Check
        is_complete_chunk = False
        accept_frame = True
        
        #& Pose Estimation
        pose_results = self.pose_estimator.detect_poses(frame)

        #~ Handle Failed Detection or Update Previous Results
        if not pose_results:
            pose_results = self.prev_pose_results
            print("[POSE ESTIMATION] No pose detected, using previous results (could be None)")
        else:
            self.prev_pose_results = pose_results
        
        if not pose_results:
            print("[POSE ESTIMATION] Insufficient data for processing")
            return is_complete_chunk, accept_frame
        
        #& Rescuer and Patient Classification
        rescuer_processed_results, patient_processed_results = self.role_classifier.classify_roles(pose_results, self.prev_rescuer_processed_results, self.prev_patient_processed_results)

        #~ Handle Failed Classifications OR Update Previous Results
        if not rescuer_processed_results:
            rescuer_processed_results = self.prev_rescuer_processed_results
            print("[ROLE CLASSIFICATION] No rescuer detected, using previous results (could be None)")
        else:
            self.prev_rescuer_processed_results = rescuer_processed_results

        if not patient_processed_results:
            patient_processed_results = self.prev_patient_processed_results
            print("[ROLE CLASSIFICATION] No patient detected, using previous results (could be None)")
        else:
            self.prev_patient_processed_results = patient_processed_results
        
        if not rescuer_processed_results or not patient_processed_results:
            print("[ROLE CLASSIFICATION] Insufficient data for processing")
            return is_complete_chunk, accept_frame
             
        #^ Set Params in Role Classifier (to draw later)
        self.role_classifier.rescuer_processed_results = rescuer_processed_results
        self.role_classifier.patient_processed_results = patient_processed_results
        print(f"[ROLE CLASSIFICATION] Updated role classifier with new results")
    
        #& Chest Estimation
        chest_params = self.chest_initializer.estimate_chest_region(patient_processed_results["keypoints"], patient_processed_results["bounding_box"], frame_width=frame.shape[1], frame_height=frame.shape[0])
       
        #~ Handle Failed Estimation or Update Previous Results
        if not chest_params:
            chest_params = self.prev_chest_params
            print("[CHEST ESTIMATION] No chest region detected, using previous results (could be None)")
        else:
            self.prev_chest_params = chest_params

        if not chest_params:
            print("[CHEST ESTIMATION] Insufficient data for processing")
            return is_complete_chunk, accept_frame

        #^ Set Params in Chest Initializer (to draw later)
        self.chest_initializer.chest_params = chest_params
        self.chest_initializer.chest_params_history.append(self.chest_initializer.chest_params)

        #& Chest Expectation
        # The estimation up to the last frame
        expected_chest_params = self.chest_initializer.estimate_chest_region_weighted_avg(frame_width=frame.shape[1], frame_height=frame.shape[0])

        #~ First "window_size" detections can't avg
        if not expected_chest_params:
            self.chest_initializer.expected_chest_params = self.chest_initializer.chest_params
        else:
            self.chest_initializer.expected_chest_params = expected_chest_params

        #& Posture Analysis
        # The midpoind of the last frame
        warnings = self.posture_analyzer.validate_posture(rescuer_processed_results["keypoints"], self.prev_midpoint, self.chest_initializer.expected_chest_params)

        if warnings:
            print(f"[POSTURE ANALYSIS] Posture issues: {', '.join(warnings)}")
            self.consecutive_frames_with_posture_errors += 1
        else:
            print("[POSTURE ANALYSIS] No posture issues detected")
            self.consecutive_frames_with_posture_errors = 0
        
        accept_frame = self.consecutive_frames_with_posture_errors < self.max_consecutive_frames_with_posture_errors

        if accept_frame:
            warnings = []  # Reset warnings if the frame is accepted

        #^ Set Params in Posture Analyzer (to draw later)
        self.posture_analyzer.warnings = warnings  
        print(f"[POSTURE ANALYSIS] Updated posture analyzer with new results")

        #& Wrist Midpoint Detection
        midpoint = self.wrists_midpoint_analyzer.detect_wrists_midpoint(rescuer_processed_results["keypoints"])

        #~ Handle Failed Detection or Update Previous Results
        if not midpoint:
            midpoint = self.prev_midpoint
            print("[WRIST MIDPOINT DETECTION] No midpoint detected, using previous results (could be None)")
        else:
            self.prev_midpoint = midpoint
        
        if not midpoint:
            print("[WRIST MIDPOINT DETECTION] Insufficient data for processing")
            return is_complete_chunk, accept_frame

        if accept_frame:
            #^ Set Params in Role Classifier (to draw later)
            self.wrists_midpoint_analyzer.midpoint = midpoint
            self.wrists_midpoint_analyzer.midpoint_history.append(midpoint)
            print(f"[WRIST MIDPOINT DETECTION] Updated wrist midpoint analyzer with new results")

            #& Shoulder Distance Calculation
            shoulder_distance = self.shoulders_analyzer.calculate_shoulder_distance(rescuer_processed_results["keypoints"])
            if shoulder_distance is not None:
                self.shoulders_analyzer.shoulder_distance = shoulder_distance
                self.shoulders_analyzer.shoulder_distance_history.append(shoulder_distance)
            print(f"[SHOULDER DISTANCE] Updated shoulder distance analyzer with new results")
        else:
            #* Chunk Completion Check
            is_complete_chunk = True
            num_warnings_before = len(self.posture_errors_for_current_error_region)

            for warning in warnings:
                self.posture_errors_for_current_error_region.add(warning)
                
                num_warnings_after = len(self.posture_errors_for_current_error_region)
                
                if num_warnings_after > num_warnings_before:
                    print(f"[POSTURE ANALYSIS] Added warning to current error region: {warning}") 
    
        return is_complete_chunk, accept_frame
    
    def _compose_frame(self, frame, accept_frame):
        # Chest Region
        if frame is not None:
            frame = self.chest_initializer.draw_expected_chest_region(frame)
            print(f"[VISUALIZATION] Drawn chest region")

        # Warning Messages
        if frame is not None:
            frame = self.posture_analyzer.display_warnings(frame)
            print(f"[VISUALIZATION] Drawn warnings")
        
        if frame is not None:
            if accept_frame:
                # Midpoint
                frame = self.wrists_midpoint_analyzer.draw_midpoint(frame)   
                print(f"[VISUALIZATION] Drawn midpoint")  
        
        return frame

    def _display_frame(self, frame):        
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
        print(f"[DISPLAY FRAME] Resized to {new_w}x{new_h} (scale: {scale:.2f})")

    def _calculate_rate_and_depth_for_chunk(self, chunk_start_frame_index, chunk_end_frame_index):
        try:
            result = self.metrics_calculator.handle_chunk(np.array(self.wrists_midpoint_analyzer.midpoint_history), chunk_start_frame_index, chunk_end_frame_index, self.fps, np.array(self.shoulders_analyzer.shoulder_distance_history), self.sampling_interval_in_frames)

            if result == False:
                print("[ERROR] Failed to calculate metrics for the chunk")
                return
            
        except Exception as e:
            print(f"[ERROR] Metric calculation failed: {str(e)}")

    def _calculate_rate_and_depth_for_all_chunks(self):
        try:
            self.metrics_calculator.calculate_rate_and_depth_for_all_chunk()
            print(f"[METRICS] Weighted averages calculated")
        except Exception as e:
            print(f"[ERROR] Failed to calculate weighted averages: {str(e)}")
            
    def _plot_full_motion_curve_for_all_chunks(self):
        try:
            self.metrics_calculator.plot_motion_curve_for_all_chunks(self.posture_analyzer.posture_errors_for_all_error_region, self.sampling_interval_in_frames, self.reporting_interval_in_frames)
            print("[PLOT] Full motion curve plotted")
        except Exception as e:
            print(f"[ERROR] Failed to plot full motion curve: {str(e)}")
  
if __name__ == "__main__":
    print(f"\n[MAIN] CPR Analysis Started")

    video_path = r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\Dataset\Hopefully Ideal Angle\5.mp4"
    
    initialization_start_time = time.time()
    analyzer = CPRAnalyzer(video_path)
    initialization_end_time = time.time()
    initialization_elapsed_time = initialization_end_time - initialization_start_time
    
    print(f"[TIMING] Initialization time: {initialization_elapsed_time:.2f}s")
    
    analyzer.run_analysis()