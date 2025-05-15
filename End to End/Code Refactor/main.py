# main.py
import cv2
import time
import tkinter as tk  # For screen size detection
from datetime import datetime
import math
import sys
import numpy as np
import os  # Added for path handling
import select
import socket
import json
from threading import Thread
from queue import Queue

from pose_estimation import PoseEstimator
from role_classifier import RoleClassifier
from chest_initializer import ChestInitializer
from metrics_calculator import MetricsCalculator
from posture_analyzer import PostureAnalyzer
from wrists_midpoint_analyzer import WristsMidpointAnalyzer
from shoulders_analyzer import ShouldersAnalyzer
from graph_plotter import GraphPlotter

from threaded_camera import ThreadedCamera
from analysis_socket_server import AnalysisSocketServer
from logging_config import cpr_logger

class CPRAnalyzer:
    """Main CPR analysis pipeline with execution tracing"""
    
    def __init__(self, source, requested_fps, output_video_path):
        cpr_logger.info(f"\n[INIT] Initializing CPR Analyzer")

        #& Add socket server
        self.socket_server = AnalysisSocketServer()
        self.socket_server.start_server()
        cpr_logger.info("[INIT] Socket server started")
        
        #& Open the camera source and get the FPS
        self.cap = ThreadedCamera(source, requested_fps)
        self.fps = self.cap.fps
        cpr_logger.info(f"[INIT] Camera FPS: {self.fps}")

        #& Generate output path with MP4 extension
        self.output_video_path = output_video_path
        self.video_writer = None
        self._writer_initialized = False
        cpr_logger.info(f"[INIT] Output path: {self.output_video_path}")
        
        #& Initialize system components
        self.pose_estimator = PoseEstimator(min_confidence=0.5)
        self.role_classifier = RoleClassifier()
        self.chest_initializer = ChestInitializer()
        self.metrics_calculator = MetricsCalculator(shoulder_width_cm=45*0.65)
        
        # Remeber the conditions if you need to adjust the thresholds
        # if avg_right > self.right_arm_angle_threshold: error
        # if avg_left < self.left_arm_angle_threshold: error

        self.posture_analyzer = PostureAnalyzer(right_arm_angle_threshold=220, left_arm_angle_threshold=160, wrist_distance_threshold=170, history_length_to_average=10)
        self.wrists_midpoint_analyzer = WristsMidpointAnalyzer()
        self.shoulders_analyzer = ShouldersAnalyzer()
        self.graph_plotter = GraphPlotter()
        cpr_logger.info("[INIT] System components initialized")

        #& Keep track of previous results for continuity
        self.prev_rescuer_processed_results = None
        self.prev_patient_processed_results = None
        self.prev_chest_params = None
        self.prev_midpoint = None
        self.prev_pose_results = None
        cpr_logger.info("[INIT] Previous results initialized")

        #& Fundamental timing parameters (in seconds)
        self.MIN_ERROR_DURATION = 1.0    # Require sustained errors for 1 second
        self.REPORTING_INTERVAL = 5.0    # Generate reports every 5 seconds
        self.SAMPLING_INTERVAL = 0.2     # Analyze every 0.2 seconds

        # Derived frame counts (calculated once during initialization)
        self.sampling_interval_frames = int(round(self.fps * self.SAMPLING_INTERVAL))
        self.error_threshold_frames = int(self.MIN_ERROR_DURATION / self.SAMPLING_INTERVAL)
        self.reporting_interval_frames = int(self.REPORTING_INTERVAL / self.SAMPLING_INTERVAL)

        # Enhanced validation using ratio checking
        ratio = self.REPORTING_INTERVAL / self.SAMPLING_INTERVAL
        assert math.isclose(ratio, round(ratio)), \
            f"Reporting interval ({self.REPORTING_INTERVAL}) must be an exact multiple of "\
            f"sampling interval ({self.SAMPLING_INTERVAL}). Actual ratio: {ratio:.2f}"

        assert self.MIN_ERROR_DURATION >= self.SAMPLING_INTERVAL, \
            f"Error detection window ({self.MIN_ERROR_DURATION}s) must be ≥ sampling interval ({self.SAMPLING_INTERVAL}s)"

        cpr_logger.info(f"\n[INIT] Temporal alignment:")
        cpr_logger.info(f" - {self.SAMPLING_INTERVAL}s sampling → {self.sampling_interval_frames} frames")
        cpr_logger.info(f" - {self.MIN_ERROR_DURATION}s error detection → {self.error_threshold_frames} samples")
        cpr_logger.info(f" - {self.REPORTING_INTERVAL}s reporting → {self.reporting_interval_frames} samples")

        #& Workaround for minor glitches
        # A frame is accepted as long as this counter does not exceed the error_threshold_frames set above.
        self.consecutive_frames_with_posture_errors = 0
        cpr_logger.info("[INIT] Consecutive frames with posture errors initialized")

        #& Initialize variables for reporting warnings
        # This variable is used when plotting the graph at the end after the whole video is processed.
        self.posture_errors_for_current_error_region = set()
        cpr_logger.info("[INIT] Posture errors for current error region initialized")

        #& Initialize variables used in reporting warnings frame by frame (not the graph)
        self.posture_warnings_from_current_frame = []
        cpr_logger.info("[INIT] Posture warnings from current frame initialized")

        self.rate_and_depth_warnings_from_the_last_report = []
        cpr_logger.info("[INIT] Rate and depth warnings from the last report initialized")

         # Warm up pose estimator with dummy data
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pose_estimator.detect_poses(dummy_frame)  # Force model loading
        cpr_logger.info("[INIT] Pose estimator initialized")

    def _initialize_video_writer(self, frame):
        """Initialize writer with safe fallback options"""
        height, width = frame.shape[:2]
        effective_fps = self.fps / max(1, self.sampling_interval_frames)
        
        # Try different codec/container combinations
        for codec, ext, fmt in [('avc1', 'mp4', 'mp4v'),  # H.264
                                ('MJPG', 'avi', 'avi'), 
                                ('XVID', 'avi', 'avi')]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(self.output_video_path, fourcc, effective_fps, (width, height))
            
            if writer.isOpened():
                self.video_writer = writer
                self._writer_initialized = True
                cpr_logger.info(f"[VIDEO WRITER] Initialized with {codec} codec")
                return
            else:
                writer.release()
        
        cpr_logger.info("[ERROR] Failed to initialize any video writer!")
        self._writer_initialized = False
        
    def run_analysis(self):
        try:
            cpr_logger.info("\n[RUN ANALYSIS] Starting analysis")
            # Wait for client connection before proceeding
            if not self.socket_server.wait_for_connection(timeout=120):
                cpr_logger.info("[ERROR] No client connected within 60 seconds")
                return

            # Start camera capture AFTER client connects
            self.cap.start_capture()
            cpr_logger.info("[RUN ANALYSIS] Camera capture started")

            main_loop_start_time = time.time()

            #& Initialize Variables
            # Handling chunks & mini chunks
            first_time_to_have_a_proccessed_frame = True
            waiting_to_start_new_chunk = False

            chunk_start_frame_index = 0
            chunk_end_frame_index = 0

            mini_chunk_start_frame_index = None
            mini_chunk_end_frame_index = 0

            frame_counter = -1

            cpr_logger.info(f"[RUN ANALYSIS] Initialized variables")

            cpr_logger.info("[RUN ANALYSIS] Starting main execution loop")
            #& Main execution loop
            while True:
                one_iteration_start_time = time.time()

                #& Get frame from camera queue
                frame = self.cap.read()

                # Check for termination sentinel
                if frame is None:
                    cpr_logger.info("Camera stream ended")
                    break

                frame_counter += 1
                cpr_logger.info(f"\n[FRAME {int(frame_counter)}]")
                
				#& Check if you want to skip the frame
                if frame_counter % self.sampling_interval_frames != 0:    
                    # Return the cashed warnings
                    formatted_warnings = self._format_warnings()
                    cpr_logger.info(f"[RUN ANALYSIS] Formatted warnings: {formatted_warnings}")

                    self.socket_server.warning_queue.put(formatted_warnings)
                    cpr_logger.info(f"[RUN ANALYSIS] Sent warnings to socket server")
                    
                    cpr_logger.info(f"[SKIP FRAME] Skipping frame {int(frame_counter)}") 
                    continue
                
                #& Rotate frame
                frame = self._handle_frame_rotation(frame)
                cpr_logger.info(f"[RUN ANALYSIS] Rotated frame")

                #& Process frame
                # Processing a frame means updating the values for the current and previous detections both in the CPR Analyzer and the system components it includes.
                # The returned flags are:
                # - is_complete_chunk: True if a "Posture Error" occurs in the frame, False otherwise.
                # - accept_frame: True if the frame is accepted for further processing, False otherwise.
                # Not that a frame containing an error could be accepted if the number of consecutive frames with errors is less than the threshold.
                is_complete_chunk, accept_frame, posture_warnings, has_appended_midpoint = self._process_frame(frame)
                self.posture_warnings_from_current_frame = posture_warnings
                cpr_logger.info(f"[RUN ANALYSIS] Processed frame")
                
                #& Compose frame
                # This function is responsible for drawing the data detected during the processing of the frame on it.
                # The frame would not be displayed yet, just composed.
                processed_frame = self._compose_frame(frame, accept_frame)
                cpr_logger.info(f"[RUN ANALYSIS] Composed frame")

                if processed_frame is not None:
                    frame = processed_frame

                #& Set the chunk start frame index for the first chunk
                # Along the video when a failure  in any step of the processing occurs, the variables are populated with the previous results to keep the analysis going.
                # The problem occurs when the first few frames have a failure in the processing, and the variables are not populated yet.
                # This is why the first chunk starts from the first frame that has been processed successfully.
                if (has_appended_midpoint is True) and first_time_to_have_a_proccessed_frame:
                    first_time_to_have_a_proccessed_frame = False
                    chunk_start_frame_index = frame_counter
                    mini_chunk_start_frame_index = frame_counter
                    cpr_logger.info(f"[RUN ANALYSIS] First processed frame detected")

                #& Set the chunk start frame index for the all chunks after the first one & append the errors detected in the error region before this chunk if any
                # When a "Posture Error" occurs, a chunk is considered complete, and the program becomes ready to start a new chunk.
                # is_complete_chunk is returned as true for every frame that has a "Posture Error" in it, and false for every other frame.
                # This is why we need to wait for a frame with a false is_complete_chunk to start a new chunk.
                if (waiting_to_start_new_chunk) and (not is_complete_chunk) and (has_appended_midpoint is True):
                    waiting_to_start_new_chunk = False
                    chunk_start_frame_index = frame_counter
                    mini_chunk_start_frame_index = frame_counter
                    cpr_logger.info(f"[RUN ANALYSIS] A new chunk is starting")

                    if len(self.posture_errors_for_current_error_region) > 0:
                        cpr_logger.info(f"[RUN ANALYSIS] Posture errors detected in the error region before this chunk")

                        error_region_start = max(chunk_end_frame_index, mini_chunk_end_frame_index) + 1
                        error_region_end = frame_counter - 1
                        
                        self.posture_analyzer.assign_error_region_data(
                            error_region_start,
                            error_region_end,
                            self.posture_errors_for_current_error_region
                        )
                        cpr_logger.info(f"[RUN ANALYSIS] Assigned error region data")

                        self.posture_errors_for_current_error_region.clear()

                        cpr_logger.info(f"[RUN ANALYSIS] Reset posture errors for current error region")

                #& Process the current chunk or mini chunk if the conditions are met
                process_chunk = (is_complete_chunk) and (not waiting_to_start_new_chunk) and (frame_counter != 0)
                process_mini_chunk = (frame_counter % self.reporting_interval_frames == 0) and (frame_counter != 0) and (mini_chunk_start_frame_index is not None) and (not is_complete_chunk) 
               
                if process_chunk: 
                    cpr_logger.info(f"[RUN ANALYSIS] Chunk completion detected")
                    
                    chunk_end_frame_index = frame_counter - 1                
                    cpr_logger.info(f"[RUN ANALYSIS] Determined the last frame of the chunk")
                    #!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    self._calculate_rate_and_depth_for_chunk(chunk_start_frame_index, chunk_end_frame_index)
                    cpr_logger.info(f"[RUN ANALYSIS] Calculated metrics for the chunk")

                elif process_mini_chunk:
                    cpr_logger.info(f"[RUN ANALYSIS] Mini chunk completion detected")

                    mini_chunk_end_frame_index = frame_counter
                    cpr_logger.info(f"[RUN ANALYSIS] Determined the last frame of the mini chunk")
                    #!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    self._calculate_rate_and_depth_for_chunk(mini_chunk_start_frame_index, mini_chunk_end_frame_index)
                    cpr_logger.info(f"[RUN ANALYSIS] Calculated metrics for the mini chunk")                       
                        
                if process_chunk or process_mini_chunk:
                    waiting_to_start_new_chunk = True

                    self.shoulders_analyzer.reset_shoulder_distances()
                    self.wrists_midpoint_analyzer.reset_midpoint_history()
                    cpr_logger.info(f"[RUN ANALYSIS] Reset shoulder distances and midpoint history")

                    self._get_rate_and_depth_warnings()
                    cpr_logger.info(f"[RUN ANALYSIS] Retrieved rate and depth warnings for the chunk")

                #& Initialize video writer if not done yet
                if frame is not None and not self._writer_initialized:
                    self._initialize_video_writer(frame)
                    cpr_logger.info(f"[VIDEO WRITER] Initialized video writer")

                #& Write frame if writer is functional
                if self._writer_initialized:
                    # Convert frame to BGR if needed
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    if len(frame.shape) == 2:  # Grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    try:
                        self.video_writer.write(frame)
                    except Exception as e:
                        cpr_logger.info(f"[WRITE ERROR] {str(e)}")
                        self._writer_initialized = False
                
                formatted_warnings = self._format_warnings()
                cpr_logger.info(f"[RUN ANALYSIS] Formatted warnings: {formatted_warnings}")

                self.socket_server.warning_queue.put(formatted_warnings)
                cpr_logger.info(f"[RUN ANALYSIS] Sent warnings to socket server")
                                
                #& Check if the user wants to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cpr_logger.info("\n[RUN ANALYSIS] 'q' pressed, exiting loop.")
                    break

                one_iteration_end_time = time.time()
                one_iteration_elapsed_time = one_iteration_end_time - one_iteration_start_time
                cpr_logger.info(f"[TIMING] One iteration elapsed time: {one_iteration_elapsed_time:.2f}s")

            main_loop_end_time = time.time()
            elapsed_time = main_loop_end_time - main_loop_start_time
            cpr_logger.info(f"[TIMING] Main loop elapsed time: {elapsed_time:.2f}s")

        except Exception as e:
            cpr_logger.info(f"[ERROR] An error occurred during main execution loop: {str(e)}")

        finally:
            report_and_plot_start_time = time.time()
            
            # Add socket server cleanup
            self.socket_server.stop_server()
            cpr_logger.info("[CLEANUP] Socket server stopped")

            self.cap.release()
            self.cap = None
            
            if self.video_writer is not None:
                self.video_writer.release()
                cpr_logger.info(f"[VIDEO WRITER] Released writer. File should be at: {os.path.abspath(self.output_video_path)}")
            cv2.destroyAllWindows()
            cpr_logger.info("[RUN ANALYSIS] Released video capture and destroyed all windows")         

            self._calculate_rate_and_depth_for_all_chunks()
            cpr_logger.info("[RUN ANALYSIS] Calculated weighted averages of the metrics across all chunks")

            self._plot_full_motion_curve_for_all_chunks()
            cpr_logger.info("[RUN ANALYSIS] Plotted full motion curve")

            self.metrics_calculator.add_warnings_to_processed_video(self.output_video_path, self.sampling_interval_frames)
            cpr_logger.info("[RUN ANALYSIS] Added warnings to processed video")

            report_and_plot_end_time = time.time()
            report_and_plot_elapsed_time = report_and_plot_end_time - report_and_plot_start_time
            cpr_logger.info(f"[TIMING] Report and plot elapsed time: {report_and_plot_elapsed_time:.2f}s")

    def _format_warnings(self):
        """Combine warnings into a simple structured response"""
        return {
            #!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            "status": "warning" if any([self.posture_warnings_from_current_frame, self.rate_and_depth_warnings_from_the_last_report]) else "ok",
            "posture_warnings": list(self.posture_warnings_from_current_frame),
            "rate_and_depth_warnings": self.rate_and_depth_warnings_from_the_last_report,
        }

    def _handle_frame_rotation(self, frame):
        #! Till now, the code has only been testes on portrait videos.
        if frame.shape[1] > frame.shape[0]:  # Width > Height
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def _process_frame(self, frame):
        #* Warnings for real time feedback
        warnings = []

        #* Chunk Completion Check
        is_complete_chunk = False
        accept_frame = True

        has_appended_midpoint = False
        
        #& Pose Estimation
        pose_results = self.pose_estimator.detect_poses(frame)

        #~ Handle Failed Detection or Update Previous Results
        if not pose_results:
            pose_results = self.prev_pose_results
            cpr_logger.info("[POSE ESTIMATION] No pose detected, using previous results (could be None)")
        else:
            self.prev_pose_results = pose_results
        
        if not pose_results:
            cpr_logger.info("[POSE ESTIMATION] Insufficient data for processing")
            return is_complete_chunk, accept_frame, warnings, has_appended_midpoint
        
        #& Rescuer and Patient Classification
        rescuer_processed_results, patient_processed_results = self.role_classifier.classify_roles(pose_results, self.prev_rescuer_processed_results, self.prev_patient_processed_results)

        #~ Handle Failed Classifications OR Update Previous Results
        if not rescuer_processed_results:
            rescuer_processed_results = self.prev_rescuer_processed_results
            cpr_logger.info("[ROLE CLASSIFICATION] No rescuer detected, using previous results (could be None)")
        else:
            self.prev_rescuer_processed_results = rescuer_processed_results

        if not patient_processed_results:
            patient_processed_results = self.prev_patient_processed_results
            cpr_logger.info("[ROLE CLASSIFICATION] No patient detected, using previous results (could be None)")
        else:
            self.prev_patient_processed_results = patient_processed_results
        
        if not rescuer_processed_results or not patient_processed_results:
            cpr_logger.info("[ROLE CLASSIFICATION] Insufficient data for processing")
            return is_complete_chunk, accept_frame, warnings, has_appended_midpoint
             
        #^ Set Params in Role Classifier (to draw later)
        self.role_classifier.rescuer_processed_results = rescuer_processed_results
        self.role_classifier.patient_processed_results = patient_processed_results
        cpr_logger.info(f"[ROLE CLASSIFICATION] Updated role classifier with new results")
    
        #& Chest Estimation
        chest_params = self.chest_initializer.estimate_chest_region(patient_processed_results["keypoints"], patient_processed_results["bounding_box"], frame_width=frame.shape[1], frame_height=frame.shape[0])
       
        #~ Handle Failed Estimation or Update Previous Results
        if not chest_params:
            chest_params = self.prev_chest_params
            cpr_logger.info("[CHEST ESTIMATION] No chest region detected, using previous results (could be None)")
        else:
            self.prev_chest_params = chest_params

        if not chest_params:
            cpr_logger.info("[CHEST ESTIMATION] Insufficient data for processing")
            return is_complete_chunk, accept_frame, warnings, has_appended_midpoint

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
            cpr_logger.info(f"[POSTURE ANALYSIS] Posture issues: {', '.join(warnings)}")
            self.consecutive_frames_with_posture_errors += 1
        else:
            cpr_logger.info("[POSTURE ANALYSIS] No posture issues detected")
            self.consecutive_frames_with_posture_errors = 0
        
        accept_frame = self.consecutive_frames_with_posture_errors < self.error_threshold_frames

        if accept_frame:
            warnings = []  # Reset warnings if the frame is accepted

        #^ Set Params in Posture Analyzer (to draw later)
        self.posture_analyzer.warnings = warnings  
        cpr_logger.info(f"[POSTURE ANALYSIS] Updated posture analyzer with new results")

        #& Wrist Midpoint Detection
        midpoint = self.wrists_midpoint_analyzer.detect_wrists_midpoint(rescuer_processed_results["keypoints"])

        #~ Handle Failed Detection or Update Previous Results
        if not midpoint:
            midpoint = self.prev_midpoint
            cpr_logger.info("[WRIST MIDPOINT DETECTION] No midpoint detected, using previous results (could be None)")
        else:
            self.prev_midpoint = midpoint
        
        if not midpoint:
            cpr_logger.info("[WRIST MIDPOINT DETECTION] Insufficient data for processing")
            return is_complete_chunk, accept_frame, warnings, has_appended_midpoint

        if accept_frame:
            #^ Set Params in Role Classifier (to draw later)
            has_appended_midpoint = True
            self.wrists_midpoint_analyzer.midpoint = midpoint
            self.wrists_midpoint_analyzer.midpoint_history.append(midpoint)
            cpr_logger.info(f"[WRIST MIDPOINT DETECTION] Updated wrist midpoint analyzer with new results")

            #& Shoulder Distance Calculation
            shoulder_distance = self.shoulders_analyzer.calculate_shoulder_distance(rescuer_processed_results["keypoints"])
            if shoulder_distance is not None:
                self.shoulders_analyzer.shoulder_distance = shoulder_distance
                self.shoulders_analyzer.shoulder_distance_history.append(shoulder_distance)
            cpr_logger.info(f"[SHOULDER DISTANCE] Updated shoulder distance analyzer with new results")
        else:
            #* Chunk Completion Check
            is_complete_chunk = True
            num_warnings_before = len(self.posture_errors_for_current_error_region)

            for warning in warnings:
                
                self.posture_errors_for_current_error_region.add(warning)
                
                num_warnings_after = len(self.posture_errors_for_current_error_region)
                
                if num_warnings_after > num_warnings_before:
                    cpr_logger.info(f"[POSTURE ANALYSIS] Added warning to current error region: {warning}") 

        return is_complete_chunk, accept_frame, warnings, has_appended_midpoint
    
    def _compose_frame(self, frame, accept_frame):
        # Chest Region
        if frame is not None:
            frame = self.chest_initializer.draw_expected_chest_region(frame)
            cpr_logger.info(f"[VISUALIZATION] Drawn chest region")

        # Warning Messages
        if frame is not None:
            frame = self.posture_analyzer.display_warnings(frame)
            cpr_logger.info(f"[VISUALIZATION] Drawn warnings")
        
        if frame is not None:
            if accept_frame:
                # Midpoint
                frame = self.wrists_midpoint_analyzer.draw_midpoint(frame)   
                cpr_logger.info(f"[VISUALIZATION] Drawn midpoint")  
    
        return frame

    def _calculate_rate_and_depth_for_chunk(self, chunk_start_frame_index, chunk_end_frame_index):
        try:
            result = self.metrics_calculator.handle_chunk(np.array(self.wrists_midpoint_analyzer.midpoint_history), chunk_start_frame_index, chunk_end_frame_index, self.fps, np.array(self.shoulders_analyzer.shoulder_distance_history), self.sampling_interval_frames)

            if result == False:
                cpr_logger.info("[ERROR] Failed to calculate metrics for the chunk")
                return
            
        except Exception as e:
            cpr_logger.info(f"[ERROR] Metric calculation failed: {str(e)}")

    def _calculate_rate_and_depth_for_all_chunks(self):
        try:
            self.metrics_calculator.calculate_rate_and_depth_for_all_chunk()
            cpr_logger.info(f"[METRICS] Weighted averages calculated")
        except Exception as e:
            cpr_logger.info(f"[ERROR] Failed to calculate weighted averages: {str(e)}")
            
    def _plot_full_motion_curve_for_all_chunks(self):
        try:
            self.graph_plotter.plot_motion_curve_for_all_chunks(self.metrics_calculator.chunks_y_preprocessed, 
                                                  self.metrics_calculator.chunks_peaks, 
                                                  self.metrics_calculator.chunks_depth, 
                                                  self.metrics_calculator.chunks_rate, 
                                                  self.metrics_calculator.chunks_start_and_end_indices, 
                                                  self.posture_analyzer.error_regions, 
                                                  self.sampling_interval_frames,
                                                  self.fps)
            cpr_logger.info("[PLOT] Full motion curve plotted")
        except Exception as e:
            cpr_logger.info(f"[ERROR] Failed to plot full motion curve: {str(e)}")
  
    def _get_rate_and_depth_warnings(self):
        #!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.rate_and_depth_warnings_from_the_last_report = self.metrics_calculator.get_rate_and_depth_warnings()
        cpr_logger.info(f"[VISUALIZATION] Rate and depth warnings data: {self.rate_and_depth_warnings_from_the_last_report}")

if __name__ == "__main__":
    cpr_logger.info(f"\n[MAIN] CPR Analysis Started")
    
    source = "https://192.168.1.9:8080/video"  # IP camera URL
    # source = 0
    requested_fps = 30
    output_video_path = r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\End to End\Code Refactor\Output\output.mp4"
    
    initialization_start_time = time.time()
    analyzer = CPRAnalyzer(source, requested_fps, output_video_path)
    initialization_end_time = time.time()
    initialization_elapsed_time = initialization_end_time - initialization_start_time
    
    cpr_logger.info(f"[TIMING] Initialization time: {initialization_elapsed_time:.2f}s")
    
    try:
        # This will now block until client connects
        analyzer.run_analysis()
    finally:
        analyzer.socket_server.stop_server()