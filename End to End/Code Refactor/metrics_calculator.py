# metrics_calculator.py
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import sys
import cv2
import os
from logging_config import cpr_logger

class MetricsCalculator:
    """Rate and depth calculation from motion data with improved peak detection"""
    
    def __init__(self, shoulder_width_cm):
        # Configuration parameters
        self.shoulder_width_cm = shoulder_width_cm

        # Parameters for cleaning the smoothed midpoints
        self.removing_impulse_noise_window_size = 5
        self.removing_impulse_noise_threshold = 3.0

        # Parameters for one chunk
        self.y_preprocessed = np.array([])

        self.peaks = np.array([])
        self.peaks_max = np.array([])
        self.peaks_min = np.array([])
                
        self.cm_px_ratio = None

        self.depth = None
        self.rate = None

        self.rate_and_depth_warnings = []
        
        # Parameters for all chunks
        self.chunks_y_preprocessed = []

        self.chunks_peaks = []

        self.chunks_depth = []
        self.chunks_rate = []

        self.weighted_depth = None
        self.weighted_rate = None

        self.chunks_start_and_end_indices = []

        self.chunks_rate_and_depth_warnings = []

        # Parameters for validation
        self.min_depth_threshold = 3.0  # cm
        self.max_depth_threshold = 6.0  # cm

        self.min_rate_threshold = 100.0  # cpm
        self.max_rate_threshold = 120.0  # cpm
        
#^ ################# Validating #######################

    def validate_midpoints_and_frames_count_in_chunk(self, y_exact, chunk_start_frame_index, chunk_end_frame_index, sampling_interval_in_frames):
        """
        Validate the number of midpoints and frames in a chunk

        Args:
            y_exact (np.ndarray): The exact y-values of the midpoints.
            chunk_start_frame_index (int): The starting frame index of the chunk.
            chunk_end_frame_index (int): The ending frame index of the chunk.
            sampling_interval_in_frames (int): The interval at which frames are sampled.
        
        Raises:
            ValueError: If the number of midpoints does not match the expected number for the given chunk.        
        """
        try:
            # Calculate expected number of sampled frames
            start = chunk_start_frame_index
            end = chunk_end_frame_index
            interval = sampling_interval_in_frames
            
            # Mathematical formula to count sampled frames
            expected_samples = (end // interval) - ((start - 1) // interval)

            # Validate
            actual_y_exact_length = len(y_exact)
            if actual_y_exact_length != expected_samples:
                cpr_logger.info(f"\nERROR: Mismatch in expected and actual samples")
                cpr_logger.info(f"Expected: {expected_samples} samples (frames {start}-{end} @ every {interval} frames)")
                cpr_logger.info(f"Actual: {actual_y_exact_length} midoints points recieived")
                sys.exit(1)

        except Exception as e:
            cpr_logger.error(f"\nCRITICAL VALIDATION ERROR: {str(e)}")
            sys.exit(1)

#^ ################# Preprocessing #######################

    def _smooth_midpoints(self, midpoints):
        """
        Smooth the y-values of the midpoints using Savitzky-Golay filter

        Args:
            y_exact (np.ndarray): The exact y-values of the midpoints.

        Returns:
            np.ndarray: The smoothed y-values.
        """
        
        if len(midpoints) > 5:  # Ensure enough data points
            try:
                y_smooth = savgol_filter(
                    midpoints[:, 1], 
                    window_length=3, 
                    polyorder=2, 
                    mode='nearest'
                )
                return y_smooth
            except Exception as e:
                cpr_logger.error(f"Smoothing error: {e}")
                y_smooth = midpoints[:, 1]  # Fallback to original
                return y_smooth
        else:
            y_smooth = midpoints[:, 1]  # Not enough points
            return y_smooth

    def _clean_midpoints(self, y_smooth):
        """
        Clean the smoothed y-values to remove impulse noise using median filtering

        Args:
            y_smooth (np.ndarray): The smoothed y-values.

        Returns:
            np.ndarray: The cleaned y-values.
        """

        if len(y_smooth) < self.removing_impulse_noise_window_size:
            return y_smooth  # Not enough points for processing
        
        y_clean = np.array(y_smooth, dtype=float)  # Copy to avoid modifying original
        half_window = self.removing_impulse_noise_window_size // 2
        
        for i in range(len(y_smooth)):
            # Get local window (handle boundaries)
            start = max(0, i - half_window)
            end = min(len(y_smooth), i + half_window + 1)
            window = y_smooth[start:end]
            
            # Calculate local median and MAD (robust statistics)
            med = np.median(window)
            mad = 1.4826 * np.median(np.abs(window - med))  # Median Absolute Deviation
            
            # Detect and replace outliers
            if abs(y_smooth[i] - med) > self.removing_impulse_noise_threshold * mad:
                # Replace with median of immediate neighbors (better than global median)
                left = y_smooth[max(0, i-1)]
                right = y_smooth[min(len(y_smooth)-1, i+1)]
                y_clean[i] = np.median([left, right])
        
        return y_clean

    def preprocess_midpoints(self, midpoints):
        """
        Preprocess the y-values of the midpoints by smoothing and cleaning

        Sets:
            y_preprocessed (np.ndarray): The preprocessed y-values.

        Args:
            y_exact (np.ndarray): The exact y-values of the midpoints.
        
        Returns:
            bool: True if preprocessing was successful, False otherwise.
        """

        y_smooth = self._smooth_midpoints(midpoints)
        y_clean = self._clean_midpoints(y_smooth)

        self.y_preprocessed = y_clean

        return len(self.y_preprocessed) > 0  # Return True if preprocessing was successful

#^ ################# Processing #######################

    def detect_midpoints_peaks(self):
        """
        Detect peaks in the preprocessed y-values using dynamic distance
        
        Sets:
            peaks (np.ndarray): The detected peaks.
            peaks_max (np.ndarray): The detected max peaks.
            peaks_min (np.ndarray): The detected min peaks.

        Returns:
            bool: True if peaks were detected, False otherwise.
        """

        if self.y_preprocessed.size == 0:
            cpr_logger.info("No smoothed values found for peak detection")
            return False
            
        try:
            distance = min(1, len(self.y_preprocessed))  # Dynamic distance based on data length

            # Detect max peaks with default prominence
            self.peaks_max, _ = find_peaks(self.y_preprocessed, distance=distance)
            
            # Detect min peaks with reduced or no prominence requirement
            self.peaks_min, _ = find_peaks(
                -self.y_preprocessed, 
                distance=distance, 
                prominence=(0.5, None)  # Adjust based on your data's characteristics
            )
            
            self.peaks = np.sort(np.concatenate((self.peaks_max, self.peaks_min)))

            return len(self.peaks) > 0
        except Exception as e:
            cpr_logger.error(f"Peak detection error: {e}")
            return False

    def calculate_cm_px_ratio(self, shoulder_distances):
        """
        Calculate the ratio of cm to pixels based on shoulder distances

        Sets:
            cm_px_ratio (float): The ratio of cm to pixels.

        Args:
            shoulder_distances (list): List of shoulder distances in pixels.
        """

        if len(shoulder_distances) > 0:
            avg_shoulder_width_px = np.mean(shoulder_distances)
            self.cm_px_ratio = self.shoulder_width_cm / avg_shoulder_width_px
        else:
            self.cm_px_ratio = None
            cpr_logger.info("No shoulder distances available for cm/px ratio calculation")
        
    def calculate_rate_and_depth_for_chunk(self, original_fps, sampling_interval_in_frames=1):
        """
        Calculate the rate and depth of the motion data for a chunk.

        Sets:
            depth (float): The calculated depth in cm.
            rate (float): The calculated rate in cpm.

        Args:
            original_fps (float): The original frames per second of the video.
            sampling_interval_in_frames (int): Number of frames skipped between samples.
        """   
        try:

            # Without Adjustment: A peak distance of 5 (downsampled frames) would incorrectly be interpreted as 5/30 = 0.167 sec (too short).
            # With Adjustment: The same peak distance 5 (downsampled frames) correctly represents 5/10 = 0.5 sec.

            effective_fps = original_fps / sampling_interval_in_frames  # Correctly reduced FPS

            # Depth calculation (unchanged)
            depth = None
            if len(self.peaks) > 1:
                depth = np.mean(np.abs(np.diff(self.y_preprocessed[self.peaks]))) * self.cm_px_ratio

            # Rate calculation (now uses effective_fps)
            rate = None
            if len(self.peaks_max) > 1:
                # Peak indices are from the downsampled signal, so we use effective_fps
                peak_intervals = np.diff(self.peaks_max)  # Already in downsampled frames
                rate = (1 / (np.mean(peak_intervals) / effective_fps)) * 60  # Correct CPM
                
            # Handle cases with no valid data
            if depth is None or rate is None:
                depth = 0
                rate = 0
                self.peaks = np.array([])

            self.depth = depth
            self.rate = rate
        except Exception as e:
            cpr_logger.error(f"Error calculating rate and depth: {e}")
    
    def assign_chunk_data(self, chunk_start_frame_index, chunk_end_frame_index):
        """
        Capture chunk data for later analysis

        Sets:
            chunks_depth (list): List of depths for each chunk.
            chunks_rate (list): List of rates for each chunk.
            chunks_start_and_end_indices (list): List of start and end indices for each chunk.
            chunks_y_preprocessed (list): List of preprocessed y-values for each chunk.
            chunks_peaks (list): List of detected peaks for each chunk.

        Args:
            chunk_start_frame_index (int): The starting frame index of the chunk.
            chunk_end_frame_index (int): The ending frame index of the chunk.
        """
        self.chunks_depth.append(self.depth)
        self.chunks_rate.append(self.rate)
        self.chunks_start_and_end_indices.append((chunk_start_frame_index, chunk_end_frame_index))

        self.chunks_y_preprocessed.append(self.y_preprocessed.copy())
        self.chunks_peaks.append(self.peaks.copy())      

        self.current_chunk_start = chunk_start_frame_index
        self.current_chunk_end = chunk_end_frame_index

        self.chunks_rate_and_depth_warnings.append(self.rate_and_depth_warnings.copy())
        
    def calculate_rate_and_depth_for_all_chunk(self):
        """
        Calculate the weighted average rate and depth for all chunks

        Sets:
            weighted_depth (float): The weighted average depth in cm.
            weighted_rate (float): The weighted average rate in cpm.
        """
        
        if not self.chunks_depth or not self.chunks_rate or not self.chunks_start_and_end_indices:
            cpr_logger.info("[WARNING] No chunk data available for averaging")
            return None
            
        if not (len(self.chunks_depth) == len(self.chunks_rate) == len(self.chunks_start_and_end_indices)):
            cpr_logger.info("[ERROR] Mismatched chunk data lists")
            return None

        total_weight = 0
        weighted_depth_sum = 0
        weighted_rate_sum = 0

        for depth, rate, (start, end) in zip(self.chunks_depth, 
                                        self.chunks_rate,
                                        self.chunks_start_and_end_indices):
            
            # Calculate chunk duration (+1 because inclusive)
            chunk_duration = end - start + 1
            
            weighted_depth_sum += depth * chunk_duration
            weighted_rate_sum += rate * chunk_duration
            total_weight += chunk_duration

        if total_weight == 0:
            self.weighted_depth = None
            self.weighted_rate = None

            cpr_logger.info("[ERROR] No valid chunks for averaging")
        else:
            self.weighted_depth = weighted_depth_sum / total_weight
            self.weighted_rate =  weighted_rate_sum / total_weight
        
            cpr_logger.info(f"[RESULTS] Weighted average depth: {self.weighted_depth:.1f} cm")
            cpr_logger.info(f"[RESULTS] Weighted average rate: {self.weighted_rate:.1f} cpm")

#^ ################# Warnings #######################

    def _get_rate_and_depth_status(self):
        """Internal validation logic"""

        depth_status = "normal"
        rate_status  = "normal"
        
        if self.depth < self.min_depth_threshold and self.depth > 0:
            depth_status = "low"
        elif self.depth > self.max_depth_threshold:
            depth_status = "high"
            
        if self.rate < self.min_rate_threshold and self.rate > 0:
            rate_status = "low"
        elif self.rate > self.max_rate_threshold:
            rate_status = "high"
            
        return depth_status, rate_status

    def get_rate_and_depth_warnings(self):
        """Get performance warnings based on depth and rate"""

        depth_status, rate_status = self._get_rate_and_depth_status()

        warnings = []
        if depth_status == "low":
            warnings.append("Depth too low!")
        elif depth_status == "high":
            warnings.append("Depth too high!")

        if rate_status == "low":
            warnings.append("Rate too slow!")
        elif rate_status == "high":
            warnings.append("Rate too fast!")

        self.rate_and_depth_warnings = warnings

        return warnings

#^ ################# Handle Chunk #######################

    def handle_chunk(self, midpoints, chunk_start_frame_index, chunk_end_frame_index, fps, shoulder_distances, sampling_interval_in_frames):
        """
        Handle a chunk of motion data by validating, preprocessing, and calculating metrics
        for the chunk.

        Args:
            y_exact (np.ndarray): The exact y-values of the midpoints.
            chunk_start_frame_index (int): The starting frame index of the chunk.
            chunk_end_frame_index (int): The ending frame index of the chunk.
            fps (float): The frames per second of the video.
            shoulder_distances (list): List of shoulder distances in pixels.

        Returns:
            bool: True if the chunk was processed successfully, False otherwise.
        """

        # The program is terminated if the validation fails
        self.validate_midpoints_and_frames_count_in_chunk(midpoints, chunk_start_frame_index, chunk_end_frame_index, sampling_interval_in_frames)

        preprocessing_reult = self.preprocess_midpoints(midpoints)
        if not preprocessing_reult:
            cpr_logger.info("Preprocessing failed, skipping chunk")
            return False
        
        self.detect_midpoints_peaks()
        if not self.detect_midpoints_peaks():
            cpr_logger.info("Peak detection failed, skipping chunk")

            self.peaks = np.array([])
            self.peaks_max = np.array([])
            self.peaks_min = np.array([])

            self.depth = 0
            self.rate = 0

            return False
        
        self.calculate_cm_px_ratio(shoulder_distances)
        if self.cm_px_ratio is None:
            cpr_logger.info("cm/px ratio calculation failed, skipping chunk")

            self.depth = 0
            self.rate = 0
            
            return False
        
        self.calculate_rate_and_depth_for_chunk(fps, sampling_interval_in_frames)
        if self.depth is None or self.rate is None:
            cpr_logger.info("Rate and depth calculation failed, skipping chunk")
            return False
        else:
            cpr_logger.info(f"Chunk {chunk_start_frame_index}-{chunk_end_frame_index} - Depth: {self.depth:.1f} cm, Rate: {self.rate:.1f} cpm")

        self.get_rate_and_depth_warnings()

        self.assign_chunk_data(chunk_start_frame_index, chunk_end_frame_index)
        cpr_logger.info(f"Chunk {chunk_start_frame_index}-{chunk_end_frame_index} processed successfully")
        return True

#^ ################# Comments #######################
# Between every two consecutive mini chunks, there wil be "sampling interval" frames unaccounted for.
# This is because when we reach the "reporting interval" number of frames, we terminate the first mini chunk.
# But we only start the next mini chunk when we detect the next successfully processed frame.
# Which is "sampling interval" frames later at the earliest.
# We can't just initialize the next mini chunk at the "reporting interval" frame, because we need to wait for the next successful frame.
# Becuase maybe the next frame is a frame with posture errors.
# For better visualization, we connect between the last point of the previous chunk and the first point of the next chunk if they are "sampling interval" frames apart.
# But that is only for visualization, all calculations are done on the original frames.

# Chunks that are too short can fail any stage of the "handle chunk" process.
# If they do, we vizualize what we have and ignore the rest.
# For example, a chunk with < 2 peaks will not be able to calculate the rate.
# So we will set it to zero and display the midpoints and detected peaks.
# If there are no peaks, we will set the rate to zero and display the midpoints.

# Problems with chunks could be:
# - Less than 3 seconds.
# - Not enough peaks to calculate depth and rate