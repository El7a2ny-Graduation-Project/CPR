# metrics_calculator.py
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import sys

class MetricsCalculator:
    """Rate and depth calculation from motion data with improved peak detection"""
    
    def __init__(self, frame_count, shoulder_width_cm):
        # Configuration parameters
        self.shoulder_width_cm = shoulder_width_cm
        self.frame_count = frame_count

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
        
        # Parameters for all chunks
        self.chunks_y_preprocessed = []

        self.chunks_peaks = []

        self.chunks_depth = []
        self.chunks_rate = []

        self.weighted_depth = None
        self.weighted_rate = None

        self.chunks_start_and_end_indices = []


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
                print(f"\nERROR: Mismatch in expected and actual samples")
                print(f"Expected: {expected_samples} samples (frames {start}-{end} @ every {interval} frames)")
                print(f"Actual: {actual_y_exact_length} midoints points recieived")
                sys.exit(1)

        except Exception as e:
            print(f"\nCRITICAL VALIDATION ERROR: {str(e)}")
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
                print(f"Smoothing error: {e}")
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
            print("No smoothed values found for peak detection")
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
            print(f"Peak detection error: {e}")
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
            print("No shoulder distances available for cm/px ratio calculation")
        
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
            print(f"Error calculating rate and depth: {e}")
    
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
        
    def calculate_rate_and_depth_for_all_chunk(self):
        """
        Calculate the weighted average rate and depth for all chunks

        Sets:
            weighted_depth (float): The weighted average depth in cm.
            weighted_rate (float): The weighted average rate in cpm.
        """
        
        if not self.chunks_depth or not self.chunks_rate or not self.chunks_start_and_end_indices:
            print("[WARNING] No chunk data available for averaging")
            return None
            
        if not (len(self.chunks_depth) == len(self.chunks_rate) == len(self.chunks_start_and_end_indices)):
            print("[ERROR] Mismatched chunk data lists")
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

            print("[ERROR] No valid chunks for averaging")
        else:
            self.weighted_depth = weighted_depth_sum / total_weight
            self.weighted_rate =  weighted_rate_sum / total_weight
        
            print(f"[RESULTS] Weighted average depth: {self.weighted_depth:.1f} cm")
            print(f"[RESULTS] Weighted average rate: {self.weighted_rate:.1f} cpm")

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
            print("Preprocessing failed, skipping chunk")
            return False
        
        self.detect_midpoints_peaks()
        if not self.detect_midpoints_peaks():
            print("Peak detection failed, skipping chunk")
            return False
        
        self.calculate_cm_px_ratio(shoulder_distances)
        if self.cm_px_ratio is None:
            print("cm/px ratio calculation failed, skipping chunk")
            return False
        
        self.calculate_rate_and_depth_for_chunk(fps, sampling_interval_in_frames)
        if self.depth is None or self.rate is None:
            print("Rate and depth calculation failed, skipping chunk")
            return False

        self.assign_chunk_data(chunk_start_frame_index, chunk_end_frame_index)
        print(f"Chunk {chunk_start_frame_index}-{chunk_end_frame_index} processed successfully")
        return True

#^ ################# Plotting #######################
    def plot_motion_curve_for_all_chunks(self, posture_errors_for_all_error_region, sampling_interval_in_frames, reporting_interval_in_frames):
        """Plot combined analysis with connected chunks and proper error regions"""
        print("[Plot Graph] Initializing complete motion curve plot...")
        
        if not self.chunks_start_and_end_indices:
            print("[Plot Graph] No chunk data available for plotting")
            return

        plt.figure(figsize=(16, 8))
        ax = plt.gca()
        
        # Sort chunks chronologically
        sorted_chunks = sorted(zip(self.chunks_start_and_end_indices, 
                                 self.chunks_depth, 
                                 self.chunks_rate),
                                 key=lambda x: x[0][0])
        print(f"[Plot Graph] Processing {len(sorted_chunks)} CPR chunks")

        # Track previous chunk's last point and end frame
        prev_last_point = None
        prev_chunk_end = None

        # Plot each chunk and handle connections
        for idx, chunk in enumerate(sorted_chunks):
            print(f"[Plot Graph] Rendering chunk {idx+1}/{len(sorted_chunks)}")
            prev_last_point, prev_chunk_end = self._plot_single_chunk(
                ax, chunk, idx, prev_last_point, prev_chunk_end, sampling_interval_in_frames
            )
        
        # Compute and plot error regions
        error_regions = self._compute_error_regions(sorted_chunks, reporting_interval_in_frames)
        print(f"[Plot Graph] Identified {len(error_regions)} error regions")
        self._print_analysis_details(sorted_chunks, error_regions, posture_errors_for_all_error_region)
        self._plot_error_regions(ax, error_regions, posture_errors_for_all_error_region)
        
        # Add weighted averages
        if hasattr(self, 'weighted_depth') and hasattr(self, 'weighted_rate'):
            print(f"[Plot Graph] Adding weighted averages: Depth={self.weighted_depth:.1f}cm, Rate={self.weighted_rate:.1f}cpm")
            ax.annotate(f"Weighted Averages:\nDepth: {self.weighted_depth:.1f}cm\nRate: {self.weighted_rate:.1f}cpm",
                        xy=(0.98, 0.98), xycoords='axes fraction',
                        ha='right', va='top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black'))
        
        # Configure legend and layout
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper right')

        plt.xlabel("Frame Number")
        plt.ylabel("Vertical Position (px)")
        plt.title("Complete CPR Analysis with Metrics")
        plt.grid(True)
        plt.tight_layout()
        print("[Plot Graph] Finalizing plot layout")
        plt.show()
        print("[Plot Graph] Plot display complete")

    def _plot_single_chunk(self, ax, chunk, idx, prev_last_point, prev_chunk_end, sampling_interval):
        (start, end), depth, rate = chunk
        chunk_frames = np.arange(start, end + 1, sampling_interval)
        y_preprocessed = self.chunks_y_preprocessed[idx]
        peaks = self.chunks_peaks[idx]
        
        # Add separator line between chunks
        if prev_chunk_end is not None:
            separator_x = prev_chunk_end + 0.5
            print(f"[Plot Graph] Adding chunk separator at frame {separator_x}")
            ax.axvline(x=separator_x, color='orange', linestyle=':', linewidth=1.5)
        
        # Check if chunks are contiguous and need connection
        if (prev_chunk_end is not None and 
            start == prev_chunk_end + sampling_interval and
            prev_last_point is not None):
            
            connect_frames = [prev_chunk_end, start]
            connect_y_preprocessed = np.vstack([prev_last_point['y_preprocessed'], y_preprocessed[0]])
            
            print(f"[Plot Graph] Connecting chunk {idx+1} to previous chunk (frames {prev_chunk_end}-{start})")
            ax.plot(connect_frames, connect_y_preprocessed, 
                    color="blue", linewidth=2)
        
        # Plot current chunk data
        print(f"[Plot Graph] Plotting chunk {idx+1} (frames {start}-{end}) with {len(peaks)} peaks")
        smooth_label = "Motion" if idx == 0 else ""
        peaks_label = "Peaks" if idx == 0 else ""

        ax.plot(chunk_frames, y_preprocessed,
                color="blue", linewidth=2,
                marker='o', markersize=4,
                markerfacecolor='blue', markeredgecolor='blue',
                label=smooth_label)
        
        # Plot peaks
        if peaks.size > 0:
            ax.plot(chunk_frames[peaks], y_preprocessed[peaks],
                    "x", color="green", markersize=8,
                    label=peaks_label)

        # Annotate chunk metrics
        if depth is not None and rate is not None:
            mid_frame = (start + end) // 2
            print(f"[Plot Graph] Chunk {idx+1} metrics: {depth:.1f}cm depth, {rate:.1f}cpm rate")
            ax.annotate(f"Depth: {depth:.1f}cm\nRate: {rate:.1f}cpm",
                    xy=(mid_frame, np.max(y_preprocessed)),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        return {'y_preprocessed': y_preprocessed[-1]}, end

    def _compute_error_regions(self, sorted_chunks, reporting_interval):
        error_regions = []
        chunk_boundaries = [chunk[0][0] for chunk in sorted_chunks]
        chunk_boundaries.append(sorted_chunks[-1][0][1])
        
        print("[Plot Graph] Computing error regions:")
        # Before first chunk
        first_chunk_start = sorted_chunks[0][0][0]
        if first_chunk_start > 0:
            print(f" - Pre-chunk region: 0-{first_chunk_start-1}")
            error_regions.append((0, first_chunk_start - 1))
        
        # Between chunks
        for i in range(1, len(sorted_chunks)):
            prev_end = sorted_chunks[i-1][0][1]
            curr_start = sorted_chunks[i][0][0]
            if curr_start - prev_end > reporting_interval + 1:
                print(f" - Inter-chunk region: {prev_end+1}-{curr_start-1}")
                error_regions.append((prev_end + 1, curr_start - 1))
        
        # After last chunk
        last_end = sorted_chunks[-1][0][1]
        if last_end < self.frame_count - 1:
            print(f" - Post-chunk region: {last_end+1}-{self.frame_count-1}")
            error_regions.append((last_end + 1, self.frame_count - 1))
        
        return [r for r in error_regions if r[1] - r[0] > 0]

    def _plot_error_regions(self, ax, error_regions, posture_errors):
        print("[Plot Graph] Rendering error regions:")
        for idx, (start, end) in enumerate(error_regions):
            print(f" - Region {idx+1}: frames {start}-{end}")
            ax.axvspan(start, end, color='gray', alpha=0.2, label='Posture Errors' if idx == 0 else "")
            
            if posture_errors:
                try:
                    region_errors = posture_errors[idx]
                    error_text = "Errors:\n" + "\n".join(region_errors) if region_errors else ""
                    mid_frame = (start + end) // 2
                    ax.text(mid_frame, np.mean(ax.get_ylim()), error_text,
                            ha='center', va='center', fontsize=9, color='red', alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.7))
                except IndexError:
                    print(f"[Plot Graph] Warning: No error data for region {idx}")

    def _print_analysis_details(self, sorted_chunks, error_regions, posture_errors):
        """Combined helper for printing chunks and error regions"""
        print("[Plot Graph]\n=== CPR Chunk Analysis ===")
        for idx, ((start, end), depth, rate) in enumerate(sorted_chunks):
            duration = end - start + 1
            print(f"[Plot Graph] Chunk {idx+1}: "
                  f"Frames {start}-{end} ({duration} frames), "
                  f"Depth: {depth:.1f}cm, Rate: {rate:.1f}cpm")

        print("\n[Plot Graph] === Error Region Analysis ===")
        for i, (start, end) in enumerate(error_regions):
            try:
                errors = posture_errors[i]
                error_str = ", ".join(errors) if errors else "No errors"
                print(f"[Plot Graph] Region {i+1}: Frames {start}-{end} - {error_str}")
            except IndexError:
                print(f"[Plot Graph] Region {i+1}: Frames {start}-{end} - Error data missing")