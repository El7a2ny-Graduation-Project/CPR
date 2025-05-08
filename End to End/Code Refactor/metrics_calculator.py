# metrics_calculator.py
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import sys

class MetricsCalculator:
    """Rate and depth calculation from motion data with improved peak detection"""
    
    def __init__(self, frame_count, shoulder_width_cm):
        self.shoulder_width_cm = shoulder_width_cm
        self.peaks = np.array([])
        self.peaks_max = np.array([])
        self.peaks_min = np.array([])
        self.y_smoothed = np.array([])
        self.cm_px_ratio = None
        self.midpoints_list = np.array([])
        self.shoulder_distances = []

        # Parameters for the final report
        self.chunks_depth = []
        self.chunks_rate = []
        self.chunks_start_and_end_indices = []

        self.chunks_midpoints = []
        self.chunks_smoothed = []
        self.chunks_peaks = []

        self.frame_count = frame_count

    def smooth_midpoints(self, midpoints):
        """Apply Savitzky-Golay filter to smooth motion data"""
        self.midpoints_list = np.array(midpoints)
        
        if len(self.midpoints_list) > 5:  # Ensure enough data points
            try:
                self.y_smoothed = savgol_filter(
                    self.midpoints_list[:, 1], 
                    window_length=10, 
                    polyorder=2, 
                    mode='nearest'
                )
                return True
            except Exception as e:
                print(f"Smoothing error: {e}")
                self.y_smoothed = self.midpoints_list[:, 1]  # Fallback to original
                return False
        else:
            self.y_smoothed = self.midpoints_list[:, 1]  # Not enough points
            return False

    def detect_peaks(self):
        """Improved peak detection with adjusted prominence for min peaks"""
        if self.y_smoothed.size == 0:
            print("No smoothed values found for peak detection")
            return False
            
        try:
            distance = min(10, len(self.y_smoothed))  # Dynamic distance based on data length

            # Detect max peaks with default prominence
            self.peaks_max, _ = find_peaks(self.y_smoothed, distance=distance)
            
            # Detect min peaks with reduced or no prominence requirement
            self.peaks_min, _ = find_peaks(
                -self.y_smoothed, 
                distance=distance, 
                prominence=(0.3, None)  # Adjust based on your data's characteristics
            )
            
            self.peaks = np.sort(np.concatenate((self.peaks_max, self.peaks_min)))

            return len(self.peaks) > 0
        except Exception as e:
            print(f"Peak detection error: {e}")
            return False

    def _validate_chunk(self, chunk_start_frame_index, chunk_end_frame_index):
        """Validate that the data length matches the expected frame range.
        Terminates the program with error code 1 if validation fails.
        
        Args:
            chunk_start_frame_index: Start frame index of the chunk
            chunk_end_frame_index: End frame index of the chunk
            
        Exits:
            If validation fails, prints error message and exits with code 1
        """
        try:
            # Calculate expected number of frames
            num_frames = chunk_end_frame_index - chunk_start_frame_index + 1
            
            # Validate midpoints data
            if len(self.midpoints_list[:, 1]) != num_frames:
                print(f"\nERROR: Data length mismatch in midpoints_list")
                print(f"Expected: {num_frames} frames ({chunk_start_frame_index}-{chunk_end_frame_index})")
                print(f"Actual: {len(self.midpoints_list[:, 1])} frames")
                sys.exit(1)
                
            # Validate smoothed data
            if len(self.y_smoothed) != num_frames:
                print(f"\nERROR: Data length mismatch in y_smoothed")
                print(f"Expected: {num_frames} frames ({chunk_start_frame_index}-{chunk_end_frame_index})")
                print(f"Actual: {len(self.y_smoothed)} frames")
                sys.exit(1)
                
        except Exception as e:
            print(f"\nCRITICAL VALIDATION ERROR: {str(e)}")
            sys.exit(1)

    def calculate_metrics(self, shoulder_distances, fps, chunk_start_frame_index, chunk_end_frame_index):
        """Calculate compression metrics with improved calculations"""        
        
        self._validate_chunk(chunk_start_frame_index, chunk_end_frame_index)
                
        self.shoulder_distances = shoulder_distances

        try:
            # Calculate pixel to cm ratio
            if len(self.shoulder_distances) > 0:
                avg_dist = np.mean(self.shoulder_distances)
                self.cm_px_ratio = self.shoulder_width_cm / avg_dist
            else:
                print("No shoulder distances available")
                return None, None

            # Depth calculation using all peaks
            depth = 0.0
            if len(self.peaks) > 1:
                depth = np.mean(np.abs(np.diff(self.y_smoothed[self.peaks]))) * self.cm_px_ratio

            # Rate calculation using only compression peaks (peaks_max)
            rate = 0.0
            if len(self.peaks_max) > 1:
                rate = 1 / (np.mean(np.diff(self.peaks_max)) / fps) * 60  # Convert to CPM

            # Store the results of this chunk for the final report if they are not None
            self.chunks_depth.append(depth)
            self.chunks_rate.append(rate)
            self.chunks_start_and_end_indices.append((chunk_start_frame_index, chunk_end_frame_index))

            self.chunks_midpoints.append(self.midpoints_list.copy())
            self.chunks_smoothed.append(self.y_smoothed.copy())
            self.chunks_peaks.append(self.peaks.copy())      


            return depth, rate
            
        except Exception as e:
            print(f"Metric calculation error: {e}")
            return None, None

    def plot_motion_curve(self, chunk_start_frame_index, chunk_end_frame_index):
        """Enhanced visualization with original and smoothed data"""
        if self.midpoints_list.size == 0:
            print("No midpoint data to plot")
            return
        
        self._validate_chunk(chunk_start_frame_index, chunk_end_frame_index)

        # Create frame index array for x-axis
        frame_indices = np.arange(chunk_start_frame_index, chunk_end_frame_index + 1)


        plt.figure(figsize=(12, 6))
        
        # Plot original and smoothed data with correct frame indices
        plt.plot(frame_indices, self.midpoints_list[:, 1], 
                label="Original Motion", 
                color="red", 
                linestyle="dashed", 
                alpha=0.6)
    
        plt.plot(frame_indices, self.y_smoothed, 
                label="Smoothed Motion", 
                color="blue", 
                linewidth=2)

        # Plot peaks if detected
        if self.peaks.size > 0:
            plt.plot(frame_indices[self.peaks], 
                    self.y_smoothed[self.peaks], 
                    "x", 
                    color="green", 
                    markersize=10, 
                    label="Peaks")
        else:
            print("No peaks to plot")

        plt.xlabel("Frame Number")
        plt.ylabel("Vertical Position (px)")
        plt.title("Compression Motion Analysis")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def calculate_weighted_averages(self):
        """Calculate weighted averages based on chunk durations
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
            print("[ERROR] Total chunk durations is zero")
            return None

        weighted_depth = weighted_depth_sum / total_weight
        weighted_rate = weighted_rate_sum / total_weight

        print(f"[RESULTS] Weighted average depth: {weighted_depth:.1f} cm")
        print(f"[RESULTS] Weighted average rate: {weighted_rate:.1f} cpm")
        
        return weighted_depth, weighted_rate

    def plot_motion_curve_for_all_chunks(self, posture_errors_for_all_error_region):
        """Plot combined analysis with metrics annotations and posture error labels"""
        if not self.chunks_start_and_end_indices:
            print("No chunk data available for plotting")
            return

        # Print chunk information before plotting
        print("\n=== Chunk Ranges ===")
        for i, (start_end, depth, rate) in enumerate(zip(self.chunks_start_and_end_indices, 
                                                    self.chunks_depth, 
                                                    self.chunks_rate)):
            print(f"Chunk {i+1}: Frames {start_end[0]}-{start_end[1]} | "
                f"Depth: {depth:.1f}cm | Rate: {rate:.1f}cpm")

        plt.figure(figsize=(16, 8))
        ax = plt.gca()
        
        # Sort chunks chronologically
        sorted_chunks = sorted(zip(self.chunks_start_and_end_indices, 
                        self.chunks_depth, 
                        self.chunks_rate),
                        key=lambda x: x[0][0])
        
        # 1. Plot all valid chunks with metrics
        for idx, ((start, end), depth, rate) in enumerate(sorted_chunks):
            chunk_frames = np.arange(start, end + 1)
            midpoints = self.chunks_midpoints[idx]
            smoothed = self.chunks_smoothed[idx]
            peaks = self.chunks_peaks[idx]
            
            # Plot data
            ax.plot(chunk_frames, midpoints[:, 1], 
                color="red", linestyle="dashed", alpha=0.6,
                label="Original Motion" if idx == 0 else "")
            ax.plot(chunk_frames, smoothed,
                color="blue", linewidth=2,
                label="Smoothed Motion" if idx == 0 else "")
            
            # Plot peaks
            if peaks.size > 0:
                ax.plot(chunk_frames[peaks], smoothed[peaks],
                    "x", color="green", markersize=8,
                    label="Peaks" if idx == 0 else "")

            # Annotate chunk metrics
            mid_frame = (start + end) // 2
            ax.annotate(f"Depth: {depth:.1f}cm\nRate: {rate:.1f}cpm",
                    xy=(mid_frame, np.max(smoothed)),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        # 2. Identify and label posture error regions
        error_regions = []
        
        # Before first chunk
        if sorted_chunks[0][0][0] > 0:
            error_regions.append((0, sorted_chunks[0][0][0]-1))
        
        # Between chunks
        for i in range(1, len(sorted_chunks)):
            prev_end = sorted_chunks[i-1][0][1]
            curr_start = sorted_chunks[i][0][0]
            if curr_start - prev_end > 1:
                error_regions.append((prev_end + 1, curr_start - 1))
        
        # After last chunk
        last_end = sorted_chunks[-1][0][1]
        if last_end < self.frame_count - 1:
            error_regions.append((last_end + 1, self.frame_count - 1))

        # Print error regions information
        print("\n=== Error Regions ===")
        for i, (start, end) in enumerate(error_regions):
            # Get errors for this region if available
            try:
                errors = posture_errors_for_all_error_region[i]
                error_str = ", ".join(errors) if errors else "No errors detected"
            except IndexError:
                error_str = "No error data"
                
            print(f"Error Region {i+1}: Frames {start}-{end} | Errors: {error_str}")

        # Shade and label error regions
        for error_region_index, region in enumerate (error_regions):
            ax.axvspan(region[0], region[1], 
                    color='gray', alpha=0.2,
                    label='Posture Errors' if region == error_regions[0] else "")
            
            # Add vertical dotted lines at boundaries
            ax.axvline(x=region[0], color='black', linestyle=':', alpha=0.5)
            ax.axvline(x=region[1], color='black', linestyle=':', alpha=0.5)
            
            # Add frame number labels - properly aligned
            y_pos = ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            
            # Start frame label - right aligned before the line
            ax.text(region[0] - 1, y_pos, 
                f"Frame {region[0]}",
                rotation=90, va='bottom', ha='right',
                fontsize=8, alpha=0.7)
            
            # End frame label - left aligned after the line
            ax.text(region[1] + 1, y_pos,
                f"Frame {region[1]}",
                rotation=90, va='bottom', ha='left',
                fontsize=8, alpha=0.7)
            
            # Add error labels if available
            if posture_errors_for_all_error_region:
                try:
                    # Get errors for this specific error region
                    region_errors = posture_errors_for_all_error_region[error_region_index]
                    
                    # Format errors text
                    error_text = "Errors:\n" + "\n".join(region_errors) if region_errors else ""
                    
                    # Position text in middle of the error region
                    mid_frame = (region[0] + region[1]) // 2
                    ax.text(mid_frame, np.mean(ax.get_ylim()), 
                        error_text,
                        ha='center', va='center', 
                        fontsize=9, color='red', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                fc='white', ec='red', alpha=0.7))               
                except IndexError:
                    print(f"No error data for region {error_region_index}")

        # 3. Add weighted averages
        if hasattr(self, 'weighted_depth') and hasattr(self, 'weighted_rate'):
            ax.annotate(f"Weighted Averages:\nDepth: {self.weighted_depth:.1f}cm\nRate: {self.weighted_rate:.1f}cpm",
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black'))

        # 4. Configure legend and layout
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper right')

        plt.xlabel("Frame Number")
        plt.ylabel("Vertical Position (px)")
        plt.title("Complete CPR Analysis with Metrics")
        plt.grid(True)
        plt.tight_layout()
        plt.show()