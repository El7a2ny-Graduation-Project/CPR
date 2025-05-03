# metrics_calculator.py
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

class MetricsCalculator:
    """Rate and depth calculation from motion data with improved peak detection"""
    
    def __init__(self, shoulder_width_cm=45):
        self.shoulder_width_cm = shoulder_width_cm
        self.peaks = np.array([])
        self.peaks_max = np.array([])
        self.peaks_min = np.array([])
        self.y_smoothed = np.array([])
        self.cm_px_ratio = None
        self.midpoints_list = np.array([])
        self.shoulder_distances = []

    def smooth_midpoints(self, midpoints):
        """Apply Savitzky-Golay filter to smooth motion data"""
        self.midpoints_list = np.array(midpoints)

        # Dynamically set window_length as 5% of the number of points
        num_points = len(self.midpoints_list)
        window_length = max(3, int(num_points * 0.05) | 1)  # Ensure it's an odd number and at least 3
        
        if len(self.midpoints_list) > 5:  # Ensure enough data points
            try:
                self.y_smoothed = savgol_filter(
                    self.midpoints_list[:, 1], 
                    window_length=window_length, 
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
        """Improved peak detection with separate max/min peaks"""
        if self.y_smoothed.size == 0:
            print("No smoothed values found for peak detection")
            return False
            
        try:
            # Dynamically set distance as 5% of the number of points
            num_points = len(self.y_smoothed)
            distance = max(1, int(num_points * 0.05))  # 5% of points, minimum of 1
    
            self.peaks_max, _ = find_peaks(self.y_smoothed, distance=distance)
            self.peaks_min, _ = find_peaks(-self.y_smoothed, distance=distance)
            self.peaks = np.sort(np.concatenate((self.peaks_max, self.peaks_min)))
            return len(self.peaks) > 0
        except Exception as e:
            print(f"Peak detection error: {e}")
            return False

    def calculate_metrics(self, shoulder_distances, effective_fps):
        """Calculate compression metrics with improved calculations"""
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
            depth = None
            if len(self.peaks) > 1:
                depth = np.mean(np.abs(np.diff(self.y_smoothed[self.peaks]))) * self.cm_px_ratio

            # Rate calculation using only compression peaks (peaks_max)
            rate = None
            if len(self.peaks_max) > 1:
                rate = 1 / (np.mean(np.diff(self.peaks_max)) / effective_fps) * 60  # Convert to CPM

            return depth, rate
            
        except Exception as e:
            print(f"Metric calculation error: {e}")
            return None, None

    def plot_motion_curve(self, sample_rate):
        """Enhanced visualization with original and smoothed data"""
        if self.midpoints_list.size == 0:
            print("No midpoint data to plot")
            return
    
        plt.figure(figsize=(12, 6))
        
        # Scale x-axis to reflect original frame numbers
        x_original = np.arange(len(self.midpoints_list)) * sample_rate
        x_smoothed = np.arange(len(self.y_smoothed)) * sample_rate
    
        # Plot original and smoothed data
        plt.plot(x_original, 
                 self.midpoints_list[:, 1], 
                 label="Original Motion", 
                 color="red", 
                 linestyle="dashed", 
                 alpha=0.6)
                 
        plt.plot(x_smoothed, 
                 self.y_smoothed, 
                 label="Smoothed Motion", 
                 color="blue", 
                 linewidth=2)
    
        # Plot peaks if detected
        if self.peaks.size > 0:
            plt.plot(x_smoothed[self.peaks], 
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