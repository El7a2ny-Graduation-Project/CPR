import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

class GraphPlotter:
    """Class to plot graphs for various metrics"""

    def __init__(self):
        self.chunks_y_preprocessed = []
        self.chunks_peaks = []
        self.chunks_depth = []
        self.chunks_rate = []
        self.chunks_start_and_end_indices = []
        self.error_regions = []
        self.sampling_interval_in_frames = 0
        self.fps = None  # Added FPS attribute

    def _assign_graph_data(self, chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, error_regions, sampling_interval_in_frames, fps):
        """Assign data members for the class"""
        self.chunks_y_preprocessed = chunks_y_preprocessed
        self.chunks_peaks = chunks_peaks
        self.chunks_depth = chunks_depth
        self.chunks_rate = chunks_rate
        self.chunks_start_and_end_indices = chunks_start_and_end_indices
        self.error_regions = error_regions
        self.sampling_interval_in_frames = sampling_interval_in_frames
        self.fps = fps  # Store FPS

        print(f"[Graph Plotter] Data members assigned with {len(self.chunks_start_and_end_indices)} chunks and {len(self.error_regions)} error regions for a sampling interval of {self.sampling_interval_in_frames} frames and FPS {self.fps}")

    def plot_motion_curve_for_all_chunks(self, chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, error_regions, sampling_interval_in_frames, fps):
        """Plot combined analysis with connected chunks and proper error regions"""
        
        self._assign_graph_data(chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, error_regions, sampling_interval_in_frames, fps)
        print("[Graph Plotter] Starting to plot motion curve for all chunks")
        
        if not self.chunks_start_and_end_indices:
            print("[Graph Plotter] No chunk data available for plotting")
            return

        plt.figure(figsize=(16, 8))
        ax = plt.gca()
        
        # Sort chunks chronologically
        sorted_chunks = sorted(zip(self.chunks_start_and_end_indices, 
                                 self.chunks_depth, 
                                 self.chunks_rate),
                                 key=lambda x: x[0][0])
        print(f"[Graph Plotter] Processing {len(sorted_chunks)} CPR chunks")

        # Track previous chunk's last point and end frame
        prev_last_point = None
        prev_chunk_end = None

        # Plot each chunk and handle connections
        for idx, chunk in enumerate(sorted_chunks):
            print(f"[Graph Plotter] Rendering chunk {idx+1}/{len(sorted_chunks)}")
            prev_last_point, prev_chunk_end = self._plot_single_chunk(ax, chunk, idx, prev_last_point, prev_chunk_end)
        
        # Convert error regions to time tuples for plotting
        computed_error_regions = [(er['start_frame']/self.fps, er['end_frame']/self.fps) for er in self.error_regions]
        print(f"[Graph Plotter] Received {len(self.error_regions)} error regions")
        
        self._print_analysis_details(sorted_chunks)
        self._plot_error_regions(ax, computed_error_regions)
        
        # Configure legend and layout
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper right')

        plt.xlabel("Time (seconds)")  # Updated label
        plt.ylabel("Vertical Position (px)")
        plt.title("Complete CPR Analysis with Metrics")
        plt.grid(True)
        plt.tight_layout()
        print(f"\n[Graph Plotter] Finalizing plot layout")
        plt.show()
        print("[Graph Plotter] Plot display complete")

    def _plot_single_chunk(self, ax, chunk, idx, prev_last_point, prev_chunk_end):
        (start_frame, end_frame), depth, rate = chunk
        # Convert frames to time
        chunk_frames = np.arange(start_frame, end_frame + 1, self.sampling_interval_in_frames)
        chunk_times = chunk_frames / self.fps  # Convert to seconds
        y_preprocessed = self.chunks_y_preprocessed[idx]
        peaks = self.chunks_peaks[idx]
        
        # Add separator line between chunks (in seconds)
        if prev_chunk_end is not None:
            separator_x = (prev_chunk_end + 0.5) / self.fps
            print(f"[Graph Plotter] Adding chunk separator at {separator_x:.2f} seconds")
            ax.axvline(x=separator_x, color='orange', linestyle=':', linewidth=1.5)
        
        # Check if chunks are contiguous and need connection (frame-based logic)
        if (prev_chunk_end is not None and 
            start_frame == prev_chunk_end + self.sampling_interval_in_frames and
            prev_last_point is not None):
            
            # Convert connection points to seconds
            connect_start = prev_chunk_end / self.fps
            connect_end = start_frame / self.fps
            connect_times = [connect_start, connect_end]
            
            print(f"[Graph Plotter] Connecting chunk {idx+1} to previous chunk (time {connect_start:.2f}-{connect_end:.2f}s)")
            ax.plot(connect_times, [prev_last_point['y_preprocessed'], y_preprocessed[0]], 
                    color="blue", linewidth=2)
        
        # Plot current chunk data
        print(f"[Graph Plotter] Plotting chunk {idx+1} (time {chunk_times[0]:.2f}-{chunk_times[-1]:.2f}s)")
        smooth_label = "Motion" if idx == 0 else ""
        peaks_label = "Peaks" if idx == 0 else ""

        ax.plot(chunk_times, y_preprocessed,
                color="blue", linewidth=2,
                marker='o', markersize=4,
                markerfacecolor='blue', markeredgecolor='blue',
                label=smooth_label)
        
        # Plot peaks
        if peaks.size > 0:
            ax.plot(chunk_times[peaks], y_preprocessed[peaks],
                    "x", color="green", markersize=8,
                    label=peaks_label)

        # Annotate chunk metrics (time-based)
        if (depth is not None and rate is not None) and (depth > 0 and rate > 0):
            mid_time = (start_frame + end_frame) / (2 * self.fps)
            print(f"[Graph Plotter] Chunk {idx+1} metrics: {depth:.1f}cm depth, {rate:.1f}cpm rate")
            ax.annotate(f"Depth: {depth:.1f}cm\nRate: {rate:.1f}cpm",
                    xy=(mid_time, np.max(y_preprocessed)),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        return {'y_preprocessed': y_preprocessed[-1]}, end_frame

    def _plot_error_regions(self, ax, computed_error_regions):
        print("[Graph Plotter] Rendering error regions:")
        for idx, (start_sec, end_sec) in enumerate(computed_error_regions):
            print(f" - Region {idx+1}: {start_sec:.2f}s to {end_sec:.2f}s")
            ax.axvspan(start_sec, end_sec, color='gray', alpha=0.2, label='Posture Errors' if idx == 0 else "")
            
            # Annotate error text
            region_data = self.error_regions[idx]
            if region_data['errors']:
                error_text = "Errors:\n" + "\n".join(region_data['errors'])
                mid_time = (start_sec + end_sec) / 2
                ax.text(mid_time, np.mean(ax.get_ylim()), error_text,
                        ha='center', va='center', fontsize=9, color='red', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.7))
    
    def _print_analysis_details(self, sorted_chunks):
        """Combined helper for printing chunks and error regions in seconds"""
        print(f"\n\n=== CPR Chunk Analysis ===")
        display_idx = 0  # Separate counter for displayed indices
        
        # Convert frame numbers to seconds using video FPS
        fps = self.fps  # Get FPS from class instance
        
        for ((start_frame, end_frame), depth, rate) in sorted_chunks:
            # Skip chunks with both values at 0
            if depth == 0 and rate == 0:
                continue
                
            # Convert frames to seconds
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            duration_sec = (end_frame - start_frame + 1) / fps  # +1 to include both endpoints
            
            print(f"[Graph Plotter] Chunk {display_idx+1}: "
                f"Time {start_sec:.2f}s - {end_sec:.2f}s ({duration_sec:.2f}s), "
                f"Depth: {depth:.1f}cm, Rate: {rate:.1f}cpm")
            
            display_idx += 1

        print(f"\n\n=== Error Region Analysis ===")
        for i, region in enumerate(self.error_regions):  # Updated to match actual attribute name
            start_frame = region['start_frame']
            end_frame = region['end_frame']
            errors = region['errors']
            
            # Convert to seconds
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            error_str = ", ".join(errors) if errors else "No errors detected"
            
            print(f"[Graph Plotter] Region {i+1}: "
                f"Time {start_sec:.2f}s - {end_sec:.2f}s - {error_str}")
        
        print(f"\n\n")