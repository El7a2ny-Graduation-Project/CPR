import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from logging_config import cpr_logger

class GraphPlotter:
    """Class to plot graphs for various metrics"""

    def __init__(self):
        self.chunks_y_preprocessed = []
        self.chunks_peaks = []
        self.chunks_depth = []
        self.chunks_rate = []
        self.chunks_start_and_end_indices = []
        self.posture_warnings_regions = []
        self.sampling_interval_in_frames = 0
        self.fps = None  # Added FPS attribute

    def _assign_graph_data(self, chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, posture_warnings_regions, sampling_interval_in_frames, fps):
        """Assign data members for the class"""
        self.chunks_y_preprocessed = chunks_y_preprocessed
        self.chunks_peaks = chunks_peaks
        self.chunks_depth = chunks_depth
        self.chunks_rate = chunks_rate
        self.chunks_start_and_end_indices = chunks_start_and_end_indices
        self.posture_warnings_regions = posture_warnings_regions
        self.sampling_interval_in_frames = sampling_interval_in_frames
        self.fps = fps  # Store FPS

        cpr_logger.info(f"[Graph Plotter] Data members assigned with {len(self.chunks_start_and_end_indices)} chunks and {len(self.posture_warnings_regions)} error regions for a sampling interval of {self.sampling_interval_in_frames} frames and FPS {self.fps}")

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
            cpr_logger.info(f"[Graph Plotter] Adding chunk separator at {separator_x:.2f} seconds")
            ax.axvline(x=separator_x, color='orange', linestyle=':', linewidth=1.5)
        
        # Check if chunks are contiguous and need connection (frame-based logic)
        if (prev_chunk_end is not None and 
            start_frame == prev_chunk_end + self.sampling_interval_in_frames and
            prev_last_point is not None):
            
            # Convert connection points to seconds
            connect_start = prev_chunk_end / self.fps
            connect_end = start_frame / self.fps
            connect_times = [connect_start, connect_end]
            
            cpr_logger.info(f"[Graph Plotter] Connecting chunk {idx+1} to previous chunk (time {connect_start:.2f}-{connect_end:.2f}s)")
            ax.plot(connect_times, [prev_last_point['y_preprocessed'], y_preprocessed[0]], 
                    color="blue", linewidth=2)
        
        # Plot current chunk data
        cpr_logger.info(f"[Graph Plotter] Plotting chunk {idx+1} (time {chunk_times[0]:.2f}-{chunk_times[-1]:.2f}s)")
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
            cpr_logger.info(f"[Graph Plotter] Chunk {idx+1} metrics: {depth:.1f}cm depth, {rate:.1f}cpm rate")
            ax.annotate(f"Depth: {depth:.1f}cm\nRate: {rate:.1f}cpm",
                    xy=(mid_time, np.max(y_preprocessed)),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        return {'y_preprocessed': y_preprocessed[-1]}, end_frame

    def _plot_error_regions(self, ax, computed_error_regions):
        cpr_logger.info("[Graph Plotter] Rendering error regions:")
        for idx, (start_sec, end_sec) in enumerate(computed_error_regions):
            cpr_logger.info(f" - Region {idx+1}: {start_sec:.2f}s to {end_sec:.2f}s")
            ax.axvspan(start_sec, end_sec, color='gray', alpha=0.2, label='Posture Errors' if idx == 0 else "")
            
            # Annotate error text
            region_data = self.posture_warnings_regions[idx]
            if region_data['posture_warnings']:
                error_text = "Errors:\n" + "\n".join(region_data['posture_warnings'])
                mid_time = (start_sec + end_sec) / 2
                ax.text(mid_time, np.mean(ax.get_ylim()), error_text,
                        ha='center', va='center', fontsize=9, color='red', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.7))
        
        # Add separators between consecutive error regions
        sorted_regions = sorted(self.posture_warnings_regions, key=lambda x: x['start_frame'])
        for i in range(1, len(sorted_regions)):
            prev_er = sorted_regions[i-1]
            current_er = sorted_regions[i]
            separator_x = (prev_er['end_frame'] + 0.5) / self.fps
            cpr_logger.info(f"[Graph Plotter] Adding error region separator at {separator_x:.2f} seconds")
            ax.axvline(x=separator_x, color='orange', linestyle=':', linewidth=1.5)
    
    def _print_analysis_details(self, sorted_chunks):
        """Combined helper for printing chunks and error regions in seconds"""
        cpr_logger.info(f"\n\n=== CPR Chunk Analysis ===")
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
            
            cpr_logger.info(f"[Graph Plotter] Chunk {display_idx+1}: "
                f"Time {start_sec:.2f}s - {end_sec:.2f}s ({duration_sec:.2f}s), "
                f"Depth: {depth:.1f}cm, Rate: {rate:.1f}cpm")
            
            display_idx += 1

        cpr_logger.info(f"\n\n=== Error Region Analysis ===")
        
        for i, region in enumerate(self.posture_warnings_regions):  # Updated to match actual attribute name
            start_frame = region['start_frame']
            end_frame = region['end_frame']
            errors = region['posture_warnings']
            
            # Convert to seconds
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            error_str = ", ".join(errors) if errors else "No errors detected"
            
            cpr_logger.info(f"[Graph Plotter] Region {i+1}: "
                f"Time {start_sec:.2f}s - {end_sec:.2f}s - {error_str}")
        
        cpr_logger.info(f"\n\n")

    def plot_motion_curve_for_all_chunks(self, chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, posture_warnings_regions, sampling_interval_in_frames, fps):
        """Plot combined analysis with connected chunks and proper error regions"""
        
        self._assign_graph_data(chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, posture_warnings_regions, sampling_interval_in_frames, fps)
        cpr_logger.info("[Graph Plotter] Starting to plot motion curve for all chunks")
        
        # Create figure even if there's only error regions to plot
        plt.figure(figsize=(16, 8))
        ax = plt.gca()

        # Plot CPR chunks if they exist
        if self.chunks_start_and_end_indices:
            sorted_chunks = sorted(zip(self.chunks_start_and_end_indices, 
                                self.chunks_depth, 
                                self.chunks_rate),
                                key=lambda x: x[0][0])
            cpr_logger.info(f"[Graph Plotter] Processing {len(sorted_chunks)} CPR chunks")

            prev_last_point = None
            prev_chunk_end = None

            for idx, chunk in enumerate(sorted_chunks):
                cpr_logger.info(f"[Graph Plotter] Rendering chunk {idx+1}/{len(sorted_chunks)}")
                prev_last_point, prev_chunk_end = self._plot_single_chunk(ax, chunk, idx, prev_last_point, prev_chunk_end)
            
            self._print_analysis_details(sorted_chunks)
        else:
            cpr_logger.info("[Graph Plotter] No chunk data available for plotting")
            # Set reasonable default axis if only plotting errors
            ax.set_ylim(0, 100)  # Example default Y-axis range for position

        # Always plot error regions if they exist
        computed_error_regions = [(er['start_frame']/self.fps, er['end_frame']/self.fps) 
                                for er in self.posture_warnings_regions]
        self._plot_error_regions(ax, computed_error_regions)

        # Configure remaining elements regardless of chunks
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        if unique:  # Only add legend if there's content
            ax.legend(*zip(*unique), loc='upper right')

        plt.xlabel("Time (seconds)")
        plt.ylabel("Vertical Position (px)")
        plt.title("Complete CPR Analysis with Metrics")
        plt.grid(True)
        plt.tight_layout()
        cpr_logger.info(f"\n[Graph Plotter] Finalizing plot layout")
        plt.show()
        cpr_logger.info("[Graph Plotter] Plot display complete")