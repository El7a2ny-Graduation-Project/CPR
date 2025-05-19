import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from logging_config import cpr_logger
from matplotlib.ticker import MultipleLocator
import os

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
        self.fps = None  

        self.error_symbols = {
            "Right arm bent!": ('o', '#A61D1D'), # circle
            "Left arm bent!": ('s', '#A61D1D'), # square
            "Left hand not on chest!": ('P', '#A61D1D'), # plus
            "Right hand not on chest!": ('*', '#A61D1D'), # star
            "Both hands not on chest!": ('D', '#A61D1D') # diamond
        }

        self.annotation_y_level = None  # Will store our target y-position

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
                color="#2F5597", linewidth=2.5)
        
        # Plot current chunk data
        cpr_logger.info(f"[Graph Plotter] Plotting chunk {idx+1} (time {chunk_times[0]:.2f}-{chunk_times[-1]:.2f}s)")
        smooth_label = "Motion" if idx == 0 else ""
        peaks_label = "Peaks" if idx == 0 else ""

        # Updated motion plot
        ax.plot(chunk_times, y_preprocessed,
                color="#2F5597", linewidth=2.5,
                marker='o', markersize=4,
                markerfacecolor='#2F5597', markeredgecolor='#2F5597',
                label=smooth_label)
        
        # Updated peaks
        if peaks.size > 0:
            ax.plot(chunk_times[peaks], y_preprocessed[peaks],
                    "x", color="#ED7D31", markersize=8,
                    label=peaks_label)

       # Annotate chunk metrics (time-based)
        if (depth is not None and rate is not None) and (depth > 0 and rate > 0):
            mid_time = (start_frame + end_frame) / (2 * self.fps)
            cpr_logger.info(f"[Graph Plotter] Chunk {idx+1} metrics: {depth:.1f}cm depth, {rate:.1f}cpm rate")
            
            # Calculate or use stored annotation y-level
            if self.annotation_y_level is None:
                # For first chunk, calculate midpoint between min and max of y_preprocessed
                y_range = np.max(y_preprocessed) - np.min(y_preprocessed)
                self.annotation_y_level = np.min(y_preprocessed) + y_range * 0.5  # 70% up from bottom
                cpr_logger.info(f"[Graph Plotter] Setting annotation y-level to {self.annotation_y_level:.2f}")
            
            # Updated annotation box using consistent y-level
            ax.annotate(f"Depth: {depth:.1f}cm\nRate: {rate:.1f}cpm",
                    xy=(mid_time, self.annotation_y_level),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', 
                            fc='#F2F2F2', ec='#595959', alpha=0.8))
        
        return {'y_preprocessed': y_preprocessed[-1]}, end_frame

    def _plot_error_regions(self, ax, computed_error_regions):
        """Visualize error regions with adaptive symbol sizing"""
        cpr_logger.info("[Graph Plotter] Rendering error regions:")

        # Size parameters
        target_width_ratio = 0.7  # Max 80% of region width
        legend_size = 80  # Fixed legend symbol size (points²)
        
        legend_handles = []
        y_mid = np.mean(ax.get_ylim())
        
        # Get figure dimensions for size conversion
        fig = ax.figure
        fig_width_points = fig.get_figwidth() * fig.dpi
        x_min, x_max = ax.get_xlim()
        data_range = x_max - x_min
        points_per_second = fig_width_points / data_range

        for idx, (start_sec, end_sec) in enumerate(computed_error_regions):
            region_width = end_sec - start_sec
            region_data = self.posture_warnings_regions[idx]
            warnings = region_data.get('posture_warnings', [])

            # Calculate max allowed width in data units (seconds)
            max_data_width = region_width * target_width_ratio
            
            # Convert legend size to data units
            legend_data_width = (np.sqrt(legend_size) / points_per_second)
            
            # Determine final symbol width (data units)
            symbol_data_width = min(legend_data_width, max_data_width)
            
            # Convert back to points² for matplotlib
            symbol_point_width = symbol_data_width * points_per_second
            symbol_size = symbol_point_width ** 2

            for error in warnings:
                if error in self.error_symbols:
                    marker, color = self.error_symbols[error]
                    
                    ax.scatter(
                        x=(start_sec + end_sec)/2,
                        y=y_mid,
                        s=symbol_size,
                        marker=marker,
                        color=color,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=5
                    )

                    # Create legend entry once
                    if not any(error == h.get_label() for h in legend_handles):
                        legend_handles.append(
                            ax.scatter([], [], 
                                    s=legend_size,
                                    marker=marker,
                                    color=color,
                                    edgecolors='black',
                                    linewidths=0.5,
                                    alpha=0.7,
                                    label=error)
                        )

            # Updated error region fill
            ax.axvspan(start_sec, end_sec, 
                    color='#FCE4D6', alpha=0.3, zorder=1)

        if not ax.get_xlabel():
            ax.set_xlabel("Time (seconds)", fontsize=10)
        if not ax.get_ylabel():
            ax.set_ylabel("Signal Value", fontsize=10)

        return legend_handles

    def plot_motion_curve_for_all_chunks(self, chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, posture_warnings_regions, sampling_interval_in_frames, fps, plot_output_path):
        """Plot combined analysis with connected chunks and proper error regions"""
        
        self._assign_graph_data(chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, posture_warnings_regions, sampling_interval_in_frames, fps)
        cpr_logger.info("[Graph Plotter] Starting to plot motion curve for all chunks")
        
        # Create figure even if there's only error regions to plot
        plt.figure(figsize=(16, 8))
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(5))

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
        
        # In the "Configure remaining elements" section (replace existing legend code):
        handles, labels = ax.get_legend_handles_labels()

        # Collect error handles from _plot_error_regions (modified to return them)
        error_legend_handles = self._plot_error_regions(ax, computed_error_regions)

        # Merge both sets of handles/labels
        if error_legend_handles:
            handles += error_legend_handles
            labels += [h.get_label() for h in error_legend_handles]

        # Remove duplicate labels
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

        # Create single horizontal legend at bottom
        if unique:
           ax.legend(
            *zip(*unique),
            loc='upper center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(unique),
            fontsize=8,
            handletextpad=0.3,
            columnspacing=1.5,
            framealpha=0.9,
            borderpad=0.7
        )
        plt.tight_layout(rect=[0, 0.025, 1, 1])

        plt.xlabel("Time (seconds)")
        plt.ylabel("Vertical Position (px)")
        plt.title("Complete CPR Analysis with Metrics", pad=20)  # Added pad parameter
        
        plt.grid(True)
        cpr_logger.info(f"\n[Graph Plotter] Finalizing plot layout")
        
        # Adjust tight_layout with additional padding
        plt.tight_layout(rect=[0, 0.025, 1, 0.95])  # Reduced top from 1 to 0.95 to make space

        if plot_output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
            plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
            cpr_logger.info(f"[Graph Plotter] Plot saved to {plot_output_path}")
        
        plt.show()
        cpr_logger.info("[Graph Plotter] Plot display complete")
       
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
