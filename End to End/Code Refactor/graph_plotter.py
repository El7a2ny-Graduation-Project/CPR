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

    def _assign_graph_data(self, chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, error_regions, sampling_interval_in_frames):
        """Assign data members for the class"""
        self.chunks_y_preprocessed = chunks_y_preprocessed
        self.chunks_peaks = chunks_peaks
        self.chunks_depth = chunks_depth
        self.chunks_rate = chunks_rate
        self.chunks_start_and_end_indices = chunks_start_and_end_indices
        self.error_regions = error_regions
        self.sampling_interval_in_frames = sampling_interval_in_frames

        print(f"[Graph Plotter] Data members assigned with {len(self.chunks_start_and_end_indices)} chunks and {len(self.error_regions)} error regions for a sampling interval of {self.sampling_interval_in_frames} frames")

    def plot_motion_curve_for_all_chunks(self, chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, error_regions, sampling_interval_in_frames):
        """Plot combined analysis with connected chunks and proper error regions"""
        
        self._assign_graph_data(chunks_y_preprocessed, chunks_peaks, chunks_depth, chunks_rate, chunks_start_and_end_indices, error_regions, sampling_interval_in_frames)
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
        
        # Convert error regions to frame tuples for plotting
        computed_error_regions = [(er['start_frame'], er['end_frame']) for er in self.error_regions]
        print(f"[Graph Plotter] Received {len(self.error_regions)} error regions")
        
        self._print_analysis_details(sorted_chunks)
        self._plot_error_regions(ax, computed_error_regions)
        
        # Configure legend and layout
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='upper right')

        plt.xlabel("Frame Number")
        plt.ylabel("Vertical Position (px)")
        plt.title("Complete CPR Analysis with Metrics")
        plt.grid(True)
        plt.tight_layout()
        print("[Graph Plotter] Finalizing plot layout")
        plt.show()
        print("[Graph Plotter] Plot display complete")

    def _plot_single_chunk(self, ax, chunk, idx, prev_last_point, prev_chunk_end):
        (start, end), depth, rate = chunk
        chunk_frames = np.arange(start, end + 1, self.sampling_interval_in_frames)
        y_preprocessed = self.chunks_y_preprocessed[idx]
        peaks = self.chunks_peaks[idx]
        
        # Add separator line between chunks
        if prev_chunk_end is not None:
            separator_x = prev_chunk_end + 0.5
            print(f"[Graph Plotter] Adding chunk separator at frame {separator_x}")
            ax.axvline(x=separator_x, color='orange', linestyle=':', linewidth=1.5)
        
        # Check if chunks are contiguous and need connection
        if (prev_chunk_end is not None and 
            start == prev_chunk_end + self.sampling_interval_in_frames and
            prev_last_point is not None):
            
            connect_frames = [prev_chunk_end, start]
            connect_y_preprocessed = np.vstack([prev_last_point['y_preprocessed'], y_preprocessed[0]])
            
            print(f"[Graph Plotter] Connecting chunk {idx+1} to previous chunk (frames {prev_chunk_end}-{start})")
            ax.plot(connect_frames, connect_y_preprocessed, 
                    color="blue", linewidth=2)
        
        # Plot current chunk data
        print(f"[Graph Plotter] Plotting chunk {idx+1} (frames {start}-{end}) with {len(peaks)} peaks")
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
            print(f"[Graph Plotter] Chunk {idx+1} metrics: {depth:.1f}cm depth, {rate:.1f}cpm rate")
            ax.annotate(f"Depth: {depth:.1f}cm\nRate: {rate:.1f}cpm",
                    xy=(mid_frame, np.max(y_preprocessed)),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        return {'y_preprocessed': y_preprocessed[-1]}, end

    def _plot_error_regions(self, ax, computed_error_regions):
        print("[Graph Plotter] Rendering error regions:")
        for idx, (start, end) in enumerate(computed_error_regions):
            print(f" - Region {idx+1}: frames {start}-{end}")
            ax.axvspan(start, end, color='gray', alpha=0.2, label='Posture Errors' if idx == 0 else "")
            
            # Get corresponding errors from original self.error_regions list
            region_data = self.error_regions[idx]
            if region_data['errors']:
                error_text = "Errors:\n" + "\n".join(region_data['errors'])
                mid_frame = (start + end) // 2
                ax.text(mid_frame, np.mean(ax.get_ylim()), error_text,
                        ha='center', va='center', fontsize=9, color='red', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.7))

    def _print_analysis_details(self, sorted_chunks):
        """Combined helper for printing chunks and error regions"""
        print("[Graph Plotter]\n=== CPR Chunk Analysis ===")
        for idx, ((start, end), depth, rate) in enumerate(sorted_chunks):
            duration = end - start + 1
            print(f"[Graph Plotter] Chunk {idx+1}: "
                  f"Frames {start}-{end} ({duration} frames), "
                  f"Depth: {depth:.1f}cm, Rate: {rate:.1f}cpm")

        print("\n[Graph Plotter] === Error Region Analysis ===")
        for i, region in enumerate(self.error_regions):
            start = region['start_frame']
            end = region['end_frame']
            errors = region['errors']
            error_str = ", ".join(errors) if errors else "No errors detected"
            print(f"[Graph Plotter] Region {i+1}: Frames {start}-{end} - {error_str}")
