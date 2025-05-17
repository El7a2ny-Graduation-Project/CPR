import cv2
import numpy as np
import os
from logging_config import cpr_logger

class WarningsOverlayer:
    def __init__(self):
        pass
    
    def _draw_warning_banner(self, frame, text, color, position):
            """Base drawing function for warning banners"""
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            x, y = position
            # Background rectangle
            cv2.rectangle(frame, 
                        (x - 10, y - text_height - 10),
                        (x + text_width + 10, y + 10),
                        color, -1)
            # Text
            cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
    def draw_rate_and_depth_warnings(self, frame, warnings):
        """Draw stacked warning messages on left side"""
        WARNING_CONFIG = {
            'Depth too low!': (125, 52, 235),             
            'Depth too high!': (125, 52, 235),
            'Rate too slow!': (235, 52, 214),
            'Rate too fast!': (235, 52, 214),      
            }
        
        y_start = 250 # After the posture error messages

        for idx, warning in enumerate(warnings):
            color = WARNING_CONFIG.get(warning, (255, 255, 255))
            self._draw_warning_banner(
                frame=frame,
                text=warning,
                color=color,
                position=(50, y_start + idx*50)  # Stack vertically
            )

    def add_warnings_to_processed_video(self, output_video_path, sampling_interval_frames, rate_and_depth_warnings):
        """Post-process video with compatible encoding"""
        cpr_logger.info("\n[POST-PROCESS] Starting warning overlay")

        # Read processed video with original parameters
        cap = cv2.VideoCapture(output_video_path)
        if not cap.isOpened():
            cpr_logger.info("[ERROR] Failed to open processed video")
            return

        # Get original video properties
        original_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        processed_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create final writer with ORIGINAL codec and parameters
        base = os.path.splitext(output_video_path)[0]
        final_path = os.path.abspath(f"{base}_final.mp4")
        writer = cv2.VideoWriter(final_path, original_fourcc, processed_fps, (width, height))
        
        warning_periods = []
        for warning_entry in rate_and_depth_warnings:
            warnings = warning_entry['rate_and_depth_warnings']
            if not warnings:
                continue
                
            # Convert original frames to processed frames
            start_processed = warning_entry['start_frame'] // sampling_interval_frames
            end_processed = warning_entry['end_frame'] // sampling_interval_frames
            
            warning_periods.append((
                int(start_processed),
                int(end_processed),
                warnings
            ))

        # Process frames with original timing
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check active warnings using video FPS
            current_time = frame_idx / processed_fps
            active_warnings = []
            for start, end, warnings in warning_periods:
                if start <= frame_idx <= end:
                    active_warnings.extend(warnings)

            if active_warnings:
                self.draw_rate_and_depth_warnings(frame, active_warnings)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        cpr_logger.info(f"\n[POST-PROCESS] Final output saved to: {final_path}")
