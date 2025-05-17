import cv2
import numpy as np
import os
import sys

from logging_config import cpr_logger

class WarningsOverlayer:
    def __init__(self):
        self.WARNING_CONFIG = {
            # Posture Warnings
            "Right arm bent!": {
                "color": (52, 110, 235),
                "position": (50, 150)  # Top position
            },
            "Left arm bent!": {
                "color": (52, 110, 235),
                "position": (50, 200)
            },
            "One-handed CPR detected!": {
                "color": (27, 150, 70),
                "position": (50, 250)
            },
            "Hands not on chest!": {
                "color": (161, 127, 18),
                "position": (50, 300)
            },
            
            # Rate/Depth Warnings
            "Depth too low!": {
                "color": (125, 52, 235),
                "position": (50, 350)  # Bottom position
            },
            "Depth too high!": {
                "color": (125, 52, 235),
                "position": (50, 350)
            },
            "Rate too slow!": {
                "color": (235, 52, 214),
                "position": (50, 350)
            },
            "Rate too fast!": {
                "color": (235, 52, 214),
                "position": (50, 350)
            }
        }
    
    def add_warnings_to_processed_video(self, output_video_path, sampling_interval_frames, rate_and_depth_warnings, posture_warnings):
        """Process both warning types with identical handling"""
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

        # Combine all warnings into unified list
        all_warnings = []
        
        # Process posture warnings
        for entry in posture_warnings:
            if warnings := entry.get('posture_warnings'):
                start = entry['start_frame'] // sampling_interval_frames
                end = entry['end_frame'] // sampling_interval_frames
                all_warnings.append((int(start), int(end), warnings))
        
        # Process rate/depth warnings
        for entry in rate_and_depth_warnings:
            if warnings := entry.get('rate_and_depth_warnings'):
                start = entry['start_frame'] // sampling_interval_frames
                end = entry['end_frame'] // sampling_interval_frames
                all_warnings.append((int(start), int(end), warnings))

        # Video processing loop
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Check active warnings for current frame
            active_warnings = []
            for start, end, warnings in all_warnings:
                if start <= frame_idx <= end:
                    active_warnings.extend(warnings)
            
            # Draw all warnings using unified config
            self._draw_warnings(frame, active_warnings)
            
            writer.write(frame)
            frame_idx += 1
        
        cap.release()
        writer.release()
        cpr_logger.info(f"\n[POST-PROCESS] Final output saved to: {final_path}")

    def _draw_warnings(self, frame, active_warnings):
        """Draw warnings using unified configuration"""
        drawn_positions = set()  # Prevent overlapping
        
        for warning_text in active_warnings:
            if config := self.WARNING_CONFIG.get(warning_text):
                x, y = config['position']
                
                # Auto-stack if position occupied
                while (x, y) in drawn_positions:
                    y += 50  # Move down by 50px
                
                self._draw_warning_banner(
                    frame=frame,
                    text=warning_text,
                    color=config['color'],
                    position=(x, y)
                )
                drawn_positions.add((x, y))
    
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
   