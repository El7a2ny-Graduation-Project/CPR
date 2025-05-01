import cv2
import numpy as np
import logging
from collections import defaultdict
from enum import Enum
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import math

#&---------------------------
from keypoints import COCOKeypoints

# Configure logging
logger = logging.getLogger("CPRMonitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

MIN_KEYPOINTS_COUNT = 10
RESCUER_CONFIDENCE_THRESHOLD = 0.65
MAX_PROXIMITY = 0.3

class CPRTracker:
    def __init__(self):
        self.pose_model = YOLO("yolo11n-pose.pt")
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.2,
            nn_budget=100,
            embedder="mobilenet",
            half=True
        )
        self.track_history = defaultdict(lambda: {
            "keypoints": [],
            "posture_scores": [],
            "positions": []
        })

    def calculate_posture_score(self, kps):
        try:
            left_shoulder = kps[COCOKeypoints.LEFT_SHOULDER.value]
            right_shoulder = kps[COCOKeypoints.RIGHT_SHOULDER.value]
            left_hip = kps[COCOKeypoints.LEFT_HIP.value]
            right_hip = kps[COCOKeypoints.RIGHT_HIP.value]

            shoulders = [(left_shoulder[0] + right_shoulder[0])/2,
                        (left_shoulder[1] + right_shoulder[1])/2]
            hips = [(left_hip[0] + right_hip[0])/2,
                   (left_hip[1] + right_hip[1])/2]
            
            horizontal_diff = abs(shoulders[0] - hips[0])
            vertical_diff = abs(shoulders[1] - hips[1])
            return min(1.0, 1 - (horizontal_diff / (vertical_diff + 1e-9)))
        except Exception as e:
            logger.warning(f"Posture calculation error: {str(e)}")
            return 0.0

    def update_tracks(self, frame):
        results = self.pose_model(frame, verbose=False, conf=0.5)[0]
        detections = self._format_detections(results)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        current_tracks = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            keypoints = self._get_keypoints_for_track(track, results)
            
            if keypoints is None:
                continue
                
            posture_score = self.calculate_posture_score(keypoints)
            
            current_tracks[track_id] = {
                "bbox": bbox,
                "keypoints": keypoints,
                "posture_score": posture_score,
                "age": track.time_since_update
            }
            
            # Update historical data (keep last 10 frames)
            self.track_history[track_id]["keypoints"].append(keypoints)
            self.track_history[track_id]["posture_scores"].append(posture_score)
            self.track_history[track_id]["positions"].append(bbox)
            if len(self.track_history[track_id]["posture_scores"]) > 10:
                self.track_history[track_id]["posture_scores"].pop(0)
                self.track_history[track_id]["keypoints"].pop(0)
                self.track_history[track_id]["positions"].pop(0)

        return current_tracks

    def _format_detections(self, results):
        return [
            (det.boxes.xyxy[0].cpu().numpy(),
             det.boxes.conf[0].cpu().item(),
             0) for det in results
        ]

    def _get_keypoints_for_track(self, track, results):
        if not results or not hasattr(track, 'detection_id'):
            return None
            
        detection_idx = track.detection_id
        if 0 <= detection_idx < len(results):
            return results[detection_idx].keypoints.xy[0].cpu().numpy()
        return None

class RoleClassifier:
    def __init__(self, frame_height):
        self.frame_height = frame_height
        self.rescuer_id = None

    def classify(self, tracks):
        valid_tracks = {
            tid: data for tid, data in tracks.items() 
            if len(data["keypoints"]) >= MIN_KEYPOINTS_COUNT
            and data["age"] < 5
        }
        
        if not valid_tracks:
            return None, None

        rescuer = self._select_rescuer(valid_tracks)
        patient = self._select_patient(rescuer["id"], valid_tracks) if rescuer else None

        return rescuer, patient

    def _select_rescuer(self, tracks):
        candidates = []
        for tid, data in tracks.items():
            posture_score = np.mean(data["posture_scores"])
            nose_y = data["keypoints"][COCOKeypoints.NOSE.value][1] / self.frame_height
            confidence = (posture_score * 0.6) + ((1 - nose_y) * 0.4)
            
            if confidence >= RESCUER_CONFIDENCE_THRESHOLD:
                candidates.append((tid, confidence, data))
        
        if not candidates:
            return None
            
        # Select candidate with highest confidence and sufficient history
        best = max(candidates, key=lambda x: (x[1], len(tracks[x[0]]["posture_scores"])))
        return {
            "id": best[0],
            "bbox": best[2]["bbox"],
            "keypoints": best[2]["keypoints"],
            "confidence": best[1]
        }

    def _select_patient(self, rescuer_id, tracks):
        rescuer_data = tracks[rescuer_id]
        rescuer_center = [
            (rescuer_data["bbox"][0] + rescuer_data["bbox"][2]) / 2,
            (rescuer_data["bbox"][1] + rescuer_data["bbox"][3]) / 2
        ]
        
        candidates = []
        for tid, data in tracks.items():
            if tid == rescuer_id:
                continue
                
            posture_score = np.mean(data["posture_scores"])
            bbox_center = [
                (data["bbox"][0] + data["bbox"][2]) / 2,
                (data["bbox"][1] + data["bbox"][3]) / 2
            ]
            
            distance = math.dist(rescuer_center, bbox_center) / self.frame_height
            patient_score = ((1 - posture_score) * 0.6) + ((1 - distance) * 0.4)
            
            if distance <= MAX_PROXIMITY and patient_score > 0.4:
                candidates.append((tid, patient_score, data))
        
        if not candidates:
            return None
            
        best = max(candidates, key=lambda x: x[1])
        return {
            "id": best[0],
            "bbox": best[2]["bbox"],
            "keypoints": best[2]["keypoints"],
            "confidence": best[1]
        }

def draw_tracking_results(frame, rescuer, patient):
    try:
        if rescuer:
            x1, y1, x2, y2 = map(int, rescuer["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Rescuer ({rescuer['confidence']:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        if patient:
            x1, y1, x2, y2 = map(int, patient["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Patient ({patient['confidence']:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    except Exception as e:
        logger.error(f"Drawing error: {str(e)}")

def main():
    video_path = r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\Keypoint Detection\Dataset\Crowds\vid2.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video source")
        return

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = CPRTracker()
    classifier = RoleClassifier(frame_height)

    cv2.namedWindow('CPR Monitoring', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break

            tracks = tracker.update_tracks(frame)
            rescuer, patient = classifier.classify(tracks)
            
            # Draw results directly on original frame
            draw_tracking_results(frame, rescuer, patient)
            
            # Display the frame
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('CPR Monitoring', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User initiated shutdown")
                break
                
    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()