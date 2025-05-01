import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import traceback
import os
import math

# Constants
MIN_KEYPOINTS_COUNT_FOR_VALID_CANDIDATE = 10

# Initialize YOLO Pose Estimation Model
pose_model = YOLO("yolo11n-pose.pt")

# Initialize DeepSORT Tracker
tracker = DeepSort(
    # Frames a track survives without detection.
    max_age=30, 
    # Detections needed to start a new track.         
    n_init=3,            
    # IoU threshold for motion-based matching.
    max_iou_distance=0.7, 
    # Threshold for appearance-based Re-ID.
    max_cosine_distance=0.2, 
    # Max stored appearance features per track.
    nn_budget=100,   
    # Overlap threshold for Non-Max Suppression.     
    nms_max_overlap=1.0 
)

# A default dictionary to store tracked persons so that when accessing a non-existing key, it initializes with a default value.
tracked_persons = defaultdict(lambda: {
    "age": 0, 
    # (x1, y1, x2, y2) format for bounding box
    "last_bbox": np.zeros((4,), dtype=np.float32),
    # (x, y) format for the 17 keypoints in COCO dataset
    "last_keypoints": np.zeros((17, 2), dtype=np.float32),
    # The velocity of the tracked person, initialized to zero
    "velocity": np.zeros((4,), dtype=np.float32)
})

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection area
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area != 0 else 0

def classify_roles(candidates, frame_height, rescuer_confidence_threshold=0.7, max_proximity=0.2):
    """
    Classify tracked persons into rescuer, patient, and passengers based on posture and spatial relationships.
    
    Args:
        candidates: List of candidate dictionaries with:
            - last_bbox: [x1, y1, x2, y2]
            - last_keypoints: List of keypoints
            - age: Track age
        frame_height: Height of video frame
        rescuer_confidence_threshold: Minimum confidence to consider as rescuer
        max_proximity: Max normalized distance (0-1) for patient-rescuer proximity
    
    Returns:
        (rescuer, patient, passengers) tuple
    """
    
    def calculate_posture_score(kps):
        """Calculate vertical posture score (0=horizontal, 1=vertical)"""
        try:
            shoulders = [(kps[5][0] + kps[6][0])/2, (kps[5][1] + kps[6][1])/2]
            hips = [(kps[11][0] + kps[12][0])/2, (kps[11][1] + kps[12][1])/2]
            return 1 - (abs(shoulders[0] - hips[0]) / (abs(shoulders[1] - hips[1]) + 1e-9))
        except (IndexError, KeyError, TypeError):
            return 0

    def bbox_center(bbox):
        """Get normalized bbox center coordinates"""
        return [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]

    # Filter valid candidates with keypoints and sufficient track age
    valid_candidates = [
        c for c in candidates
        if len(c.get("last_keypoints", [])) >= MIN_KEYPOINTS_COUNT_FOR_VALID_CANDIDATE  # Require COCO keypoints
        and c.get("age", 0) > 2  # Minimum 3-frame tracking history
    ]

    # Identify potential rescuers
    rescuer_candidates = []
    for candidate in valid_candidates:
        posture_score = calculate_posture_score(candidate["last_keypoints"])
        nose_position = candidate["last_keypoints"][0][1] / frame_height
        
        # Combined confidence score
        rescue_confidence = (posture_score * 0.6) + ((1 - nose_position) * 0.4)
        
        if rescue_confidence >= rescuer_confidence_threshold:
            rescuer_candidates.append((candidate, rescue_confidence))
    
    # Select best rescuer
    rescuer = None
    if rescuer_candidates:
        rescuer = max(rescuer_candidates, key=lambda x: (x[1], x[0]["age"]))[0]

    # Identify patient and passengers
    patient = None

    if rescuer:
        # Find most likely patient (closest horizontal person)
        rescuer_center = bbox_center(rescuer["last_bbox"])
        patient_candidates = []
        
        for candidate in valid_candidates:
            if candidate["id"] == rescuer["id"]:
                continue
                
            # Calculate horizontal posture score
            posture = 1 - calculate_posture_score(candidate["last_keypoints"])
            distance = math.dist(
                rescuer_center,
                bbox_center(candidate["last_bbox"])
            ) / frame_height
            
            if distance <= max_proximity and posture < 0.5:
                patient_score = (posture * 0.6) + ((1 - distance) * 0.4)
                patient_candidates.append((candidate, patient_score))
            
            # Print all the parameters responsible for the patient detection if the variable has a value
            print("------------------------------------------------------------------------------------------")
            print(f"Candidate ID: {candidate['id']}")
            print(f"Posture: {posture:.2f}, Distance: {distance:.2f}, Patient Score:")
            print("------------------------------------------------------------------------------------------")
        
        if patient_candidates:
            patient = max(patient_candidates, key=lambda x: x[1])[0]

    return rescuer, patient

def draw_tracking_results(frame, rescuer, patient):
    """Draw DeepSORT tracking results on the frame"""
    # Draw rescuer (A) with green box
    if rescuer is not None and rescuer["last_bbox"].size > 0:
        try:
            x1, y1, x2, y2 = map(int, rescuer["last_bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Rescuer A ({rescuer['age']}f)", (x1, y1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error drawing rescuer: {str(e)}")

    # Draw patient (B) with red box
    if patient is not None and patient["last_bbox"].size > 0:
        try:
            x1, y1, x2, y2 = map(int, patient["last_bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Patient B ({patient['age']}f)", (x1, y1-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error drawing patient: {str(e)}")

def process_frame(frame):

    #& Step 1: YOLO Pose Detection
    results = pose_model(frame, verbose=False, conf=0.5)[0]
    
    # Convert detections to DeepSort format which is a list of tuples like this one (xyxy_coords, confidence_score, class_id)
    detections_for_tracking = [
        (detection.boxes.xyxy[0].cpu().numpy(),
         detection.boxes.conf[0].cpu().item(),
         0) for detection in results
    ]

    #& Step 2: Update Tracker with YOLO detections
    if len(detections_for_tracking) == 0:
        print("No detections for tracking")
        return None, None
    
    tracklets = tracker.update_tracks(detections_for_tracking, frame=frame)

    #& Step 3: Update Tracked Persons with Keypoints and Motion Prediction
    current_ids = set()

    for tracklet in tracklets:
        # Confirm that the tracklet has been detected in more that n_init frames
        if not tracklet.is_confirmed():
            continue
        
        # Grab tracklet ID and bounding box
        track_id = tracklet.track_id
        current_bbox = tracklet.to_ltrb()
        
        # Get previous state of the tracked person (if this was the first detection, all values are initialized to as specified in the default dictionary)
        prev_data = tracked_persons[track_id]
        prev_bbox = prev_data["last_bbox"]
        prev_velocity = prev_data["velocity"]

        is_first_detection = np.all(prev_bbox == 0)

        # Motion prediction
        if not is_first_detection:
            # If this tracklet has been previously detected, use the information from the last detection 
            # to predict where is this person expected to be in the current frame
            predicted_bbox = prev_bbox + prev_velocity
        else:
            # If this is the first detection, we use the current bounding box as the predicted position (in other words, we are not predicting anything)
            predicted_bbox = current_bbox

        # Find best detection match using predicted position
        best_detection_match = None
        best_iou = 0.0

        for detection in results:
            det_bbox = detection.boxes.xyxy[0].cpu().numpy()
            
            # Calculate IoU with predicted position
            iou = calculate_iou(predicted_bbox, det_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_detection_match = detection

        # Update tracking data
        tracked_persons[track_id]["age"] += 1
        tracked_persons[track_id]["last_bbox"] = current_bbox
        
        # Calculate velocity based on actual movement
        if not is_first_detection:
            tracked_persons[track_id]["velocity"] = current_bbox - prev_bbox
            
        if best_detection_match:
            tracked_persons[track_id]["last_keypoints"] = best_detection_match.keypoints.xy[0].cpu().numpy()
            
        current_ids.add(track_id)

    # Age decay and cleanup
    for track_id in list(tracked_persons.keys()):
        if track_id not in current_ids:
            tracked_persons[track_id]["age"] = max(0, tracked_persons[track_id]["age"] - 1)

    # Step 4: Find Rescuer and Patient with lower age threshold
    candidates = [{"id": tid, **data} for tid, data in tracked_persons.items() if data["age"] > 2]

    rescuer, patient = classify_roles(
        candidates=candidates,
        frame_height=frame.shape[0],
        rescuer_confidence_threshold=0.65,
        max_proximity=0.5
    )

    return rescuer, patient

def main():
    file_path = r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\Keypoint Detection\Dataset\Crowds\vid3.mp4"  # Update with your path
    
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found at: {file_path}")

        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise IOError("Failed to open video source")

        # Get original video dimensions
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Original video dimensions: {orig_width}x{orig_height}")

        # Create resizable window
        cv2.namedWindow('CPR Monitoring', cv2.WINDOW_NORMAL)

        while True:
            print("Attempting to read frame...")
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            if not ret:
                print("Warning: Frame read failed - End of stream or hardware issue?")
                break

            try:
                # Process frame and get modified frame with YOLO drawings
                rescuer, patient = process_frame(frame)
                
                # Draw tracking results on top of YOLO drawings
                draw_tracking_results(frame, rescuer, patient)
                
            except Exception as e:
                print(f"Error in frame processing: {str(e)}")
                continue

            # Calculate display scaling
            h, w = frame.shape[:2]
            scale = min(1280/w, 720/h)  # 1280x720 max display size
            new_dim = (int(w * scale), int(h * scale))
            
            # Resize frame with annotations
            display_frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
            
            # Update window size
            cv2.resizeWindow('CPR Monitoring', new_dim[0], new_dim[1])
            
            # Display resized frame
            cv2.imshow('CPR Monitoring', display_frame)

            # Exit handling (keep your existing code)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User terminated execution with Q key")
                break

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
