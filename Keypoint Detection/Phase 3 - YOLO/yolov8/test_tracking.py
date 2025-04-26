import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import traceback
import os

# Initialize YOLOv8 Pose Estimation Model
pose_model = YOLO("yolo11n-pose.pt")  # Official Ultralytics model

# Optimized DeepSORT Tracker Parameters
tracker = DeepSort(
    max_age=15,          # Increased from 5 (keep tracks longer through occlusions)
    n_init=2,            # Reduced from 3 (confirm tracks faster)
    max_iou_distance=0.8, # Increased from 0.7 (tolerate larger position changes)
    max_cosine_distance=0.3, # Tighter appearance matching (prevents ID switches)
    nn_budget=50,        # Add appearance feature budget
    nms_max_overlap=1.0  # Allow full overlap between tracks
)

# Tracked persons storage - initialize with empty arrays instead of None
tracked_persons = defaultdict(lambda: {
	"age": 0, 
	"last_bbox": np.empty((4,), dtype=np.float32),  # 4 elements for bbox
	"last_keypoints": np.empty((17, 2), dtype=np.float32)  # 17 keypoints
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

def is_rescuer(candidate, frame_height):
	"""Identify rescuer using posture analysis and keypoint positions."""
	# Check for valid data presence
	if (candidate["last_bbox"] is None or 
		candidate["last_keypoints"] is None or
		len(candidate["last_bbox"]) == 0 or
		len(candidate["last_keypoints"]) == 0):
		return False

	# Convert numpy arrays to lists for consistent handling
	bbox = candidate["last_bbox"].tolist() if isinstance(candidate["last_bbox"], np.ndarray) else candidate["last_bbox"]
	keypoints = candidate["last_keypoints"].tolist() if isinstance(candidate["last_keypoints"], np.ndarray) else candidate["last_keypoints"]

	# Bounding box analysis
	x1, y1, x2, y2 = bbox
	width = x2 - x1
	height = y2 - y1
	aspect_ratio = width / height

	# Keypoint analysis - ensure valid keypoint structure
	if len(keypoints) < 1:
		return False
	nose_y = keypoints[0][1]  # COCO nose index=0

	# Rescuer criteria (vertical posture + high head position)
	return aspect_ratio < 0.65 and nose_y < frame_height * 0.3

def process_frame(frame):
	frame_height = frame.shape[0]

	# Step 1: YOLO Pose Detection
	results = pose_model(frame, verbose=False, conf=0.7)[0]
	
	# Convert detections to DeepSort format: [(xyxy, confidence, class_id), ...]
	detections_for_tracking = [
		(detection.boxes.xyxy[0].cpu().numpy(),  # Bounding box in xyxy format
		 detection.boxes.conf[0].cpu().item(),    # Confidence score
		 0)                                       # Class ID (0=person)
		for detection in results
	]

	# Step 2: Update Tracker with new API
	tracklets = tracker.update_tracks(detections_for_tracking, frame=frame)

	# Step 3: Update Tracked Persons with Keypoints
	current_ids = set()
	for tracklet in tracklets:
		if not tracklet.is_confirmed():
			continue

		track_id = tracklet.track_id
		tracklet_bbox = tracklet.to_ltrb()  # Use to_ltrb() instead of to_tlbr()

		# Find matching detection with highest IoU
		best_match = None
		best_iou = 0.0
		for detection in results:
			det_bbox = detection.boxes.xyxy[0].cpu().numpy().tolist()
			iou = calculate_iou(tracklet_bbox, det_bbox)
			if iou > best_iou:
				best_iou = iou
				best_match = detection

		# Update tracked person data
		tracked_persons[track_id]["age"] += 1
		tracked_persons[track_id]["last_bbox"] = tracklet_bbox
		if best_match:
			tracked_persons[track_id]["last_keypoints"] = best_match.keypoints.xy[0].cpu().numpy().tolist()
		current_ids.add(track_id)

	# Age decay for non-detected persons
	for track_id in list(tracked_persons.keys()):
		if track_id not in current_ids:
			tracked_persons[track_id]["age"] = max(0, tracked_persons[track_id]["age"] - 1)

	# Step 4: Find Rescuer and Patient (remainder unchanged)
	rescuer = None
	patient = None
	candidates = [{"id": tid, **data} for tid, data in tracked_persons.items() if data["age"] > 2]

	for candidate in candidates:
		if is_rescuer(candidate, frame_height):
			if not rescuer or candidate["age"] > rescuer["age"]:
				rescuer = candidate
		else:
			if not patient or candidate["age"] > patient["age"]:
				patient = candidate

	return rescuer, patient

def draw_annotations(frame, rescuer, patient):
	# Draw rescuer (A) with green box
	if rescuer is not None and rescuer["last_bbox"].size > 0:
		try:
			x1, y1, x2, y2 = map(int, rescuer["last_bbox"])
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(frame, "Rescuer A", (x1, y1-10),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
		except Exception as e:
			print(f"Error drawing rescuer: {str(e)}")

	# Draw patient (B) with red box
	if patient is not None and patient["last_bbox"].size > 0:
		try:
			x1, y1, x2, y2 = map(int, patient["last_bbox"])
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.putText(frame, "Patient B", (x1, y1-10),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
		except Exception as e:
			print(f"Error drawing patient: {str(e)}")
	
	# Add temporary boxes for new tracks
	if patient is None:
		for tid, data in tracked_persons.items():
			if data["age"] <= 1 and data["last_bbox"].size > 0:
				x1, y1, x2, y2 = map(int, data["last_bbox"])
				cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
				cv2.putText(frame, f"Unconfirmed {tid}", (x1, y1-30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

	return frame

def main():
	file_path = r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\Keypoint Detection\Dataset\Tracking\video_1.mp4"
	
	try:
		# Verify file exists before opening
		if not os.path.exists(file_path):
			raise FileNotFoundError(f"Video file not found at: {file_path}")

		cap = cv2.VideoCapture(file_path)
		
		# Check if video opened successfully
		if not cap.isOpened():
			raise IOError("Failed to open video source. Check file format/codec support.")
			
		print(f"Successfully opened video source. Frame dimensions: {int(cap.get(4))}x{int(cap.get(3))}")

		while True:
			print("Attempting to read frame...")
			ret, frame = cap.read()
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			
			if not ret:
				print("Warning: Frame read failed - End of stream or hardware issue?")
				break

			try:
				print("Processing frame...")
				rescuer, patient = process_frame(frame)
			except Exception as e:
				print(f"Error in frame processing: {str(e)}")
				traceback.print_exc()
				continue  # Skip this frame but continue processing

			try:
				print("Drawing annotations...")
				frame = draw_annotations(frame, rescuer, patient)
			except Exception as e:
				print(f"Error in annotation drawing: {str(e)}")
				traceback.print_exc()
				continue

			# Verify frame data before display
			if frame is None or frame.size == 0:
				print("Warning: Invalid frame data - Skipping display")
				continue

			# Display resized frame
			cv2.imshow('CPR Monitoring', frame)

			# Exit immediately on 'q' press with 1ms delay for responsiveness
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				print("User terminated execution with Q key")
				break
			
			
			# Check for window closure
			if cv2.getWindowProperty('CPR Monitoring', cv2.WND_PROP_VISIBLE) < 1:
				print("Window closed by user")
				break
				
			# Exit on 'q' with 50ms delay
			if cv2.waitKey(50) & 0xFF == ord('q'):
				print("User terminated execution")
				break

	except Exception as e:
		print(f"Fatal error: {str(e)}")
		traceback.print_exc()
	finally:
		print("Cleaning up resources...")
		if 'cap' in locals() and cap.isOpened():
			cap.release()
		cv2.destroyAllWindows()
		print("Cleanup complete")

if __name__ == "__main__":
	main()
