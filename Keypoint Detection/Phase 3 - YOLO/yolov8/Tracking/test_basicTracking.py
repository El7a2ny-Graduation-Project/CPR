import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def track_people(video_path):
    # Initialize models
    detector = YOLO("yolo11n.pt")  # Detection model
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.3)  # Tracker

    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get YOLO detections (people only)
        results = detector(frame, classes=[0], verbose=False)[0]
        
        # Format detections for DeepSORT: (xyxy, confidence, class)
        detections = [
            (box.xyxy[0].cpu().numpy(), 
             box.conf[0].cpu().item(), 
             0)  # class_id=0 for person
            for box in results.boxes
        ]

        # Update tracker with YOLO's raw boxes
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw YOLO's original boxes (bypass tracker predictions)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw track IDs using YOLO's box positions
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            # Get the MOST RECENT YOLO detection for this track
            track_id = track.track_id
            matched_box = next(
                (b for b in results.boxes 
                 if track.detection_box is not None 
                 and (b.xyxy[0] == track.detection_box).all()), 
                None
            )
            
            if matched_box:
                x1, y1, x2, y2 = map(int, matched_box.xyxy[0])
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_people(r"C:\Users\Fatema Kotb\Documents\CUFE 25\Year 04\GP\Spring\El7a2ny-Graduation-Project\CPR\Keypoint Detection\Dataset\Crowds\vid2.mp4")