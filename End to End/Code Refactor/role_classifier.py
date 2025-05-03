# role_classifier.py
import cv2
import numpy as np
from keypoints import CocoKeypoints

class RoleClassifier:
    """Role classification and tracking using image processing"""
    
    def __init__(self, proximity_thresh=0.3):
        self.proximity_thresh = proximity_thresh
        self.rescuer_id = None
        self.midpoints = []
        self.shoulder_distances = []
        self.rescuer_keypoints = None

    def track_rescuer_midpoints(self, keypoints, frame):
        """Track and draw rescuer's wrist midpoints (using absolute pixel coordinates)"""
        try:
            # Store keypoints for shoulder distance calculation
            self.rescuer_keypoints = keypoints
            
            # Get wrist coordinates directly in pixels
            lw = keypoints[CocoKeypoints.LEFT_WRIST.value]
            rw = keypoints[CocoKeypoints.RIGHT_WRIST.value]
            
            # Calculate midpoint in pixel space
            midpoint = (
                int((lw[0] + rw[0]) / 2),
                int((lw[1] + rw[1]) / 2)
            )
            self.midpoints.append(midpoint)
            
            # Draw tracking
            cv2.circle(frame, midpoint, 8, (0,255,0), -1)
            cv2.putText(frame, "MIDPOINT", (midpoint[0]+5, midpoint[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            return frame
        except Exception as e:
            print(f"Midpoint tracking error: {e}")
            return frame

    def update_shoulder_distance(self):
        """Helper function to calculate and store shoulder distance"""
        if self.rescuer_keypoints is None:
            return

        try:
            left_shoulder = self.rescuer_keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            right_shoulder = self.rescuer_keypoints[CocoKeypoints.RIGHT_SHOULDER.value]
            
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            self.shoulder_distances.append(shoulder_dist)
            print(f"Shoulder distance: {shoulder_dist:.2f} pixels")
        except Exception as e:
            print(f"Shoulder distance calculation error: {e}")

    def _calculate_verticality_score(self, keypoints):
        """Calculate posture verticality score (0=horizontal, 1=vertical) based on shoulder-hip alignment.
        """
        try:
            # Get shoulder and hip keypoints using enum
            left_shoulder = keypoints[CocoKeypoints.LEFT_SHOULDER.value]
            right_shoulder = keypoints[CocoKeypoints.RIGHT_SHOULDER.value]
            left_hip = keypoints[CocoKeypoints.LEFT_HIP.value]
            right_hip = keypoints[CocoKeypoints.RIGHT_HIP.value]
            
            # Calculate midpoints
            shoulder_midpoint = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )
            hip_midpoint = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )
            
            # Calculate differences
            horizontal_diff = abs(shoulder_midpoint[0] - hip_midpoint[0])
            vertical_diff = abs(shoulder_midpoint[1] - hip_midpoint[1])
            
            return 0 if horizontal_diff > 1.5 * vertical_diff else 1
            
        except (IndexError, KeyError, TypeError) as e:
            print(f"Verticality score calculation error: {e}")
            return -1  # Invalid score
    
    def _calculate_bounding_box_center(self, bounding_box):
        """Calculate the center coordinates of a bounding box.
        """
        x1, y1, x2, y2 = bounding_box
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    def classify_roles(self, results):
        """
        Classify roles of detected results based on keypoints and bounding boxes.
        
        Returns:
            Tuple: (rescuer_dict, patient_dict) or (None, None) if not found
        """
        # Step 1: Preprocessing & feature extraction
        processed_results = []
        
        for i, (box, keypoints) in enumerate(zip(results.boxes.xywh.cpu().numpy(), 
                                            results.keypoints.xy.cpu().numpy())):
            try:
                # Convert box to [x1,y1,x2,y2] format
                x_center, y_center, width, height = box
                bounding_box = [
                    x_center - width/2,  # x1
                    y_center - height/2, # y1 
                    x_center + width/2,  # x2
                    y_center + height/2  # y2
                ]
                
                # Calculate features
                verticality_score = self._calculate_verticality_score(keypoints)
                bounding_box_center = self._calculate_bounding_box_center(bounding_box)
                
                # Store processed results
                processed_results.append({
                    'original_index': i,
                    'bounding_box': bounding_box,
                    'bounding_box_center': bounding_box_center,
                    'verticality_score': verticality_score,
                    'keypoints': keypoints,
                })
                
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue
        
        # Step 2: Identify the patient (horizontal posture)
        patient_candidates = [res for res in processed_results 
                            if res['verticality_score'] == 0]
        
        # If more than one horizontal person, select person with lowest center (likely lying down)
        if len(patient_candidates) > 1:
            patient_candidates = sorted(patient_candidates, 
                                    key=lambda x: x['bounding_box_center'][1])[:1]  # Sort by y-coordinate
        
        patient = patient_candidates[0] if patient_candidates else None
        
        # Step 3: Identify the rescuer
        rescuer = None
        if patient:
            # Find vertical people who aren't the patient
            potential_rescuers = [
                res for res in processed_results 
                if res['verticality_score'] == 1 
                and res['original_index'] != patient['original_index']
            ]
            
            if potential_rescuers:
                # Select rescuer closest to patient
                rescuer = min(potential_rescuers,
                            key=lambda x: self._calculate_distance(
                                x['bounding_box_center'], 
                                patient['bounding_box_center']))
        
        return rescuer, patient

    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5

    def draw_bounding_box_for_roles(self, frame, rescuer_processed_results, patient_processed_results):
        # Draw rescuer (A) with green box
        if rescuer_processed_results:
            try:
                x1, y1, x2, y2 = map(int, rescuer_processed_results["bounding_box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.putText(frame, "Rescuer A", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error drawing rescuer: {str(e)}")

        # Draw patient (B) with red box
        if patient_processed_results:
            try:
                x1, y1, x2, y2 = map(int, patient_processed_results["bounding_box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, "Patient B", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error drawing patient: {str(e)}")

        return frame