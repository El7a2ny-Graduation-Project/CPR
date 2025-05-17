# role_classifier.py
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator  # Import YOLO's annotator
from logging_config import cpr_logger


class RoleClassifier:
    """Role classification and tracking using image processing"""
    
    def __init__(self, proximity_thresh=0.3):
        self.proximity_thresh = proximity_thresh
        self.rescuer_id = None
        self.rescuer_processed_results = None
        self.patient_processed_results = None

    def _calculate_verticality_score(self, bounding_box):
        """Calculate posture verticality score (0=horizontal, 1=vertical) using bounding box aspect ratio."""
        try:
            x1, y1, x2, y2 = bounding_box
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Handle edge cases with invalid dimensions
            if width == 0 or height == 0:
                return -1
            
            return 1 if height > width else 0  # 1 for vertical, 0 for horizontal
            
        except (TypeError, ValueError) as e:
            cpr_logger.error(f"Verticality score calculation error: {e}")
            return -1
        
    def _calculate_bounding_box_center(self, bounding_box):
        """Calculate the center coordinates of a bounding box.
        """
        x1, y1, x2, y2 = bounding_box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5
 
    def _calculate_bbox_areas(self, rescuer_bbox, patient_bbox):
        """
        Calculate bounding box areas for rescuer and patient.
        
        Args:
            rescuer_bbox: [x1, y1, x2, y2] coordinates of rescuer's bounding box
            patient_bbox: [x1, y1, x2, y2] coordinates of patient's bounding box
        
        Returns:
            Tuple: (rescuer_area, patient_area) in pixels
        """
        def compute_area(bbox):
            if bbox is None:
                return 0
            width = bbox[2] - bbox[0]  # x2 - x1
            height = bbox[3] - bbox[1]  # y2 - y1
            return abs(width * height)  # Absolute value to handle negative coordinates
        
        return compute_area(rescuer_bbox), compute_area(patient_bbox)
    
    def classify_roles(self, results, prev_rescuer_processed_results=None, prev_patient_processed_results=None):
        """
        Classify roles of rescuer and patient based on detected keypoints and bounding boxes.
        """

        processed_results = []
        
        # Calculate combined area threshold if previous boxes exist
        threshold = None
        if prev_rescuer_processed_results and prev_patient_processed_results:
            prev_rescuer_bbox = prev_rescuer_processed_results["bounding_box"]
            prev_patient_bbox = prev_patient_processed_results["bounding_box"]

            rescuer_area = (prev_rescuer_bbox[2]-prev_rescuer_bbox[0])*(prev_rescuer_bbox[3]-prev_rescuer_bbox[1])
            patient_area = (prev_patient_bbox[2]-prev_patient_bbox[0])*(prev_patient_bbox[3]-prev_patient_bbox[1])
            threshold = rescuer_area + patient_area
        
        for i, (box, keypoints) in enumerate(zip(results.boxes.xywh.cpu().numpy(), 
                                            results.keypoints.xy.cpu().numpy())):
            try:
                # Convert box to [x1,y1,x2,y2] format
                x_center, y_center, width, height = box
                bounding_box = [
                    x_center - width/2,  # x1
                    y_center - height/2,  # y1 
                    x_center + width/2,   # x2
                    y_center + height/2   # y2
                ]
                
                # Skip if box exceeds area threshold (when threshold exists)
                if threshold:
                    box_area = width * height
                    if box_area > threshold * 1.2:  # 20% tolerance
                        cpr_logger.info(f"Filtered oversized box {i} (area: {box_area:.1f} > threshold: {threshold:.1f})")
                        continue
                
                # Calculate features
                verticality_score = self._calculate_verticality_score(bounding_box)
                #!We already have the center coordinates from the bounding box, no need to recalculate it.
                bounding_box_center = self._calculate_bounding_box_center(bounding_box)
                
                # Store valid results
                processed_results.append({
                    'original_index': i,
                    'bounding_box': bounding_box,
                    'bounding_box_center': bounding_box_center,
                    'verticality_score': verticality_score,
                    'keypoints': keypoints,
                })
            
            except Exception as e:
                cpr_logger.error(f"Error processing detection {i}: {e}")
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
                #! Useless condition because the patient was horizontal
                and res['original_index'] != patient['original_index']
            ]
            
            if potential_rescuers:
                # Select rescuer closest to patient
                rescuer = min(potential_rescuers,
                            key=lambda x: self._calculate_distance(
                                x['bounding_box_center'], 
                                patient['bounding_box_center']))
        
        return rescuer, patient
    
    def draw_rescuer_and_patient(self, frame):        
        # Create annotator object
        annotator = Annotator(frame)
        
        # Draw rescuer (A) with green box and keypoints
        if self.rescuer_processed_results:
            try:
                x1, y1, x2, y2 = map(int, self.rescuer_processed_results["bounding_box"])
                annotator.box_label((x1, y1, x2, y2), "Rescuer A", color=(0, 255, 0))
                
                if "keypoints" in self.rescuer_processed_results:
                    keypoints = self.rescuer_processed_results["keypoints"]
                    annotator.kpts(keypoints, shape=frame.shape[:2])
            except Exception as e:
                cpr_logger.error(f"Error drawing rescuer: {str(e)}")

        # Draw patient (B) with red box and keypoints
        if self.patient_processed_results:
            try:
                x1, y1, x2, y2 = map(int, self.patient_processed_results["bounding_box"])
                annotator.box_label((x1, y1, x2, y2), "Patient B", color=(0, 0, 255))
                
                if "keypoints" in self.patient_processed_results:
                    keypoints = self.patient_processed_results["keypoints"]
                    annotator.kpts(keypoints, shape=frame.shape[:2])
            except Exception as e:
                cpr_logger.error(f"Error drawing patient: {str(e)}")

        return annotator.result()
    