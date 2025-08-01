import numpy as np
import cv2 as cv
import hashlib
import colorsys
from abc import ABC, abstractmethod
from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction


class BaseTracker(ABC):
    def __init__(
        self, 
        det_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_obs: int = 50,
        nr_classes: int = 80,
        per_class: bool = False,
        asso_func: str = 'iou',
        is_obb: bool = False
    ):
        """
        Initialize the BaseTracker object with detection threshold, maximum age, minimum hits, 
        and Intersection Over Union (IOU) threshold for tracking objects in video frames.

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age of a track before it is considered lost.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class  # Track per class or not
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func+"_obb" if is_obb else asso_func
        self.is_obb = is_obb
        
        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        self._first_dets_processed = False
        
        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []
        
        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            print("self.max_obs", self.max_obs)

    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method 
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")
    
    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # Check if there are detections
        if dets.size > 0:
            class_indices = np.where(dets[:, 5] == cls_id)[0]
            class_dets = dets[class_indices]
            
            if embs is not None:
                # Assert that if embeddings are provided, they have the same number of elements as detections
                assert dets.shape[0] == embs.shape[0], "Detections and embeddings must have the same number of elements when both are provided"
                
                if embs.size > 0:
                    class_embs = embs[class_indices]
                    self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
                else:
                    class_embs = None
        return class_dets, class_embs
    
    @staticmethod
    def setup_decorator(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """
        def wrapper(self, *args, **kwargs):
            # If setup hasn't been done yet, perform it
            # Even if dets is empty (e.g., shape (0, 7)), this check will still pass if it's Nx7
            if not self._first_dets_processed:
                dets = args[0]
                if dets is not None:
                    if dets.ndim == 2 and dets.shape[1] == 6:
                        self.is_obb = False
                        self._first_dets_processed = True
                    elif dets.ndim == 2 and dets.shape[1] == 7:
                        self.is_obb = True
                        self._first_dets_processed = True

            if not self._first_frame_processed:
                img = args[1]
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

                # Mark that the first frame setup has been done
                self._first_frame_processed = True

            # Call the original method (e.g., update)
            return method(self, *args, **kwargs)
        
        return wrapper
    
    
    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """
        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            
            #handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))
            
            if self.per_class:
                # Initialize an array to store the tracks for each class
                per_class_tracks = []
                
                # same frame count for all classes
                frame_count = self.frame_count

                for cls_id in range(self.nr_classes):
                    # Get detections and embeddings for the current class
                    class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
                    
                    LOGGER.debug(f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings {class_embs.shape if class_embs is not None else None}")

                    # Activate the specific active tracks for this class id
                    self.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # Reset frame count for every class
                    self.frame_count = frame_count
                    
                    # Update detections using the decorated method
                    tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                    # Save the updated active tracks
                    self.per_class_active_tracks[cls_id] = self.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                
                # Increase frame count by 1
                self.frame_count = frame_count + 1

                return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)
        return wrapper


    def check_inputs(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        if self.is_obb:
            assert (
                dets.shape[1] == 7
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (cx,cy,w,h,angle,conf,cls)"
        else :
            assert (
                dets.shape[1] == 6
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (x1,y1,x2,y2,conf,cls)"


    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        
        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]
        
        return bgr

    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        if self.is_obb:
            
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}, a: {box[4]:.2f}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else :

            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),
                thickness
            )
            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img


    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations. Each point
        in the trajectory is represented by a circle, with the thickness increasing for more
        recent observations to visualize the path of movement.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories.
        - observations (list): A list of bounding box coordinates representing the historical
        observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
        - id (int): The unique identifier of the tracked object for color consistency in visualization.

        Returns:
        - np.ndarray: The image array with the trajectories drawn on it.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float (i + 1)) * 1.2)
            if self.is_obb:
                img = cv.circle(
                    img,
                    (int(box[0]), int(box[1])),
                    2,
                    color=self.id_to_color(int(id)),
                    thickness=trajectory_thickness 
                )
            else:

                img = cv.circle(
                    img,
                    (int((box[0] + box[2]) / 2),
                    int((box[1] + box[3]) / 2)), 
                    2,
                    color=self.id_to_color(int(id)),
                    thickness=trajectory_thickness
                )
        return img

    def plot_results(self, img: np.ndarray, keypoints, show_trajectories: bool, 
                thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes tracks, trajectories, and pose keypoints with connections.
        
        Parameters:
        - img: Input image (numpy array)
        - keypoints: Ultralytics Keypoints object from results
        - show_trajectories: Whether to show movement trajectories
        - thickness: Line thickness for drawings
        - fontscale: Text size
        
        Returns:
        - Image with visualizations (numpy array)
        """


        # print the shape of the keypoints
        print("keypoints shape", keypoints.shape)

        # Get image dimensions
        height, width = img.shape[:2]
        
        # COCO keypoint connections (1-based indices)
        SKELETON = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], 
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]]
        
        # Plot keypoints and connections if available
        if keypoints is not None:
            # Convert Keypoints object to numpy array [num_people, 17, 2/3]
            kpts_data = keypoints.data.cpu().numpy() if hasattr(keypoints, 'data') else None
            
            if kpts_data is not None:
                for person_kpts in kpts_data:  # For each detected person
                    # Get confidence if available (shape may be [17,2] or [17,3])
                    has_conf = person_kpts.shape[-1] == 3
                    
                    # Draw keypoints
                    for i, kp in enumerate(person_kpts):
                        x, y = kp[:2]
                        conf = kp[2] if has_conf else 1.0
                        
                        if conf > 0.5:  # Confidence threshold
                            pt = (int(x * width), int(y * height))
                            cv.circle(img, pt, thickness+2, (0, 255, 0), -1)
                    
                    # Draw connections
                    for i, j in SKELETON:
                        if (i-1 < len(person_kpts) and j-1 < len(person_kpts)):
                            kp1 = person_kpts[i-1]
                            kp2 = person_kpts[j-1]
                            
                            # Check confidence if available
                            kp1_visible = (kp1[2] > 0.5) if has_conf else True
                            kp2_visible = (kp2[2] > 0.5) if has_conf else True
                            
                            if kp1_visible and kp2_visible:
                                pt1 = (int(kp1[0] * width), int(kp1[1] * height))
                                pt2 = (int(kp2[0] * width), int(kp2[1] * height))
                                cv.line(img, pt1, pt2, (255, 0, 0), thickness)

        # Original tracking visualization (unchanged)
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations and len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations and len(a.history_observations) > 2:
                    box = a.history_observations[-1]
                    img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                    if show_trajectories:
                        img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        
        return img

    # def plot_results(self, img: np.ndarray, keypoints, show_trajectories: bool, 
    #             thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
    #     """
    #     Fixed version for Ultralytics Keypoints object
    #     Handles both xy coordinates and confidence scores properly
    #     """
    #     # Get image dimensions
    #     height, width = img.shape[:2]
        
    #     # COCO keypoint connections (1-based indices)
    #     SKELETON = [
    #         [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], 
    #         [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
    #         [2, 4], [3, 5], [4, 6], [5, 7]]
        
    #     if keypoints is not None:
    #         try:
    #             # Extract coordinates and confidence separately
    #             kpts_xy = keypoints.xy[0].cpu().numpy()  # Shape: [17, 2]
    #             kpts_conf = keypoints.conf[0].cpu().numpy()  # Shape: [17]
                
    #             print(f"Keypoints XY shape: {kpts_xy.shape}")
    #             print(f"Keypoints Conf shape: {kpts_conf.shape}")
                
    #             # Draw keypoints
    #             for i, ((x, y), conf) in enumerate(zip(kpts_xy, kpts_conf)):
    #                 if conf > 0.5:  # Confidence threshold
    #                     x, y = int(x), int(y)
    #                     if 0 <= x < width and 0 <= y < height:
    #                         cv.circle(img, (x, y), thickness+2, (0, 255, 0), -1)
    #                         print(f"KP {i+1} at ({x},{y}) conf={conf:.2f}")
                
    #             # Draw connections
    #             for i, j in SKELETON:
    #                 idx1, idx2 = i-1, j-1  # Convert to 0-based
    #                 if kpts_conf[idx1] > 0.5 and kpts_conf[idx2] > 0.5:
    #                     x1, y1 = map(int, kpts_xy[idx1])
    #                     x2, y2 = map(int, kpts_xy[idx2])
    #                     if (0 <= x1 < width and 0 <= y1 < height and 
    #                         0 <= x2 < width and 0 <= y2 < height):
    #                         cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness)
    #                         print(f"Conn {i}-{j}: ({x1},{y1})→({x2},{y2})")
                
    #         except Exception as e:
    #             print(f"Keypoints processing error: {str(e)}")
    #             import traceback
    #             traceback.print_exc()
        
    #      # Original tracking visualization (unchanged)
    #     if self.per_class_active_tracks is not None:
    #         for k in self.per_class_active_tracks.keys():
    #             active_tracks = self.per_class_active_tracks[k]
    #             for a in active_tracks:
    #                 if a.history_observations and len(a.history_observations) > 2:
    #                     box = a.history_observations[-1]
    #                     img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
    #                     if show_trajectories:
    #                         img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
    #     else:
    #         for a in self.active_tracks:
    #             if a.history_observations and len(a.history_observations) > 2:
    #                 box = a.history_observations[-1]
    #                 img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
    #                 if show_trajectories:
    #                     img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        
    #     return img