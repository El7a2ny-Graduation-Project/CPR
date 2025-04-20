import cv2
import mediapipe as mp
import numpy as np
import time
import os
from scipy.signal import savgol_filter, find_peaks

class Keypoints:
	"""
	Class to hold keypoint indices for pose landmarks.
	"""
	RIGHT_WRIST = 15
	LEFT_WRIST = 16

class Detector:
	"""
	Base class for detection and analysis of trajectories.
	"""
	def __init__(self):
		self.y = []
		self.y_smoothed = []
		self.peaks_max = []
		self.peaks = []
	
	def smoothTrajectory(self, window=10, poly=2):
		"""
		Smooth the trajectory using Savitzky-Golay filter.
		
		Parameters:
		window (int): The length of the filter window (must be a positive odd integer).
		poly (int): The order of the polynomial used to fit the samples.
		"""
		self.y_smoothed = savgol_filter(self.y, window, poly) if len(self.y) > window else self.y
	
	def detectPeaks(self, distance=10):
		"""
		Detect peaks in the smoothed trajectory.
		
		Parameters:
		distance (int): Required minimal horizontal distance (in samples) between neighboring peaks.
		"""
		self.peaks_max = find_peaks(self.y_smoothed, distance=distance)[0]
		peaks_min = find_peaks(-np.array(self.y_smoothed), distance=distance)[0]
		self.peaks = np.sort(np.concatenate((self.peaks_max, peaks_min)))
	
	def calculateRate(self, fps):
		"""
		Calculate the rate of detected peaks.
		
		Parameters:
		fps (int): Frames per second of the video.
		
		Returns:
		float: The calculated rate in Hz.
		"""
		return 1 / (np.mean(np.diff(self.peaks_max)) / fps) if len(self.peaks_max) > 1 else None

	def calculateDepth(self):
		"""
		Calculate the average depth of detected peaks.
		
		Returns:
		float: The average depth in pixels.
		"""
		return np.mean(np.abs(np.diff(self.y_smoothed[self.peaks]))) if len(self.peaks) > 1 else None

class PoseDetector(Detector):
	"""
	Class for detecting and analyzing pose keypoints.
	"""
	def __init__(self):
		super().__init__()
		self.pose = mp.solutions.pose.Pose()
	
	def process(self, img):
		"""
		Process an image to detect pose keypoints and extract the midpoint of the wrists.
		
		Parameters:
		img (numpy.ndarray): The input image.
		
		Returns:
		numpy.ndarray: The image with the detected midpoint drawn.
		"""
		results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		if results.pose_landmarks:
			# Get image dimensions
			h, w = img.shape[:2]
			
			# Extract the right and left wrist coordinates, scaling them to image dimensions
			rw, lw = [(int(l.x * w), int(l.y * h)) for l in [results.pose_landmarks.landmark[k] for k in [Keypoints.RIGHT_WRIST, Keypoints.LEFT_WRIST]]]
			
			# Compute the midpoint between the right and left wrist
			midpoint = tuple(np.mean([rw, lw], axis=0).astype(int))
			
			# Store only the y-coordinate of the midpoint for further analysis
			self.y.append(midpoint[1])
			self.y.append(midpoint[1])
			
			# Draw the two wrists on the image as red circles
			cv2.circle(img, rw, 10, (0, 0, 255), -1)
			cv2.circle(img, lw, 10, (0, 0, 255), -1)

			# Draw a line between the two wrists on the image as a red line
			cv2.line(img, rw, lw, (0, 0, 255), 2)

			# Draw the midpoint on the image as a blue circle
			cv2.circle(img, midpoint, 10, (255, 255, 0), -1)
		return img

class BallDetector(Detector):
	"""
	Class for detecting and analyzing a ball in the image.
	"""
	def __init__(self):
		super().__init__()
		self.lower_orange = np.array([5, 120, 120])
		self.upper_orange = np.array([25, 255, 255])    
		
	def process(self, img):
		"""
		Process an image to detect a ball and extract its position.
		
		Parameters:
		img (numpy.ndarray): The input image.
		
		Returns:
		numpy.ndarray: The image with the detected ball drawn.
		"""
		# Convert the image to HSV color space for better color filtering
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		# Create a binary mask where orange pixels are white and the rest are black
		mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
		
		# Gray Scale the image
		gray = cv2.GaussianBlur(cv2.bitwise_and(img, img, mask=mask), (9, 9), 2)
		gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
		
		# Apply Hough Circle Transform to detect circles in the blurred image        
		circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 30, param1=50, param2=30, minRadius=5, maxRadius=30)
		
		if circles is not None:
			x, y, r = map(int, circles[0][0])
			self.y.append(y)
			# Draw the detected ball on the image as a green circle
			cv2.circle(img, (x, y), r, (0, 255, 0), 2)
			# Draw the center of the detected ball on the image as a yellow dot
			cv2.circle(img, (x, y), 2, (0, 255, 255), 3)
		else:
			self.y.append(self.y[-1] if self.y else 0)  # Prevent empty list error
		return img

def main():
	"""
	Main function to start the keypoints and ball detection process.
	"""
	print("Starting Keypoints & Ball Detection...")

	video_path = r"CPR\Dataset\ball_8.mp4"
	print(f"Checking file: {video_path}")
	print("Exists:", os.path.exists(video_path))
	print("Absolute path:", os.path.abspath(video_path))

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened(): return print("Error: Could not open video file.")
	
	pose_detector, ball_detector = PoseDetector(), BallDetector()
	pTime, fps = time.time(), 0
	
	while cap.isOpened():
		success, img = cap.read()
		if not success: break
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		img = pose_detector.process(img)
		img = ball_detector.process(img)
		
		fps = int(1 / (time.time() - pTime)) if pTime else 0
		cv2.putText(img, f"FPS: {fps}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
		cv2.imshow("Tracking", img)
		pTime = time.time()
		if cv2.waitKey(1) & 0xFF == ord('q'): break
	
	cap.release()
	cv2.destroyAllWindows()
	
	# Process results
	pose_detector.smoothTrajectory()
	ball_detector.smoothTrajectory()
	pose_detector.detectPeaks()
	ball_detector.detectPeaks()
	
	print(f"Midpoint Compression Depth: {pose_detector.calculateDepth()} px")
	print(f"Midpoint Compression Rate: {pose_detector.calculateRate(fps)} Hz")
	print(f"Ball Compression Depth: {ball_detector.calculateDepth()} px")
	print(f"Ball Compression Rate: {ball_detector.calculateRate(fps)} Hz")

if __name__ == "__main__":
	main()