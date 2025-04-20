import cv2
import mediapipe as mp
from enum import Enum
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import time
import os

# Define Enum for specific keypoints (shoulders and wrists)
class Keypoints(Enum):
	RIGHT_SHOULDER = 11
	LEFT_SHOULDER = 12
	RIGHT_WRIST = 15
	LEFT_WRIST = 16

class poseDetector():
	def __init__(self, mode=False, upBody=True, smooth=True,
				 detectionCon=0.5, trackCon=0.5):
		"""
		Initializes the pose detector with given parameters.
		"""
		self.mode = mode
		self.upBody = upBody
		self.smooth = smooth
		self.detectionCon = detectionCon
		self.trackCon = trackCon
		self.mpDraw = mp.solutions.drawing_utils
		self.mpPose = mp.solutions.pose
		
		# Initialize the Mediapipe Pose model
		self.pose = self.mpPose.Pose(
			static_image_mode=self.mode,
			model_complexity=1,
			smooth_landmarks=self.smooth,
			enable_segmentation=False,
			min_detection_confidence=self.detectionCon,
			min_tracking_confidence=self.trackCon
		)
		
		# Arrays to store midpoints and smoothed values and peaks
		self.midpoints = np.empty((0, 2), dtype=int)
		self.y_smoothed = np.empty(0, dtype=float)
		self.peaks = np.empty(0, dtype=int)
		
	def findPose(self, img, draw=True):
		"""
		Processes an image to detect the pose landmarks.
		"""
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
		self.results = self.pose.process(imgRGB)  # Run pose estimation
		return img

	def findPosition(self, img, draw=True):
		"""
		Extracts landmark positions and calculates the wrist midpoint.
		"""
		lmList = np.empty((0, 4))  # Initialize an empty NumPy array

		if self.results.pose_landmarks:
			h, w, c = img.shape
			# Extract landmark positions (ID, X, Y, visibility)
			lmList = np.array([[id, int(lm.x * w), int(lm.y * h), lm.visibility] 
							   for id, lm in enumerate(self.results.pose_landmarks.landmark)])

		if draw and lmList.size > 0:
			keypoints = [Keypoints.RIGHT_WRIST, Keypoints.LEFT_WRIST]
			points = {keypoint: lmList[keypoint.value, 1:] for keypoint in keypoints}

			# Get wrist coordinates
			RIGHT_WRIST = points[Keypoints.RIGHT_WRIST][:2]
			LEFT_WRIST = points[Keypoints.LEFT_WRIST][:2]
			# Compute the midpoint between wrists
			midpoint = np.mean([RIGHT_WRIST, LEFT_WRIST], axis=0).astype(int)

			# Append the midpoint to the list
			self.midpoints = np.vstack([self.midpoints, midpoint])

			# Draw the midpoint on the image
			cv2.circle(img, tuple(midpoint), 10, (255, 255, 0), cv2.FILLED)
			cv2.putText(img, 'Midpoint', (midpoint[0] + 10, midpoint[1] + 10),
						cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

		return lmList

	def smoothMidpoints(self):
		"""
		Applies Savitzky-Golay filter to smooth the y-coordinates of midpoints.
		"""
		if self.midpoints.shape[0] > 5:  # Ensure enough data points for smoothing
			y_smoothed = savgol_filter(self.midpoints[:, 1], window_length=10, polyorder=2, mode='nearest')
		else:
			y_smoothed = self.midpoints[:, 1]  # Use original data if not enough points
		
		self.y_smoothed = y_smoothed
		return y_smoothed
	
	# Detect the peaks of the motion curve
	def detectPeaks(self):
		"""
		Detects the peaks of the motion curve.
		"""
		if self.y_smoothed.size == 0:
			print("No smoothed values found.")
			return  # Do nothing if there are no smoothed values
				
		peaks_max, _ = find_peaks(self.y_smoothed, distance=10)
		peaks_min, _ = find_peaks(-self.y_smoothed, distance=10)
		peaks = np.sort(np.concatenate((peaks_max, peaks_min)))

		self.peaks = peaks

		return peaks
	
	def plotMidpoints(self):
		"""
		Plots the original and smoothed y-coordinates of the midpoints.
		"""
		if self.midpoints.size == 0:
			return  # Do nothing if there are no midpoints
		
		y = self.midpoints[:, 1]
		y_smoothed = self.y_smoothed

		# Plot motion curve
		plt.plot(y, label="Original Motion", color="red", linestyle="dashed", alpha=0.6)
		plt.plot(y_smoothed, label="Smoothed Motion", color="blue", linewidth=2)

		peaks = self.peaks
		if peaks.size > 0:
			plt.plot(peaks, y_smoothed[peaks], "x", color="green", markersize=10, label="Peaks")
		else:
			print("No peaks detected.")
		
		plt.xlabel("Frame Number")
		plt.ylabel("Y Position")
		plt.title("Motion Curve")
		plt.grid(True)
		plt.legend()
		plt.show()

# Main function for video processing
def main():
	print("Starting pose detection...")

	video_path = r"CPR\Code Refactor\ball_8.mp4"
	print(f"Checking file: {video_path}")
	print("Exists:", os.path.exists(video_path))
	print("Absolute path:", os.path.abspath(video_path))

	cap = cv2.VideoCapture(video_path)
	
	# Validate if the video file opened successfully
	if not cap.isOpened():
		print("Error: Could not open video file.")
		return
	
	detector = poseDetector()
	pTime = 0  # Previous time for FPS calculation

	while True:
		success, img = cap.read()

		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

		if not success:
			print("Video processing complete or failed.")
			break
		
		img = detector.findPose(img)
		lmList = detector.findPosition(img, draw=True)
		
		cTime = time.time()
		fps = 1 / (cTime - pTime) if pTime else 0
		pTime = cTime
		
		cv2.putText(img, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
		cv2.imshow("Image", img)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("Exiting video processing...")
			break
	
	# Plot the motion curve
	detector.smoothMidpoints()
	detector.detectPeaks()
	detector.plotMidpoints()

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
