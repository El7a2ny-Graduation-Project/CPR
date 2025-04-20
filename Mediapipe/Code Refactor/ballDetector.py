import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import os

class ballDetector:
	def __init__(self):
		self.lower_orange = np.array([5, 120, 120])
		self.upper_orange = np.array([25, 255, 255])
		self.y = np.array([])
		self.y_smoothed = np.array([])
		self.peaks_max = np.array([])
		self.peaks_min = np.array([])
		self.peaks = np.array([])
		self.sum_radius = 0
		self.avg_radius = 0
		self.pixel_to_cm = 0

	def calculatePixelToCM(self, radius_cm=2):
		if self.y.size > 0:
			self.avg_radius = self.sum_radius / self.y.size
			self.pixel_to_cm = radius_cm / self.avg_radius

	def smoothBallTrajectory(self):
		if self.y.size > 10:
			self.y_smoothed = savgol_filter(self.y, window_length=10, polyorder=2)
		else:
			self.y_smoothed = self.y
		return self.y_smoothed

	def detectPeaks(self, smoothed_y):
		self.peaks_max = find_peaks(-smoothed_y, distance=10)[0]
		self.peaks_min = find_peaks(smoothed_y, distance=10)[0]
		self.peaks = np.sort(np.concatenate((self.peaks_max, self.peaks_min)))

	def calculateCompressionRate(self, fps):
		if self.peaks_max.size > 1:
			peak_to_peak_frames = np.diff(self.peaks_max)
			avg_peak_to_peak_frames = np.mean(peak_to_peak_frames)
			avg_period_seconds = avg_peak_to_peak_frames / fps
			return 1 / avg_period_seconds
		return None

	def calculateCompressionDepth(self, smoothed_y):
		if self.peaks.size > 1:
			compression_depth_px = np.abs(np.diff(smoothed_y[self.peaks_max]))
			return np.mean(compression_depth_px) * self.pixel_to_cm
		return None

	def plotBallTrajectory(self, smoothed_y):
		plt.plot(self.y, label="Original Motion", color="red", linestyle="dashed", alpha=0.6)
		plt.plot(smoothed_y, label="Smoothed Motion", color="blue", linewidth=2)
		plt.xlabel("Frame Number")
		plt.ylabel("Y Position")
		plt.title("Smoothed Ball Motion Curve")
		plt.legend()
		plt.gca().invert_yaxis()
		plt.show()


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

	fps = cap.get(cv2.CAP_PROP_FPS)
	detector = ballDetector()

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, detector.lower_orange, detector.upper_orange)
		gray = cv2.GaussianBlur(cv2.bitwise_and(frame, frame, mask=mask), (9, 9), 2)
		gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
		circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 30, param1=50, param2=30, minRadius=5, maxRadius=50)
		
		if circles is not None:
			x, y, r = min(circles[0], key=lambda c: c[2])
			detector.y = np.append(detector.y, y)
			detector.sum_radius += r
			cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
			cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)

		cv2.imshow("Ball Tracking", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	
	detector.calculatePixelToCM()
	smoothed_y = detector.smoothBallTrajectory()
	detector.detectPeaks(smoothed_y)
	frequency_hz = detector.calculateCompressionRate(fps)
	avg_compression_depth_cm = detector.calculateCompressionDepth(smoothed_y)
	
	print(f"Motion Frequency: {frequency_hz:.2f} Hz" if frequency_hz else "Not enough peaks detected")
	print(f"Average Compression Depth: {avg_compression_depth_cm:.2f} cm" if avg_compression_depth_cm else "Not enough peaks detected")
	
	detector.plotBallTrajectory(smoothed_y)

if __name__ == "__main__":
	main()
