import threading
from queue import Queue
import queue
import cv2
from logging_config import cpr_logger

class ThreadedCamera:
    def __init__(self, source, requested_fps = 30):

        # The constructor of OpenCV's VideoCapture class automatically opens the camera
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"[VIDEO CAPTURE] Unable to open camera source: {source}")
        cpr_logger.info(f"[VIDEO CAPTURE] Camera source opened: {source}")

        # Attempt to configure the camera to the requested FPS
        # Which is set to the value we have been working on with recorded videos
        # .set() returns True if the camera acknowledged the request, not if it actually achieved the FPS.
        set_success = self.cap.set(cv2.CAP_PROP_FPS, requested_fps)

        # Get the actual FPS from the camera
        # This is the FPS that the camera is actually using, which may differ from the requested FPS.
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = actual_fps

        cpr_logger.info(f"[VIDEO CAPTURE] Requested FPS: {requested_fps}, Set Success: {set_success}, Actual FPS: {actual_fps}")

        # The buffer should be able to hold a lag of up to 2 seconds
        number_of_seconds_to_buffer = 5
        queue_size = int(actual_fps * number_of_seconds_to_buffer)
        self.q = Queue(maxsize=queue_size)
        cpr_logger.info(f"[VIDEO CAPTURE] Queue size: {queue_size}")

        # Set a flag to indicate that the camera is running
        self.running = threading.Event()
        self.running.set()  # Initial state = running
        cpr_logger.info(f"[VIDEO CAPTURE] Camera running: {self.running.is_set()}")

        self.number_of_dropped_frames = 0

        self.thread = None

    def start_capture(self):
        # Clear any existing frames in queue
        while not self.q.empty():
            self.q.get()

        # threading.Thread() initialize a new thread
        # target=self._reader specify the method (_reader) the thread will execute
        self.thread = threading.Thread(target=self._reader)
        cpr_logger.info(f"[VIDEO CAPTURE] Thread initialized: {self.thread}")
        
        # Set the thread as a daemon thread:
        #   Daemon threads automatically exit when the main program exits
        #   They run in the background and don't block program termination
        self.thread.daemon = True
        cpr_logger.info(f"[VIDEO CAPTURE] Thread daemon: {self.thread.daemon}")

        # Start the thread execution:
        #   Call the _reader method in parallel with the main program
        self.thread.start()

    def _reader(self):
        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                cpr_logger.info("Camera disconnected")
                self.q.put(None)  # Sentinel for clean exit
                break
                
            try:
                self.q.put(frame, timeout=0.1)
            except queue.Full:
                cpr_logger.error("Frame dropped")
                self.number_of_dropped_frames += 1

    def read(self):
        return self.q.get()

    def release(self):
        cpr_logger.info(f"Number of dropped frames: {self.number_of_dropped_frames}")
        
        self.running.clear()
        
        # First release the capture to unblock pending reads
        self.cap.release()  # MOVED THIS LINE UP
        
        # Then join the thread
        self.thread.join(timeout=1.0)
        
        if self.thread.is_alive():
            cpr_logger.info("Warning: Thread didn't terminate cleanly")
        # Removed redundant self.cap.release()

    def isOpened(self):
        return self.running.is_set() and self.cap.isOpened()

    def __del__(self):
        if self.running.is_set():  # Only release if not already done
            self.release()