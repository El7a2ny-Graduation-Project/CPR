import threading
from queue import Queue
import queue
import cv2

class ThreadedCamera:
    def __init__(self, source, requested_fps = 30):

        # The constructor of OpenCV's VideoCapture class automatically opens the camera
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"[VIDEO CAPTURE] Unable to open camera source: {source}")
        print(f"[VIDEO CAPTURE] Camera source opened: {source}")

        # Attempt to configure the camera to the requested FPS
        # Which is set to the value we have been working on with recorded videos
        # .set() returns True if the camera acknowledged the request, not if it actually achieved the FPS.
        set_success = self.cap.set(cv2.CAP_PROP_FPS, requested_fps)

        # Get the actual FPS from the camera
        # This is the FPS that the camera is actually using, which may differ from the requested FPS.
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = actual_fps

        print(f"[VIDEO CAPTURE] Requested FPS: {requested_fps}, Set Success: {set_success}, Actual FPS: {actual_fps}")

        # The buffer should be able to hold a lag of up to 2 seconds
        queue_size = int(actual_fps * 2)
        self.q = Queue(maxsize=queue_size)
        print(f"[VIDEO CAPTURE] Queue size: {queue_size}")

        # Set a flag to indicate that the camera is running
        self.running = threading.Event()
        self.running.set()  # Initial state = running
        print(f"[VIDEO CAPTURE] Camera running: {self.running.is_set()}")

        # threading.Thread() initialize a new thread
        # target=self._reader specify the method (_reader) the thread will execute
        self.thread = threading.Thread(target=self._reader)
        print(f"[VIDEO CAPTURE] Thread initialized: {self.thread}")
        
        # Set the thread as a daemon thread:
        #   Daemon threads automatically exit when the main program exits
        #   They run in the background and don't block program termination
        self.thread.daemon = True
        print(f"[VIDEO CAPTURE] Thread daemon: {self.thread.daemon}")

        # Start the thread execution:
        #   Call the _reader method in parallel with the main program
        self.thread.start()

    def _reader(self):
        dropped_frames = 0

        while self.running.is_set():
            ret, frame = self.cap.read()
           
            if not ret:
                print("[ERROR] Failed to read camera frame")
                break

            try:
                # Add timeout and handle full queue
                self.q.put(frame, timeout=0.1)  # Non-blocking
            except queue.Full:
                # If the queue is full, we can either drop the frame or handle it
                # Here we just print a message and continue
                dropped_frames += 1
                print(f"[VIDEO CAPTURE] Queue is full, dropping frame")
                print(f"[VIDEO CAPTURE] Dropped frames: {dropped_frames}")

    def read(self):
        return self.q.get()

    def release(self):
        self.running.clear()
        self.thread.join(timeout=1.0)  # Wait max 1 second
        if self.thread.is_alive():
            print("Warning: Thread didn't terminate cleanly")
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    def __del__(self):
        if self.running.is_set():  # Only release if not already done
            self.release()