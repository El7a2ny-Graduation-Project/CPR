import cv2
import threading
import time
from queue import Queue
from collections import deque

class VideoCapture:
    def __init__(self, name, IP):
        self.cap = cv2.VideoCapture(name)
        self.cap.open(IP)  # Comment for non-IP cameras
        self.q = Queue()
        self.running = True
        self.frame_count = 0
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            self.q.put((frame, self.frame_count))

    def read(self):
        return self.q.get()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

cap = VideoCapture(0, "https://192.168.1.9:8080/video")

cv2.namedWindow('frame')

try:
    prev_time = time.time()
    fps_deque = deque(maxlen=30)  # Smoothed FPS over 10 frames
    
    while True:
        frame, frame_num = cap.read()
        
        # Calculate FPS
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time
        
        if delta_time > 0:
            fps = 1 / delta_time
            fps_deque.append(fps)
        
        avg_fps = sum(fps_deque)/len(fps_deque) if fps_deque else 0
        
        # Add frame number and FPS text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()