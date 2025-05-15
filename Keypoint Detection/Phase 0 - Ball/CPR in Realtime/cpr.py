import cv2
import threading
from queue import Queue

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name, IP):
        self.cap = cv2.VideoCapture(name)
        self.cap.open(IP)  # comment this line if you don't need to operate using mobile camera
        self.q = Queue()
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.running = False
        self.thread.join()  # Ensure the thread has finished
        self.cap.release()

cap = VideoCapture(0, "https://192.168.1.9:8080/video")

cv2.namedWindow('frame')

try:
    while True:
        # Capture frame from camera
        frame = cap.read()
        #! frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # Exit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()