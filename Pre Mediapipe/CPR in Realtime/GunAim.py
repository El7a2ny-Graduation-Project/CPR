import imutils
import serial
import numpy as np
from queue import Queue
import cv2, threading
import time


# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name, IP):
    self.cap = cv2.VideoCapture(name)
    self.cap.open(IP) #comment this line if you don't need to operate using mobile camera
    self.q = Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

cap = VideoCapture(0, "https://192.168.1.13:8080/video")

ser = serial.Serial("COM5", '9600', timeout=2)
time.sleep(1)

cv2.namedWindow('image')
cv2.namedWindow('Offset')
#
def nothing(x):
    pass
#
# # create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

cv2.setTrackbarPos('HMin', 'image', 39)
cv2.setTrackbarPos('SMin', 'image', 100)
cv2.setTrackbarPos('VMin', 'image', 60)
# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 92)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 200)

cv2.createTrackbar('Offset_y','Offset',0,180,nothing)
cv2.createTrackbar('Error_Y','Offset',0,180,nothing)
cv2.createTrackbar('Offset_X','Offset',0,180,nothing)
cv2.createTrackbar('Error_X','Offset',0,180,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('Offset_X', 'Offset', 90)
cv2.setTrackbarPos('Error_X', 'Offset', 10)
cv2.setTrackbarPos('Offset_y', 'Offset', 90)
cv2.setTrackbarPos('Error_Y', 'Offset', 10)

# state 1: the arduino is ready to receive
# state 2: the arduino is has received data and is processing
state = "1"

while True:
    time.sleep(0.5)
    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    Offset_Y = cv2.getTrackbarPos('Offset_y','Offset')
    Error_Y = cv2.getTrackbarPos('Error_Y','Offset')
    Offset_X = cv2.getTrackbarPos('Offset_X','Offset')
    Error_X = cv2.getTrackbarPos('Error_X','Offset')

    # # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])


    if state == "2":
        l = ser.readline()
        if l == bytes(b'1\n'):
            state = "1"
    if state == "1":
        # Capture frame from camera
        frame = cap.read()
        frame = imutils.resize(frame, width=600, height=800)
        frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)

        blurFrame = cv2.GaussianBlur(frame, (17, 17), 0)


        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)
        # mask1 = cv2.inRange(hsv, lower, upper)
        # mask1 = cv2.inRange(hsv, (28, 50, 50), (81, 140, 150)) #green
        mask1 = cv2.inRange(hsv, (39, 100, 60), (92, 255, 200)) #green ball
        # mask1 = cv2.inRange(hsv, (111, 135,105), (179,255,255))
        mask = mask1 #+mask2

        # Apply morphological operations to remove noise
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        center2 = None
        cnts = sorted(cnts,key=lambda x:cv2.contourArea(x),reverse=True)

        rows, cols, _ = frame.shape
        center_y = int(rows / 2)
        center_x = int(cols / 2)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid

            c = cnts[0]
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)


                print("centerY: " + str(center_y + Error_Y))
                print(y + 90 - Offset_Y)
                if x + 90-Offset_X > center_x + 15:
                    ser.write("r\n".encode('utf-8'))
                    print("Right")
                    l = ser.readline()
                    if l == bytes(b'2\n'):
                        state = "2"
                elif x +90- Offset_X < center_x - 15:
                    ser.write("l\n".encode('utf-8'))
                    print("Left")
                    l = ser.readline()
                    if l == bytes(b'2\n'):
                        state = "2"
                if y + 90 - Offset_Y > center_y + Error_Y:
                    ser.write("d\n".encode('utf-8'))
                    print("down")
                    l = ser.readline()
                    if l == bytes(b'2\n'):
                        state = "2"
                elif y + 90 - Offset_Y < center_y - Error_Y:
                    ser.write("u\n".encode('utf-8'))
                    print("up")
                    l = ser.readline()
                    if l == bytes(b'2\n'):
                        state = "2"
                else:
                    ser.write("s\n".encode('utf-8'))
                    print("Done")
                    l = ser.readline()
                    if l == bytes(b'2\n'):
                        state = "2"

    # Display the resulting frame
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
