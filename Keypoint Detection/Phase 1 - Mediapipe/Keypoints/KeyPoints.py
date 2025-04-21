import cv2
import mediapipe as mp
import time
import math
from enum import Enum

# Define Enum for specific keypoints (shoulders and hips)
class Keypoints(Enum):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

class poseDetector():
    def __init__(self, mode=False, upBody=True, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        
        # Pose constructor
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                visibility = lm.visibility
                lmList.append([id, cx, cy, visibility])

        # Return and print all 33 keypoints
        if draw:
            for lm in lmList:
                id, cx, cy, visibility = lm
                # Draw the keypoints
                #cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                #cv2.putText(img, f'{id}', (cx + 10, cy + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            
            # Draw connections between keypoints
            for connection in self.mpPose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(lmList) and end_idx < len(lmList):
                    start_point = (lmList[start_idx][1], lmList[start_idx][2])
                    end_point = (lmList[end_idx][1], lmList[end_idx][2])
                    #cv2.line(img, start_point, end_point, (0, 255, 0), 2)
            
            # Print x, y, and visibility for the keypoints
                #print(f'ID: {id} - X: {cx}, Y: {cy}, Visibility: {visibility}')
            
            # Calculate angle in the triangle making the points of ids 11,13,15
            self.lmList = lmList
            angle_left = self.findAngle(img, 11, 13, 15)
            angle_right = self.findAngle(img, 12, 14, 16)
            print(f'Left Arm Angle: {angle_left}')
            print(f'Right Arm Angle: {angle_right}')

        
        return lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - 
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture('real.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=True)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
