## Analysing Results (ecxluding the rate & depth calculation):
- Crowds:
	- vid1: YOLO fails to detect the patient in most frames.
	- vid2: YOLO fails to detect the patient in all frames.
	- vid3: The camera's skwed angle prevents YOLO from detecting the patient's legs so the whole skeleton is compressed near the head.
- Tracking:
	- video_1: Same as vid3.
	- video_2: Having the patient's head on the left side of the image is not handled yet.
	- video_3: Perfection.
	- video_4: Perfection.