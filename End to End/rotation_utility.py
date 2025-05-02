import cv2

def rotate_video(input_path, output_path, angle=90):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        #print("Error opening video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if angle in [90, 270] else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if angle in [90, 270] else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec if needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame
        if angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Rotated video saved as {output_path}")