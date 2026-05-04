import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    ## To turn off unnecessary warnings

import cv2
import pose
import numpy as np
from feedback import suggest
from metrics import calculate_metrics

def main():
    # PATH = "samples/test3.mp4"           ## for video path
    PATH = 0                                 ## for webcam
    
    
    wait = 1
    cap = cv2.VideoCapture(PATH)

    if not cap.isOpened():
        print("Error: Could not open video stream.") 
        return
    
    detector = pose.PoseDetector()

    # --- init once after first frame read ---
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        return

    h, w, _ = frame.shape
    ar = w/h
    H = 620
    W = int(H*ar)
        
    sidebar_width = 300
    sidebar_height = 620 

    # --- setup video writer ---
    if isinstance(PATH, str):
        base, ext = os.path.splitext(PATH)
        out_path = base + "_out" + ext
    else:
        out_path = "webcam_out.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (W + sidebar_width, H))

    # process first frame + loop rest
    while True:
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

        sidebar = np.ones((sidebar_height, sidebar_width, 3), dtype="uint8") * 255
        y_offset = 25

        cv2.putText(
            sidebar,
            f"DEBUG MODE",(10, y_offset),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(
            sidebar,
            f"Original Video Resolution: {w}x{h}",
            (10, y_offset + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        disp_1, keypoints, sidebar = detector.get_keypoints(frame, sidebar)

        if keypoints is not None:
            sidebar, metrics = calculate_metrics(keypoints, sidebar)
            disp_1, sidebar = suggest(frame, metrics, sidebar)

        disp_resized = cv2.resize(disp_1, (W, H))
        sidebar_resized = cv2.resize(sidebar, (sidebar_width, H))
        stacked = np.hstack((disp_resized, sidebar_resized))

        cv2.imshow('Annotated frame', disp_resized)
        cv2.imshow('Debug info', sidebar)

        out.write(stacked)

        key = cv2.waitKey(wait)
        if key == 27:
            break
        if key == 32:
            wait = 1 - wait

        frame = None  # trigger next read

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()