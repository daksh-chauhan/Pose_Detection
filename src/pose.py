import mediapipe as mp
import cv2
import numpy as np
import config

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )

    def get_keypoints(self, frame, sidebar):

        h, w, _ = frame.shape

        results = self.pose.process(frame[:, :, ::-1])

        if not results.pose_landmarks:
            keypoints = None

        else:

            lm = results.pose_landmarks.landmark
            idx = self.mp_pose.PoseLandmark

            keypoints = {
                "left_shoulder": (lm[idx.LEFT_SHOULDER], lm[idx.LEFT_SHOULDER].visibility),
                "right_shoulder": (lm[idx.RIGHT_SHOULDER], lm[idx.RIGHT_SHOULDER].visibility),
                "left_hip": (lm[idx.LEFT_HIP], lm[idx.LEFT_HIP].visibility),
                "right_hip": (lm[idx.RIGHT_HIP], lm[idx.RIGHT_HIP].visibility),
                "left_knee": (lm[idx.LEFT_KNEE], lm[idx.LEFT_KNEE].visibility),
                "right_knee": (lm[idx.RIGHT_KNEE], lm[idx.RIGHT_KNEE].visibility),
                "left_toe": (lm[idx.LEFT_FOOT_INDEX], lm[idx.LEFT_FOOT_INDEX].visibility),
                "right_toe": (lm[idx.RIGHT_FOOT_INDEX], lm[idx.RIGHT_FOOT_INDEX].visibility),
                "left_ankle": (lm[idx.LEFT_ANKLE], lm[idx.LEFT_ANKLE].visibility),
                "right_ankle": (lm[idx.RIGHT_ANKLE], lm[idx.RIGHT_ANKLE].visibility),
            }

            for name, (lm, vis) in keypoints.items():
                if vis < config.VISIBILITY_THRESHOLD or lm.x < 0 or lm.x > 1 or lm.y < 0 or lm.y > 1:
                    keypoints = None


        frame, sidebar = self.draw(frame, keypoints, sidebar)
        
        return frame, keypoints, sidebar


    def draw(self, frame, keypoints, sidebar):

        h, w, _ = frame.shape
        y_offset = 80

        def pt(lm):
            return int(lm.x * w), int(lm.y * h)


        if keypoints is None:
            cv2.putText(
                sidebar,
                "No keypoints detected",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )
            return frame, sidebar

        # points
        for _, (lm, _) in keypoints.items():
            cv2.circle(frame, pt(lm), 5, (0, 255, 0), -1)

        # connections
        def connect(a, b):
            cv2.line(frame, pt(keypoints[a][0]), pt(keypoints[b][0]), (255, 0, 0), 2)

        connect("left_shoulder", "left_hip")
        connect("left_hip", "left_knee")
        connect("left_knee", "left_ankle")
        connect("left_toe", "left_ankle")

        connect("right_shoulder", "right_hip")
        connect("right_hip", "right_knee")
        connect("right_knee", "right_ankle")
        connect("right_toe", "right_ankle")

        connect("left_shoulder", "right_shoulder")
        connect("left_hip", "right_hip")


        for name, (lm, vis) in keypoints.items():
            x, y = pt(lm)
            text = f"{name}: ({x},{y}) v={vis:.2f}"

            cv2.putText(
                sidebar,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 20

        return frame, sidebar
