import numpy as np
import cv2
import config 

def viewtype(l_shoulder, r_shoulder, l_hip, r_hip):
    
    l_shoulder_x = l_shoulder[0].x
    l_shoulder_y = l_shoulder[0].y
    r_shoulder_x = r_shoulder[0].x
    l_y1 = l_hip[0].y


    x = abs(l_shoulder_x - r_shoulder_x)
    y = abs(l_y1 - l_shoulder_y)

    aspect_ratio = x / (y+1e-5)
    return aspect_ratio

    # NOTE: side-view classification disabled for now. 
    if aspect_ratio > config.VIEWTYPE_THRSHOLD:
        return 0
    else: 
        return 1
    
def calc_angle(coord1, coord2):

    x1, y1 = coord1[0].x, coord1[0].y
    x2, y2 = coord2[0].x, coord2[0].y
    
    angle = np.arctan(abs(y1 - y2)/abs(x1-x2))*180/np.pi

    conf = coord1[1]*coord2[1]

    return angle, conf


def calc_distance(pivot, up, down):

    x_up = up[0].x
    x_down = down[0].x
    x_pivot = pivot[0].x

    dist = x_down - x_up
    if x_pivot < x_down:
        dist*=-1

    conf = up[1] * down[1] * pivot[1]

    return dist, conf


def calculate_metrics(keypoints, sidebar):

    l_shoulder = keypoints.get("left_shoulder")
    r_shoulder = keypoints.get("right_shoulder")
    l_hip = keypoints.get("left_hip")
    r_hip = keypoints.get("right_hip")
    l_knee = keypoints.get("left_knee")
    r_knee = keypoints.get("right_knee")
    l_toe = keypoints.get("left_toe")
    r_toe = keypoints.get("right_toe")
    l_ankle = keypoints.get("left_ankle")
    r_ankle = keypoints.get("right_ankle")

    y_offset = 300

    aspect_ratio = viewtype(l_shoulder, r_shoulder, l_hip, r_hip)
    # text = "Side view" if aspect_ratio else "Not side view"
    text = aspect_ratio
    cv2.putText(
        sidebar,
        f"View Type: {text}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

    y_offset = 310

    if not aspect_ratio:
        return sidebar, None
    
    thigh_angles = calc_angle(l_hip, l_knee), calc_angle(r_hip, r_knee)
    back_angles = calc_angle(l_hip, l_shoulder), calc_angle(r_hip, r_shoulder)
    knee_toe_dists = calc_distance(l_ankle, l_knee, l_toe), calc_distance(r_ankle, r_knee, r_toe)

    metrics = {
        "Back Angles": {
            "vals": [back_angles[0][0], back_angles[1][0]],
            "confs": [back_angles[0][1], back_angles[1][1]],
        },
        "Thigh Angles": {
            "vals": [thigh_angles[0][0], thigh_angles[1][0]],
            "confs": [thigh_angles[0][1], thigh_angles[1][1]],
        },
        "Knee-toe Dist": {
            "vals": [knee_toe_dists[0][0], knee_toe_dists[1][0]],
            "confs": [knee_toe_dists[0][1], knee_toe_dists[1][1]],
        },
    }


    for name, data in metrics.items():
        vals = data["vals"]
        confs = data["confs"]

        # print(name)
        y_offset += 20
        cv2.putText(
            sidebar,
            f"{name}: ({vals[0]:.2f}, {vals[1]:.2f})",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1
        )

        y_offset += 20
        cv2.putText(
            sidebar,
            f"conf: ({confs[0]:.2f}, {confs[1]:.2f})",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (100, 100, 100), 
            1
        )


    return sidebar, metrics