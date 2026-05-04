import cv2
import numpy as np
import config

def draw_border(frame, color='black', strength=0.6, thickness=5):
    h, w, _ = frame.shape
    thickness = int(thickness*min(h,w)/100)
    # color map (BGR)
    colors = {
        'black': (0, 0, 0),
        'red': (0, 0, 255),
        'orange': (0, 128, 255),
        'yellow' : (0, 255, 255),
        'green': (0, 255, 0)
    }
    tint = np.array(colors.get(color.lower(), (0, 0, 0)), dtype=np.float32)

    # --- create gradient mask ---
    y = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    x = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]

    dist = np.minimum(x, y).astype(np.float32)

    # normalize to [0,1] within thickness
    mask = np.clip((thickness - dist) / thickness, 0, 1)

    # smooth falloff (feels nicer)
    mask = mask ** 1.5

    mask = mask[..., None]  # shape (h, w, 1)

    # --- apply tint ---
    frame_f = frame.astype(np.float32)
    tinted = frame_f * (1 - mask * strength) + tint * (mask * strength)

    return np.clip(tinted, 0, 255).astype(np.uint8)


def conf_weighted_avg(metric):
    vals = metric['vals']
    confs = metric['confs']
    if np.sum(confs) == 0:
        return 0
    
    return np.dot(vals, confs)/np.sum(confs)


def review(metrics):
    
    faults = ['']*3
   
    back_angle = conf_weighted_avg(metrics['Back Angles'])
    thigh_angle = conf_weighted_avg(metrics['Thigh Angles'])
    knee_toe_dist = conf_weighted_avg(metrics['Knee-toe Dist'])

    processed_metrics = {"Back Angle": back_angle, "Thigh Angle": thigh_angle, "Knee-toe distance": knee_toe_dist}

    if back_angle < config.BACK_ANGLE_THRESHOLD:
        faults[0] = 'Straighten your back'
    if thigh_angle > config.DEPTH_ANGLE_THRESHOLD:
        faults[1] = 'Keep hips at right angle'
    if knee_toe_dist > config.KNEE_FORWARD_TOLERANCE:
        faults[2] = 'Maintain knees behind toes'

    return faults, processed_metrics

# # # ## Just for reference. Do NOT UNCOMMENT
# # #    metrics = {
# # #         "Thigh Angles": {
# # #             "vals": [thigh_angles[0][0], thigh_angles[1][0]],
# # #             "confs": [thigh_angles[0][1], thigh_angles[1][1]],
# # #         },
# # #         "Back Angles": {
# # #             "vals": [back_angles[0][0], back_angles[1][0]],
# # #             "confs": [back_angles[0][1], back_angles[1][1]],
# # #         },
# # #         "Knee-toe Dist": {
# # #             "vals": [knee_toe_dists[0][0], knee_toe_dists[1][0]],
# # #             "confs": [knee_toe_dists[0][1], knee_toe_dists[1][1]],
# # #         },
# # #     }

def suggest(frame, metrics, sidebar):

    if metrics is None:
        color = 'black'
        faults = ['No Keypoints Detected']

    else:
        faults, processed_metrics = review(metrics)
        n_faults = len([i for i in faults if i])
        if n_faults==3:
            color = 'red'
        elif n_faults==2:
            color = 'orange'
        elif n_faults==1:
            color = 'yellow'
        elif n_faults==0:
            color = 'green'

    frame = draw_border(frame, color)

    y_offset = 460

    for name,val in processed_metrics.items():

        cv2.putText(
            sidebar,
            f"{name}: {val:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        y_offset += 20

    y_offset += 20

    for i,fault in enumerate(faults):

        cv2.putText(
            sidebar,
            f"{fault}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        y_offset += 20

        cv2.putText(
            frame,
            f"{fault}",
            (20, 20+i*25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


    return frame, sidebar