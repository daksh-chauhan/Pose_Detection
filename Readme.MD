## Sample Outputs

### test3_out.mp4
<video src="https://raw.githubusercontent.com/daksh-chauhan/Pose_Detection/main/samples/test3_out.mp4" controls width="700"></video>

### test4_out.mp4
<video src="https://raw.githubusercontent.com/daksh-chauhan/Pose_Detection/main/samples/test4_out.mp4" controls width="700"></video>

---


## Notes

* This implementation is designed for **side-profile squat analysis**.

* Pose estimation is dependent on keypoint visibility and confidence.
  If keypoints are not reliably detected, feedback is intentionally withheld to avoid incorrect suggestions.

* The system provides real-time visual feedback using posture metrics such as back angle, thigh angle, and knee-toe alignment.

## Usage

```
conda env create -f environment.yml
conda activate pose-env
python src/main.py
```

*In case of a keyboard interrupt-related error, simply re-run the script without deactivating the environment.*
