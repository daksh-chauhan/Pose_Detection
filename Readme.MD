## Sample Outputs

![test3](samples/test3.gif)
![test4](samples/test4.gif)

Full videos:
- [test3_out.mp4](https://github.com/daksh-chauhan/Pose_Detection/blob/main/samples/test3_out.mp4)
- [test4_out.mp4](https://github.com/daksh-chauhan/Pose_Detection/blob/main/samples/test4_out.mp4)


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
