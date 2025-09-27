# Simple Motion Planner for Pointing Models

This demo is for executing pointing models (e.g. ER1, FSD, RoboPoint) on XArm 6 with a RealSense Depth Camera.

## How to Use?

You need first update in `xarm_env.py` the `CAMERA_TO_WORLD` and `ARM_TO_WORLD` to match your robot's configuration. This process is called "camera extrinsic calibration".

Then, you can run the `python pick_place_er1_trace.py` to execute the pointing models.

## License

This project is licensed under the MIT License. See the LICENSE file for details.