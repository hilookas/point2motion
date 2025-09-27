import time
from xarm_env import XarmEnv
import traceback
from realsense_env import RealsenseEnv
from xarm_env import CAMERA_TO_WORLD, ARM_TO_WORLD, TCP_TO_EEF
# from self.o3d_vis import Open3dVisualizer
import cv2
import numpy as np
import math
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import os
import atexit
from argparse import ArgumentParser

class VideoRecorder:
    def __init__(self, filename):
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MP4V"), 30, (640, 480))
        atexit.register(self.finish)
        
    def render(self, obs, action=None):
        self.video_writer.write(cv2.cvtColor(np.asarray(obs["im_rgbd"].color), cv2.COLOR_RGB2BGR))
    
    def finish(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

def get_xyz_from_uvd(u, v, d, intrinsic_matrix, depth_scale):
    # https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

class MotionPlanner:
    def __init__(self, model_type="er1_trace"):
        self.model_type = model_type
        os.makedirs(f"log_{self.model_type}", exist_ok=True)
        
        self.env = XarmEnv(force_control_enabled=True)
        self.rs_env = RealsenseEnv()
        
    def move_arm(self, goal_point):
        self.goal_point = goal_point

        last_time = time.time()
        while True:
            obs = self.env.step(self.goal_point)
            obs |= self.rs_env.step()
            # self.o3d_vis.render(obs, None)
            self.video_rec.render(obs, None)
            
            img_rgb = np.asarray(obs["im_rgbd"].color)
            img_depth = np.asarray(obs["im_rgbd"].depth)            
            
            pos_dist = math.dist(self.goal_point[:3], obs["cart_pos"][:3])
            rot_dist = pr.quaternion_dist(
                pr.quaternion_from_extrinsic_euler_xyz(self.goal_point[3:6]),
                pr.quaternion_from_extrinsic_euler_xyz(obs["cart_pos"][3:6])
            )
            
            print(f"pos_dist {pos_dist}mm, rot_dist {rot_dist / math.pi * 180}deg")
            
            if (pos_dist < 20):
                break

            print(f"fps: {1/(time.time() - last_time)}")
            last_time = time.time()
        
        return obs
                    
    def move_gripper(self, gripper_open=False):
        start_time = time.time()
        last_time = time.time()
        while time.time() - start_time < 1:
            obs = self.env.step(self.goal_point, gripper_action=840 if gripper_open else 0)
            obs |= self.rs_env.step()
            # self.o3d_vis.render(obs, None)
            self.video_rec.render(obs, None)

            print(f"fps: {1/(time.time() - last_time)}")
            last_time = time.time()
        
        return obs
            
    def preexecute(self):
        self.ts_str = time.strftime('%Y%m%d_%H%M%S')
        
        # self.o3d_vis = Open3dVisualizer()
        self.video_rec = VideoRecorder(f"log_{self.model_type}/{self.ts_str}.mp4")
        
        obs = self.env.reset()
        obs |= self.rs_env.reset()
        # self.o3d_vis.render(obs, None)
        self.video_rec.render(obs, None)
        
        obs = self.move_arm(np.array([ 491, 0, 450, 3.14, 0, 0]))
        
        return obs
            
    def plan(self, task_instruction, obs):
        img_rgb = np.asarray(obs["im_rgbd"].color)
        img_depth = np.asarray(obs["im_rgbd"].depth)
        
        assert self.model_type == "er1_trace"
        from waypoint_er1_trace import get_waypoints
        
        image_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(f"log_{self.model_type}/{self.ts_str}_{task_instruction}.png", image_cv)

        cv2.imshow(task_instruction, image_cv)
        cv2.waitKey(100)
        
        (u, v), (u2, v2), points, text_result = get_waypoints(img_rgb, task_instruction)
        
        with open(f"log_{self.model_type}/{self.ts_str}_{task_instruction}_pick_place_result.txt", "w") as f:
            f.write(text_result)
        
        for point in points:
            image_cv = cv2.circle(image_cv, (int(point[0]), int(point[1])), 5, (0, 0, 127), -1)
        image_cv = cv2.circle(image_cv, (int(u), int(v)), 5, (0, 0, 255), -1)
        image_cv = cv2.circle(image_cv, (int(u2), int(v2)), 5, (255, 0, 0), -1)

        cv2.imwrite(f"log_{self.model_type}/{self.ts_str}_{task_instruction}_result.png", image_cv)
        
        cv2.imshow(task_instruction, image_cv)
        cv2.waitKey(100)
        
        debug_image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
        d = img_depth[int(v), int(u)].item()
        pick_goal_uvd = (u, v, d)
        d2 = img_depth[int(v2), int(u2)].item()
        place_goal_uvd = (u2, v2, d2)
        
        pick_goal_xyz = get_xyz_from_uvd(*pick_goal_uvd, self.rs_env.intrinsic_matrix, self.rs_env.depth_scale)
        place_goal_xyz = get_xyz_from_uvd(*place_goal_uvd, self.rs_env.intrinsic_matrix, self.rs_env.depth_scale)

        point2camera = np.array([[*pick_goal_xyz, 1]]).T
        point2world = CAMERA_TO_WORLD @ point2camera
        pick_point2arm = np.linalg.inv(ARM_TO_WORLD) @ point2world
        pick_point2arm[2, 0] += TCP_TO_EEF[2, 3]
        assert pick_point2arm[2, 0] > 0.0

        point2camera = np.array([[*place_goal_xyz, 1]]).T
        point2world = CAMERA_TO_WORLD @ point2camera
        place_point2arm = np.linalg.inv(ARM_TO_WORLD) @ point2world
        place_point2arm[2, 0] += TCP_TO_EEF[2, 3]
        assert place_point2arm[2, 0] > 0.0
        
        place_point2arm[2, 0] += 0.020

        mid_point2arm = (pick_point2arm + place_point2arm) / 2
        mid_point2arm[2, 0] += 0.300

        trace = [
            (*(pick_point2arm[:3, 0] * 1000), 3.14, 0, 0), 
            (*(mid_point2arm[:3, 0] * 1000), 3.14, 0, 0), 
            (*(place_point2arm[:3, 0] * 1000), 3.14, 0, 0)
        ]
        
        return trace, text_result, debug_image_rgb
    
    def execute(self, trace):
        print("execute")
        
        self.move_arm(trace[0])

        self.move_gripper(gripper_open=False)
        
        for tr in trace[1:-1]:
            self.move_arm(tr)

        self.move_arm(trace[-1])
        
        self.move_gripper(gripper_open=True)
        
        self.move_arm(np.array([ 491, 0, 450, 3.14, 0, 0]))
        
        self.video_rec.finish()
        print("finish")

def main():
    parser = ArgumentParser()

    parser.add_argument('task_instruction',
                        type=str,
                        default="Pick up the sponge and put it on the plate.",
                        help='Task instruction.')

    args = parser.parse_args()
    
    model_type = "er1_trace"
    task_instruction = args.task_instruction
    
    planner = MotionPlanner(model_type)
    
    try:
        obs = planner.preexecute()
        
        img_rgb = np.asarray(obs["im_rgbd"].color)
        
        trace, text_result, debug_image_rgb = planner.plan(task_instruction, obs)
        
        print("press any key")
        cv2.waitKey(0)
        
        planner.execute(trace)

    except KeyboardInterrupt:
        pass

    except:
        print(traceback.format_exc())

    finally:
        pass

if __name__ == "__main__":
    main()
