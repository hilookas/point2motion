import atexit
import open3d as o3d
import numpy as np
from open3d_RGBD import getOpen3DFromTrimeshScene
from trimesh_URDF import getURDF
from trimesh_render import lookAt
from xarm_env import CAMERA_TO_WORLD, ARM_TO_WORLD

class Open3dVisualizer:
    def __init__(self):
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        plane_mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.01)
        plane_mesh_box.translate([-0.5, -0.5, -0.01])
        plane_mesh_box.paint_uniform_color([0.5, 0.0, 0.0])  # Gray color
        self.pcd = o3d.geometry.PointCloud()
        self.arm_mesh = o3d.geometry.TriangleMesh()
        
        self.urdf, self.controller = getURDF("/home/ubuntu/miniforge3/envs/dp3/lib/python3.10/site-packages/pybullet_data/xarm/xarm6_robot_white.urdf")
        
        # Initialize the pointcloud viewer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud")

        self.vis.add_geometry(origin)
        self.vis.add_geometry(plane_mesh_box)
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.arm_mesh)
        
        self.vis.get_render_option().point_size = 1
        self.vis.get_render_option().background_color = np.asarray([0, 0, 0])
        
        view_control = self.vis.get_view_control()
        view_control.set_constant_z_far(1000)
        
        # Retrieve the camera parameters
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        # Set the extrinsic parameters, yz_flip is for Open3D camera configuration
        camera_pose = lookAt(eye=np.array([0., -1., 1.]), target=np.array([0. ,0., 0.]), up=np.array([0.0, 0.0, 1.0]), yz_flip=True)
        camera_params.extrinsic = np.linalg.inv(camera_pose)
        # Set the camera parameters
        view_control.convert_from_pinhole_camera_parameters(camera_params)

        atexit.register(self.cleanup)

    def render(self, obs, action=None) -> None:
        new_pcd: o3d.geometry.PointCloud = obs["pcd"].to_legacy()
        new_pcd.transform(CAMERA_TO_WORLD)

        self.pcd.points = new_pcd.points
        self.pcd.colors = new_pcd.colors
        self.vis.update_geometry(self.pcd)
        
        # Interact with the URDF
        for name, real_angle in zip(["link1", "link2", "link3", "link4", "link5", "link6"], obs['servo_angle']):
            self.controller[name].interact(real_angle)
        
        trimesh_scene = self.urdf.getMesh()

        color_map = {
            'base.stl': np.array([0.8,0.8,0.8]),
            'link1.stl': np.array([0.7,0.7,0.7]),
            'link2.stl': np.array([0.8,0.8,0.8]),
            'link3.stl': np.array([0.7,0.7,0.7]),
            'link4.stl': np.array([0.8,0.8,0.8]),
            'link5.stl': np.array([0.7,0.7,0.7]),
            'link6.stl': np.array([0.8,0.8,0.8]),
        }
        new_arm_mesh = getOpen3DFromTrimeshScene(trimesh_scene, color_map=color_map)
        new_arm_mesh.transform(ARM_TO_WORLD)
        
        self.arm_mesh.vertices = new_arm_mesh.vertices
        self.arm_mesh.triangles = new_arm_mesh.triangles
        self.arm_mesh.vertex_colors = new_arm_mesh.vertex_colors
        self.vis.update_geometry(self.arm_mesh)

        # Update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()

    def cleanup(self):
        self.vis.destroy_window()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--speed', type=int, default=1)
    args = parser.parse_args()

    # import zarr
    # root = zarr.open(args.path)
    # print(root.tree())

    import h5py
    root = h5py.File(args.path, 'r')

    o3d_vis = Open3dVisualizer()

    jump = 0
    for img, depth, intrinsic_matrix, depth_scale in zip(root["image_color"], root["image_depth"], root["intrinsic_matrix"], root["depth_scale"]):
        if jump < args.speed - 1:
            jump += 1
            continue
        else:
            jump = 0

        im_rgbd: o3d.geometry.RGBDImage = o3d.geometry.RGBDImage(
            color=o3d.geometry.Image(img),
            depth=o3d.geometry.Image(depth),
        )
        # print(im_rgbd)
        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            im_rgbd,
            o3d.core.Tensor(intrinsic_matrix, dtype=o3d.core.Dtype.Float32),
            depth_scale=depth_scale
        )

        o3d_vis.render({
            "pcd": new_pcd
        }, None)
