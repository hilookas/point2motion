import atexit
import open3d as o3d

class RealsenseEnv:
    def __init__(self, serial: str="", record: bool=False):
        print(o3d.t.io.RealSenseSensor.list_devices())

        self.rs = o3d.t.io.RealSenseSensor()
        config = o3d.t.io.RealSenseSensorConfig({
            "serial": serial,
            "color_format": "RS2_FORMAT_RGB8",
            "color_resolution": "640,480",
            "depth_format": "RS2_FORMAT_Z16",
            "depth_resolution": "640,480",
            "fps": "30",
            "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE"
        })
        if record:
            self.rs.init_sensor(config, 0, "debug.bag")
            self.rs.start_capture(True) # True: start recording with capture
        else:
            self.rs.init_sensor(config, 0)
            self.rs.start_capture()

        self.intrinsic_matrix = self.rs.get_metadata().intrinsics.intrinsic_matrix
        self.depth_scale = self.rs.get_metadata().depth_scale

        atexit.register(self.cleanup)

    def _get_observation(self) -> dict:
        im_rgbd: o3d.t.geometry.RGBDImage = self.rs.capture_frame(True, True)
        pcd: o3d.t.geometry.PointCloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics=o3d.core.Tensor(self.intrinsic_matrix, dtype=o3d.core.Dtype.Float32), depth_scale=self.depth_scale)

        return {
            "im_rgbd": im_rgbd,
            "intrinsic_matrix": self.intrinsic_matrix,
            "depth_scale": self.depth_scale,
            "pcd": pcd
        }

    def reset(self, action=None) -> dict:
        return self._get_observation()

    def step(self, action=None) -> dict:
        return self._get_observation()

    def cleanup(self):
        self.rs.stop_capture()

if __name__ == "__main__":
    '''
    View Intel Realsense D405 pointcloud in Open3D viewer
    Src: https://github.com/isl-org/Open3D/issues/6221
    '''

    from o3d_vis import Open3dVisualizer
    import traceback
    
    rs_env = RealsenseEnv()
    o3d_vis = Open3dVisualizer()
        
    try:
        while True:
            rs_obs = rs_env.step()
            rs_obs |= {
                "servo_angle": [1, 0, 0, 0, 0, 0]
            }
            o3d_vis.render(rs_obs, None)

    except KeyboardInterrupt:
        pass

    except:
        print(traceback.format_exc())

    finally:
        pass