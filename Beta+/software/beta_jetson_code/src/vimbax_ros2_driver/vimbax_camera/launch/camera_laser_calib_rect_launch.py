import launch
from launch_ros.actions import ComposableNodeContainer
from launch.actions import IncludeLaunchDescription
from launch_ros.descriptions import ComposableNode
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    output_width = 1936
    output_height = 1216

    img_rectify_node = ComposableNode(
        name = 'img_rectify_node',
        package = 'isaac_ros_image_proc',
        plugin = 'nvidia::isaac_ros::image_proc::RectifyNode',
        parameters = [{
            'output_width': output_width,
            'output_height': output_height,
            'history': 'keep_last'
        }],
        remappings = [
            ('image_raw', '/vimbax_camera_beta/image_raw'),
            ('camera_info', '/vimbax_camera_beta/camera_info'),
            ('image_rect', '/vimbax_camera_beta/image_rect'),
            ('camera_info_rect', '/vimbax_camera_beta/camera_info_rect')
        ]
    )

    vimbax_camera_node = ComposableNode(
        name = 'vimbax_camera_beta',
        package = 'vimbax_camera',
        namespace = 'vimbax_camera_beta',
        plugin = 'vimbax_camera::VimbaXCameraNode',
        parameters=[{
                "buffer_count": 3,
                "settings_file": "/workspaces/isaac_ros-dev/src/vimbax_ros2_driver/vimbax_camera/config/laser_calibration.xml",
                "use_ros_time": True,
                "camera_frame_id": "beta_camera",
                "camera_info_url": "file:///workspaces/isaac_ros-dev/src/vimbax_ros2_driver/vimbax_camera/calibration/DEV_1AB22C0438A4.yaml",
                "history": "keep_last",
                "autostream": 1
            }],
            extra_arguments=[{'use_intra_process_comms': True}]
    )

    rectification_container = ComposableNodeContainer(
        name = 'rectification_container',
        package = 'rclcpp_components',
        executable = 'component_container_mt',
        composable_node_descriptions = [vimbax_camera_node],
        namespace = '',
        output = 'screen'
    )

    return launch.LaunchDescription([rectification_container])
