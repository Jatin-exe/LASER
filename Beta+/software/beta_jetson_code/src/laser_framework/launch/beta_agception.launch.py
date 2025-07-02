import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime

def generate_launch_description():
    # Declare a launch argument for the bag file base name
    bag_file_name_arg = DeclareLaunchArgument(
        'bag_file_name',
        default_value='default_bag',  # Default value if none is provided
        description='Base name of the output bag file'
    )

    # Paths to the individual launch files
    camera_launch_path = os.path.join(
        get_package_share_directory('vimbax_camera'),
        'launch',
        'vimbax_camera_rect_launch.py'
    )

    perception_launch_path = os.path.join(
        get_package_share_directory('beta_perception'),
        'launch',
        'perception.launch.py'
    )

    # Include the launch files
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(camera_launch_path)
    )

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(perception_launch_path)
    )

    # Stop Signal Node (Python Node)
    stop_signal_node = Node(
        package='laser_framework',  # Replace with the package containing stop_signal.py
        executable='stop_signal.py',  # Replace with the name of the Python executable
        name='stop_signal_node',
        output='screen'
    )

    # Event handler to shut down all nodes when stop_signal exits
    shutdown_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=stop_signal_node,
            on_exit=[launch.actions.Shutdown()]
        )
    )

    # Return the combined launch description
    return LaunchDescription([
        camera_launch,
        perception_launch,
        stop_signal_node,
        shutdown_handler
    ])

