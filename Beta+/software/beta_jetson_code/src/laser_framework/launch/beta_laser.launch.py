import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, ExecuteProcess, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessExit
import os

def generate_launch_description():
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
    
    tracking_launch_path = os.path.join(
        get_package_share_directory('beta_tracking'),
        'launch',
        'tracking.launch.py'
    )
    
    planner_launch_path = os.path.join(
        get_package_share_directory('beta_path_planning'),
        'launch',
        'planner.launch.py'
    )

    # Include the launch files
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(camera_launch_path)
    )

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(perception_launch_path)
    )
    
    tracking_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(tracking_launch_path)
    )
    
    planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(planner_launch_path)
    )
    
    # Define the targeting node that should start immediately
    system_targeting_node = Node(
        package='laser_framework',
        executable='system_targeting.py',
        name='system_targeting',
        output='screen'
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
        tracking_launch,
        planner_launch,
        system_targeting_node,
        stop_signal_node,
        shutdown_handler
    ])

