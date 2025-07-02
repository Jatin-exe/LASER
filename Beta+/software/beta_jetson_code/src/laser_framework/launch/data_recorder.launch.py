import launch
from launch import LaunchDescription
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime

def resolve_and_record_bag(context, *args, **kwargs):
    # Define the constant directory path where all bag files will be stored
    bag_file_directory = '/workspaces/isaac_ros-dev/src/laser_framework/data_sets'  # Replace with your desired directory
    os.makedirs(bag_file_directory, exist_ok=True)

    # Resolve the value of the bag file name from the LaunchConfiguration
    bag_file_name = LaunchConfiguration('bag_file_name').perform(context)

    # Add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_bag_file_path = os.path.join(bag_file_directory, f'{bag_file_name}_{timestamp}')

    # Create the ExecuteProcess action
    bag_record_action = Node(
            package='laser_framework',  # Replace with your actual package name
            executable='keyframe',  # Replace with the executable name from CMakeLists.txt
            name='keyframe',
            parameters=[
                {'image_topic': '/vimbax_camera_beta/image_raw'},
                {'save_directory': unique_bag_file_path},
                {'max_keyframes': 100},
            ],
            output='screen'
        )

    # Return the action inside a list
    return [GroupAction([bag_record_action])]

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
        bag_file_name_arg,  # Add the launch argument
        camera_launch,
        # perception_launch,
        OpaqueFunction(function=resolve_and_record_bag),  # Use OpaqueFunction to resolve and execute the bag record action
        stop_signal_node,
        shutdown_handler
    ])

