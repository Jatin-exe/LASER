import launch
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare user-defined arguments
    grid_x_arg = DeclareLaunchArgument('grid_x', default_value='3', description='Grid size in X direction')
    grid_y_arg = DeclareLaunchArgument('grid_y', default_value='3', description='Grid size in Y direction')
    dwell_arg = DeclareLaunchArgument('dwell', default_value='75', description='Dwell time for the laser')

    # Path to the existing launch file
    camera_launch_path = os.path.join(
        get_package_share_directory('vimbax_camera'),
        'launch',
        'camera_laser_calib_rect_launch.py'
    )

    # Include the existing launch file
    camera_container = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(camera_launch_path)
    )

    # Define the targeting node that should start immediately
    system_targeting_node = Node(
        package='laser_framework',
        executable='system_targeting.py',
        name='system_targeting',
        output='screen'
    )

    # Define the calibration node but delay its start until the required topics are available
    system_calib_beta_node = Node(
        package='laser_framework',
        executable='system_calib_beta_least_squares.py',
        name='system_calib_beta',
        output='screen',
        parameters=[{
            'grid_x': LaunchConfiguration('grid_x'),
            'grid_y': LaunchConfiguration('grid_y'),
            'dwell': LaunchConfiguration('dwell')
        }]
    )

    # Timer to start system_calib_beta_node after a short delay to allow topics to become available
    start_calib_beta_node = TimerAction(
        period=5.0,  # Adjust based on expected startup time for topics
        actions=[system_calib_beta_node]
    )

    # Event handler to shut down all nodes when system_calib_beta_node exits
    shutdown_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=system_calib_beta_node,
            on_exit=[launch.actions.Shutdown()]
        )
    )

    return launch.LaunchDescription([
        grid_x_arg,
        grid_y_arg,
        dwell_arg,
        camera_container,
        system_targeting_node,
        start_calib_beta_node,  # Delays system_calib_beta_node start until after the camera launch
        shutdown_handler  # Register the shutdown handler for system_calib_beta_node
    ])

