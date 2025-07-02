from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare the launch argument for stay_duration
    dwell_duration_arg = DeclareLaunchArgument(
        'dwell', 
        default_value='0.2',  # Default value for stay_duration
        description='Dwell duration parameter for the planner node'
    )

    # Node configuration
    planner_node = Node(
        package='beta_path_planning',
        executable='planner_node',
        name='beta_path_planner',
        output='screen',
        parameters=[
            {'dwell': LaunchConfiguration('dwell')}  # Use the launch argument
        ]
    )

    planner_vis_node = Node(
        package='beta_path_planning',
        executable='planner_viz.py',
        name='beta_path_planner_viz',
        output='screen'
    )

    target_pub_node = Node(
        package='beta_path_planning',
        executable='camera_laser_transformer',
        name='beta_path_planner_target_pub',
        output='screen'
    )

    return LaunchDescription([
        dwell_duration_arg,  # Add the launch argument
        planner_node,        # Add the node
        planner_vis_node,
        target_pub_node
    ])
