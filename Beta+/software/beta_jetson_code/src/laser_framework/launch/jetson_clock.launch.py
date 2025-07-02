from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # First Node: System Clock Publisher
        Node(
            package='laser_framework',
            executable='system_clock_publihser',
            name='system_clock_publihser',
            output='screen'
        ),
        
        # Second Node: SSD Storage Publisher
        Node(
            package='laser_framework',
            executable='storage_stats.py',
            name='storage_stats',
            output='screen'
        ),
        
        Node(
            package='beta_nn_update',
            executable='update_node.py',
            name='update_node',
            output='screen'
        )
    ])
