from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        default_value='/workspaces/isaac_ros-dev/src/data_sets/pucks_1',  # Replace with a default path if desired
        description='Path to the bag file to play'
    )

    bag_file_path = LaunchConfiguration('bag_file')

    return LaunchDescription([
        bag_file_arg,
        # Node for laser tracking
        Node(
            package='beta_tracking',
            executable='tracking',
            name='tracking',
            output='screen',
            parameters=[{
            'reliability_option': 'reliable',
            'annotation_options': '',
            'log_level': 'TRACE'
            }],
            remappings=[
                ('/image_topic','/camera/image_raw'),
                ('/detection_topic','/detections')
            ],
            arguments=['--ros-args', '--log-level', 'DEBUG']
        ),
        # ExecuteProcess to play the bag file
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_file_path, '--remap', '/vimbax_camera_beta/image_rect:=/camera/image_raw', '/laser/weed_detections:=/detections', '--clock'],
            output='screen'
        )
    ])
