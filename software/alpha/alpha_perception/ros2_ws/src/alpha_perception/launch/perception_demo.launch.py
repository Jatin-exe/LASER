from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare model variant argument
    model_variant_arg = DeclareLaunchArgument(
        'model_variant',
        default_value='nano',
        description='Model variant to use: nano, small, medium, large, xlarge'
    )

    # Declare GUI toggle argument
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Enable OpenCV GUI window for visualization (true/false)'
    )

    model_variant = LaunchConfiguration('model_variant')
    gui = LaunchConfiguration('gui')

    return LaunchDescription([
        model_variant_arg,
        gui_arg,

        Node(
            package='alpha_perception',
            executable='perception',
            name='alpha_perception',
            output='screen',
            parameters=[
                {'model_variant': model_variant},
                {'gui': gui}
            ],
            remappings=[('/image_raw', '/image_raw')]
        ),

        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play',
                '/ros2_ws/src/alpha_perception/datasets/04-23-2024_4w',
                '--clock'
            ],
            output='screen'
        )
    ])
