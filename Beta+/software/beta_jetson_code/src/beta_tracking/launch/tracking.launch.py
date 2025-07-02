from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    detection_score_arg = DeclareLaunchArgument(
        'detection_score',
        default_value='0.95',
        description='Detection score threshold'
    )

    refinement_offset_arg = DeclareLaunchArgument(
        'refinement_offset',
        default_value='0.1',
        description='Refinement offset'
    )

    refinement_mode_arg = DeclareLaunchArgument(
        'refinement_mode',
        default_value='TM_CCOEFF_NORMED',
        description='Refinement mode'
    )

    refinement_strategy_arg = DeclareLaunchArgument(
        'refinement_strategy',
        default_value='LAST_ANCHOR',
        description='Refinement strategy'
    )

    reliability_option_arg = DeclareLaunchArgument(
        'reliability_option',
        default_value='reliable',
        description='QoS reliability option (best_effort, reliable, or system_default)'
    )

    annotation_options_arg = DeclareLaunchArgument(
        'annotation_options',
        default_value='default',
        description='JSON string of custom annotation options'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='WARNING',
        description='Logging level (e.g., TRACE, DEBUG, INFO, WARNING, ERROR, or CRITICAL)'
    )

    downsample_arg = DeclareLaunchArgument(
        'fft_downscale_factor',
        default_value='0.05',
        description='Factor to downscale the FFT image (0 to disable downscaling)'
    )

    # Create the node
    tracking_node = Node(
        package='beta_tracking',
        executable='tracking',
        name='tracking',
        output='screen',
        parameters=[{
            'detection_score': LaunchConfiguration('detection_score'),
            'refinement_offset': LaunchConfiguration('refinement_offset'),
            'refinement_mode': LaunchConfiguration('refinement_mode'),
            'refinement_strategy': LaunchConfiguration('refinement_strategy'),
            'reliability_option': LaunchConfiguration('reliability_option'),
            'annotation_options': LaunchConfiguration('annotation_options'),
            'log_level': LaunchConfiguration('log_level'),
            'fft_downscale_factor': LaunchConfiguration('fft_downscale_factor')
        }],
        remappings=[
            ('image_topic', '/vimbax_camera_beta/image_raw'),
            ('detection_topic', '/laser/weed_detections'),
            ('annotated_image', '/laser/tracker_image'),
            ('tracker_output', '/laser/tracker')
        ]
    )

    return LaunchDescription([
        detection_score_arg,
        refinement_offset_arg,
        refinement_mode_arg,
        refinement_strategy_arg,
        reliability_option_arg,
        annotation_options_arg,
        log_level_arg,
        downsample_arg,
        tracking_node
    ])
