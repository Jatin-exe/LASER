# launch/beta_perception_launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess

def generate_launch_description():
    # Declare launch argument
    crop_type_arg = DeclareLaunchArgument(
        'crop_type',
        default_value='carrot',
        description='Crop type to route to the correct perception model'
    )

    # LaunchConfiguration to get the user input
    crop_type = LaunchConfiguration('crop_type')
    
    return LaunchDescription([

        # Start the Triton server as a separate process
        ExecuteProcess(
            cmd=['/opt/tritonserver/bin/tritonserver', '--backend-config=tensorrt,version=lean', '--model-repository=/workspaces/isaac_ros-dev/src/beta_perception/weights', '--cuda-memory-pool-byte-size=0:3221225472', '--pinned-memory-pool-byte-size=536870912'],
            output='screen'
        ),

        # Start the visualization node
        Node(
            package='beta_perception',
            executable='visualization',
            name='visualization_node',
            output='screen'
        ),
        
        # Start the detector node
        Node(
            package='beta_perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{'crop_type': crop_type}]
        ),
    ])
