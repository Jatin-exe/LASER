#!/usr/bin/python3

import os
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import UInt16MultiArray
import numpy as np
from scipy.interpolate import RBFInterpolator

class LaserTargetSubscriber(Node):
    def __init__(self):
        super().__init__('camera_laser_transformer')

        self.rbf_pixel_to_laser, self.rbf_laser_to_pixel = self.load_calibration_data()

        # Set up QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to the /laser/current_target topic
        self.subscription = self.create_subscription(
            PointStamped,
            '/laser/current_target',
            self.listener_callback,
            qos_profile
        )

        self.posePub_ = self.create_publisher(UInt16MultiArray, "/laser/location", qos_profile=qos_profile)

    def load_calibration_data(self):
        calibration_file = '/workspaces/isaac_ros-dev/src/laser_framework/config/camera_laser_calib.json'
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                data = json.load(f)

            # Extract pixel and laser points
            pixel_points = np.array(data["pixel_points"])
            laser_points = np.array(data["laser_points"])

            # Load RBF parameters if available
            rbf_params = data.get("rbf_params", {})
            kernel = rbf_params.get("kernel", 'multiquadric')
            epsilon = rbf_params.get("epsilon", 3.0)
            smoothing = rbf_params.get("smoothing", 1.0)
            degree = rbf_params.get("degree", 2)

            # Create RBF interpolators
            rbf_pixel_to_laser = RBFInterpolator(
                pixel_points, laser_points,
                kernel=kernel,
                epsilon=epsilon,
                smoothing=smoothing,
                degree=degree
            )
            rbf_laser_to_pixel = RBFInterpolator(
                laser_points, pixel_points,
                kernel=kernel,
                epsilon=epsilon,
                smoothing=smoothing,
                degree=degree
            )
            self.get_logger().info("Calibration data loaded and RBF models created successfully.")
            return rbf_pixel_to_laser, rbf_laser_to_pixel 
        else:
            self.get_logger().error(f"Calibration file not found at {calibration_file}")
            return None, None

    def fireLaser(self, x, y, dwell):
        msg = UInt16MultiArray()
        msg.data = [int(x), int(y), int(dwell)]
        self.posePub_.publish(msg)

    def listener_callback(self, msg):
        pixel_coords = np.array([[msg.point.x, msg.point.y]])
        laser_coords = self.rbf_pixel_to_laser(pixel_coords)
        laser_x, laser_y = laser_coords[0]
        
        # Fire laser at calculated laser coordinates with the dwell time
        self.fireLaser(laser_x, laser_y, msg.point.z)


def main(args=None):
    rclpy.init(args=args)
    node = LaserTargetSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
