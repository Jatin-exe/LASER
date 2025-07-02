#!/usr/bin/python3

import os
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import UInt16MultiArray
import numpy as np

class LaserTargetSubscriber(Node):
    def __init__(self):
        super().__init__('camera_laser_transformer')

        self.pixel_to_laser_transform, self.laser_to_pixel_transform = self.load_calibration_data()

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
            pixel_to_laser_transform = np.array(data["calib_pixel_to_laser"])
            laser_to_pixel_transform = np.array(data["calib_laser_to_pixel"])

            self.get_logger().info("Calibration data loaded and RBF models created successfully.")
            return pixel_to_laser_transform, laser_to_pixel_transform
        else:
            self.get_logger().error(f"Calibration file not found at {calibration_file}")
            return None, None
        
    def forwardTransform(self, source_x, source_y):
        transformed_source_x = (self.pixel_to_laser_transform[0] + self.pixel_to_laser_transform[1] * source_x + self.pixel_to_laser_transform[2] * source_x**2 +
                        self.pixel_to_laser_transform[3] * source_x**3 + self.pixel_to_laser_transform[4] * source_y + 
                        self.pixel_to_laser_transform[5] * source_y**2 + self.pixel_to_laser_transform[6] * source_x * source_y +
                        self.pixel_to_laser_transform[7] * source_x**2 * source_y + self.pixel_to_laser_transform[8] * source_x * source_y**2)

        transformed_source_y = (self.pixel_to_laser_transform[9] + self.pixel_to_laser_transform[10] * source_x + self.pixel_to_laser_transform[11] * source_x**2 +
                        self.pixel_to_laser_transform[12] * source_x**3 + self.pixel_to_laser_transform[13] * source_y + 
                        self.pixel_to_laser_transform[14] * source_y**2 + self.pixel_to_laser_transform[15] * source_x * source_y +
                        self.pixel_to_laser_transform[16] * source_x**2 * source_y + self.pixel_to_laser_transform[17] * source_x * source_y**2)
        return transformed_source_x, transformed_source_y
        
    def inverseTransform(self, source_x, source_y):
        transformed_source_x = (self.laser_to_pixel_transform[0] + self.laser_to_pixel_transform[1] * source_x + self.laser_to_pixel_transform[2] * source_x**2 +
                        self.laser_to_pixel_transform[3] * source_x**3 + self.laser_to_pixel_transform[4] * source_y + 
                        self.laser_to_pixel_transform[5] * source_y**2 + self.laser_to_pixel_transform[6] * source_x * source_y +
                        self.laser_to_pixel_transform[7] * source_x**2 * source_y + self.laser_to_pixel_transform[8] * source_x * source_y**2)

        transformed_source_y = (self.laser_to_pixel_transform[9] + self.laser_to_pixel_transform[10] * source_x + self.laser_to_pixel_transform[11] * source_x**2 +
                        self.laser_to_pixel_transform[12] * source_x**3 + self.laser_to_pixel_transform[13] * source_y + 
                        self.laser_to_pixel_transform[14] * source_y**2 + self.laser_to_pixel_transform[15] * source_x * source_y +
                        self.laser_to_pixel_transform[16] * source_x**2 * source_y + self.laser_to_pixel_transform[17] * source_x * source_y**2)
        return transformed_source_x, transformed_source_y

    def fireLaser(self, x, y, dwell):
        msg = UInt16MultiArray()
        msg.data = [int(x), int(y), int(dwell)]
        self.posePub_.publish(msg)

    def listener_callback(self, msg):
        laser_x, laser_y = self.forwardTransform(msg.point.x, msg.point.y)
        
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
