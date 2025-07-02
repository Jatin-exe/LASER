#!/usr/bin/python3

from typing import List, Dict
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Image
from scipy.interpolate import RBFInterpolator
import cv2
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import json
import os

class PointAndShoot(Node):
    def __init__(self):
        super().__init__("point_and_shoot")
        self.declare_parameter("dwell", 75)
        self.dwell = self.get_parameter("dwell").get_parameter_value().integer_value

        # Callback groups for handling each component separately
        self.pose_callback_group = MutuallyExclusiveCallbackGroup()
        self.camera_callback_group = MutuallyExclusiveCallbackGroup()

        # Set up QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Initialize CvBridge and other attributes
        self.bridge_ = CvBridge()
        self.rbf_pixel_to_laser, self.rbf_laser_to_pixel = self.load_calibration_data()
        
        # Publisher and subscription with callback groups
        self.posePub_ = self.create_publisher(UInt16MultiArray, "/laser/location", qos_profile=qos_profile, callback_group=self.pose_callback_group)
        self.cameraSub_ = self.create_subscription(Image, "/vimbax_camera_beta/image_rect", self.callback_, qos_profile=qos_profile, callback_group=self.camera_callback_group)

        # Register the mouse callback for displaying pixel location on click
        cv2.namedWindow("Camera Image")
        cv2.setMouseCallback("Camera Image", self.on_mouse_click)

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
        
    def fireLaser(self, x, y):
        msg = UInt16MultiArray()
        msg.data = [int(x), int(y), self.dwell]
        self.posePub_.publish(msg)

    def callback_(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge_.imgmsg_to_cv2(msg, "rgb8")

            # Resize the image
            resized_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5)

            # Display the resized image using OpenCV
            cv2.imshow("Camera Image", resized_image)
            cv2.waitKey(1)  # A small wait to refresh the display window

        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: %s" % e)

    def on_mouse_click(self, event, x, y, flags, param):
        # Check if the left mouse button is clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale up the coordinates to match the original image size
            scaled_x, scaled_y = int(x * 2), int(y * 2)
            self.get_logger().info(f"Mouse clicked at scaled pixel location: (x={scaled_x}, y={scaled_y})")

            # Transform the scaled pixel to laser coordinates if the RBF model is loaded
            if self.rbf_pixel_to_laser and self.rbf_laser_to_pixel:
                pixel_coords = np.array([[scaled_x, scaled_y]])
                laser_coords = self.rbf_pixel_to_laser(pixel_coords)
                laser_x, laser_y = laser_coords[0]
                
                # Fire laser at calculated laser coordinates
                self.fireLaser(laser_x, laser_y)
                self.get_logger().info(f"Corresponding laser coordinates: (x={int(laser_x)}, y={int(laser_y)})")

                # Convert laser coordinates back to pixel space
                back_to_pixel_coords = self.rbf_laser_to_pixel(np.array([[laser_x, laser_y]]))
                back_pixel_x, back_pixel_y = back_to_pixel_coords[0]
                self.get_logger().info(f"Back-converted pixel coordinates: (x={int(back_pixel_x)}, y={int(back_pixel_y)})")
            else:
                self.get_logger().warn("RBF model is not loaded; cannot transform pixel to laser coordinates.")


def main():
    rclpy.init()
    pointAndShoot_ = PointAndShoot()
    rclpy.spin(pointAndShoot_)
    pointAndShoot_.destroy_node()
    rclpy.shutdown()
    # Release OpenCV windows on shutdown
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

