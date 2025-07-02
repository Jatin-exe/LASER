#!/usr/bin/python3

from typing import List, Dict
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Image, CompressedImage
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
        self.pixel_to_laser_transform, self.laser_to_pixel_transform, self.roi_corners = self.load_calibration_data()
        
        # Publisher and subscription with callback groups
        self.posePub_ = self.create_publisher(UInt16MultiArray, "/laser/location", qos_profile=qos_profile, callback_group=self.pose_callback_group)
        self.cameraSub_ = self.create_subscription(Image, "/vimbax_camera_beta/image_raw", self.callback_, qos_profile=qos_profile, callback_group=self.camera_callback_group)
        self.pointShootImage_ = self.create_publisher(CompressedImage, "/laser/point_shoot_image", qos_profile=qos_profile, callback_group=self.camera_callback_group)
        self.pixelLocSub_ = self.create_subscription(UInt16MultiArray, "/beta_gui/pixel_location", self.pixelCallback_, qos_profile=qos_profile, callback_group=self.pose_callback_group)

        # Register the mouse callback for displaying pixel location on click
        cv2.namedWindow("Camera Image")
        cv2.setMouseCallback("Camera Image", self.on_mouse_click)

    def load_calibration_data(self):
        calibration_file = '/workspaces/isaac_ros-dev/src/laser_framework/config/camera_laser_calib.json'
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                data = json.load(f)

            # Extract pixel and laser points
            pixel_to_laser_transform = np.array(data["calib_pixel_to_laser"])
            laser_to_pixel_transform = np.array(data["calib_laser_to_pixel"])
            
            # Extract ROI corners
            roi_corners = [
                tuple(data["roi"]["top_left"]),
                tuple(data["roi"]["top_right"]),
                tuple(data["roi"]["bottom_right"]),
                tuple(data["roi"]["bottom_left"])
            ]

            self.get_logger().info("Calibration data and ROI loaded successfully.")
            return pixel_to_laser_transform, laser_to_pixel_transform, roi_corners
        else:
            self.get_logger().error(f"Calibration file not found at {calibration_file}")
            return None, None, None
    
    def fireLaser(self, x, y):
        msg = UInt16MultiArray()
        msg.data = [int(x), int(y), self.dwell]
        self.posePub_.publish(msg)
        
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

    def crop_roi(self, image):
        if self.roi_corners:
            x_min = min(p[0] for p in self.roi_corners)
            y_min = min(p[1] for p in self.roi_corners)
            x_max = max(p[0] for p in self.roi_corners)
            y_max = max(p[1] for p in self.roi_corners)
            return image[y_min:y_max, x_min:x_max], x_min, y_min
        return image, 0, 0

    def callback_(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge_.imgmsg_to_cv2(msg, "bgr8")

            # Crop ROI from the image
            cropped_image, self.roi_x_min, self.roi_y_min = self.crop_roi(cv_image)

            _, encoded_image = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 10])  # Adjust quality (0-100)

            # Create a CompressedImage message
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header  # Copy header from original message
            compressed_msg.header.frame_id = "point_shoot_frame"
            compressed_msg.format = "rgb8; jpeg compressed bgr8"  # Format must be specified
            compressed_msg.data = encoded_image.tobytes()  # Convert to bytes

            # Publish the compressed image
            self.pointShootImage_.publish(compressed_msg)

            # Display the resized image using OpenCV
            #cv2.imshow("Camera Image", cropped_image)
            #cv2.waitKey(1)  # A small wait to refresh the display window

        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: %s" % e)

    def pixelCallback_(self, msg):
        scaled_x, scaled_y = int(msg.data[0]) + self.roi_x_min, int(msg.data[1]) + self.roi_y_min
        # Transform the scaled pixel to laser coordinates if the RBF model is loaded
        if self.pixel_to_laser_transform is not None and self.laser_to_pixel_transform is not None:
            laser_x, laser_y = self.forwardTransform(scaled_x, scaled_y)
            
            # Fire laser at calculated laser coordinates
            self.fireLaser(laser_x, laser_y)

    def on_mouse_click(self, event, x, y, flags, param):
        # Check if the left mouse button is clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale up the coordinates to match the cropped image size
            scaled_x, scaled_y = int(x) + self.roi_x_min, int(y) + self.roi_y_min
            self.get_logger().info(f"Mouse clicked at scaled pixel location: (x={scaled_x}, y={scaled_y})")

            # Transform the scaled pixel to laser coordinates if the RBF model is loaded
            if self.pixel_to_laser_transform is not None and self.laser_to_pixel_transform is not None:
                laser_x, laser_y = self.forwardTransform(scaled_x, scaled_y)
                
                # Fire laser at calculated laser coordinates
                self.fireLaser(laser_x, laser_y)
                self.get_logger().info(f"Corresponding laser coordinates: (x={int(laser_x)}, y={int(laser_y)})")
            else:
                self.get_logger().warn("Calib was missing.")

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
