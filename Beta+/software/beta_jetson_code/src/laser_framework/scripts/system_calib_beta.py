#!/usr/bin/python3

from typing import List, Dict
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import UInt16MultiArray, UInt16
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import json
from io import BytesIO
import statistics

class Calibration(Node):
    def __init__(self):
        super().__init__("calibration")

        # Retrieve grid size parameters from ROS, with defaults if not set
        self.declare_parameter("grid_x", 3)
        self.declare_parameter("grid_y", 3)
        self.declare_parameter("dwell", 75)
        self.grid_x_points = self.get_parameter("grid_x").get_parameter_value().integer_value
        self.grid_y_points = self.get_parameter("grid_y").get_parameter_value().integer_value
        self.dwell = self.get_parameter("dwell").get_parameter_value().integer_value

        # Callback groups for handling each component separately
        self.pose_callback_group = MutuallyExclusiveCallbackGroup()
        self.camera_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()

        # Correspondence dictionary to store laser and image coordinates
        self.correspondences: Dict[int, Dict[str, List[int]]] = {}

        # Set up QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publisher and subscription with callback groups
        self.posePub_ = self.create_publisher(UInt16MultiArray, "/laser/location", qos_profile=qos_profile, callback_group=self.pose_callback_group)
        self.cameraSub_ = self.create_subscription(Image, "/vimbax_camera_beta/image_rect", self.callback_, qos_profile=qos_profile, callback_group=self.camera_callback_group)
        self.calibProgressPub_ = self.create_publisher(UInt16, "/laser/calibration_progress", qos_profile = qos_profile, callback_group=self.camera_callback_group)
        self.calibImgPub_ = self.create_publisher(Image, "/laser/calibration_image", qos_profile = qos_profile, callback_group=self.timer_callback_group)

        # Initialize CvBridge and other attributes
        self.bridge_ = CvBridge()
        self.latest_image = None
        self.capture_next_frames = False
        self.frames_to_capture = 10  # Number of frames to capture at each point
        self.laser_positions = []  # Store (cX, cY) positions of each frame
        self.position_tolerance = 10  # Tolerance for (cX, cY) variation in pixels
        self.grid_points = self.create_grid_points()  # Grid points based on grid size
        self.current_grid_index = 0  # Start at the first grid point
        self.retry_in_progress = False  # Flag to indicate if a retry is in progress
        self.validation_in_progress = False
        self.rbf_pixel_to_laser = None
        self.rbf_laser_to_pixel = None
        self.epsilon = 3.0
        self.smoothing = 1.0
        self.degree = 2

        # Timer callback for automated grid traversal with separate callback group
        self.traversal_timer = self.create_timer(1, self.move_to_next_grid_point, callback_group=self.timer_callback_group)

    def create_grid_points(self):
        x_start, x_end = 500, 9800
        y_start, y_end = 2000, 10500

        # Calculate spacing based on the grid size
        x_spacing = round((x_end - x_start) / (self.grid_x_points - 1))
        y_spacing = round((y_end - y_start) / (self.grid_y_points - 1))

        # Create grid points based on specified number of points along x and y axes
        grid_points = [
            (
                min(x_start + i * x_spacing, x_end),  # Ensure x does not exceed x_end
                min(y_start + j * y_spacing, y_end)   # Ensure y does not exceed y_end
            )
            for i in range(self.grid_x_points) for j in range(self.grid_y_points)
        ]
        return grid_points

    def findLaser(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding to isolate the laser spot
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(main_contour)
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            cv2.drawContours(img, [main_contour], -1, (0, 255, 0), 2)
            cv2.circle(img, (cX, cY), 7, (0, 0, 0), -1)

            return img, (cX, cY)
        else:
            return img, (None, None)

    def fireLaser(self, x, y):
        msg = UInt16MultiArray()
        msg.data = [x, y, self.dwell]
        self.posePub_.publish(msg)

    def callback_(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge_.imgmsg_to_cv2(msg, "rgb8")

            # Capture frames if in capture mode
            if self.capture_next_frames and self.frames_to_capture > 0 and self.rbf_pixel_to_laser is None:
                cv_image, (cX, cY) = self.findLaser(cv_image)
                
                # Only record positions if a valid laser spot was detected
                if cX is not None and cY is not None:
                    self.laser_positions.append((cX, cY))
                
                self.frames_to_capture -= 1

                # Stop capture mode if all frames have been captured
                if self.frames_to_capture == 0:
                    self.capture_next_frames = False
                    self.check_position_consistency()
                            
            if self.rbf_pixel_to_laser is None:
                ros_image = self.bridge_.cv2_to_imgmsg(cv_image, encoding="rgb8")
                
                # Set the header of the new ROS Image message to match the original
                ros_image.header = msg.header
                ros_image.header.frame_id = "calibration_frame"
                self.calibImgPub_.publish(ros_image)
            

        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: %s" % e)

    def check_position_consistency(self):
        # Only proceed if at least 2 frames had valid laser positions
        if len(self.laser_positions) < 3:
            self.retry_capture()
            return
        
        # Calculate the average (cX, cY) position
        positions = np.array(self.laser_positions)
        mean_position = np.mean(positions, axis=0).astype(int).tolist()

        # Record the laser and mean image coordinates for the current grid point
        grid_key = self.current_grid_index + 1
        laser_position = self.grid_points[self.current_grid_index]
        self.correspondences[grid_key] = {
            "laser": list(laser_position),
            "image": mean_position
        }

        #self.get_logger().info("{}".format(self.correspondences))
        # Move to the next grid point after successful capture
        self.current_grid_index += 1  
        self.retry_in_progress = False  # End retry state after successful capture
        # Assuming `total_grid_points` is the product of grid_x_points and grid_y_points
        total_grid_points = self.grid_x_points * self.grid_y_points

        # Calculate the progress percentage based on the current grid index
        progress_percentage = int((self.current_grid_index / total_grid_points) * 100)

        # Publish the progress percentage
        self.calibProgressPub_.publish(self.genProgressMsg(progress_percentage))

        if (grid_key) % self.grid_y_points == 0:
            row_data = [self.correspondences[key] for key in range(grid_key - self.grid_y_points + 1, grid_key + 1)]
            self.check_for_outliers(start_index_global=(grid_key - self.grid_y_points), row_data=row_data, x_tolerance=50, y_tolerance=50)

    def check_for_outliers(self, start_index_global, row_data, x_tolerance=50, y_tolerance=50):
        self.validation_in_progress = True
        x_outliers_found = False
        
        # Extract x values from the row_data
        x_values = [entry["image"][0] for entry in row_data]  # All x values in row

        # Check x values for consistency within x_tolerance using median
        median_x = statistics.median(x_values)
        for i, x in enumerate(x_values, start=1):
            if abs(x - median_x) > x_tolerance:
                grid_key = start_index_global + i
                self.get_logger().info(f"Outlier detected and removed in x-coordinates at grid key {grid_key}: {x} deviates from median {median_x} by more than {x_tolerance}")
                x_outliers_found = True
                
                # Remove the outlier from correspondences
                if grid_key in self.correspondences:
                    del self.correspondences[grid_key]

        if not x_outliers_found:
            self.get_logger().info("X values look good within tolerance.")

        self.validation_in_progress = False


    def genProgressMsg(self, progess):
        msg = UInt16()
        msg.data = progess
        return msg

    def retry_capture(self):
        # Set retry flag to true to block automatic traversal during retry
        self.retry_in_progress = True
        x, y = self.grid_points[self.current_grid_index]
        self.fireLaser(x, y)  # Fire the laser again for retry
        self.laser_positions = []  # Reset collected positions
        self.capture_next_frames = True  # Re-enable capture
        self.frames_to_capture = 10  # Set frames to capture again

    def move_to_next_grid_point(self):
        # Only proceed if a retry is not in progress
        if not self.retry_in_progress:
            if self.current_grid_index < len(self.grid_points) and self.rbf_pixel_to_laser is None and not self.validation_in_progress:
                x, y = self.grid_points[self.current_grid_index]
                self.fireLaser(x, y)
                self.capture_next_frames = True
                self.frames_to_capture = 10  # Set number of frames to capture
                self.laser_positions = []  # Reset laser positions for new capture
            elif self.current_grid_index >= len(self.grid_points) and self.rbf_pixel_to_laser is None and not self.validation_in_progress:
                # All grid points completed, create RBF interpolators
                self.create_rbf_interpolators()
                final_calib_image = self.visualize_correspondences()
                # Convert the OpenCV image to a ROS Image message
                ros_image = self.bridge_.cv2_to_imgmsg(final_calib_image, encoding="rgb8")
                # Set the header of the new ROS Image message to a new frame id and timestamp
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "calibration_frame"  # Set a unique frame_id for this calibration image
                self.calibImgPub_.publish(ros_image)
                self.export_calibration_data()
            elif self.current_grid_index >= len(self.grid_points) and self.rbf_pixel_to_laser is not None:
                raise SystemExit

    def create_rbf_interpolators(self):
        # Extract points for RBF training
        pixel_points = np.array([v["image"] for v in self.correspondences.values()])
        laser_points = np.array([v["laser"] for v in self.correspondences.values()])

        self.rbf_pixel_to_laser = RBFInterpolator(
            pixel_points, laser_points,
            kernel='multiquadric',  # Suitable for non-linear, sparse data
            epsilon=self.epsilon,            # Adjust based on point density
            smoothing=self.smoothing,          # To handle noise
            degree=self.degree                # For a flexible quadratic baseline
        )

        self.rbf_laser_to_pixel = RBFInterpolator(
            laser_points, pixel_points,
            kernel='multiquadric',
            epsilon=self.epsilon,
            smoothing=self.smoothing,
            degree=self.degree
        )

    def visualize_correspondences(self):
        # Extract only the pixel points and labels
        pixel_points = np.array([v["image"] for v in self.correspondences.values()])
        labels = list(self.correspondences.keys())

        # Plot only the raw pixel points
        plt.figure(figsize=(12, 6))
        plt.scatter(pixel_points[:, 0], pixel_points[:, 1], c='blue', marker='x', label='Pixel Points')

        # Label each pixel point
        for i, label in enumerate(labels):
            plt.text(pixel_points[i, 0], pixel_points[i, 1], f"{label}", color="blue", fontsize=8)

        # Configure plot appearance
        plt.legend()
        plt.title("Raw Pixel Points with Grid Labels")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        
        # Save the plot to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format = 'png')
        buf.seek(0)

        # Convert buffer to a NumPy array
        img_arr = np.frombuffer(buf.getvalue(), dtype = np.uint8)
        buf.close()

        # Use OpenCV to read the image from memory buffer
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img


    def calculate_roi(self):
        """Calculate the ROI based on the first, last, and corner points of the grid using 1-based indexing."""
        grid_size = self.grid_x_points * self.grid_y_points

        # Ensure we have enough points
        if grid_size > len(self.correspondences):
            return None

        # Calculate the 1-based indices of the 4 corner points
        index_top_left = 1  # First point in the grid
        index_top_right = self.grid_x_points  # Last point in the first row
        index_bottom_left = grid_size - self.grid_x_points + 1  # First point in the last row
        index_bottom_right = grid_size  # Last point in the last row

        # Fetch the pixel coordinates of the corner points
        roi = {
            "top_left": self.correspondences[index_top_left]["image"],
            "top_right": self.correspondences[index_top_right]["image"],
            "bottom_left": self.correspondences[index_bottom_left]["image"],
            "bottom_right": self.correspondences[index_bottom_right]["image"],
        }
        
        return roi

    # Methods for saving and loading data that incorporate the ROI
    def export_calibration_data(self, filename="camera_laser_calib.json"):
        """Save pixel and laser points along with RBF parameters and ROI to a JSON file."""
        pixel_points = [v["image"] for v in self.correspondences.values()]
        laser_points = [v["laser"] for v in self.correspondences.values()]

        roi = self.calculate_roi()  # Get ROI data

        rbf_params = {
            "kernel": 'multiquadric',
            "epsilon": self.epsilon,
            "smoothing": self.smoothing,
            "degree": self.degree
        }

        data = {
            "pixel_points": pixel_points,
            "laser_points": laser_points,
            "roi": roi,
            "rbf_params": rbf_params
        }

        with open('/workspaces/isaac_ros-dev/src/laser_framework/config/{}'.format(filename), 'w') as f:
            json.dump(data, f, indent=4)

def main():
    rclpy.init()
    calib_ = Calibration()
    rclpy.spin(calib_)
    calib_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

