#!/usr/bin/python3

from typing import List, Dict
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import UInt16MultiArray, UInt16, Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
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

        qos_profile_laser_image = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=500
        )
        
        # Publisher and subscription with callback groups
        self.posePub_ = self.create_publisher(UInt16MultiArray, "/laser/location", qos_profile=qos_profile, callback_group=self.pose_callback_group)
        self.cameraSub_ = self.create_subscription(Image, "/vimbax_camera_beta/image_raw", self.callback_, qos_profile=qos_profile, callback_group=self.camera_callback_group)
        self.calibProgressPub_ = self.create_publisher(UInt16, "/laser/calibration_progress", qos_profile = qos_profile, callback_group=self.camera_callback_group)
        self.calibImgPub_ = self.create_publisher(CompressedImage, "/laser/calibration_image", qos_profile = qos_profile_laser_image, callback_group=self.timer_callback_group)
        self.rmsePub_ = self.create_publisher(Float32MultiArray, "/laser/rmse_values", qos_profile = qos_profile, callback_group=self.camera_callback_group)

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

        self.calib_pixel_to_laser = np.zeros(18)
        self.calib_laser_to_pixel = np.zeros(18)

        # Timer callback for automated grid traversal with separate callback group
        self.traversal_timer = self.create_timer(2, self.move_to_next_grid_point, callback_group=self.timer_callback_group)

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

    def findLaser(self, img, min_area=20):  # min_area: Minimum contour area to consider
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding to isolate the laser spot
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if valid_contours:
            # Find the largest valid contour
            main_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(main_contour)
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Draw the largest valid contour
            cv2.drawContours(img, [main_contour], -1, (0, 255, 0), 2)
            cv2.circle(img, (cX, cY), 7, (0, 0, 255), -1)

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
            if self.capture_next_frames and self.frames_to_capture > 0 and np.all(self.calib_pixel_to_laser == 0) and np.all(self.calib_laser_to_pixel == 0):
                cv_image, (cX, cY) = self.findLaser(cv_image)
                
                # Only record positions if a valid laser spot was detected
                if cX is not None and cY is not None:
                    self.laser_positions.append((cX, cY))
                
                self.frames_to_capture -= 1

                # Stop capture mode if all frames have been captured
                if self.frames_to_capture == 0:
                    self.capture_next_frames = False
                    self.check_position_consistency()

            if np.all(self.calib_pixel_to_laser == 0) and np.all(self.calib_laser_to_pixel == 0):
                for grid_key, info in self.correspondences.items():
                    cv2.circle(cv_image, (info["image"][0], info["image"][1]), 10, (255, 255, 255), -1)

                resized_image = cv2.resize(cv_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)

                # Encode the OpenCV image as a compressed JPEG
                _, encoded_image = cv2.imencode('.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 10])  # Adjust quality (0-100)

                # Create a CompressedImage message
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header  # Copy header from original message
                compressed_msg.header.frame_id = "calibration_frame"
                compressed_msg.format = "rgb8; jpeg compressed bgr8"  # Format must be specified
                compressed_msg.data = encoded_image.tobytes()  # Convert to bytes

                # Publish the compressed image
                self.calibImgPub_.publish(compressed_msg)
            
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
        progress_percentage = min(99, int((self.current_grid_index / total_grid_points) * 100))

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


    def tranformResiduals(self, params, source_points, target_points):
        source_x, source_y = source_points[:, 0], source_points[:, 1]
        target_x, target_y = target_points[:, 0], target_points[:, 1]
        transformed_source_x = (params[0] + params[1] * source_x + params[2] * source_x**2 +
                        params[3] * source_x**3 + params[4] * source_y + 
                        params[5] * source_y**2 + params[6] * source_x * source_y +
                        params[7] * source_x**2 * source_y + params[8] * source_x * source_y**2)
        transformed_source_y = (params[9] + params[10] * source_x + params[11] * source_x**2 +
                                params[12] * source_x**3 + params[13] * source_y + 
                                params[14] * source_y**2 + params[15] * source_x * source_y +
                                params[16] * source_x**2 * source_y + params[17] * source_x * source_y**2)

        residuals = np.concatenate([transformed_source_x - target_x, transformed_source_y - target_y])
        return residuals
    
    def transformSolver(self, source_points, target_points, calib):
        for i in range(500):
            initial_params = calib[:18]
            # Define your termination criteria
            max_iterations = int(1e12)
            ftol = 1e-12
            xtol = 1e-12
            # Additional parameters for the optimization
            method = 'dogbox'  # The optimization method to use ('lm', 'trf', or 'dogbox')
            loss = 'soft_l1'   # The loss function to use ('linear', 'soft_l1', 'huber', or 'cauchy')
            jac = 'cs'    # The method to compute the Jacobian ('2-point', '3-point', 'cs', or a callable)
            # Additional optimization options
            options = {
                'max_nfev': max_iterations,  # Maximum number of iterations
                'ftol': ftol,                # Tolerance for convergence in the function values
                'xtol': xtol,                # Tolerance for convergence in the parameter values
                'verbose': 0,                 # Verbosity level (0: no output, 1: minimal, 2: detailed)
                'gtol': 1e-12,                # Tolerance for convergence in the gradient (default: 1e-8)
                'tr_solver': 'exact',        # Trust-region solver ('exact' or 'lsmr', default: 'exact')
            }
            # Perform the optimization
            result = least_squares(
                self.tranformResiduals,
                initial_params,
                jac=jac,
                args=(source_points, target_points),
                method=method,
                loss=loss,
                **options
            )
            source_x, source_y = source_points[:, 0], source_points[:, 1]
            transformed_source_x = (result.x[0] + result.x[1] * source_x + result.x[2] * source_x**2 +
                            result.x[3] * source_x**3 + result.x[4] * source_y + 
                            result.x[5] * source_y**2 + result.x[6] * source_x * source_y +
                            result.x[7] * source_x**2 * source_y + result.x[8] * source_x * source_y**2)

            transformed_source_y = (result.x[9] + result.x[10] * source_x + result.x[11] * source_x**2 +
                                    result.x[12] * source_x**3 + result.x[13] * source_y + 
                                    result.x[14] * source_y**2 + result.x[15] * source_x * source_y +
                                    result.x[16] * source_x**2 * source_y + result.x[17] * source_x * source_y**2)
            calib[:18] = result.x
        # Calculate residuals for RMSE computation
        residuals_x = transformed_source_x - target_points[:, 0]
        residuals_y = transformed_source_y - target_points[:, 1]

        # Calculate RMSE for x and y
        rmse_x = np.sqrt(np.mean(residuals_x**2))
        rmse_y = np.sqrt(np.mean(residuals_y**2))

        # Optimization cost is already part of the result
        optimization_cost = result.cost
        return np.column_stack((transformed_source_x, transformed_source_y)), rmse_x.item(), rmse_y.item(), optimization_cost, calib


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
            if self.current_grid_index < len(self.grid_points) and np.all(self.calib_pixel_to_laser == 0) and np.all(self.calib_laser_to_pixel == 0) and not self.validation_in_progress:
                x, y = self.grid_points[self.current_grid_index]
                self.fireLaser(x, y)
                self.capture_next_frames = True
                self.frames_to_capture = 10  # Set number of frames to capture
                self.laser_positions = []  # Reset laser positions for new capture
            elif self.current_grid_index >= len(self.grid_points) and np.all(self.calib_pixel_to_laser == 0) and np.all(self.calib_laser_to_pixel == 0) and not self.validation_in_progress:
                # All grid points completed, create RBF interpolators
                merged_img, rsmePixelX, rsmePixelY, pixelCost, rsmeLaserX, rmseLaserY, laserCost = self.calibrateLaserAndCamera()
                # Convert the OpenCV image to a ROS Image message
                resized_image = cv2.resize(merged_img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)

                _, encoded_image = cv2.imencode('.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 75])  # Adjust quality (0-100)

                # Create a CompressedImage message
                compressed_msg = CompressedImage()
                compressed_msg.header.stamp = self.get_clock().now().to_msg()  # Copy header from original message
                compressed_msg.header.frame_id = "calibration_frame"
                compressed_msg.format = "rgb8; jpeg compressed bgr8"  # Format must be specified
                compressed_msg.data = encoded_image.tobytes()  # Convert to bytes

                # Publish the compressed image
                self.calibImgPub_.publish(compressed_msg)

                msg = Float32MultiArray()
                msg.data = [rsmePixelX, rsmePixelY]  # Publish RMSE values
                self.rmsePub_.publish(msg)

                self.calibProgressPub_.publish(self.genProgressMsg(100))
                
                self.export_calibration_data()
            elif self.current_grid_index >= len(self.grid_points) and not np.all(self.calib_pixel_to_laser == 0) and not np.all(self.calib_laser_to_pixel == 0):
                raise SystemExit
            
    def calibrateLaserAndCamera(self):
        new_pixelPts, rsmePixelX, rsmePixelY, pixelCost, self.calib_pixel_to_laser = self.transformSolver(np.array([v["image"] for v in self.correspondences.values()]), np.array([v["laser"] for v in self.correspondences.values()]), self.calib_pixel_to_laser)
        new_laserPts, rsmeLaserX, rmseLaserY, laserCost, self.calib_laser_to_pixel = self.transformSolver(np.array([v["laser"] for v in self.correspondences.values()]), np.array([v["image"] for v in self.correspondences.values()]), self.calib_laser_to_pixel)
        pixel_to_laser_img = self.plotPts("Pixel to Laser Transform", new_pixelPts, [v["laser"] for v in self.correspondences.values()])
        laser_to_pixel_img = self.plotPts("Laser to Pixel Transform", [v["image"] for v in self.correspondences.values()], new_laserPts)
        if pixel_to_laser_img.shape[0] != laser_to_pixel_img.shape[0]:
            height = max(pixel_to_laser_img.shape[0], laser_to_pixel_img.shape[0])
            pixel_to_laser_img = cv2.resize(pixel_to_laser_img, (pixel_to_laser_img.shape[1], height))
            laser_to_pixel_img = cv2.resize(laser_to_pixel_img, (laser_to_pixel_img.shape[1], height))
    
        # Concatenate images horizontally
        merged_img = np.hstack((pixel_to_laser_img, laser_to_pixel_img))
        return pixel_to_laser_img, rsmePixelX, rsmePixelY, pixelCost, rsmeLaserX, rmseLaserY, laserCost


    def plotPts(self, title, pixelPts, laserPts):
        fig, ax = plt.subplots(num = str(title))
        for i, (pixel_pt, laser_pt, id) in enumerate(zip(pixelPts, laserPts, list(self.correspondences.keys()))):
            ax.scatter(pixel_pt[0], pixel_pt[1], color=f'C{id}', label=f'Pixel ID {id}')
            ax.scatter(laser_pt[0], laser_pt[1], color=f'C{id}', marker='x', label=f'Laser ID {id}')
            ax.annotate(str(id), (pixel_pt[0], pixel_pt[1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color=f'C{i}')
            ax.annotate(str(id), (laser_pt[0], laser_pt[1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color=f'C{i}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
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
        """Save pixel to laser param, laser to pixel param and ROI to a JSON file."""

        roi = self.calculate_roi()  # Get ROI data

        data_pTol_to_save = self.calib_pixel_to_laser.tolist()
        data_lTop_to_save = self.calib_laser_to_pixel.tolist()

        data = {
            "calib_pixel_to_laser": data_pTol_to_save,
            "calib_laser_to_pixel": data_lTop_to_save,
            "roi": roi
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

