#!/usr/bin/python3

from typing import List
import rclpy
from rclpy.context import Context
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Int64MultiArray, Int16, Float32MultiArray
import time
import numpy as np
from sensor_msgs.msg import Image
import matplotlib
import matplotlib.pyplot as plt 
from cv_bridge import CvBridge, CvBridgeError
import cv2
from scipy.optimize import least_squares
import yaml
import os
from rclpy.qos import QoSProfile
import random
from io import BytesIO

class calibration(Node):
    def __init__(self):
        super().__init__("calibration")
        # Declare parameters with default values
        self.declare_parameter('dwell', 800)
        self.declare_parameter('cam_expo', 650)
        
        # Retrieve and use the parameters
        self.dwell = int(self.get_parameter('dwell').value)
        self.cam_expo = int(self.get_parameter('cam_expo').value)
        self.bridge_ = CvBridge()
        self.totalPts = 0
        self.pixelPts = np.empty((100, 2), dtype = np.float32)
        self.laserPts = np.empty((100, 2), dtype = np.float32)
        self.id = np.arange(1, 101)
        self.posePub_ = self.create_publisher(Int64MultiArray, "/laser/location", QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.calibImgPub_ = self.create_publisher(Image, "/laser/calibration_image", QoSProfile(depth = 1000, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.calibProgressPub_ = self.create_publisher(Int16, "/laser/calibration_progress", QoSProfile(depth = 1000, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.calibErrorPub_ = self.create_publisher(Float32MultiArray, "/laser/calibration_errors", QoSProfile(depth = 1000, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.callback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.calib = np.zeros(26)
        self.ROI = []
        self.warmUp = 0

    def createMsg(self, x, y):
        msg = Int64MultiArray()
        msg.data = [x, y, 0]
        return msg
    
    def createMsgHigh(self, x, y):
        msg = Int64MultiArray()
        msg.data = [x, y, 800]
        return msg

    def drawGrid(self, width, height):
        msg = Int64MultiArray()
        if 0 <= self.totalPts < 100:
            x, y = 1000 + self.totalPts // 10 * width, (1500 + ((self.totalPts % 10)) * height)
            self.laserPts[self.totalPts][:] = np.array([x, y], dtype = np.float32)
            msg.data = [x, y, self.dwell]
            self.posePub_.publish(msg)

    def findLaser(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to isolate the bright red laser spot.
        # You may need to adjust the threshold value (150 here) depending on the brightness of your laser.
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image.
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Proceed only if at least one contour was found.
        if contours:
            # In case of multiple contours, you may need a logic to select the contour representing the laser.
            # For simplicity, we're assuming the largest contour is the laser spot.
            main_contour = max(contours, key=cv2.contourArea)

            # Compute the moments of the contour, which will help us find the centroid.
            M = cv2.moments(main_contour)

            # Calculate the coordinates of the centroid using the moments.
            if M["m00"] != 0:  # Check to avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0  # Assign some default value or handle the case appropriately

            # Draw the contour and the center of the shape on the image.
            cv2.drawContours(img, [main_contour], -1, (0, 255, 0), 2)
            cv2.circle(img, (cX, cY), 7, (0, 0, 0), -1)

            return img, (cX, cY)
        else:
            # If no contours were found, return the original image and None.
            return img, None
        """
        r, g, b = cv2.split(img)
        max_red_pixel_coords = np.unravel_index(r.argmax(), r.shape)
        max_red_pixel_color = img[max_red_pixel_coords]
        cv2.circle(img, (max_red_pixel_coords[1], max_red_pixel_coords[0]), 5, (255, 0, 0), 5)
        return img, max_red_pixel_coords
        """

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
    
    def transformSolver(self, source_points, target_points):
        for i in range(500):
            initial_params = self.calib[:18]
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
            self.calib[:18] = result.x
        # Calculate residuals for RMSE computation
        residuals_x = transformed_source_x - target_points[:, 0]
        residuals_y = transformed_source_y - target_points[:, 1]

        # Calculate RMSE for x and y
        rmse_x = np.sqrt(np.mean(residuals_x**2))
        rmse_y = np.sqrt(np.mean(residuals_y**2))

        # Optimization cost is already part of the result
        optimization_cost = result.cost
        return np.column_stack((transformed_source_x, transformed_source_y)), rmse_x.item(), rmse_y.item(), optimization_cost

    def plotPts(self, title):
        plt.ion()
        fig, ax = plt.subplots(num = str(title))
        for i, (pixel_pt, laser_pt, id) in enumerate(zip(self.pixelPts, self.laserPts, self.id)):
            ax.scatter(pixel_pt[0], pixel_pt[1], color=f'C{id}', label=f'Pixel ID {id}')
            ax.scatter(laser_pt[0], laser_pt[1], color=f'C{id}', marker='x', label=f'Laser ID {id}')
            ax.annotate(str(id), (pixel_pt[0], pixel_pt[1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color=f'C{i}')
            ax.annotate(str(id), (laser_pt[0], laser_pt[1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color=f'C{i}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.ioff()
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


    def saveToYaml(self):
        file_path = "/home/laudando/laser_ws/src/laser_framework/config/demo.yaml"
        if os.path.exists(file_path):
            self.get_logger().info("The file %s already exists. It will be overwritten." % file_path)
        with open(file_path, 'w') as yaml_file:
            self.calib[18] = self.ROI[0][0]
            self.calib[19] = self.ROI[0][1]
            self.calib[20] = self.ROI[1][0]
            self.calib[21] = self.ROI[1][1]
            self.calib[22] = self.ROI[2][0]
            self.calib[23] = self.ROI[2][1]
            self.calib[24] = self.ROI[3][0]
            self.calib[25] = self.ROI[3][1]
            yaml.dump(self.calib.tolist(), yaml_file)

    def genProgressMsg(self, progess):
        msg = Int16()
        msg.data = progess
        return msg
    
    def createBlackImg(self):
        black_image = np.zeros((1088, 1920, 3), np.uint8)
        text = "Initializing....."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_color = (255, 255, 255)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Position the text in the center
        text_x = (1920 - text_size[0]) // 2
        text_y = (1088 + text_size[1]) // 2

        # Apply the text to the black image
        cv2.putText(black_image, text, (text_x, text_y), font, font_scale, font_color, thickness)
        return black_image
    
    def genErrorMsg(self, rsme_x, rsme_y, cost):
        msg = Float32MultiArray()
        msg.data = [rsme_x, rsme_y, cost]
        return msg

    def callback_(self, msg):
        if self.warmUp < 250:
            self.warmUp = self.warmUp + 1
            self.posePub_.publish(self.createMsg(random.randint(1000, 6000), random.randint(1000, 8000)))
            #self.calibImgPub_.publish(self.bridge_.cv2_to_imgmsg(cv2.resize(self.createBlackImg(), dsize = (0, 0), fx = 0.25, fy = 0.25), encoding = "bgr8", header = msg.header))
            return
        elif self.warmUp < 275:
            self.warmUp = self.warmUp + 1
            self.posePub_.publish(self.createMsgHigh(1000, 1500))
            #self.calibImgPub_.publish(self.bridge_.cv2_to_imgmsg(cv2.resize(self.createBlackImg(), dsize = (0, 0), fx = 0.25, fy = 0.25), encoding = "bgr8", header = msg.header))
            return
        if 0 <= self.totalPts <= 100:
            try:
                self.drawGrid(650, 750)
                cv_image = self.bridge_.imgmsg_to_cv2(msg, "bgr8")
                laserImg, center = self.findLaser(cv_image)
                self.pixelPts[self.totalPts - 1][:] = np.array(center[::-1], dtype = np.float32)
                if self.totalPts == 1 or self.totalPts == 10 or self.totalPts == 91 or self.totalPts == 100:
                    self.ROI.append(np.array(center[::-1], dtype = np.float32).tolist())
                #self.calibProgressPub_.publish(self.genProgressMsg(self.totalPts))
                self.totalPts = self.totalPts + 1
                #self.calibImgPub_.publish(self.bridge_.cv2_to_imgmsg(cv2.resize(laserImg, dsize = (0, 0), fx = 0.25, fy = 0.25), encoding = "bgr8", header = msg.header))
                cv2.imshow("TEST", cv2.resize(laserImg, dsize = (0, 0), fx = 0.25, fy = 0.25))
                cv2.waitKey(self.cam_expo)
                
            except CvBridgeError as e:
                self.get_logger().error("%s" % e)
        else:
            self.get_logger().info("Starting Calibration!")
            before_calib = self.plotPts("Original Mapped Points")
            self.pixelPts, rsmeX, rsmeY, cost = self.transformSolver(self.pixelPts, self.laserPts)
            after_calib = self.plotPts("Post Calibration Mapped Points")
            calibration_image_msg = self.bridge_.cv2_to_imgmsg(after_calib, encoding = "bgr8", header = msg.header)
            for i in range(25):
                self.calibImgPub_.publish(calibration_image_msg)
                self.calibErrorPub_.publish(self.genErrorMsg(rsmeX, rsmeY, cost))
            self.get_logger().info("Calibration Complete!")
            self.saveToYaml()
            plt.show()
            raise SystemExit
            

def main():
    rclpy.init()
    calib_ = calibration()
    rclpy.spin(calib_)
    calib_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
