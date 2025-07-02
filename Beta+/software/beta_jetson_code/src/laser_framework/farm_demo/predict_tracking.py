#!/usr/bin/python3

import sys
sys.path.insert(0, '/home/laudando/laser_ws/src/laser_framework/scripts/')
from typing import List
import rclpy
from rclpy.context import Context
from rclpy.node import Node
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from std_msgs.msg import Int64MultiArray, Float64MultiArray, Int32
from centeroidTracker import CentroidTracker
import matplotlib.pyplot as plt
from scipy import signal
import yaml
import time
import statistics
from KalmanFilter import KalmanFilter


class detector(Node):
    def __init__(self):
        super().__init__("detector")
        self.bridge = CvBridge()
        self.model_ = YOLO('/home/laudando/weights/treadmill-8n/pucks.pt') # '/home/hari/laser_ws/src/laser_framework/scripts/network_training/runs/detect/train9/weights/best.pt'
        self.model_.to('cuda:0')
        self.model_.fuse() # Fuse Conv2d and BatchNorm2d layers, improves computation efficiency
        self.__modelWarmUp__() # Model Warm Up
        self.coeff = self.__loadYaml__()
        self.ROI = [int(float(item)) for item in self.coeff[18:]]
        self.tracker_ = CentroidTracker(maxDisappeared = 0, maxDistance = 230)
        self.activeTracking_ = {}
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, 0)
        self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", 0)
        self.shutDownSub_ = self.create_subscription(Int32, "/laser_gui/shutdown", self.shutDownCallback_, 1)

    def __modelWarmUp__(self):
        for i in range(10):
            self.model_.predict("/home/laudando/40.png", device = 0, verbose = False) # Model Warm Up

    def __loadYaml__(self):
        file_path = "/home/laudando/laser_ws/src/laser_framework/config/demo.yaml"
        with open(file_path, 'r') as yaml_file:
            file = yaml.safe_load(yaml_file)
        return file
    
    def createMsg_(self, x, y):
        msg = Int64MultiArray()
        msg.data = [int(x), int(y), 50]
        return msg
    
    def transformFrames_(self, source_x, source_y):
        # Assuming source_x and source_y are NumPy arrays
        x2 = source_x ** 2
        x3 = source_x ** 3
        y2 = source_y ** 2
        y3 = source_y ** 3

        transformed_source_x = (self.coeff[0] + self.coeff[1] * source_x + self.coeff[2] * x2 +
                                self.coeff[3] * x3 + self.coeff[4] * source_y + 
                                self.coeff[5] * y2 + self.coeff[6] * source_x * source_y +
                                self.coeff[7] * x2 * source_y + self.coeff[8] * source_x * y2)

        transformed_source_y = (self.coeff[9] + self.coeff[10] * source_x + self.coeff[11] * x2 +
                                self.coeff[12] * x3 + self.coeff[13] * source_y + 
                                self.coeff[14] * y2 + self.coeff[15] * source_x * source_y +
                                self.coeff[16] * x2 * source_y + self.coeff[17] * source_x * y2)

        return transformed_source_x, transformed_source_y
    
    def shoot(self, last_x, last_y):
        targetX, targetY = self.transformFrames_(last_x, last_y)
        self.targetPub_.publish(self.createMsg_(targetX, targetY))

    def is_within_roi(self, box, roi):
        def is_point_inside_polygon(x, y, polygon):
            n = len(polygon)
            inside = False
            p1x, p1y = polygon[0]

            for i in range(1, n + 1):
                p2x, p2y = polygon[i % n]

                if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside

                p1x, p1y = p2x, p2y

            return inside
        return is_point_inside_polygon(box[0], box[1], [roi[:2], roi[2:4], roi[6:8], roi[4:6]])

    def objectDetection_(self, frame):
        results = self.model_.predict(frame, device = 0, conf = 0.4, verbose = False)
        centerPts = []
        for xyxy in results[0].keypoints.xy.cpu().numpy().astype(int):
            if len(xyxy) > 0:
                centerPts.append(np.squeeze(xyxy).tolist())
        return centerPts
    
    def trackingSystem_(self, detection):
        local_map = []
        for tracker_id, xy in detection.items():
            local_map.append(tracker_id)
            if tracker_id not in self.activeTracking_:
                self.activeTracking_[tracker_id] = [[xy[0], xy[1]]]
            else:
                self.activeTracking_[tracker_id].append([xy[0], xy[1]])
        for tracker_id in self.activeTracking_.copy():
            if tracker_id not in local_map:
                del self.activeTracking_[tracker_id]

    def polyPredict(self, time_step, targetLoc):
        x_values = [point[0] for point in targetLoc]
        y_values = [point[1] for point in targetLoc]
        times = np.arange(1, len(x_values) + 1)
        
        # Fit a polynomial to the data
        degree = 1  # Degree of the polynomial
        coefficients_x = np.polyfit(times, x_values, degree)
        coefficients_y = np.polyfit(times, y_values, degree)

        # Create a polynomial function based on the coefficients
        poly_x = np.poly1d(coefficients_x)
        poly_y = np.poly1d(coefficients_y)

        # Predict x and y values for the next time steps using the polynomial function
        time_steps_to_predict = np.arange(times[-1] + 1, times[-1] + (time_step + 1))
        predicted_x_values = poly_x(time_steps_to_predict)
        predicted_y_values = poly_y(time_steps_to_predict)

        return predicted_x_values, predicted_y_values

    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            centerPts = self.objectDetection_(cv_image)
            objects = self.tracker_.track(centerPts, 0, 0)
            self.trackingSystem_(objects)
            for tracking_id, targetLoc in self.activeTracking_.items():
                if len(targetLoc) > 1:
                    predX, predY = self.polyPredict(3, targetLoc)
                    filter_ = KalmanFilter(0.0001, 2, 0, 1, 0.1, 0)
                    x1, y1 = filter_.update([predX[-1], predY[-1]])
                    x, y = filter_.predict()
                    if self.is_within_roi([y[0, 0], y[0, 1]], self.ROI):
                        self.shoot(y[0, 0], y[0, 1])
        except CvBridgeError as e:
            self.get_logger().error("%s" % e)
                        
    def shutDownCallback_(self, msg):
        if msg.data == 1:
            raise SystemExit 

def main():
    rclpy.init()
    detector_ = detector()
    rclpy.spin(detector_)
    detector_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
