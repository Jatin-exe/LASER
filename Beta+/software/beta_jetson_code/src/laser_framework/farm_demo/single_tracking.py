#!/usr/bin/python3

import sys
sys.path.insert(0, '/home/hari/laser_ws/src/laser_framework/scripts/')
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
from rclpy.qos import QoSProfile
from KalmanFilter import KalmanFilter


class detector(Node):
    def __init__(self):
        super().__init__("detector")
        self.bridge = CvBridge()
        self.model_ = YOLO('/home/hari/laser_ws/src/laser_framework/weeds.pt') # '/home/hari/laser_ws/src/laser_framework/scripts/network_training/runs/detect/train9/weights/best.pt'
        self.model_.to('cuda:0')
        self.model_.fuse() # Fuse Conv2d and BatchNorm2d layers, improves computation efficiency
        self.__modelWarmUp__() # Model Warm Up
        self.coeff = self.__loadYaml__()
        self.ROI = [int(float(item)) for item in self.coeff[18:]]
        self.tracker_ = CentroidTracker(maxDisappeared = 2, maxDistance = 230)
        self.activeTracking_ = {}
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        #self.screenPub_ = self.create_publisher(Image, "/laser/image", QoSProfile(depth = 100, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        #self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.shutDownSub_ = self.create_subscription(Int32, "/laser_gui/shutdown", self.shutDownCallback_, 1)

    def __modelWarmUp__(self):
        for i in range(10):
            pass
            #self.model_.predict("/home/laudando/40.png", device = 0, verbose = False) # Model Warm Up

    def __loadYaml__(self):
        file_path = "/home/hari/laser_ws/src/laser_framework/config/demo.yaml"
        with open(file_path, 'r') as yaml_file:
            file = yaml.safe_load(yaml_file)
        return file
    
    def createMsg_(self, x, y):
        msg = Int64MultiArray()
        msg.data = [int(x), int(y), 400]
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
        #self.targetPub_.publish(self.createMsg_(targetX, targetY))

    def is_within_roi(self, box, roi):
        def is_point_inside_polygon(x, y, polygon):
            n = len(polygon)
            inside = False
            p1y, p1x = polygon[0]

            for i in range(1, n + 1):
                p2y, p2x = polygon[i % n]

                if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside

                p1x, p1y = p2x, p2y

            return inside
        return is_point_inside_polygon(box[0], box[1], [roi[:2], roi[2:4], roi[6:8], roi[4:6]])

    def is_within_roi_or_orange_mask(self, point):
        # Check if the point is within the general ROI
        inside_roi = self.is_within_roi([point[1], point[0]], self.ROI)
        
        # Check if the point is within the specific "orange mask" criteria
        inside_orange_mask = self.is_within_orange_mask([point[0], point[1]])
        
        # Return True if the point is inside either the ROI or the orange mask
        return inside_roi or inside_orange_mask

    def is_within_orange_mask(self, point):
        x, y = point  # Unpack the point coordinates

        # Define the line for the orange mask and its vertical bounds
        x1, y1, x2, y2 = self.ROI[4], self.ROI[5], self.ROI[0], self.ROI[1]
        m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        b = y1 - m * x1 if (x2 - x1) != 0 else 0
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Check if the point is within the vertical bounds
        if y < y_min or y > y_max:
            return False

        # For non-vertical lines, check if the point is to the right of the line
        if m != float('inf') and y > (m * x + b):
            return True
        
        # For vertical lines, check if the point is to the right of x1
        if m == float('inf') and x > x1:
            return True

        return False

    def objectDetection_(self, frame):
        results = self.model_.predict(frame, device = 0, verbose = True, conf = 0.65)
        centerPts, area = [], []
        # Move data from GPU to CPU and convert to numpy in one go
        keypoints = results[0].keypoints.xy.cpu().numpy().astype(int)
        boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
        for xyxy, ap in zip(keypoints, boxes):
            if len(xyxy) > 0:
                centerPts.append(np.squeeze(xyxy).tolist())
                area.append(self.computeArea(np.squeeze(ap)))
        return centerPts, area
    
    def computeArea(self, xywh):
        return xywh[2] * xywh[3]

    def polyPredict(self, targetLoc):
        x_values = [point[0] for point in targetLoc]
        y_values = [point[1] for point in targetLoc]

        # Extract seconds and nanoseconds from ROS headers
        timestamps_sec = [header[2].stamp.sec for header in targetLoc]
        timestamps_nsec = [header[2].stamp.nanosec for header in targetLoc]

        # Convert nanoseconds to seconds and add to the seconds list
        timestamps = [sec + nsec * 1e-9 for sec, nsec in zip(timestamps_sec, timestamps_nsec)]

        # Fit a polynomial to the data
        degree = 1  # Degree of the polynomial
        coefficients_x = np.polyfit(timestamps, x_values, degree)
        coefficients_y = np.polyfit(timestamps, y_values, degree)

        # Create polynomial functions based on the coefficients
        return np.poly1d(coefficients_x), np.poly1d(coefficients_y)


    def newPredict(self, new_timestamp, poly_x, poly_y):
        # Extract seconds and nanoseconds from the current ROS timestamp
        current_sec = new_timestamp.sec
        current_nsec = new_timestamp.nanosec

        # Add 100 nanoseconds to the nanoseconds part
        
        current_nsec += 2.5 * 1e7

        if current_nsec >= int(1e9):
            current_sec += 1
            current_nsec -= int(1e9)
        
        # Convert nanoseconds to seconds and add to the seconds list
        current_time = current_sec + current_nsec * 1e-9

        # Predict x and y values for the new timestamp using the polynomial function
        predicted_x_value = poly_x(current_time)
        predicted_y_value = poly_y(current_time)

        return predicted_x_value, predicted_y_value
    
    def put_masks(self, cv_image):
        # Create a coordinate grid
        y_coords, x_coords = np.indices(cv_image.shape[:2], dtype=float)

        # Initialize the mask with zeros (no mask)
        mask = np.zeros(cv_image.shape, dtype=np.uint8)

        # Apply red masks for each line specifically above or below as instructed
        lines_red_conditions = [
            ((self.ROI[0], self.ROI[1]), (self.ROI[2], self.ROI[3]), 'above'),  # Line 1 - mask above
            ((self.ROI[2], self.ROI[3]), (self.ROI[6], self.ROI[7]), 'left'),   # Line 2 - mask left
            ((self.ROI[6], self.ROI[7]), (self.ROI[4], self.ROI[5]), 'below')   # Line 3 - mask below
        ]

        for (x1, y1), (x2, y2), position in lines_red_conditions:
            m = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            b = y1 - m * x1 if m != float('inf') else 0
            
            if position == 'above':
                condition = y_coords < (m * x_coords + b)
            elif position == 'below':
                condition = y_coords > (m * x_coords + b)
            elif position == 'left':
                if m != float('inf'):
                    condition = x_coords < ((y_coords - b) / m)
                else:
                    condition = x_coords < x1

            mask[condition] = [0, 0, 255]  # Apply red mask

        # Calculate slope (m) and y-intercept (b) for the green mask line
        x1, y1, x2, y2 = self.ROI[4], self.ROI[5], self.ROI[0], self.ROI[1]
        m_green = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        b_green = y1 - m_green * x1

        # Determine the vertical bounds for the green mask
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Apply green mask within bounds and to the right of the line
        green_condition = np.logical_and.reduce([
            y_coords >= y_min,
            y_coords <= y_max,
            y_coords > (m_green * x_coords + b_green)
        ])

        mask[np.logical_and(green_condition, np.all(mask == [0, 0, 0], axis=-1))] = [0, 165, 255]  # Apply green mask

        # Blend the mask with the original image for transparency
        alpha = 0.4  # Transparency factor
        blended_image = cv2.addWeighted(cv_image, 1 - alpha, mask, alpha, 0)

        # Update the original image
        return blended_image
    
    def draw_tracking_info(self, frame, objects):
        for objectID, (centroid, area) in objects.items():
            # Unpack the centroid
            cX, cY = centroid

            # Draw the object ID near the centroid
            text = f"ID {objectID}"
            cv2.putText(frame, text, (int(cX - 10), int(cY - 50)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
            
            # Optionally, draw the centroid
            cv2.circle(frame, (int(cX), int(cY)), 4, (0, 255, 0), -1)

            # If you want to visualize the area (e.g., as a circle radius or rectangle size),
            # you might want to scale the area value to make it visually meaningful.
            # Here's an example of drawing a circle with radius proportional to sqrt(area) to represent the area.
            radius = int((area / 3.14) ** 0.5)  # Assuming area = pi * r^2 to calculate radius
            cv2.circle(frame, (int(cX), int(cY)), int(radius), (0, 255, 255), 2)
        return frame
    
    def trackingSystem_(self, detection, header):
        # Convert local_map to a set for efficient set operations
        current_ids = set(detection.keys())
        
        # Update existing tracks or add new ones
        for tracker_id, xya in detection.items():
            if tracker_id not in self.activeTracking_:
                self.activeTracking_[tracker_id] = [[[xya[0][0], xya[0][1], header]], 0]
            else:
                self.activeTracking_[tracker_id][0].append([xya[0][0], xya[0][1], header])

        # Determine which tracker IDs to remove using set difference
        stale_ids = set(self.activeTracking_.keys()) - current_ids
        
        # Remove stale tracker IDs efficiently
        for stale_id in stale_ids:
            del self.activeTracking_[stale_id]

    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            centerPts, areas = self.objectDetection_(cv_image)
            objects = self.tracker_.graph_track(centerPts, -2, 0, areas)
            self.trackingSystem_(objects, msg.header)
            for tracking_id, targetLoc in self.activeTracking_.items():
                if len(targetLoc[0]) > 1 and (self.is_within_roi_or_orange_mask([targetLoc[0][-1][0], targetLoc[0][-1][1]])):
                    poly_x, poly_y = self.polyPredict(targetLoc[0])
                    time = self.get_clock().now().to_msg()
                    predX, predY = self.newPredict(time, poly_x, poly_y)
                    if self.is_within_roi([predY, predX], self.ROI) and targetLoc[1] < 15:
                        self.shoot(predY, predX)
                        targetLoc[1] += 1
                        cv2.circle(cv_image, (int(predX), int(predY)), 10, (0, 0, 0), 10)
                        break
            objects_image = self.draw_tracking_info(cv_image, objects)
            outImgMsg = self.bridge.cv2_to_imgmsg(cv2.resize(objects_image, dsize = (0, 0), fx = 0.5, fy = 0.5), encoding="bgr8", header=msg.header)
            #self.screenPub_.publish(outImgMsg)
            #annotated_image = self.put_masks(cv_image)
            cv2.imshow("FRAME", cv2.resize(objects_image, dsize = (0, 0), fx = 0.5, fy = 0.5))
            cv2.waitKey(1)
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
