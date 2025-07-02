#!/usr/bin/python3

import sys
sys.path.insert(0, '/home/laudando/laser_ws/src/laser_framework/scripts/')
from typing import List
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from std_msgs.msg import Int64MultiArray, Int32
from centeroidTracker import CentroidTracker
import yaml
import cv2
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

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
        self.tracker_ = CentroidTracker(maxDisappeared = 5, maxDistance = 230)
        self.activeTracking_ = {}
        self.poly_x = None
        self.poly_y = None
        self.start_time = None
        self.start = True
        self.id = None
        self.duration = 300
        self.callback_group1 = MutuallyExclusiveCallbackGroup()
        self.callback_group2 = MutuallyExclusiveCallbackGroup()
        self.shutdown_callback_group = ReentrantCallbackGroup()
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE), callback_group = self.callback_group1)
        self.timer = self.create_timer(0.005, self.timer_callback, callback_group = self.callback_group2)
        self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE), callback_group = self.shutdown_callback_group)
        self.shutDownSub_ = self.create_subscription(Int32, "/laser_gui/shutdown", self.shutDownCallback_, 1, callback_group = self.shutdown_callback_group)

    def __modelWarmUp__(self):
        for i in range(10):
            self.model_.predict("/home/laudando/40.png", device = 0, verbose = False) # Model Warm Up

    def __loadYaml__(self):
        file_path = "/home/laudando/laser_ws/src/laser_framework/config/demo.yaml"
        with open(file_path, 'r') as yaml_file:
            file = yaml.safe_load(yaml_file)
        return file
    
    def createMsg_(self, x, y, duration):
        msg = Int64MultiArray()
        msg.data = [int(x), int(y), int(duration)]
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
    
    def shoot(self, last_x, last_y, duration):
        targetX, targetY = self.transformFrames_(last_x, last_y)
        self.targetPub_.publish(self.createMsg_(targetX, targetY, duration))

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

    def objectDetection_(self, frame):
        results = self.model_.predict(frame, device = 0, verbose = False)
        centerPts, area = [], []
        for xyxy, ap in zip(results[0].keypoints.xy.cpu().numpy().astype(int), results[0].boxes.xywh.cpu().numpy().astype(int)):
            if len(xyxy) > 0:
                centerPts.append(np.squeeze(xyxy).tolist())
                area.append(self.computeArea(np.squeeze(ap)))
        return centerPts, area
    
    def computeArea(self, xywh):
        return xywh[2] * xywh[3]
    
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
        local_map = []
        for tracker_id, xya in detection.items():
            local_map.append(tracker_id)
            if tracker_id not in self.activeTracking_:
                self.activeTracking_[tracker_id] = [[[xya[0][0], xya[0][1], header]], 0]
            else:
                self.activeTracking_[tracker_id][0].append([xya[0][0], xya[0][1], header])
        for tracker_id in self.activeTracking_.copy():
            if tracker_id not in local_map:
                del self.activeTracking_[tracker_id]

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
        current_nsec += 5.75 * 1e7

        if current_nsec >= int(1e9):
            current_sec += 1
            current_nsec -= int(1e9)
        # Convert nanoseconds to seconds and add to the seconds list
        current_time = current_sec + current_nsec * 1e-9

        # Predict x and y values for the new timestamp using the polynomial function
        predicted_x_value = poly_x(current_time)
        predicted_y_value = poly_y(current_time)

        return predicted_x_value, predicted_y_value - 3

    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            centerPts, areas = self.objectDetection_(cv_image)
            objects = self.tracker_.graph_track(centerPts, -20, 0, areas)
            self.trackingSystem_(objects, msg.header)
            # Filter for targets within the orange mask, correctly passing coordinates
            orange_mask_targets = [(tracking_id, targetLoc) for tracking_id, targetLoc in self.activeTracking_.items()
                                if self.is_within_orange_mask([targetLoc[0][-1][0], targetLoc[0][-1][1]])]
            """
            for tracking_id, targetLoc in self.activeTracking_.items():
                if len(targetLoc[0]) > 1 and self.id == tracking_id and targetLoc[1] == 0 and not self.start:
                    self.poly_x, self.poly_y = self.polyPredict(targetLoc[0])
                    self.get_logger().info("Update Pose {} {}".format(self.id, tracking_id))
                    break
                elif len(targetLoc[0]) > 1 and self.is_within_orange_mask([targetLoc[0][-1][0], targetLoc[0][-1][1]]) and targetLoc[1] == 0 and self.id != tracking_id and self.start:
                    self.id = tracking_id
                    self.poly_x, self.poly_y = self.polyPredict(targetLoc[0])
                    self.get_logger().info("Outside Update {} {}".format(self.id, tracking_id))
                    break
                else:
                    self.poly_x = None
                    self.poly_y = None
                    self.id = None
            """
        except CvBridgeError as e:
            self.get_logger().error("%s" % e)

    def compute_time_diff_in_ms(self, start_time_msg, end_time_msg):
        time_diff_ns = (end_time_msg.sec * 1e9 + end_time_msg.nanosec) - (start_time_msg.sec * 1e9 + start_time_msg.nanosec)
        time_diff_ms = time_diff_ns / 1e6
        return time_diff_ms
    
    def compute_time_difference(self, curr_time):
        return self.compute_time_diff_in_ms(self.start_time, curr_time) < self.duration
    
    def complete_target_shooting(self):
        self.activeTracking_[self.id][1] = 1  # Mark current target as completed
        self.poly_x = None
        self.poly_y = None
        self.id = None
        self.start = True
        self.start_time = None
        self.get_logger().info("Target shooting completed. Ready for next target.")

    def timer_callback(self):
        if self.poly_x and self.poly_y and self.id:
            time = self.get_clock().now().to_msg()
            predX, predY = self.newPredict(time, self.poly_x, self.poly_y)
            if self.is_within_roi([predY, predX], self.ROI) and self.start and self.activeTracking_[self.id][1] == 0:
                self.start_time = time
                self.shoot(predY, predX, self.duration)
                self.get_logger().info("START")
                self.start = False
            elif self.is_within_roi([predY, predX], self.ROI) and not self.start and self.compute_time_difference(time):
                self.shoot(predY, predX, 0)
                self.get_logger().info("TRACKING")
            elif self.is_within_roi([predY, predX], self.ROI) and not self.start and not self.compute_time_difference(time):
                self.complete_target_shooting()
            elif not self.is_within_roi([predY, predX], self.ROI) and not self.start:
                self.complete_target_shooting()
    
    def shutDownCallback_(self, msg):
        if msg.data == 1:
            raise SystemExit 

def main():
    rclpy.init()
    detector_ = detector()
    executor = MultiThreadedExecutor()
    executor.add_node(detector_)
    executor.spin()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
