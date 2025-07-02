#!/usr/bin/python3

import sys
sys.path.insert(0, '/home/hari/laser_ws/src/laser_framework/scripts/')
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
import time as t
import message_filters
import cv2
from laser_framework.msg import DetectionArray
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

class detector(Node):
    def __init__(self):
        super().__init__("detector")
        self.bridge = CvBridge()
        self.coeff = self.__loadYaml__()
        self.ROI = [int(float(item)) for item in self.coeff[18:]]
        self.tracker_ = CentroidTracker(maxDisappeared = 0, maxDistance = 230)
        self.activeTracking_ = {}
        self.counter = 0
        self.id = [None, 0, None]
        self.poly_x = None
        self.poly_y = None
        self.values = None
        self.incomplete = 0
        self.pub_group = MutuallyExclusiveCallbackGroup()
        # Create subscribers using message_filters
        self.detection_sub = message_filters.Subscriber(self, DetectionArray, "/laser/weed_detections", qos_profile = QoSProfile(depth = 1000, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.camera_sub = message_filters.Subscriber(self, Image, "/image_raw", qos_profile = QoSProfile(depth = 1000, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        #self.detectionSub_ = self.create_subscription(DetectionArray, "/laser/weed_detections", self.detectionCallback, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.screenPub_ = self.create_publisher(Image, "/laser/image", QoSProfile(depth = 1000, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE), callback_group = self.pub_group)
        #self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.shutDownSub_ = self.create_subscription(Int32, "/laser_gui/shutdown", self.shutDownCallback_, 1)
        self.ts = message_filters.TimeSynchronizer([self.detection_sub, self.camera_sub], 1000)
        self.ts.registerCallback(self.sync_callback)

    def __loadYaml__(self):
        file_path = "/home/hari/laser_ws/src/laser_framework/config/demo.yaml"
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
            cv2.putText(frame, text, (int(cX - 25), int(cY - 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
            
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
                if len(self.activeTracking_[tracker_id][0]) > 3:
                    self.activeTracking_[tracker_id][0].pop(0)

        # Determine which tracker IDs to remove using set difference
        stale_ids = set(self.activeTracking_.keys()) - current_ids
        
        # Remove stale tracker IDs efficiently
        for stale_id in stale_ids:
            del self.activeTracking_[stale_id]

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

    def extractKpsAndArea(self, msg, score = 0.5):
        centerPts, areas = [], []
        for box in msg.boxes:
            if box.score > score:
                centerPts.append([box.keypoint.x, box.keypoint.y])
                areas.append(box.area)
        
        return centerPts, areas
    
    def check_proximity(self, predX, predY, targetLoc):
        # Return True if both differences are within the threshold of 50.0, else return False
        return True if abs(predX - targetLoc[0][-1][0]) < 75.0 and abs(predY - targetLoc[0][-1][1]) < 75.0 else False

    def check_future_possible(self, new_timestamp, poly_x, poly_y):
        # Extract seconds and nanoseconds from the current ROS timestamp
        current_sec = new_timestamp.sec
        current_nsec = new_timestamp.nanosec

        # Add 300 milliseconds to the nanoseconds part
        current_nsec += 300 * 1e6  # 300 ms = 300,000,000 ns

        # Check if current_nsec exceeds one billion nanoseconds, and adjust
        if current_nsec >= int(1e9):
            current_sec += 1
            current_nsec -= int(1e9)

        # Convert nanoseconds to seconds and add to the seconds list
        current_time = current_sec + current_nsec * 1e-9


        # Predict x and y values for the new timestamp using the polynomial function
        predicted_x_value = poly_x(current_time)
        predicted_y_value = poly_y(current_time)

        return self.is_within_roi([predicted_y_value, predicted_x_value], self.ROI)
    
    def check_future_possible_out(self, new_timestamp, poly_x, poly_y):
        # Extract seconds and nanoseconds from the current ROS timestamp
        current_sec = new_timestamp.sec
        current_nsec = new_timestamp.nanosec

        # Add 300 milliseconds to the nanoseconds part
        current_nsec += 300 * 1e6  # 300 ms = 300,000,000 ns

        # Check if current_nsec exceeds one billion nanoseconds, and adjust
        if current_nsec >= int(1e9):
            current_sec += 1
            current_nsec -= int(1e9)

        # Convert nanoseconds to seconds and add to the seconds list
        current_time = current_sec + current_nsec * 1e-9


        # Predict x and y values for the new timestamp using the polynomial function
        predicted_x_value = poly_x(current_time)
        predicted_y_value = poly_y(current_time)

        return predicted_x_value, predicted_y_value


    def detectionCallback(self, msg):
        centerPts, areas = self.extractKpsAndArea(msg)
        objects = self.tracker_.graph_track(centerPts, -2, 0, areas)
        self.trackingSystem_(objects, msg.header)
        if self.id and (self.is_within_roi_or_orange_mask([self.activeTracking_[self.id][0][-1][0], self.activeTracking_[self.id][0][-1][1]])):
            #poly_x, poly_y = self.polyPredict(self.activeTracking_[self.id][0])
            predX, predY = self.newPredict(self.get_clock().now().to_msg(), self.poly_x, self.poly_y)
            if self.is_within_roi([predY, predX], self.ROI) and self.activeTracking_[self.id][1] < 9:
                self.shoot(predY, predX, 0)
                #self.shoot(predY, predX, 300 - (self.activeTracking_[self.id][1] * 33))
                self.activeTracking_[self.id][1] += 1
            elif not self.is_within_roi([predY, predX], self.ROI) or self.activeTracking_[self.id][1] >= 9:
                self.shoot(predY, predX, 1)
                self.id = None
                self.poly_x = None
                self.poly_y = None
            return
        for tracking_id, targetLoc in self.activeTracking_.items():
            if len(targetLoc[0]) == 2 and (self.is_within_roi_or_orange_mask([targetLoc[0][-1][0], targetLoc[0][-1][1]])):
                self.poly_x, self.poly_y = self.polyPredict(targetLoc[0])
                predX, predY = self.newPredict(self.get_clock().now().to_msg(), self.poly_x, self.poly_y)
                if self.is_within_roi([predY, predX], self.ROI) and targetLoc[1] == 0:
                    self.shoot(predY, predX, 300)
                    targetLoc[1] += 1
                    self.id = tracking_id
                    break

    def sync_callback(self, detection_msg, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error("%s" % e)
        centerPts, areas = self.extractKpsAndArea(detection_msg)
        objects = self.tracker_.graph_track(centerPts, -2, 0, areas)
        self.trackingSystem_(objects, detection_msg.header)
        if self.id[0]:
            for values in self.values:
                cv2.circle(cv_image, (int(values[0]), int(values[1])), 3, (0, 0, 0), 3)
            if self.id[0] in self.activeTracking_:
                self.poly_x, self.poly_y = self.polyPredict(self.activeTracking_[self.id[0]][0])
                self.id[2] = [[[self.activeTracking_[self.id[0]][0][-1][0], self.activeTracking_[self.id[0]][0][-1][1]]]]
            predX, predY = self.newPredict(self.get_clock().now().to_msg(), self.poly_x, self.poly_y)
            if self.is_within_roi([predY, predX], self.ROI) and self.id[1] < 9:
                cv2.circle(cv_image, (int(predX), int(predY)), 10, (255, 0, 0), 10)
                self.id[1] += 1
                if self.id[1] >= 9:
                    cv2.circle(cv_image, (int(predX), int(predY)), 10, (0, 0, 255), 10)
                    self.id = [None, 0, None]
                    self.poly_x = None
                    self.poly_y = None
                    self.values = None    
            elif not self.is_within_roi([predY, predX], self.ROI) or self.id[1] >= 9 or not self.check_proximity(predX, predY, self.id[2]):
                cv2.circle(cv_image, (int(predX), int(predY)), 10, (0, 0, 255), 10)
                if self.id[1] < 9:
                    self.incomplete += 1
                    self.get_logger().info("Times {} {}".format(self.incomplete, self.id[1]))
                self.id = [None, 0, None]
                self.poly_x = None
                self.poly_y = None
                self.values = None
            try:
                objects_image = self.draw_tracking_info(cv_image, objects)
                cv2.imwrite("/home/hari/recording_data_frames/{}.jpg".format(self.counter), cv2.resize(objects_image, dsize = (0, 0), fx = 0.5, fy = 0.5))
                self.counter += 1
            except CvBridgeError as e:
                self.get_logger().error("%s" % e)
            return
        for tracking_id, targetLoc in self.activeTracking_.items():
            if len(targetLoc[0]) == 3 and (self.is_within_roi_or_orange_mask([targetLoc[0][-1][0], targetLoc[0][-1][1]])):
                poly_x, poly_y = self.polyPredict(targetLoc[0])
                time = self.get_clock().now().to_msg()
                predX, predY = self.newPredict(time, poly_x, poly_y)
                if self.is_within_roi([predY, predX], self.ROI) and targetLoc[1] == 0 and self.check_proximity(predX, predY, targetLoc) and self.check_future_possible(time, poly_x, poly_y):
                    for values in targetLoc[0]:
                        cv2.circle(cv_image, (int(values[0]), int(values[1])), 3, (0, 0, 0), 3)
                    f_predX, f_predY = self.check_future_possible_out(time, poly_x, poly_y)
                    cv2.circle(cv_image, (int(f_predX), int(f_predY)), 10, (125, 125, 125), 10)
                    cv2.circle(cv_image, (int(predX), int(predY)), 10, (0, 0, 0), 10)
                    self.id = [tracking_id, 1, [[[targetLoc[0][-1][0], targetLoc[0][-1][1]]]]]
                    targetLoc[1] += 1
                    self.poly_x, self.poly_y = poly_x, poly_y
                    self.values = targetLoc[0]
                    break
        try:
            objects_image = self.draw_tracking_info(cv_image, objects)
            cv2.imwrite("/home/hari/recording_data_frames/{}.jpg".format(self.counter), cv2.resize(objects_image, dsize = (0, 0), fx = 0.5, fy = 0.5))
            self.counter += 1
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
