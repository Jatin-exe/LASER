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
        self.time_step = 0.005
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, 0)
        self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", 1)
        self.shutDownSub_ = self.create_subscription(Int32, "/laser_gui/shutdown", self.shutDownCallback_, 1)

    def __modelWarmUp__(self):
        for i in range(50):
            self.model_.predict(np.random.randint(low = 0, high = 255, size=(384, 640, 3)), device = 0, verbose = False) # Model Warm Up

    def __loadYaml__(self):
        file_path = "/home/laudando/laser_ws/src/laser_framework/config/demo.yaml"
        with open(file_path, 'r') as yaml_file:
            file = yaml.safe_load(yaml_file)
        return file
    
    def createMsg_(self, x, y):
        msg = Int64MultiArray()
        msg.data = [int(x), int(y), 3]
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

        # Check if the center is inside the ROI polygon
        return is_point_inside_polygon(box[0], box[1], [roi[:2], roi[2:4], roi[6:8], roi[4:6]])

    def forwardPass_(self, img):
        results = self.model_.predict(img, verbose = False, device = 0) # classes = [1, 2]
        boxes = []
        for xyxy in results[0].cpu().keypoints.xy.numpy().astype(int):
            #if self.is_within_roi([int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)], self.ROI):
            if len(xyxy) > 0:
                boxes.append([xyxy[0][0], xyxy[0][1]])
        return np.array(boxes).astype(int)
    
    def imageAnnotations_(self, img, boxes, objects):
        out = img
        for (id, cenPt) in objects.items():
            out = cv2.putText(out, str(id), cenPt, cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 4, -1)
        return out
    
    def tracking_(self, detection):
        local_map = []
        for tracker_id, xy in detection.items():
            local_map.append(tracker_id)
            if tracker_id not in self.activeTracking_:
                self.activeTracking_[tracker_id] = [[[xy[0], xy[1]]], 0, False]
            else:
                self.activeTracking_[tracker_id][0].append([xy[0], xy[1]])
                if len(self.activeTracking_[tracker_id][0]) > 2:
                        self.activeTracking_[tracker_id][0].pop(0)
        active_tracking_copy = self.activeTracking_.copy()
        for tracker_id in active_tracking_copy:
            if tracker_id not in local_map:
                del self.activeTracking_[tracker_id]
        avg_offsetX = 0
        avg_offsetY = 0
        sum_offsetX = 0
        sum_offsetY = 0
        i = 0
        for tracker_id, state in self.activeTracking_.items():
            if len(self.activeTracking_[tracker_id][0]) == 2:
                sum_offsetX += self.activeTracking_[tracker_id][0][-1][0] - self.activeTracking_[tracker_id][0][0][0]
                sum_offsetY += self.activeTracking_[tracker_id][0][-1][1] - self.activeTracking_[tracker_id][0][0][1]
                i += 1
        if i != 0:
            avg_offsetX = sum_offsetX / i
            avg_offsetY = sum_offsetY / i
        return avg_offsetX, avg_offsetY
    
    def shootingDecision_(self, offset = 0):
        for tracking_id, state in self.activeTracking_.items():
            x, y = state[0][-1][0], state[0][-1][1]
            if state[1] == 0 and self.is_within_roi([x, y], self.ROI):
                first_iteration = True
                large_increment = -60
                small_increment = -4
                while self.is_within_roi([x, y], self.ROI):
                    start_time = time.time()
                    if first_iteration:
                        x += large_increment
                        first_iteration = False
                    x += small_increment
                    if self.is_within_roi([x, y], self.ROI):
                        self.shoot(x, y)   
                    sleep_time = self.time_step - (time.time() - start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)  # Adjusting the sleep time dynamically
    
    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            #cv2.line(cv_image, (self.ROI[0], self.ROI[1]), (self.ROI[2], self.ROI[3]), (0, 255, 0), 2) # Green line
            #cv2.line(cv_image, (self.ROI[2], self.ROI[3]), (self.ROI[6], self.ROI[7]), (0, 255, 0), 2) # Green line
            #cv2.line(cv_image, (self.ROI[6], self.ROI[7]), (self.ROI[4], self.ROI[5]), (0, 255, 0), 2) # Green line
            #cv2.line(cv_image, (self.ROI[4], self.ROI[5]), (self.ROI[0], self.ROI[1]), (0, 255, 0), 2) # Green line
            results = self.forwardPass_(cv_image)
            objects = self.tracker_.track(results, -35, 0) 
            dispX, dispY = self.tracking_(objects)
            self.shootingDecision_(dispX)
            #cv_image = self.imageAnnotations_(cv_image, results, objects)
            #frame = cv2.resize(cv_image, dsize = (0, 0), fx = 0.5, fy = 0.5)
            #cv2.imshow("Tracking", frame)
            #cv2.waitKey(1)
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
