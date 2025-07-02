#!/usr/bin/python3

import sys
sys.path.insert(0, '/home/laudando/laser_ws/src/laser_framework/scripts/')
from typing import List
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile
from ultralytics import YOLO
import numpy as np
from std_msgs.msg import Int64MultiArray, Int32
import yaml
import time


class detector(Node):
    def __init__(self):
        super().__init__("detector")
        self.declare_parameter('dwell', 100)
        self.dwell_ = int(self.get_parameter('dwell').value)
        self.bridge = CvBridge()
        self.model_ = YOLO('/home/laudando/weights/treadmill-8n/weeds.pt') # '/home/hari/laser_ws/src/laser_framework/scripts/network_training/runs/detect/train9/weights/best.pt'
        self.model_.to('cuda:0')
        self.model_.fuse() # Fuse Conv2d and BatchNorm2d layers, improves computation efficiency
        self.__modelWarmUp__() # Model Warm Up
        self.coeff = self.__loadYaml__()
        self.ROI = [int(float(item)) for item in self.coeff[18:]]
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.networkImagePub_ = self.create_publisher(Image, "/laser/network_image", 100)
        self.networkTargetsPub_ = self.create_publisher(Int32, "/laser/network_targets", 100)
        self.nodeShutDownPub_ = self.create_publisher(Int32, "/laser/network_shutdown", 100)

    def __modelWarmUp__(self):
        for i in range(10):
            self.model_.predict("/home/laudando/700.png", device = 0, verbose = False) # Model Warm Up

    def __loadYaml__(self):
        file_path = "/home/laudando/laser_ws/src/laser_framework/config/demo.yaml"
        with open(file_path, 'r') as yaml_file:
            file = yaml.safe_load(yaml_file)
        return file
    
    def createMsg_(self, x, y):
        msg = Int64MultiArray()
        msg.data = [int(x), int(y), self.dwell_]
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

    def objectDetection_(self, frame):
        out = frame
        results = self.model_.predict(frame, device = 0, verbose = True)
        centerPts = []
        for (xyxy, boxes) in zip(results[0].keypoints.xy.cpu().numpy().astype(int), results[0].boxes.xyxy.cpu().numpy().astype(int)):
            if len(xyxy) > 0 and self.is_within_roi([xyxy[0][0], xyxy[0][1]], self.ROI):
                centerPts.append(np.squeeze(xyxy).tolist())
                cv2.rectangle(out, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 255, 255), 4)
                cv2.circle(out, (centerPts[-1][0], centerPts[-1][1]), 4, (255, 0, 0), 4)
        return centerPts, out
    
    def genTargetMsg(self, value):
        msg = Int32()
        msg.data = value
        self.networkTargetsPub_.publish(msg)

    def genShutdownMsg(self, value):
        msg = Int32()
        msg.data = value
        self.nodeShutDownPub_.publish(msg)

    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            centerPts, out = self.objectDetection_(cv_image)
            cropped_image = out[self.ROI[3]:self.ROI[5], self.ROI[2]:self.ROI[4]]
            network_image_msg = self.bridge.cv2_to_imgmsg(cropped_image, encoding = "bgr8", header = msg.header)
            self.networkImagePub_.publish(network_image_msg)
            self.genTargetMsg(len(centerPts))
            for xyxy in centerPts:
                if self.is_within_roi([xyxy[1], xyxy[0]], self.ROI):
                    self.shoot(xyxy[1], xyxy[0])
                    time.sleep((self.dwell_ / 1000) + 0.2)
            self.genShutdownMsg(1)
            raise SystemExit 
        except CvBridgeError as e:
            self.get_logger().error("%s" % e)

def main():
    rclpy.init()
    detector_ = detector()
    rclpy.spin(detector_)
    detector_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
