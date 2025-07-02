#!/usr/bin/python3

import sys
sys.path.insert(0, '/home/laudando/laser_ws/src/laser_framework/scripts/')
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from std_msgs.msg import Int64MultiArray, Int32
import yaml
import time

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
        self.time_step = 0.0075 #0.005
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, 0)
        self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", 1)
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
            if len(xyxy) > 0:
                boxes.append(np.squeeze(xyxy).tolist())
        return boxes
    
    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.forwardPass_(cv_image)
            for xyxy in results:
                x, y = xyxy[0], xyxy[1]
                first_iteration = True
                entered = False
                large_increment = -30
                small_increment = -6.5 #-3.8
                while True:
                    start_time = time.time()
                    if first_iteration:
                        x += large_increment
                        first_iteration = False
                    x += small_increment
                    if self.is_within_roi([x, y], self.ROI):
                        self.shoot(x, y)
                        entered = True
                    elif not self.is_within_roi([x, y], self.ROI) and entered:
                        time.sleep(0.03)
                        msg = Int64MultiArray()
                        msg.data = [7000, 4000, 0]
                        self.targetPub_.publish(msg)
                        time.sleep(0.6)
                        break
                    sleep_time = self.time_step - (time.time() - start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)  # Adjusting the sleep time dynamically 
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
