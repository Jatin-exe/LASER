#!/usr/bin/python3

from typing import List
import rclpy
from rclpy.context import Context
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import yaml
from std_msgs.msg import Int64MultiArray, Int32MultiArray
from std_msgs.msg import Int32
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image
from scipy.optimize import fsolve
from rclpy.qos import QoSProfile
import time
import signal

class ClickShoot(Node):
    def __init__(self):
        super().__init__('click_shoot')
        self.declare_parameter('dwell', 100)
        self.dwell_ = int(self.get_parameter('dwell').value)
        self.bridge_ = CvBridge()
        self.coeff = self.__loadYaml__()
        self.ROI = [int(float(item)) for item in self.coeff[18:]]
        self.cameraSub_ = self.create_subscription(Image, "/image_raw", self.cameraCallback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.croppedPub_ = self.create_publisher(Image, "/laser/cropped_image", QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.targetPub_ = self.create_publisher(Int64MultiArray, "/laser/location", QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.shutDownSub_ = self.create_subscription(Int32, "/laser_gui/shutdown", self.shutDownCallback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))
        self.locGuiSub_ = self.create_subscription(Int32MultiArray, "/laser_gui/shoot_locations", self.shootCallback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))

    def __loadYaml__(self):
        file_path = "/home/laudando/laser_ws/src/laser_framework/config/demo.yaml"
        with open(file_path, 'r') as yaml_file:
            file = yaml.safe_load(yaml_file)
        return file
    
    def createMsg(self, x, y, duration):
        msg = Int64MultiArray()
        msg.data = [int(x), int(y), int(duration)]
        return msg
    
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
    
    def trasformFrames(self, source_x, source_y):
        transformed_source_x = (self.coeff[0] + self.coeff[1] * source_x + self.coeff[2] * source_x**2 +
                        self.coeff[3] * source_x**3 + self.coeff[4] * source_y + 
                        self.coeff[5] * source_y**2 + self.coeff[6] * source_x * source_y +
                        self.coeff[7] * source_x**2 * source_y + self.coeff[8] * source_x * source_y**2)

        transformed_source_y = (self.coeff[9] + self.coeff[10] * source_x + self.coeff[11] * source_x**2 +
                                self.coeff[12] * source_x**3 + self.coeff[13] * source_y + 
                                self.coeff[14] * source_y**2 + self.coeff[15] * source_x * source_y +
                                self.coeff[16] * source_x**2 * source_y + self.coeff[17] * source_x * source_y**2)
        return transformed_source_x, transformed_source_y
    
    def shoot(self, last_x, last_y):
        futurePixPt = np.array([[last_x, last_y]])
        targetX, targetY = self.trasformFrames(futurePixPt[0][0], futurePixPt[0][1])
        self.targetPub_.publish(self.createMsg(targetX, targetY))

    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge_.imgmsg_to_cv2(msg, "bgr8")
            cv2.line(cv_image, (self.ROI[0], self.ROI[1]), (self.ROI[2], self.ROI[3]), (0, 255, 0), 2) # Green line
            cv2.line(cv_image, (self.ROI[2], self.ROI[3]), (self.ROI[6], self.ROI[7]), (0, 255, 0), 2) # Green line
            cv2.line(cv_image, (self.ROI[6], self.ROI[7]), (self.ROI[4], self.ROI[5]), (0, 255, 0), 2) # Green line
            cv2.line(cv_image, (self.ROI[4], self.ROI[5]), (self.ROI[0], self.ROI[1]), (0, 255, 0), 2) # Green line
            cv2.circle(cv_image, (self.ROI[2], self.ROI[3]), 5, (255, 255, 255), 5)
            cv2.circle(cv_image, (self.ROI[4], self.ROI[5]), 5, (255, 255, 255), 5)
            cropped_image = cv_image[self.ROI[3]:self.ROI[5], self.ROI[2]:self.ROI[4]]
            cropped_image_msg = self.bridge_.cv2_to_imgmsg(cropped_image, encoding="bgr8", header=msg.header)
            self.croppedPub_.publish(cropped_image_msg)
            #cv2.imshow("FRAME", cv_image)
            #cv2.setMouseCallback("FRAME", self.mouse_click_callback)
            #cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error("%s" % e)
    
    def mouse_click_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and self.is_within_roi([int(y + self.ROI[3]), int(x + self.ROI[2])], self.ROI):
            xL, yL = self.trasformFrames(y, x) #xL, yL = self.trasformFrames(y + self.ROI[3], x + self.ROI[2])
            self.targetPub_.publish(self.createMsg(xL, yL, self.dwell_))
            #time.sleep(self.dwell_ / 1250)

    def shutDownCallback_(self, msg):
        if msg.data == 1:
            raise SystemExit       

    def shootCallback_(self, msg):
        x, y = msg.data[0], msg.data[1]
        if self.is_within_roi([int(y + self.ROI[3]), int(x + self.ROI[2])], self.ROI):
            xL, yL = self.trasformFrames(y + self.ROI[3], x + self.ROI[2])  
            self.targetPub_.publish(self.createMsg(xL, yL, msg.data[2]))
            #time.sleep(msg.data[2] / 900)

def main(args=None):
    rclpy.init(args = args)
    ClickShoot_ = ClickShoot()
    rclpy.spin(ClickShoot_)
    ClickShoot_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
