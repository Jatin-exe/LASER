#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import serial
from std_msgs.msg import UInt16MultiArray, Int16MultiArray
import struct
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaserAPI(Node):
    def __init__(self):
        super().__init__("laser_position_transmitter")

        # Initialize cv_bridge
        self.bridge = CvBridge()

        # Create callback groups for asynchronous behavior
        self.timer_cb_group = ReentrantCallbackGroup()
        self.sub_cb_group = ReentrantCallbackGroup()

        # Initialize laser
        self.laser_ = self.__laserInit__("/dev/stm32")

        # QoS settings for the subscription: reliable and depth of 1
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscription with its own callback group and QoS settings
        self.poseSub_ = self.create_subscription(
            UInt16MultiArray, 
            "/laser/location", 
            self.callback_, 
            qos_profile, 
            callback_group=self.sub_cb_group
        )

        # Timer callback with its own callback group (1ms interval)
        self.timer_ = self.create_timer(0.00001, self.timer_callback, callback_group=self.timer_cb_group)

        # Laser info pub
        self.laserInfoPub_ = self.create_publisher(Int16MultiArray, "/laser/info", qos_profile, callback_group=self.timer_cb_group)

    def __laserInit__(self, port):
        laser = serial.Serial()
        laser.port = str(port)
        laser.baudrate = 9600
        laser.timeout = 0
        laser.write_timeout = 0
        laser.open()
        if laser.is_open:
            self.get_logger().info("Successfully opened port {}".format(laser.port))
            return laser
        else:
            self.get_logger().error("Failed to open port {}".format(laser.port))
            return None

    def sendPose(self, x, y, duration):
        # Generate and format the command
        data_packet = str(int(x)).zfill(5) + str(int(y)).zfill(5) + str(int(duration)).zfill(3)
        command = bytes(data_packet, "ascii")
        self.laser_.write(command)

    def read_laser(self):
        try:
            response = self.laser_.read(20)
            if response:
                data = struct.unpack('hhIIII', response)
                self.publishLaserInfo(data)
        except Exception as e:
            self.get_logger().error("Failed to read laser data: {}".format(e))

    def publishLaserInfo(self, data):
        out_data = Int16MultiArray()
        out_data.data = [data[0], data[1], data[3], data[4], data[5]]
        self.laserInfoPub_.publish(out_data)

    def timer_callback(self):
        # This function will be called periodically by the timer
        self.read_laser()

    def callback_(self, msg):
        self.sendPose(msg.data[0], msg.data[1], msg.data[2])

    def destroy_node(self):
        if self.laser_ is not None and self.laser_.is_open:
            self.laser_.close()
            self.get_logger().info("Closed serial port {}".format(self.laser_.port))
        super().destroy_node()

def main():
    rclpy.init()

    # Create the laserPose node
    laserapi_ = LaserAPI()

    # Create a MultiThreadedExecutor to handle callbacks asynchronously
    executor = MultiThreadedExecutor()

    # Add the node to the executor
    executor.add_node(laserapi_)

    try:
        # Spin the executor to handle callbacks
        executor.spin()
    finally:
        laserapi_.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

