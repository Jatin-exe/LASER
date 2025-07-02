#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Int64MultiArray
import struct
from rclpy.qos import QoSProfile

class laserPose(Node):
    def __init__(self):
        super().__init__("laser_position_transmitter")
        self.laser_ = self.__laserInit__("/dev/stm32")
        self.poseSub_ = self.create_subscription(Int64MultiArray, "/laser/location", self.callback_, QoSProfile(depth = 0, reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT, durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE))

    def __laserInit__(self, port):
        laser = serial.Serial()
        laser.port = str(port)
        laser.baudrate = 115200
        laser.timeout = 0
        laser.write_timeout = 0
        laser.open()
        if laser.is_open:
            self.get_logger().info("Successfully opened port {}".format(laser.port))
            return laser
        else:
            self.get_logger().error("Failed to open port {}".format(laser.port))
            return None
        
    def sendPose(self, x, y, duration, laser_power):
        self.laser_.write(bytes((str(int(x)).zfill(5) + str(int(y)).zfill(5) + str(int(duration)).zfill(3)+str(int(laser_power)).zfill(4)), "ascii"))
    
    def read_laser(self):
        try:
            response = self.laser_.read(20)
            data = response
            
            # Ensure the received data is of the expected length
            expected_length = 20
            if len(data) != expected_length:
                return None  # Adjust handling as appropriate for your application

            # Attempt to unpack the data with the specified format
            return struct.unpack('hhffibbbb', data)
        
        except struct.error as e:
            return None  # Adjust handling as appropriate for your application
        except Exception as e:
            return None  # General catch for other unexpected errors
    
    def genLaseMsg(self, x, y):
        bundle = Int64MultiArray()
        bundle.data = [int(x), int(y)]
        return bundle

    def callback_(self, msg):
        self.sendPose(msg.data[0], msg.data[1], msg.data[2], 0)


    def destroy_node(self):
        if self.laser_ is not None and self.laser_.is_open:
            self.laser_.close()
            self.get_logger().info("Closed serial port {}".format(self.laser_.port))
        super().destroy_node()

def main():
    rclpy.init()
    pose_ = laserPose()
    rclpy.spin(pose_)
    pose_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
