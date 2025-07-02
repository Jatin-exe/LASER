#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import shutil
import os

class SSDStoragePublisher(Node):
    def __init__(self):
        super().__init__("ssd_storage_publisher")
        self.ssd_path = "/"  # Path to the SSD mount point
        self.publisher_ = self.create_publisher(Float64MultiArray, "/laser/storage_stats", 10)
        self.timer = self.create_timer(2.0, self.publish_ssd_storage_stats)  # Publish every 2 seconds

    def publish_ssd_storage_stats(self):
        if os.path.ismount(self.ssd_path):  # Check if the SSD is mounted
            total, used, free = shutil.disk_usage(self.ssd_path)

            # Convert to GB
            total_gb = total / (1024 ** 3)
            used_gb = used / (1024 ** 3)
            available_gb = free / (1024 ** 3)

            msg = Float64MultiArray()
            msg.data = [total_gb, available_gb, used_gb]  # Order: Total, Used Available
            self.publisher_.publish(msg)
        else:
            msg = Float64MultiArray()
            msg.data = [0.0, 0.0, 0.0]  # Order: Total, Used Available
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SSDStoragePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
