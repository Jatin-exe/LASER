#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class StopNode(Node):
    def __init__(self):
        super().__init__('stop_node_subscriber')
        self.subscription = self.create_subscription(
            Bool,
            '/beta_gui/stop',
            self.stop_callback,
            10  # QoS depth
        )
        self.get_logger().info("StopNode is running. Waiting for /beta_gui/stop messages...")

    def stop_callback(self, msg):
        if msg.data:  # If the message is True
            self.get_logger().info("Received stop command. Shutting down the node...")
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = StopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
