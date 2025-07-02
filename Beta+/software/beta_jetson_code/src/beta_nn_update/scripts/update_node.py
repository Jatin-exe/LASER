#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_model_updater_srv.srv import CheckModelUpdate

from local_model_scan import LocalModelScanner
from s3_model_scan import S3ModelResolver


class ModelUpdaterNode(Node):
    def __init__(self):
        super().__init__('laser_crop_updater')

        self.get_logger().info("🔁 Initializing model scanners...")
        self.local_scanner = LocalModelScanner()
        self.s3_resolver = S3ModelResolver()

        self.srv = self.create_service(
            CheckModelUpdate,
            '/laser/check_model_update',
            self.check_model_update_callback
        )

        self.get_logger().info("🤖 Waiting for model update requests...")

    def check_model_update_callback(self, request, response):
        self.get_logger().info("🔍 Received request to check for updates.")

        local_versions = self.local_scanner.scan()
        latest_versions = self.s3_resolver.get_latest_versions()

        self.get_logger().info(f"📦 Local: {local_versions}")
        self.get_logger().info(f"☁️  S3:    {latest_versions}")

        updates = {
            crop: (local_versions.get(crop), latest_versions[crop])
            for crop in latest_versions
            if latest_versions[crop] and local_versions.get(crop) != latest_versions[crop]
        }

        response.update_available = len(updates) > 0
        response.crops = list(updates.keys())
        response.current_versions = [v[0] or "N/A" for v in updates.values()]
        response.available_versions = [v[1] for v in updates.values()]

        if updates:
            self.get_logger().info(f"🚨 Updates available: {updates}")
        else:
            self.get_logger().info("✅ No updates needed.")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ModelUpdaterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
