#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from beta_tracking.msg import TrackerOutput
from beta_path_planning.msg import LaserPath
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
from cv_bridge import CvBridge
import numpy as np
import json


class MultiSubscriberNode(Node):
    def __init__(self):
        super().__init__('planner_viz')

        # Subscriptions
        self.tracker_sub = Subscriber(self, TrackerOutput, '/laser/tracker')
        self.path_sub = Subscriber(self, LaserPath, '/laser/path_planned')
        self.future_position_sub = Subscriber(self, PointStamped, '/laser/current_target')
        self.image_sub = Subscriber(self, Image, '/vimbax_camera_beta/image_raw')

        # Publisher for the resized image
        self.resized_image_pub = self.create_publisher(Image, '/laser/planner_viz', 100)

        # Synchronize the subscriptions
        self.sync = ApproximateTimeSynchronizer(
            [self.tracker_sub, self.path_sub, self.future_position_sub, self.image_sub],
            queue_size=100,
            slop=0.1
        )
        self.sync.registerCallback(self.callback)

        self.bridge = CvBridge()  # For converting ROS Image messages to OpenCV images

        # Load ROI from JSON file
        self.roi_polygon = self.load_roi_from_json("/workspaces/isaac_ros-dev/src/laser_framework/config/camera_laser_calib.json")

    def load_roi_from_json(self, file_path):
        """Load ROI polygon from the JSON file."""
        try:
            with open(file_path, 'r') as file:
                roi_data = json.load(file)
                roi = roi_data["roi"]
                return np.array([
                    tuple(roi["top_left"]),
                    tuple(roi["top_right"]),
                    tuple(roi["bottom_right"]),
                    tuple(roi["bottom_left"])
                ], dtype=np.int32)
        except Exception as e:
            self.get_logger().error(f"Failed to load ROI from JSON: {e}")
            return np.array([])

    def callback(self, tracker_msg, path_msg, future_position_msg, image_msg):
        # Convert the ROS Image message to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Draw the ROI on the image
        if self.roi_polygon.size > 0:
            cv2.polylines(cv_image, [self.roi_polygon], isClosed=True, color=(0, 0, 0), thickness=5)

        # Draw the targets from TrackerOutput
        positions = {}
        for target in tracker_msg.target_list:
            x, y = int(target.target_point.x), int(target.target_point.y)
            target_id = target.id

            positions[target_id] = (x, y)
            cv2.circle(cv_image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.putText(cv_image, f"ID: {target_id}", (x + 15, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 0), thickness=4)

        # Draw the path from LaserPath
        if path_msg.data:
            path_ids = path_msg.data
            for i in range(len(path_ids) - 1):
                id1, id2 = path_ids[i], path_ids[i + 1]
                if id1 in positions and id2 in positions:
                    cv2.line(cv_image, positions[id1], positions[id2], color=(255, 0, 0), thickness=5)

        # Highlight the current future position
        future_x, future_y = int(future_position_msg.point.x), int(future_position_msg.point.y)
        remaining_time_ms = int(future_position_msg.point.z)

        cv2.circle(cv_image, (future_x, future_y), radius=20, color=(255, 0, 0), thickness=-1)
        cv2.putText(cv_image, f"Remaining: {remaining_time_ms} ms", (future_x + 10, future_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 0), thickness=3)

        # Resize the image for visualization
        resized_image = cv2.resize(cv_image, dsize=(0, 0), fx=0.5, fy=0.5)

        # Publish the resized image
        try:
            resized_image_msg = self.bridge.cv2_to_imgmsg(resized_image, encoding='bgr8')
            self.resized_image_pub.publish(resized_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish resized image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MultiSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
