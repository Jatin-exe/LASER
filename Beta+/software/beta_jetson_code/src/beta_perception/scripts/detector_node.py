#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from beta_perception.msg import DetectionArray, BoundingBox, Keypoint

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('beta_perception')
        
        # Initialize the Triton client
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
        
        # QoS settings for the subscription: reliable and depth of 1
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            Image,
            '/vimbax_camera_beta/image_rect',  # Replace with your actual image topic
            self.image_callback,
            qos_profile=qos_profile
        )
        self.detectionArrayPub = self.create_publisher(DetectionArray, "/laser/weed_detections", qos_profile=qos_profile)
        self.HIL_start_time = self.get_clock().now()
        self.bridge = CvBridge()  # Initialize CvBridge

    def create_triton_input(self, name, data, data_type):
        """Create Triton input with specified name, data, and data type."""
        triton_input = grpcclient.InferInput(name, data.shape, np_to_triton_dtype(data_type))
        triton_input.set_data_from_numpy(data)
        return triton_input

    def image_callback(self, msg):
        self.HIL_start_time = self.get_clock().now()
        # Convert ROS Image message to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return
        

        # Prepare the Triton input
        triton_input_image = self.create_triton_input("input_image", cv_image, np.uint8)
        
        try:
            # Synchronous inference
            response = self.triton_client.infer(
                model_name="perception_ensemble",
                inputs=[triton_input_image],
                outputs=[
                    grpcclient.InferRequestedOutput("boxes"),
                    grpcclient.InferRequestedOutput("areas"),
                    grpcclient.InferRequestedOutput("scores"),
                    grpcclient.InferRequestedOutput("keypoints"),
                    grpcclient.InferRequestedOutput("ids"),
                ]
            )
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # Process the inference response
        self.process_inference_response(response, msg.header)

        # Log timing
        loop_duration = (self.get_clock().now() - self.HIL_start_time).nanoseconds / 1e6  # Convert to ms
        self.get_logger().info(f"Perception Time: {loop_duration:.2f} ms")

    def process_inference_response(self, response, header):
        """Process the inference response and publish the results."""
        try:
            # Retrieve the outputs
            boxes = response.as_numpy("boxes")
            areas = response.as_numpy("areas")
            scores = response.as_numpy("scores")
            keypoints = response.as_numpy("keypoints")
            ids = response.as_numpy("ids")

            # Prepare and publish the detection results
            detection_msg = self.prepare_detection_results_message(keypoints, scores, boxes, areas, ids, header)
            self.detectionArrayPub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing inference response: {e}")

    def prepare_detection_results_message(self, kps, scores, boxes, areas, class_ids, header):        
        detection_array_msg = DetectionArray()
        detection_array_msg.header = header  # Set the header

        # Populate the DetectionArray with bounding boxes and associated data
        for i in range(len(boxes)):
            if class_ids[i] == 0:  # Only process detections with class ID 0
                box_msg = BoundingBox()
                
                # Set the bounding box properties
                box_msg.x = int(boxes[i][0])  # Assuming boxes are tuples or lists in Python
                box_msg.y = int(boxes[i][1])
                box_msg.width = int(boxes[i][2])
                box_msg.height = int(boxes[i][3])
                
                # Set additional properties
                box_msg.score = float(scores[i])
                box_msg.area = int(areas[i])

                # Add the keypoint associated with this bounding box
                kp_msg = Keypoint()
                kp_msg.x = int(kps[i, 0])  # Access the (i, 0) element of the numpy array
                kp_msg.y = int(kps[i, 1])  # Access the (i, 1) element

                # Attach keypoint to bounding box
                box_msg.keypoint = kp_msg

                # Add the bounding box to the DetectionArray
                detection_array_msg.boxes.append(box_msg)

        return detection_array_msg

def main(args=None):
    rclpy.init(args=args)
    perception = PerceptionNode()
    try:
        rclpy.spin(perception)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        perception.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == '__main__':
    main()
