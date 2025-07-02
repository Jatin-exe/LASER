#ifndef PERCEPTION_HPP_
#define PERCEPTION_HPP_

#include <memory>  // For smart pointers

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int16.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// ROS2 custom message includes
#include "beta_perception/msg/detection_array.hpp"
#include "beta_perception/msg/bounding_box.hpp"
#include "beta_perception/msg/keypoint.hpp"

// Triton client and gRPC includes
#include "grpc_client.h"


class PerceptionNode : public rclcpp::Node {
public:
    PerceptionNode();
    ~PerceptionNode();

private:
    // ROS 2 Components
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<beta_perception::msg::DetectionArray>::SharedPtr detection_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt16>::SharedPtr inference_time_pub_;

    // Triton Client
    std::unique_ptr<triton::client::InferenceServerGrpcClient> triton_client_;

    // QoS Settings
    rclcpp::Time start_time_;

    // Callback Functions
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    
    // Crop type
    std::string crop_type_;

    // Helper Functions
    triton::client::InferInput* createTritonInput(const std::string& name, const cv::Mat& image, const std::string& datatype);
    triton::client::InferInput* createTritonStringInput(const std::string& name, const std::string& value);
    void processInferenceResponse(
        const triton::client::InferResult* result,
        const std_msgs::msg::Header& header
    );
    beta_perception::msg::DetectionArray prepareDetectionResultsMessage(
    const int32_t* keypoints, const float* scores, const int32_t* boxes, const int32_t* areas,
    const int32_t* class_ids, const std_msgs::msg::Header& header, size_t num_detections);

    float fp16ToFloat(uint16_t fp16);
    std::string serializeString(const std::string& str);

    double total_duration_;   // Accumulates total inference durations
    size_t inference_count_;
};

#endif // PERCEPTION_HPP_
