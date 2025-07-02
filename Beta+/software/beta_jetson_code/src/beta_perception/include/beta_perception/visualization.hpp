// visualization.hpp

#ifndef VISUALIZATION_HPP_
#define VISUALIZATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include "beta_perception/msg/detection_array.hpp"
#include "beta_perception/msg/bounding_box.hpp"
#include "beta_perception/msg/keypoint.hpp"

class Visualization : public rclcpp::Node
{
public:
    Visualization();

private:
    void synchronized_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg, const beta_perception::msg::DetectionArray::ConstSharedPtr& detection_msg);

    message_filters::Subscriber<sensor_msgs::msg::Image> image_subscriber_;
    message_filters::Subscriber<beta_perception::msg::DetectionArray> detection_subscriber_;

    // Synchronizer
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, beta_perception::msg::DetectionArray> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;

    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr annotated_image_publisher_;
    
};

#endif  // VISUALIZATION_HPP_
