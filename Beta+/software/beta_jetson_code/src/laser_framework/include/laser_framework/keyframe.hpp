#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>  // For cv::imshow
#include <filesystem>
#include <deque>
#include <string>
#include <random>

class KeyframeSaver : public rclcpp::Node
{
public:
    explicit KeyframeSaver(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    cv::Ptr<cv::ORB> orb_;
    cv::BFMatcher matcher_;

    cv::Mat last_keyframe_;
    std::vector<cv::KeyPoint> last_keypoints_;
    cv::Mat last_descriptors_;

    int inlier_threshold_;

    void saveKeyframeImage(const cv::Mat& image, const rclcpp::Time& timestamp);

    std::string save_directory_;
    double save_probability_;  // 50% chance to save a keyframe

    bool refinement_mode_;

    int max_keyframes_;
    std::deque<std::string> saved_keyframes_;  // Stores filenames of saved keyframes
};
