#ifndef CAMERA_LASER_TRANSFORMER_HPP_
#define CAMERA_LASER_TRANSFORMER_HPP_

#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int16_multi_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nlohmann/json.hpp>


namespace beta_transformer
{
    class CameraLaserTransformer : public rclcpp::Node
    {
    public:
        CameraLaserTransformer();
        ~CameraLaserTransformer();
    private:
        // Subscription to TrackerOutput topic
        rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr plannerSub_;
        // Callback for tracker messages
        void positionCallback(const geometry_msgs::msg::PointStamped msg);

        rclcpp::Publisher<std_msgs::msg::UInt16MultiArray>::SharedPtr posePublisher_;

        // Function to load Calib from JSON file
        std::pair<std::vector<float>, std::vector<float>> loadCalibFromJSON(const std::string &file_path);

        std::pair<int, int> forawardTransform(const float source_x, const float source_y); 
        std::pair<int, int> inverseTransform(const float source_x, const float source_y); 

        void fireLaser(const int x, const int y, const int dwell);

        std::pair<std::vector<float>, std::vector<float>> transforms_;
    };
}
#endif