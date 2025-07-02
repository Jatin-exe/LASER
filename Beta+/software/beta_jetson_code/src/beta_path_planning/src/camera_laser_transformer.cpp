#include "camera_laser_transformer.hpp"

namespace beta_transformer
{
    CameraLaserTransformer::CameraLaserTransformer() : Node("camera_laser_transformer")
    {
        this->transforms_ = this->loadCalibFromJSON("/workspaces/isaac_ros-dev/src/laser_framework/config/camera_laser_calib.json");

        this->plannerSub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/laser/current_target", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile),
        std::bind(&CameraLaserTransformer::positionCallback, this, std::placeholders::_1));

        this->posePublisher_ = this->create_publisher<std_msgs::msg::UInt16MultiArray>("/laser/location", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile));
    }

    CameraLaserTransformer::~CameraLaserTransformer()
    {
    }

    std::pair<std::vector<float>, std::vector<float>> CameraLaserTransformer::loadCalibFromJSON(const std::string &file_path)
    {
        // Read JSON file
        std::ifstream file(file_path);
        if (!file.is_open()) throw std::runtime_error("Could not open JSON file: " + file_path);

        nlohmann::json json_data;
        file >> json_data;
        
        // Parse calib_pixel_to_laser values into a vector
        std::vector<float> pixel_to_laser = json_data.at("calib_pixel_to_laser").get<std::vector<float>>();
        
        // Parse calib_laser_to_pixel values into a vector
        std::vector<float> laser_to_pixel = json_data.at("calib_laser_to_pixel").get<std::vector<float>>();

        return {pixel_to_laser, laser_to_pixel};
    }

    std::pair<int, int> CameraLaserTransformer::forawardTransform(const float source_x, const float source_y)
    {
        // Access pixel_to_laser_transform from transforms_
        const auto &pixel_to_laser = this->transforms_.first;

        float transformed_source_x = pixel_to_laser[0] + pixel_to_laser[1] * source_x +
                                    pixel_to_laser[2] * source_x * source_x +
                                    pixel_to_laser[3] * source_x * source_x * source_x +
                                    pixel_to_laser[4] * source_y +
                                    pixel_to_laser[5] * source_y * source_y +
                                    pixel_to_laser[6] * source_x * source_y +
                                    pixel_to_laser[7] * source_x * source_x * source_y +
                                    pixel_to_laser[8] * source_x * source_y * source_y;

        float transformed_source_y = pixel_to_laser[9] + pixel_to_laser[10] * source_x +
                                    pixel_to_laser[11] * source_x * source_x +
                                    pixel_to_laser[12] * source_x * source_x * source_x +
                                    pixel_to_laser[13] * source_y +
                                    pixel_to_laser[14] * source_y * source_y +
                                    pixel_to_laser[15] * source_x * source_y +
                                    pixel_to_laser[16] * source_x * source_x * source_y +
                                    pixel_to_laser[17] * source_x * source_y * source_y;

        return {static_cast<int>(transformed_source_x), static_cast<int>(transformed_source_y)};
    }


    std::pair<int, int> CameraLaserTransformer::inverseTransform(const float source_x, const float source_y)
    {
        // Access laser_to_pixel_transform from transforms_
        const auto &laser_to_pixel = this->transforms_.second;

        float transformed_source_x = laser_to_pixel[0] + laser_to_pixel[1] * source_x +
                                    laser_to_pixel[2] * source_x * source_x +
                                    laser_to_pixel[3] * source_x * source_x * source_x +
                                    laser_to_pixel[4] * source_y +
                                    laser_to_pixel[5] * source_y * source_y +
                                    laser_to_pixel[6] * source_x * source_y +
                                    laser_to_pixel[7] * source_x * source_x * source_y +
                                    laser_to_pixel[8] * source_x * source_y * source_y;

        float transformed_source_y = laser_to_pixel[9] + laser_to_pixel[10] * source_x +
                                    laser_to_pixel[11] * source_x * source_x +
                                    laser_to_pixel[12] * source_x * source_x * source_x +
                                    laser_to_pixel[13] * source_y +
                                    laser_to_pixel[14] * source_y * source_y +
                                    laser_to_pixel[15] * source_x * source_y +
                                    laser_to_pixel[16] * source_x * source_x * source_y +
                                    laser_to_pixel[17] * source_x * source_y * source_y;

        return {static_cast<int>(transformed_source_x), static_cast<int>(transformed_source_y)};
    }

    void CameraLaserTransformer::fireLaser(const int x, const int y, const int dwell)
    {
        // Create a message to publish the laser coordinates and dwell time
        std_msgs::msg::UInt16MultiArray laser_location_msg;

        // Populate the message with x, y, and dwell values
        laser_location_msg.data = {static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint16_t>(dwell)};

        // Publish the message
        this->posePublisher_->publish(laser_location_msg);
    }

    void CameraLaserTransformer::positionCallback(const geometry_msgs::msg::PointStamped msg)
    {

        // Convert pixel coordinates to laser coordinates using forwardTransform
        auto [laser_x, laser_y] = this->forawardTransform(msg.point.x, msg.point.y);

        // Fire the laser at the transformed coordinates with the specified dwell time
        this->fireLaser(laser_x, laser_y, msg.point.z);
    }
}

// Main function
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<beta_transformer::CameraLaserTransformer>());
    rclcpp::shutdown();
    return 0;
}
