#include "system_clock_publisher.hpp"

SystemClockPublisher::SystemClockPublisher()
    : Node("system_clock_publisher")
{
    // Create publishers
    clock_publisher_ = this->create_publisher<std_msgs::msg::Int64>("/laser/jetson_clock", 1);
    stm32_publisher_ = this->create_publisher<std_msgs::msg::Bool>("/laser/laser_controller_status", 1);

    // Timer to publish the system clock every second
    timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&SystemClockPublisher::publish_system_clock, this));
}

bool SystemClockPublisher::is_stm32_available()
{
    struct stat buffer;
    return (stat("/dev/stm32", &buffer) == 0);
}

void SystemClockPublisher::publish_system_clock()
{
    // Get current system time
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *time_info = std::localtime(&now_c);

    // Format time as HHMMSS integer
    int64_t time_int = (time_info->tm_hour * 10000) + (time_info->tm_min * 100) + (time_info->tm_sec);

    // Create and publish the time message
    auto time_message = std_msgs::msg::Int64();
    time_message.data = time_int;
    clock_publisher_->publish(time_message);

    // Check if /dev/stm32 is available and publish Bool message
    auto stm32_message = std_msgs::msg::Bool();
    stm32_message.data = is_stm32_available();
    stm32_publisher_->publish(stm32_message);
}

// Main function
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SystemClockPublisher>());
    rclcpp::shutdown();
    return 0;
}

