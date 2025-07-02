#ifndef SYSTEM_CLOCK_PUBLISHER_HPP_
#define SYSTEM_CLOCK_PUBLISHER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/bool.hpp>
#include <chrono>
#include <ctime>
#include <sys/stat.h>

class SystemClockPublisher : public rclcpp::Node
{
public:
    SystemClockPublisher();

private:
    void publish_system_clock();
    bool is_stm32_available();
    rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr clock_publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stm32_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

#endif // SYSTEM_CLOCK_PUBLISHER_HPP_

