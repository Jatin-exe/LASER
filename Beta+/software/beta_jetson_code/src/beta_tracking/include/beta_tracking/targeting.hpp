#ifndef TARGETING_HPP_
#define TARGETING_HPP_

#include <rclcpp/rclcpp.hpp>
#include "beta_tracking/msg/target.hpp"
#include "beta_tracking/msg/tracker_output.hpp"
#include <std_msgs/msg/u_int16_multi_array.hpp>


class Targeting : public rclcpp::Node
{
private:
    rclcpp::Subscription<beta_tracking::msg::TrackerOutput>::SharedPtr trackerSubscription_;
    rclcpp::Publisher<std_msgs::msg::UInt16MultiArray>::SharedPtr laserLocationPublisher_;
    void trackerCallback_(const beta_tracking::msg::TrackerOutput::ConstSharedPtr& tracker_msg);
public:
    Targeting();
    ~Targeting();
};

#endif