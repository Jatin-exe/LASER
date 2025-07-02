#include "targeting.hpp"

Targeting::Targeting() : Node("targeting_node")
{
    // Initialize the subscriber to the TrackerOutput topic
    this->trackerSubscription_ = this->create_subscription<beta_tracking::msg::TrackerOutput>(
        "/laser/tracker", 
        rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable)
                  .durability(rclcpp::DurabilityPolicy::Volatile), 
        std::bind(&Targeting::trackerCallback_, this, std::placeholders::_1));

    // Initialize the publisher for laser location
    this->laserLocationPublisher_ = this->create_publisher<std_msgs::msg::UInt16MultiArray>(
        "/laser/location", 
        rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable)
                  .durability(rclcpp::DurabilityPolicy::Volatile));


    RCLCPP_INFO(this->get_logger(), "Targeting node has been initialized.");
}

Targeting::~Targeting()
{
    RCLCPP_INFO(this->get_logger(), "Targeting node is shutting down.");
}

void Targeting::trackerCallback_(const beta_tracking::msg::TrackerOutput::ConstSharedPtr& tracker_msg)
{
    /*
    Logic for tracker goes here
    */
}

// Main function to run the node
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Targeting>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
