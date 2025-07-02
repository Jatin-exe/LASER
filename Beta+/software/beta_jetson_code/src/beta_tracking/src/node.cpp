#include <cv_bridge/cv_bridge.h>
#include <beta_perception/msg/detection_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <beta_tracking/logging.h>
#include <beta_tracking/FFTBoostedTemplateTracker.h>

class ExecTimer {
private:
  double &seconds;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;

public:
  ExecTimer() = default;
  explicit ExecTimer(double &seconds) : seconds(seconds) {
    start_time = std::chrono::high_resolution_clock::now();
  }
  ~ExecTimer() {
    end_time = std::chrono::high_resolution_clock::now();
    seconds = 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_time - start_time)
                           .count();
  }
};

class LaserTrackingNode : public rclcpp::Node {
public:
  LaserTrackingNode() : Node("targeting_node") {
    // Setup the tracker, lets start with the IOU tracker
    laser::tracking::logging::set_log_function(
        [&](const std::string &message,
            laser::tracking::logging::LogLevel lvl) {
          static auto logger = rclcpp::get_logger("targeting_node");
          switch (lvl) {
          case laser::tracking::logging::LogLevel::INFO:
            RCLCPP_INFO(logger, "%s", message.c_str());
            break;
          case laser::tracking::logging::LogLevel::WARNING:
            RCLCPP_WARN(logger, "%s", message.c_str());
            break;
          case laser::tracking::logging::LogLevel::ERROR:
            RCLCPP_ERROR(logger, "%s", message.c_str());
            break;
          case laser::tracking::logging::LogLevel::DEBUG:
            RCLCPP_DEBUG(logger, "%s", message.c_str());
            break;
          default:
            RCLCPP_ERROR(logger, "Unknown log level");
            break;
          }
        });
    tracker =
        laser::tracking::Tracker::build(laser::tracking::TrackerType::IOU);
    rclcpp::QoS qos_profile(10); // History depth of 10
    qos_profile.reliability(
        RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT); // Reliable communication

    image_subscriber_.subscribe(this, "image_topic",
                                qos_profile.get_rmw_qos_profile());
    detection_subscriber_.subscribe(this, "detection_topic",
                                    qos_profile.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::TimeSynchronizer<
        sensor_msgs::msg::Image, beta_perception::msg::DetectionArray>>(
        image_subscriber_, detection_subscriber_, 10);

    sync_->registerCallback(std::bind(&LaserTrackingNode::synchronized_callback,
                                      this, std::placeholders::_1,
                                      std::placeholders::_2));

    annotated_image_publisher_ =
        this->create_publisher<sensor_msgs::msg::Image>("annotated_image",
                                                        qos_profile);
  }

private:
  void synchronized_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
      const beta_perception::msg::DetectionArray::ConstSharedPtr
          &detection_msg) {
    double seconds = 0;
    {
      auto timer = ExecTimer(seconds);
      // measure executon time
      RCLCPP_DEBUG(
          this->get_logger(),
          "Received synchronized messages with timestamps: %u.%u and %u.%u",
          image_msg->header.stamp.sec, image_msg->header.stamp.nanosec,
          detection_msg->header.stamp.sec, detection_msg->header.stamp.nanosec);
      // Create a new data record
      auto record =
          laser::tracking::InputRecord::build(image_msg, detection_msg);
      // Now send it to the tracker
      tracker->update(record);
      // Get the annotated image from the tracker
      sensor_msgs::msg::Image annotated_image;
      tracker->annotate_image(annotated_image);
      // update the time stamp of the annoted image
      annotated_image.header.stamp = image_msg->header.stamp;
      // Publish the annotated image
      annotated_image_publisher_->publish(annotated_image);
    }
    RCLCPP_INFO(
        this->get_logger(),
        "Tracking took %.9f seconds to process the image and detections",
        seconds);
  }

  message_filters::Subscriber<sensor_msgs::msg::Image> image_subscriber_;
  message_filters::Subscriber<beta_perception::msg::DetectionArray>
      detection_subscriber_;
  std::shared_ptr<message_filters::TimeSynchronizer<
      sensor_msgs::msg::Image, beta_perception::msg::DetectionArray>>
      sync_;

  cv::Mat current_image_;
  std::vector<beta_perception::msg::BoundingBox> current_detections_;
  std::unordered_map<std::string, beta_perception::msg::BoundingBox>
      tracked_objects_;
  std::unique_ptr<laser::tracking::Tracker> tracker;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      annotated_image_publisher_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<LaserTrackingNode>();
  executor.add_node(node);
  executor.spin();
  RCLCPP_INFO(node->get_logger(), "Laser tracking node shutting down");
  rclcpp::shutdown();
  return 0;
}