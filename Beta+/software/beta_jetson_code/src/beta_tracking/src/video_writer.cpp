#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class VideoWriterNode : public rclcpp::Node {
public:
  VideoWriterNode() : Node("video_writer_node"), video_writer_initialized_(false) {
    rclcpp::QoS qos_profile(10); // History depth of 10
    qos_profile.reliability(
        RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT); // Reliable communication
    image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_topic", qos_profile,
        std::bind(&VideoWriterNode::image_callback, this, std::placeholders::_1));
  }

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &image_msg) {
    cv::Mat frame = cv_bridge::toCvShare(image_msg, "bgr8")->image;

    if (!video_writer_initialized_) {
      initialize_video_writer(frame.size());
    }

    video_writer_.write(frame);
    RCLCPP_INFO(this->get_logger(), "Frame written to video");
  }

  void initialize_video_writer(const cv::Size &frame_size) {
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = 30.0;
    video_writer_.open("output_video.avi", codec, fps, frame_size, true);
    if (!video_writer_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "Could not open the output video file for write");
    } else {
      video_writer_initialized_ = true;
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
  cv::VideoWriter video_writer_;
  bool video_writer_initialized_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VideoWriterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}