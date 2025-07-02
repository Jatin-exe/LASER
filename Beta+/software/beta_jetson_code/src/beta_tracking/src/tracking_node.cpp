#include "cv_bridge/cv_bridge.h"
#include "beta_perception/msg/detection_array.hpp"
#include "beta_tracking/msg/tracker_output.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <nlohmann/json.hpp>
#include "beta_tracking/FFTBoostedTemplateTracker.h"
#include "beta_tracking/path.h"
#include "beta_tracking/tracker.h"
#include "beta_tracking/utils.h"

namespace laser::tracking {

bool use_trace_logging = false;

class TrackingNode : public rclcpp::Node {
public:
  TrackingNode() : Node("tracking_node") {
    //cv::namedWindow("Annotated Image", cv::WINDOW_NORMAL);
    // Using a custom TRACE log level if specified because ROS2 doesn't support
    // TRACE'
    laser::tracking::logging::set_log_function(
        [&](const std::string &message,
            laser::tracking::logging::LogLevel lvl) {
          static auto logger = rclcpp::get_logger("tracking_node");
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
          case laser::tracking::logging::LogLevel::TRACE:
            if (use_trace_logging) {
              RCLCPP_DEBUG(logger, "%s", message.c_str());
            }
            break;
          default:
            RCLCPP_ERROR(logger, "Unknown log level");
            break;
          }
        });

    this->declare_parameter<std::string>("log_level", "info");
    std::string log_level_str = this->get_parameter("log_level").as_string();
    // convert this to lower case
    std::transform(log_level_str.begin(), log_level_str.end(),
                   log_level_str.begin(), ::tolower);
    auto logger = rclcpp::get_logger("tracking_node");

    if (log_level_str == "debug") {
      logger.set_level(rclcpp::Logger::Level::Debug);
    } else if (log_level_str == "info") {
      logger.set_level(rclcpp::Logger::Level::Info);
    } else if (log_level_str == "warn") {
      logger.set_level(rclcpp::Logger::Level::Warn);
    } else if (log_level_str == "error") {
      logger.set_level(rclcpp::Logger::Level::Error);
    } else if (log_level_str == "fatal") {
      logger.set_level(rclcpp::Logger::Level::Fatal);
    } else if (log_level_str == "trace") {
      use_trace_logging = true;
      logger.set_level(rclcpp::Logger::Level::Debug);
    } else {
      RCLCPP_WARN(logger, "Invalid log level '%s'. Using 'info' as default.",
                  log_level_str.c_str());
      rclcpp::get_logger("tracking_node")
          .set_level(rclcpp::Logger::Level::Info);
    }

    RCLCPP_INFO(logger, "Log level set to: %s", log_level_str.c_str());

    // Initialize the FFTBoostedTemplateTracker
    FFTBoostedTracking::Options tracker_options;
    this->declare_parameter<float>("detection_score",
                                   FFTBoostedTracking::DEFAULT_DETECTION_SCORE);
    this->declare_parameter<float>(
        "refinement_offset", FFTBoostedTracking::DEFAULT_REFINEMENT_OFFSET);
    this->declare_parameter<std::string>("refinement_mode", "TM_CCOEFF_NORMED");
    this->declare_parameter<std::string>("refinement_strategy", "LAST_ANCHOR");
    this->declare_parameter<std::string>("reliability_option", "best_effort");
    // Declare parameter for annotation options
    this->declare_parameter<std::string>("annotation_options", "default");
    this->declare_parameter<double>("fft_downscale_factor", 1.0);

    if (this->has_parameter("fft_downscale_factor")) {
      tracker_options.fft_downscale_factor =
          this->get_parameter("fft_downscale_factor").as_double();
      RCLCPP_INFO(logger, "FFT downscale factor set to: %f",
                  tracker_options.fft_downscale_factor.value());
    }

    tracker_options.detection_score =
        this->get_parameter("detection_score").as_double();
    tracker_options.refinement_offset =
        this->get_parameter("refinement_offset").as_double();
    tracker_options.refinement_mode =
        FFTBoostedTracking::string_to_template_mode(
            this->get_parameter("refinement_mode").as_string());
    tracker_options.refinement_strategy =
        FFTBoostedTracking::string_to_refinement_strategy(
            this->get_parameter("refinement_strategy").as_string());

    // Load custom annotation options if provided in the parameter server
    std::string annotation_options_str =
        this->get_parameter("annotation_options").as_string();
    if (annotation_options_str != "default" &&
        !annotation_options_str.empty()) {
      try {
        nlohmann::json j = nlohmann::json::parse(annotation_options_str);
        tracker_options.optional_annotation_options = AnnotationOptions(j);
        RCLCPP_INFO(this->get_logger(), "Loaded custom annotation options");
      } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Error parsing annotation options: %s",
                     e.what());
      }
    } else if (annotation_options_str == "default") {
      RCLCPP_INFO(this->get_logger(), "Using default annotation options");
      tracker_options.optional_annotation_options = AnnotationOptions();
    } else {
      if (!annotation_options_str.empty()) {
        RCLCPP_WARN(this->get_logger(),
                    "Invalid annotation options. Using no annotation options");
      } else {
        RCLCPP_WARN(
            this->get_logger(),
            "No annotation options provided. Using no annotation options");
      }
    }

    tracker_options.validate_or_throw();
    tracker = std::make_unique<laser::tracking::FFTBoostedTracking>(
        std::move(tracker_options));
    priority_planner = std::make_unique<laser::tracking::SimpleCostedPath>();

    rclcpp::QoS qos_profile(10);
    std::string reliability_option =
        this->get_parameter("reliability_option").as_string();

    if (reliability_option == "best_effort") {
      qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    } else if (reliability_option == "reliable") {
      qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    } else if (reliability_option == "system_default") {
      // System default is already set by default, so we don't need to change
      // anything
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "Invalid reliability option '%s'. Using system default.",
                  reliability_option.c_str());
    }

    auto image_subs_options = rclcpp::SubscriptionOptions();
    image_subscriber_.subscribe(this, "image_topic",
                                qos_profile.get_rmw_qos_profile());
    detection_subscriber_.subscribe(this, "detection_topic",
                                    qos_profile.get_rmw_qos_profile());
    sync_ = std::make_shared<message_filters::TimeSynchronizer<
        sensor_msgs::msg::Image, beta_perception::msg::DetectionArray>>(
        image_subscriber_, detection_subscriber_, 1);

    sync_->registerCallback(std::bind(&TrackingNode::synchronized_callback,
                                      this, std::placeholders::_1,
                                      std::placeholders::_2));

    annotated_image_publisher_ =
        this->create_publisher<sensor_msgs::msg::Image>("annotated_image",
                                                        qos_profile);
    tracker_output_publisher_ =
        this->create_publisher<beta_tracking::msg::TrackerOutput>(
            "tracker_output", qos_profile);
  }

private:
  void synchronized_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
      const beta_perception::msg::DetectionArray::ConstSharedPtr
          &detection_msg) {
    //auto time_now = this->get_clock()->now();
    // Get the image
    auto incoming_image = utils::extract_image_from_ros(image_msg);
    logging::debug("Received synchronized data");
    // Get the detections
    auto anchors = FFTBoostedAnchor::build(*detection_msg, *image_msg);
    logging::debug("Received {} detections", anchors.size());
    // Perform tracking
    logging::debug("Performing tracking");
    auto tracker_output = tracker->track(incoming_image, std::move(anchors));
    logging::debug("Tracking completed");
    // Plan the target order
    logging::debug("Planning target order");
    priority_planner->update_path(tracker_output.get_tracked_instances());
    logging::debug("Target order planning completed");
    // Extract the target order
    auto &ordered_instances = priority_planner->get_path();
    // build the output message
    beta_tracking::msg::TrackerOutput tracker_output_msg;
    tracker_output_msg.target_list.reserve(ordered_instances.size());
    for (const auto &instance : ordered_instances) {
      // build the target message
      auto target_msg = beta_tracking::msg::Target();
      target_msg.id = instance->get_tracking_id().to_string();
      target_msg.box_tl =
          utils::extract_point_from_cv(instance->get_bounding_box().tl());
      target_msg.box_br =
          utils::extract_point_from_cv(instance->get_bounding_box().br());
      target_msg.target_point =
          utils::extract_point_from_cv(instance->get_target_coord());
      tracker_output_msg.target_list.push_back(std::move(target_msg));
    }
    tracker_output_msg.header = image_msg->header;
    // Publish the output message
    tracker_output_publisher_->publish(tracker_output_msg);
    //auto end_time = this->get_clock()->now();
    //logging::info("Time taken for tracking and annotation: {} ms",
                  //(end_time - time_now).nanoseconds() / 1000000);
    // Annotate the image
    auto annotated_image = tracker_output.get_annotated_image();
    if (annotated_image) {
      logging::debug("Publishing annotated image with size {}x{}x{}",
                     annotated_image->cols, annotated_image->rows,
                     annotated_image->channels());
      auto image = annotated_image.value();
      auto header = tracker_output_msg.header;
      auto ros_image = utils::extact_image_from_cv(image, std::move(header));
      annotated_image_publisher_->publish(ros_image);
      // Display the annotated image in an OpenCV window
      //cv::imshow("Annotated Image", annotated_image.value());
      //cv::waitKey(1); // Wait for a key press to close the window
      //      if(!tracker_output.get_tracked_instances().empty())
      //      {
      //        while(true)
      //        {
      //          if(cv::waitKey(1) == 'q')
      //          {
      //            exit(-1);
      //          }
      //        }
      //      }else
      //      {
      //        cv::waitKey(1);
      //      }
    }
  }

  message_filters::Subscriber<sensor_msgs::msg::Image> image_subscriber_;
  message_filters::Subscriber<beta_perception::msg::DetectionArray>
      detection_subscriber_;
  std::shared_ptr<message_filters::TimeSynchronizer<
      sensor_msgs::msg::Image, beta_perception::msg::DetectionArray>>
      sync_;

  std::unique_ptr<laser::tracking::FFTBoostedTracking> tracker{};
  std::unique_ptr<SimpleCostedPath> priority_planner;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      annotated_image_publisher_;
  rclcpp::Publisher<beta_tracking::msg::TrackerOutput>::SharedPtr
      tracker_output_publisher_{};

public:
  ~TrackingNode() override {
    // Destroy the OpenCV window when the node is destroyed
    cv::destroyAllWindows();
  }
};
} // namespace laser::tracking

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<laser::tracking::TrackingNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
