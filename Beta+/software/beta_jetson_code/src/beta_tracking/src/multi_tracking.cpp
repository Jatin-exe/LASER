

#include <beta_tracking/logging.h>
#include <beta_tracking/multi_tracking.h>
#include <cv_bridge/cv_bridge.h>
#include <beta_perception/msg/detection_array.hpp>
#include <utility>

namespace laser::tracking::logging {

std::function<void(const std::string &, LogLevel)> log_function =
    default_log_function;

void default_log_function(const std::string &message, LogLevel level) {
  switch (level) {
  case LogLevel::INFO:
    std::cout << "[INFO] " << message << std::endl;
    break;
  case LogLevel::WARNING:
    std::cerr << "[WARNING] " << message << std::endl;
    break;
  case LogLevel::ERROR:
    std::cerr << "[ERROR] " << message << std::endl;
    break;
  case LogLevel::DEBUG:
    std::cerr << "[DEBUG] " << message << std::endl;
    break;
  }
}
void set_log_function(
    std::function<void(const std::string &, LogLevel)> log_func) {
  log_function = std::move(log_func);
}

}
laser::tracking::InputRecord laser::tracking::InputRecord::build(
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
    const beta_perception::msg::DetectionArray::ConstSharedPtr &detection_msg) {
  auto new_record = InputRecord();

  // Convert the ROS image message to an OpenCV image
  try {
    new_record.image_ = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8)->image;
  } catch (cv_bridge::Exception &e) {
    throw std::runtime_error("cv_bridge exception: " + std::string(e.what()));
  }

  // Convert the DetectionArray message to a vector of Detection objects
  if (detection_msg) {
    for (const auto &detection : detection_msg->boxes) {
      new_record.detections_.emplace_back(std::move(Detection(detection)));
    }
  }

  return new_record;
}
// namespace laser::tracking::logging

namespace laser::tracking {

// Turn a BoundingBox from the laser_framework package into an OpenCV
// rectangle/bounding box
cv::Rect
utils::make_cv_roi(const beta_perception::msg::BoundingBox &laser_box) {
  // Ensure the bounding box dimensions are valid
  if (laser_box.width <= 0 || laser_box.height <= 0) {
    throw std::invalid_argument("Bounding box dimensions must be positive.");
  }
  // Create and return the OpenCV rectangle
  return {static_cast<int>(laser_box.x), static_cast<int>(laser_box.y),
          static_cast<int>(laser_box.width),
          static_cast<int>(laser_box.height)};
}

cv::Mat TrackingState::get_annotated_image(
    const TrackedInstance::AnnotationOptions &options) const {
  // Copy the source image and annotate it with the tracked instances
  cv::Mat annotated_image = source_image.clone();
  for (const auto &tracked_instance : tracked_instances) {
    tracked_instance.second.annotate_image(annotated_image, options);
  }
  return annotated_image;
}
void TrackingState::update_instance_roi(const Detection &new_roi,
                                        uint64_t uid) {
  // find the uid in the tracked instances and update its ROI
  bool has_uid = tracked_instances.contains(uid);
  if (has_uid) {
    auto updated_instance = tracked_instances.at(uid).update_clone(new_roi);
    tracked_instances.erase(uid);
    tracked_instances.emplace(uid, std::move(updated_instance));
  }
}
void TrackedInstance::annotate_image(
    cv::Mat &image, const TrackedInstance::AnnotationOptions &options) const {
  // Annotate the image with the tracked instance
  // Draw a rectangle around the bounding box of the tracked instance
  cv::rectangle(image, detection_roi.bounding_box, options.bounding_box_color,
                options.bounding_box_thickness);

  std::string text = "ID: " + std::to_string(uid);
  int font_face = cv::FONT_HERSHEY_PLAIN;
  cv::Point2d text_origin;
  switch (options.id_position) {
  case AnnotationOptions::IDPosition::TOP_LEFT:
    text_origin = detection_roi.bounding_box.tl();
    break;
  case AnnotationOptions::IDPosition::BOTTOM_RIGHT:
    text_origin = detection_roi.bounding_box.br();
    break;
  case AnnotationOptions::IDPosition::CENTER:
    text_origin = cv::Point(
        detection_roi.bounding_box.x + detection_roi.bounding_box.width / 2,
        detection_roi.bounding_box.y + detection_roi.bounding_box.height / 2);
    break;
  case AnnotationOptions::IDPosition::BOTTOM_CENTER:
    text_origin = cv::Point(
        detection_roi.bounding_box.x + detection_roi.bounding_box.width / 2,
        detection_roi.bounding_box.y + detection_roi.bounding_box.height);
    break;
  case AnnotationOptions::IDPosition::TOP_CENTER:
    text_origin = cv::Point(detection_roi.bounding_box.x +
                                detection_roi.bounding_box.width / 2,
                            detection_roi.bounding_box.y);
    break;
  case AnnotationOptions::IDPosition::TOP_RIGHT:
    text_origin = cv::Point(detection_roi.bounding_box.x +
                                detection_roi.bounding_box.width,
                            detection_roi.bounding_box.y);
    break;
  case AnnotationOptions::IDPosition::BOTTOM_LEFT:
    text_origin = cv::Point(detection_roi.bounding_box.x,
                            detection_roi.bounding_box.y +
                                detection_roi.bounding_box.height);
    break;
  default:
    text_origin = detection_roi.bounding_box.tl();
    break;
  }

  cv::putText(image, text, text_origin, font_face, options.id_font_size,
              options.id_color, 50);
}
std::unique_ptr<Tracker>
Tracker::build(TrackerType type, const std::optional<TrackerOptions> &options) {
  switch (type) {
  case TrackerType::IOU:
    if (options.has_value()) {
      return IOUTracker::build(options.value());
    } else {
      return IOUTracker::build(IOUTracker::IOUTrackerOptions());
    }
  default:
    throw std::runtime_error("Unsupported tracker type");
  }
}
std::unique_ptr<Tracker> Tracker::build(const std::string &json_string) {
  try {
    auto j = nlohmann::json::parse(json_string);
    if (!j.contains("type") || !j["type"].is_string()) {
      throw std::invalid_argument("JSON must contain a string field 'type'");
    }
    std::string type_str = j["type"];
    TrackerType type;
    if (type_str == "IOU") {
      type = TrackerType::IOU;
    } else if (type_str == "ORB") {
      type = TrackerType::ORB;
    } else if (type_str == "SuperPoints") {
      type = TrackerType::SuperPoints;
    } else {
      throw std::invalid_argument("Unsupported tracker type: " + type_str);
    }

    // Parse options based on the tracker type
    if (type == TrackerType::IOU) {
      IOUTracker::IOUTrackerOptions options;
      if (j.contains("min_overlap") && j["min_overlap"].is_number_float()) {
        options.min_overlap = j["min_overlap"];
      }
      if (j.contains("algorithm") && j["algorithm"].is_string()) {
        std::string algo_str = j["algorithm"];
        if (algo_str == "KCF") {
          options.algorithm = IOUTracker::Algorithm::KCF;
        } else if (algo_str == "CSRT") {
          options.algorithm = IOUTracker::Algorithm::CSRT;
        } else {
          throw std::invalid_argument("Unsupported tracker algorithm: " +
                                      algo_str);
        }
      }
      if (j.contains("algorithm_params") && j["algorithm_params"].is_object()) {
        options.params_json = j["algorithm_params"].dump();
      }
      return build(type, options);
    }

    // Add parsing for other tracker types if needed

  } catch (const std::exception &e) {
    std::cerr << "Error building tracker from JSON: " << e.what() << std::endl;
    return nullptr;
  }
  return nullptr;
}

// Calculate the Intersection over Union (IoU) between two rectangles
double utils::calculate_iou(const cv::Rect &rect1, const cv::Rect &rect2) {
  // Calculate the intersection rectangle
  int x_left = std::max(rect1.x, rect2.x);
  int y_top = std::max(rect1.y, rect2.y);
  int x_right = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
  int y_bottom = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

  // Check if there is an intersection
  if (x_right < x_left || y_bottom < y_top) {
    return 0.0; // No intersection
  }

  // Calculate the area of the intersection rectangle
  int intersection_area = (x_right - x_left) * (y_bottom - y_top);

  // Calculate the area of both rectangles
  int rect1_area = rect1.width * rect1.height;
  int rect2_area = rect2.width * rect2.height;

  // Calculate the area of the union
  int union_area = rect1_area + rect2_area - intersection_area;

  // Check for division by zero
  if (union_area == 0) {
    throw std::runtime_error(
        "Union area is zero, which should not happen with valid rectangles.");
  }

  // Calculate and return the IoU
  return static_cast<double>(intersection_area) / union_area;
}

} // namespace laser::tracking
