#pragma once

#include <eigen3/Eigen/Dense>
#include <beta_perception/msg/detection_array.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <sensor_msgs/msg/image.hpp>
#include <beta_tracking/anchor.h>
#include <beta_tracking/annotations.h>
#include <utility>

namespace laser::tracking {

class TrackingID {
public:
  TrackingID() = default;
  virtual ~TrackingID() = default;
  [[nodiscard]] virtual std::string to_string() const = 0;
  virtual bool operator==(const TrackingID &other) const = 0;
  virtual bool operator!=(const TrackingID &other) const = 0;
};

class TrackedInstance {
protected:
  TrackedInstance() = default;

public:
  virtual ~TrackedInstance() = default;
  virtual void update(const TrackedInstance& other) = 0;
  [[nodiscard]] virtual cv::Rect get_bounding_box() const = 0;
  [[maybe_unused]] [[nodiscard]] virtual std::vector<cv::Point2i> get_keypoints() const = 0;
  [[nodiscard]] virtual std::vector<cv::Point2f> get_mask() = 0;
  [[nodiscard]] virtual const TrackingID& get_tracking_id() const = 0;
  [[nodiscard]] virtual cv::Point2f get_target_coord() const = 0;
  [[maybe_unused]] [[nodiscard]] virtual cv::Mat
  annotate_image(const cv::Mat &image,
                 std::optional<AnnotationOptions> annotation_options) const = 0;
  [[nodiscard]] virtual std::unique_ptr<TrackedInstance> clone() const = 0;
};

}; // namespace laser::tracking