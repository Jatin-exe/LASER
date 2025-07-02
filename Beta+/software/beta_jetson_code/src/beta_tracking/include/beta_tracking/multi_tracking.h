#pragma once

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
#include <utility>
#include <eigen3/Eigen/Dense>

namespace laser::tracking::utils {

cv::Mat make_cv_mat(const sensor_msgs::msg::Image &image);

/**
 * @brief Converts a BoundingBox from the laser_framework package into an OpenCV
 * rectangle.
 *
 * @param laser_box The bounding box from the laser_framework package.
 * @return cv::Rect The corresponding OpenCV rectangle.
 */
cv::Rect make_cv_roi(const beta_perception::msg::BoundingBox &laser_box);

/**
 * @brief Calculates the Intersection over Union (IoU) of two rectangles.
 *
 * @param rect1 The first rectangle.
 * @param rect2 The second rectangle.
 * @return double The IoU value.
 */
double calculate_iou(const cv::Rect &rect1, const cv::Rect &rect2);

}; // namespace laser::tracking::utils

namespace laser::tracking {

// Default values
constexpr double DEFAULT_ID_FONT_SIZE = 5.0;
constexpr std::array<int, 3> DEFAULT_ID_COLOR_ARRAY = {255, 255, 255};  // White
constexpr std::array<int, 3> DEFAULT_BOX_COLOR_ARRAY = {255, 255, 255}; // White
constexpr double DEFAULT_BOX_THICKNESS = 10;

/**
 * @brief Enum representing different types of tracking algorithms.
 */
enum class TrackerType {
  ORB, ///< Tracking using ORB features defined from a bounding box or polytope.
  IOU, ///< Tracking using Intersection over Union (IoU) to match detected
       ///< bounding boxes.
  SuperPoints, ///< Tracking using SuperPoints, a feature descriptor defined
               ///< from a bounding box or polytope.
};

/**
 * @brief Data class representing a detection for tracking.
 */
class Detection {
public:
  const cv::Rect bounding_box; ///< The bounding box of the detection.

  Detection() = delete;

  /**
   * @brief Constructs a Detection object from an OpenCV rectangle.
   *
   * @param bounding_box The bounding box of the detection.
   */
  explicit Detection(cv::Rect bounding_box)
      : bounding_box(std::move(bounding_box)) {}

  /**
   * @brief Constructs a Detection object from a laser_framework BoundingBox.
   *
   * @param laser_box The bounding box from the laser_framework package.
   */
  explicit Detection(const beta_perception::msg::BoundingBox &laser_box)
      : bounding_box(utils::make_cv_roi(laser_box)){};

  virtual ~Detection() = default;
};

/**
 * @brief Class representing input data for tracking from an upstream source.
 */
class InputRecord {
protected:
  cv::Mat image_; ///< The image associated with the input record.
  std::vector<Detection>
      detections_; ///< The detections associated with the input record.

  InputRecord() = default;

public:
  virtual ~InputRecord() = default;

  /**
   * @brief Gets the image associated with the input record.
   *
   * @return const cv::Mat& The image.
   */
  const cv::Mat &image() const { return image_; };

  /**
   * @brief Gets the detections associated with the input record.
   *
   * @return const std::vector<Detection>& The detections.
   */
  const std::vector<Detection> &detections() const { return detections_; };

  /**
   * @brief Builds an InputRecord from ROS image and detection messages.
   *
   * @param image_msg The ROS image message.
   * @param detection_msg The ROS detection message.
   * @return InputRecord The constructed InputRecord.
   */
  static InputRecord
  build(const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
        const beta_perception::msg::DetectionArray::ConstSharedPtr
            &detection_msg);
};

/**
 * @brief Class representing an instance that is currently being tracked.
 */
class TrackedInstance {
public:
  /**
   * @brief A class representing options for annotating images for the tracked
   * instance.
   */
  class AnnotationOptions {
  public:
    cv::Scalar bounding_box_color =
        cv::Scalar(DEFAULT_BOX_COLOR_ARRAY[0], DEFAULT_BOX_COLOR_ARRAY[1],
                   DEFAULT_BOX_COLOR_ARRAY[2]); ///< The color of the bounding
                                                ///< box and id to be drawn.
    double bounding_box_thickness =
        DEFAULT_BOX_THICKNESS; ///< The line width of the bounding
                               ///< box.
    double id_font_size =
        DEFAULT_ID_FONT_SIZE; ///< The font size of the id to be drawn.
    cv::Scalar id_color =
        cv::Scalar(DEFAULT_ID_COLOR_ARRAY[0], DEFAULT_ID_COLOR_ARRAY[1],
                   DEFAULT_ID_COLOR_ARRAY[2]); ///< The color of the id text.

    /**
     * @brief Enum representing the position of the id relative to the bounding
     * box.
     */
    enum class IDPosition {
      TOP_LEFT,
      BOTTOM_RIGHT,
      CENTER,
      BOTTOM_CENTER,
      TOP_CENTER,
      TOP_RIGHT,
      BOTTOM_LEFT,
    };

    IDPosition id_position =
        IDPosition::TOP_CENTER; ///< The position of the id relative to the
                                ///< bounding box.

    AnnotationOptions(){};
    virtual ~AnnotationOptions() = default;
  };

public:
  const uint64_t uid = 0; ///< The unique identifier of the tracked instance.
  const Detection
      detection_roi; ///< The current bounding box of the tracked instance.

  /**
   * @brief Constructs a TrackedInstance object.
   *
   * @param detection_roi The detection associated with the tracked instance.
   * @param uid The unique identifier of the tracked instance.
   */
  TrackedInstance(const Detection &detection_roi, uint64_t uid)
      : uid(uid), detection_roi(detection_roi) {}

  TrackedInstance(const TrackedInstance &other)
      : uid(other.uid), detection_roi(other.detection_roi){};

  /**
   * @brief Annotates an image with the tracked instance's bounding box and id,
   * along with annotation options.
   *
   * @param image The image to annotate.
   * @param options The annotation options.
   */
  void
  annotate_image(cv::Mat &image,
                 const AnnotationOptions &options = AnnotationOptions()) const;

  /**
   * @brief Updates the tracked instance with a new detection and returns a new
   * instance.
   *
   * @param new_roi The new detection.
   * @return TrackedInstance The updated tracked instance.
   */
  TrackedInstance update_clone(const Detection &new_roi) const {
    return {new_roi, uid};
  }
};

/**
 * @brief Class representing the state of tracking.
 */
class TrackingState {
protected:
  cv::Mat source_image; ///< The last image that was used for tracking.
  std::map<uint64_t, TrackedInstance>
      tracked_instances; ///< The instances that are currently being tracked.

  TrackingState() = default;

public:
  virtual ~TrackingState() = default;

  /**
   * @brief Gets the source image used for tracking.
   *
   * @return cv::Mat The source image.
   */
  cv::Mat get_source_image() const { return source_image; }

  /**
   * @brief Gets the tracked instances.
   *
   * @return const std::map<uint64_t, TrackedInstance>& The tracked instances.
   */
  const std::map<uint64_t, TrackedInstance> &get_tracked_instances() const {
    return tracked_instances;
  }

  /**
   * @brief Gets an annotated image with the tracked instances.
   *
   * @param options The annotation options.
   * @return cv::Mat The annotated image.
   */
  cv::Mat
  get_annotated_image(const TrackedInstance::AnnotationOptions &options =
                          TrackedInstance::AnnotationOptions()) const;

  /**
   * @brief Updates the region of interest for a tracked instance.
   *
   * @param new_roi The new detection.
   * @param uid The unique identifier of the tracked instance.
   */
  void update_instance_roi(const Detection &new_roi, uint64_t uid);
};

/**
 * @brief Abstract base class for different tracking algorithms.
 */
class Tracker {
public:
  /**
   * @brief Options for configuring the tracker.
   */
  class TrackerOptions {
  public:
    TrackerOptions() = default;
    virtual ~TrackerOptions() = default;
  };

protected:
  explicit Tracker(TrackerType type) : type(type) {}

public:
  const TrackerType type; ///< The type of the tracker.

  virtual ~Tracker() = default;

  /**
   * @brief Resets the tracker to an initial state.
   */
  virtual void reset() = 0;

  /**
   * @brief Updates the tracker with new data.
   *
   * @param new_data The new input data.
   */
  virtual void update(const InputRecord &new_data) = 0;

  /**
   * @brief Returns the current tracked instances.
   *
   * @return std::map<uint64_t, TrackedInstance> The tracked instances.
   */
  virtual std::map<uint64_t, TrackedInstance> get_tracked_instances() const = 0;

  /**
   * @brief Annotates a ROS image with the tracked instances.
   *
   * @param ros_image The ROS image to annotate.
   */
  virtual void annotate_image(sensor_msgs::msg::Image &ros_image) const = 0;

  /**
   * @brief Builds a tracker of the specified type.
   *
   * @param type The type of the tracker.
   * @param options Optional tracker options.
   * @return std::unique_ptr<Tracker> The constructed tracker.
   */
  static std::unique_ptr<Tracker>
  build(TrackerType type,
        const std::optional<TrackerOptions> &options = std::nullopt);

  /**
   * @brief Builds a tracker from a JSON string.
   *
   * @param json_string The JSON string containing tracker configuration.
   * @return std::unique_ptr<Tracker> The constructed tracker.
   */
  static std::unique_ptr<Tracker> build(const std::string &json_string);
};




/**
 * @brief Class implementing an IOU-based tracking algorithm.
 *
 * The IOUTracker class is responsible for tracking objects using an Intersection over Union (IoU) based approach.
 * It supports multiple tracking algorithms, including KCF, CSRT, and GOTURN.
 */
class IOUTracker : public Tracker {
private:
  /**
   * @brief Converts JSON configuration to CSRT tracker parameters.
   *
   * @param j The JSON object containing the configuration.
   * @param p The CSRT tracker parameters to be populated.
   */
  static void from_json_to_CSRT_params(const nlohmann::json &j,
                                       cv::TrackerCSRT::Params &p);

  /**
   * @brief Converts JSON configuration to KCF tracker parameters.
   *
   * @param j The JSON object containing the configuration.
   * @param p The KCF tracker parameters to be populated.
   */
  static void from_json_to_KCF_params(const nlohmann::json &j,
                                      cv::TrackerKCF::Params &p);

public:
  /**
   * @brief Enumeration of supported tracking algorithms.
   */
  enum class Algorithm {
    KCF,    ///< Kernelized Correlation Filters
    CSRT,   ///< Discriminative Correlation Filter with Channel and Spatial Reliability
    GOTURN, ///< Generic Object Tracking Using Regression Networks
  };

  /**
   * @brief Options for configuring the IOUTracker.
   */
  class IOUTrackerOptions : public TrackerOptions {
  public:
    Algorithm algorithm = Algorithm::CSRT; ///< The tracking algorithm to use.
    double min_overlap = 0.5; ///< Minimum overlap for IoU matching.
    std::optional<std::string> params_json{
        std::nullopt}; ///< Optional JSON string for algorithm parameters.
    IOUTrackerOptions() = default;
    ~IOUTrackerOptions() override = default;
  };

private:
  uint64_t uid_counter{0}; ///< Counter for generating unique identifiers for tracked instances.

  /**
   * @brief Internal state of the IOUTracker.
   */
  class IOUTrackingState : public TrackingState {
  protected:
    std::map<uint64_t, cv::Ptr<cv::Tracker>> trackers; ///< Map of unique IDs to tracker instances.
    friend IOUTracker;
  };

  Algorithm algorithm{Algorithm::CSRT}; ///< The selected tracking algorithm.
  double min_overlap{0.75}; ///< Minimum overlap threshold for IoU matching.
  std::unique_ptr<IOUTrackingState> current_state{nullptr}; ///< Current tracking state.

  /**
   * @brief Constructs a new IOUTracker instance.
   */
  IOUTracker() : Tracker(TrackerType::IOU){};

  /**
   * @brief Checks if the tracker has a current state.
   *
   * @return true if the tracker has a state, false otherwise.
   */
  bool has_state() const { return current_state != nullptr; };

  /**
   * @brief Builds a tracker instance for a given detection.
   *
   * @param detection_roi The region of interest for the detection.
   * @param image The image in which the detection is made.
   * @param params_json Optional JSON string for algorithm parameters.
   * @return cv::Ptr<cv::Tracker> The constructed tracker instance.
   */
  cv::Ptr<cv::Tracker> build_tracker_instance(
      const Detection &detection_roi, const cv::Mat &image,
      std::optional<std::string> params_json = std::nullopt) const;

public:
  ~IOUTracker() override = default;

  /**
   * @brief Resets the tracker to its initial state.
   */
  void reset() override;

  /**
   * @brief Updates the tracker with new input data.
   *
   * @param new_data The new input data to update the tracker with.
   */
  void update(const InputRecord &new_data) override;

  /**
   * @brief Returns the current tracked instances.
   *
   * @return std::map<uint64_t, TrackedInstance> The tracked instances.
   */
  std::map<uint64_t, TrackedInstance> get_tracked_instances() const override;

  /**
   * @brief Annotates a ROS image with the tracked instances.
   *
   * @param ros_image The ROS image to annotate.
   */
  void annotate_image(sensor_msgs::msg::Image &ros_image) const override;

  /**
   * @brief Builds an IOUTracker with the specified options.
   *
   * @param options The options for configuring the tracker.
   * @return std::unique_ptr<Tracker> The constructed IOUTracker.
   */
  static std::unique_ptr<Tracker> build(const TrackerOptions &options);
};

namespace estimation {

//  /**
//   * @brief Class for estimating the shift between two images using a fixed window approach
//   */
//  class FixedWindowShiftEstimator {
//  protected:
//    cv::Mat prev_frame; //The previous image that was used for estimation
//    Eigen::Affine3d transform; //Most recent transformation estimate
//    cv::Rect2d fixed_window; //The ROI in the image that will be used for estimation
//    uint32_t window_expansion_offset; //The number of pixels to expand the fixed window for searching
//  public:
//    virtual ~FixedWindowShiftEstimator() = default;
//    Eigen::Affine3d update(const cv::Mat &new_frame);
//    cv::Mat annotate_image(const cv::Mat &image) const;
//  };

  cv::Point2f estimate_pairwise_shift(const cv::Mat &prev_frame, const cv::Mat &curr_frame);
};

} // namespace laser::tracking