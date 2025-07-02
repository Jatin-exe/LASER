#pragma once

#include <cmath>
#include <fmt/format.h>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread-pool/include/BS_thread_pool.hpp>
#include <beta_tracking/anchor.h>
#include <beta_tracking/logging.h>
#include <beta_tracking/tracked_instance.h>
#include <beta_tracking/tracker.h>
#include <beta_tracking/utils.h>
#include <utility>
#include <vector>

namespace laser::tracking {

class FFTFeatures {
public:
  cv::Rect2i bounding_box;
  std::vector<cv::Point2i> keypoints;

  FFTFeatures() = delete;

  FFTFeatures(cv::Rect2i bbox, const std::vector<cv::Point2i> &kps)
      : bounding_box(std::move(bbox)), keypoints(kps){};

  void translate(const cv::Point2f &translation) {
    for (auto &kp : keypoints) {
      kp.x += translation.x;
      kp.y += translation.y;
    }

    bounding_box.y += translation.y;
    bounding_box.x += translation.x;
  }

  [[nodiscard]] cv::Mat
  annotate_image(const cv::Mat &image,
                 std::optional<AnnotationOptions> annotation_options =
                     AnnotationOptions()) const;
};

class FFTBoostedAnchor : public Anchor {
public:
  const FFTFeatures features;
  cv::Mat source_image;

protected:
  FFTBoostedAnchor(cv::Rect2f bbox, const std::vector<cv::Point2i> &kps,
                   cv::Mat img)
      : features(std::move(bbox), kps), source_image(std::move(img)){};

public:
  FFTBoostedAnchor() = delete;

  ~FFTBoostedAnchor() override = default;

  [[nodiscard]] cv::Rect get_bounding_box() const override {
    return features.bounding_box;
  };

  [[nodiscard]] std::vector<cv::Point2i> get_keypoints() const override {
    return features.keypoints;
  };

  [[nodiscard]] std::vector<cv::Point2f> get_mask() const override {
    throw std::runtime_error("No mask available for FFTBoostedAnchor");
  };

  [[maybe_unused]] [[nodiscard]] cv::Mat annotate_image(
      const cv::Mat &image,
      std::optional<AnnotationOptions> annotation_options) const override;

  [[nodiscard]] std::string to_string() const override {
    return fmt::format("FFTBoostedAnchor[bbox={}, kps={}",
                       utils::cv_rect_to_string(this->features.bounding_box),
                       utils::cv_points_to_string(features.keypoints));
  };

  [[nodiscard]] std::unique_ptr<Anchor> clone() const {
    return std::make_unique<FFTBoostedAnchor>(*this);
  };

  [[nodiscard]] const cv::Mat &get_source_image() const override {
    return source_image;
  }

  static std::unique_ptr<Anchor>
  build(const beta_perception::msg::BoundingBox &msg,
        const sensor_msgs::msg::Image &img);

  static std::vector<std::unique_ptr<Anchor>>
  build(const beta_perception::msg::DetectionArray &msg,
        const sensor_msgs::msg::Image &img);

  static std::unique_ptr<Anchor> build(const cv::Mat &image,
                                       const FFTFeatures &features);
};

class FFTBoostedTrackingID : public TrackingID {
protected:
  uint64_t id{};
  static uint64_t id_counter;

  FFTBoostedTrackingID() = default;

public:
  ~FFTBoostedTrackingID() override = default;

  static FFTBoostedTrackingID create(const TrackingID &other);

  static FFTBoostedTrackingID create();

  [[nodiscard]] std::string to_string() const override {
    return fmt::format("id={}", id);
  };

  bool operator==(const TrackingID &other) const override {
    try {
      const auto &other_cast =
          dynamic_cast<const FFTBoostedTrackingID &>(other);
      return id == other_cast.id;
    } catch (const std::bad_cast &) {
      throw std::runtime_error(std::string(__func__) +
                               "Invalid cast to FFTBoostedTrackingID");
    }
  }

  bool operator!=(const TrackingID &other) const override {
    try {
      return !(*this == other);
    } catch (const std::runtime_error &) {
      throw std::runtime_error(std::string(__func__) +
                               "Invalid cast to FFTBoostedTrackingID");
    }
  }

  bool operator<(const TrackingID &other) const {
    try {
      const auto &other_cast =
          dynamic_cast<const FFTBoostedTrackingID &>(other);
      return id < other_cast.id;
    } catch (const std::bad_cast &) {
      throw std::runtime_error(std::string(__func__) +
                               "Invalid cast to FFTBoostedTrackingID");
    }
  }
  [[nodiscard]] uint64_t as_int() const { return id; };
};

class FFTBoostedInstance : public TrackedInstance {
public:
  enum class REFINEMENT_STRATEGY { LAST_ANCHOR, PREVIOUS_DETECTION };

protected:
  // Runtime features of the tracked instance.
  std::unique_ptr<FFTFeatures> features;
  // The unique identifier for the tracked instance.
  std::unique_ptr<FFTBoostedTrackingID> id;
  // The last anchor used to update the tracked instance.
  std::unique_ptr<FFTBoostedAnchor> last_anchor;
  cv::Rect2f template_roi;
  cv::Mat template_image;

  FFTBoostedInstance() = default;

public:
  ~FFTBoostedInstance() override = default;

  void update(const TrackedInstance &other) override {
    const auto &other_cast = dynamic_cast<const FFTBoostedInstance &>(other);
    features = std::make_unique<FFTFeatures>(*other_cast.features);
  }

  [[nodiscard]] cv::Rect get_bounding_box() const override {
    return features->bounding_box;
  }

  [[nodiscard]] std::vector<cv::Point2i> get_keypoints() const override {
    return features->keypoints;
  }

  [[nodiscard]] std::vector<cv::Point2f> get_mask() override {
    throw std::runtime_error("No mask available for FFTBoostedInstance");
  }

  [[nodiscard]] TrackingID &get_tracking_id() const override { return *id; }

  [[maybe_unused]] [[nodiscard]] cv::Mat annotate_image(
      const cv::Mat &image,
      std::optional<AnnotationOptions> annotation_options) const override;

  void translate(const cv::Point2f &translation) {
    features->translate(translation);
  }

  [[nodiscard]] double
  generate_matching_score(const FFTBoostedAnchor &anchor) const;

  void assign_anchor(const FFTBoostedAnchor &anchor) {
    try
    {
      last_anchor = std::make_unique<FFTBoostedAnchor>(anchor);
      template_roi = anchor.get_bounding_box();
      template_image = anchor.source_image(template_roi);
      logging::trace("ROI: {} {} {} {}", template_roi.x, template_roi.y, template_roi.width, template_roi.height);
      this->features = std::make_unique<FFTFeatures>(last_anchor->features);
    }
    catch(...)
    {
      template_roi = anchor.get_bounding_box();
      logging::trace("ROI: {} {} {} {}", template_roi.x, template_roi.y, template_roi.width, template_roi.height);
      throw std::runtime_error("Anchor error!!!!!!");
    }
  }

  void
  refine(const cv::Mat &new_image, float offset,
         REFINEMENT_STRATEGY strategy = REFINEMENT_STRATEGY::PREVIOUS_DETECTION,
         cv::TemplateMatchModes mode = cv::TM_CCOEFF_NORMED);

  static std::unique_ptr<FFTBoostedInstance>
  create(const FFTBoostedAnchor &anchor);

  static std::unique_ptr<FFTBoostedInstance>
  create(const FFTBoostedInstance &other);

  [[nodiscard]] cv::Point2f get_target_coord() const override {
    return this->features->keypoints[0];
  }

  [[nodiscard]] std::unique_ptr<TrackedInstance> clone() const override {
    return create(*this);
  }
};

class FFTBoostedTracking {
protected:
public:
  static constexpr auto DEFAULT_REFINEMENT_OFFSET = 0.1f;
  static constexpr auto DEFAULT_REFINEMENT_MODE = cv::TM_CCOEFF_NORMED;
  static constexpr auto DEFAULT_REFINEMENT_STRATEGY =
      FFTBoostedInstance::REFINEMENT_STRATEGY::PREVIOUS_DETECTION;
  static constexpr auto DEFAULT_DETECTION_SCORE = 50.0f;
  static cv::TemplateMatchModes
  string_to_template_mode(const std::string &mode_str) {
    // convert to upper case
    auto mode_str_upper = mode_str;
    std::transform(mode_str_upper.begin(), mode_str_upper.end(),
                   mode_str_upper.begin(), ::toupper);
    if (mode_str == "TM_SQDIFF_NORMED")
      return cv::TM_SQDIFF_NORMED;
    else if (mode_str == "TM_SQDIFF")
      return cv::TM_SQDIFF;
    else if (mode_str == "TM_CCOEFF_NORMED")
      return cv::TM_CCOEFF_NORMED;
    else if (mode_str == "TM_CCOEFF")
      return cv::TM_CCOEFF;
    else if (mode_str == "TM_CCORR_NORMED")
      return cv::TM_CCORR_NORMED;
    else if (mode_str == "TM_CCORR")
      return cv::TM_CCORR;
    else
      throw std::invalid_argument("Invalid template matching mode: " +
                                  mode_str);
  }

  static FFTBoostedInstance::REFINEMENT_STRATEGY
  string_to_refinement_strategy(const std::string &strategy_str) {
    // convert to upper case
    auto strategy_str_upper = strategy_str;
    std::transform(strategy_str_upper.begin(), strategy_str_upper.end(),
                   strategy_str_upper.begin(), ::toupper);
    if (strategy_str == "PREVIOUS_DETECTION")
      return FFTBoostedInstance::REFINEMENT_STRATEGY::PREVIOUS_DETECTION;
    else if (strategy_str == "LAST_ANCHOR")
      return FFTBoostedInstance::REFINEMENT_STRATEGY::LAST_ANCHOR;
    else
      throw std::invalid_argument("Invalid refinement strategy: " +
                                  strategy_str);
  }
  class Options {
  public:
    double detection_score = DEFAULT_DETECTION_SCORE;
    double refinement_offset = DEFAULT_REFINEMENT_OFFSET;
    std::optional<double> fft_downscale_factor = std::nullopt;
    uint32_t num_worker_threads = 8;
    cv::TemplateMatchModes refinement_mode = DEFAULT_REFINEMENT_MODE;
    FFTBoostedInstance::REFINEMENT_STRATEGY refinement_strategy =
        DEFAULT_REFINEMENT_STRATEGY;
    // If null opt then bypass any annotation
    std::optional<AnnotationOptions> optional_annotation_options = std::nullopt;
    void validate_or_throw() const {
      if (detection_score < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "Detection score must be non-negative: {}", detection_score));
      }
      if (refinement_offset < 0.0f) {
        throw std::invalid_argument(fmt::format(
            "Refinement offset must be non-negative: {}", refinement_offset));
      }
      if (refinement_mode != cv::TM_SQDIFF_NORMED &&
          refinement_mode != cv::TM_SQDIFF &&
          refinement_mode != cv::TM_CCOEFF_NORMED &&
          refinement_mode != cv::TM_CCOEFF &&
          refinement_mode != cv::TM_CCORR_NORMED &&
          refinement_mode != cv::TM_CCORR) {
        throw std::invalid_argument("Invalid refinement mode");
      }
      if (refinement_strategy !=
              FFTBoostedInstance::REFINEMENT_STRATEGY::PREVIOUS_DETECTION &&
          refinement_strategy !=
              FFTBoostedInstance::REFINEMENT_STRATEGY::LAST_ANCHOR) {
        throw std::invalid_argument("Invalid refinement strategy");
      }
    }

    std::string to_string() const {
      return fmt::format(
          "detection_score: {}, refinement_offset: {}, refinement_mode: {}, "
          "refinement_strategy: {}, optional_annotation_options: {}",
          detection_score, refinement_offset, static_cast<int>(refinement_mode),
          static_cast<int>(refinement_strategy),
          optional_annotation_options.has_value() ? "set" : "null");
    }
  };

protected:
  static cv::Point2f calculate_image_shift_fft(const cv::Mat &prev_image,
                            const cv::Mat &curr_image,
                            std::optional<double> scale = std::nullopt);
  const Options tracker_setting{};

  //  float detection_score = DEFAULT_DETECTION_SCORE;
  //  float refinement_offset = DEFAULT_REFINEMENT_OFFSET;
  //  cv::TemplateMatchModes refinement_mode = DEFAULT_REFINEMENT_MODE;
  //  FFTBoostedInstance::REFINEMENT_STRATEGY refinement_strategy =
  //      DEFAULT_REFINEMENT_STRATEGY;
  void loop_once(const cv::Mat &image,
                 std::vector<std::unique_ptr<FFTBoostedAnchor>> &&anchors);

  std::vector<cv::Mat> images;
  std::vector<std::unique_ptr<FFTBoostedInstance>> instances;
  std::unique_ptr<BS::thread_pool> refinement_pool;

public:
  FFTBoostedTracking() {
    refinement_pool =
        std::make_unique<BS::thread_pool>(tracker_setting.num_worker_threads);
  }
  explicit FFTBoostedTracking(Options &&options);
  virtual ~FFTBoostedTracking() = default;
  TrackerOutput track(const cv::Mat &image,
                      std::vector<std::unique_ptr<Anchor>> &&anchors);
};

} // namespace laser::tracking