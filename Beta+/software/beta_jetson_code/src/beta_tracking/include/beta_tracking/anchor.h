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
#include <beta_tracking/annotations.h>
#include <utility>

namespace laser::tracking {

    class Anchor {
    protected:
        Anchor() = default;

    public:
        virtual ~Anchor() = default;

        [[nodiscard]] virtual cv::Rect get_bounding_box() const = 0;

        [[nodiscard]] virtual std::vector<cv::Point2i> get_keypoints() const = 0;

        [[nodiscard]] virtual std::vector<cv::Point2f> get_mask() const = 0;

        virtual const cv::Mat &get_source_image() const = 0;

        [[maybe_unused]] [[nodiscard]] virtual cv::Mat
        annotate_image(const cv::Mat &image,
                       std::optional<AnnotationOptions> annotation_options) const = 0;

        virtual std::string to_string() const = 0;

        static std::vector<std::unique_ptr<Anchor>>
        build(const beta_perception::msg::DetectionArray &msg);
    };


}; // namespace laser::tracking