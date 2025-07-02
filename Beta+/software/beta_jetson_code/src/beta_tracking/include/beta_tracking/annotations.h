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
#include <utility>

namespace laser::tracking {

// Default values
constexpr double DEFAULT_ID_FONT_SIZE = 5.0;
constexpr std::array<int, 3> DEFAULT_ID_COLOR_ARRAY = {255, 0, 0};  // White
constexpr std::array<int, 3> DEFAULT_BOX_COLOR_ARRAY = {255, 0, 0}; // White
constexpr std::array<int, 3> DEFAULT_FEATUREPOINT_COLOR_ARRAY = {255, 0, 0}; // White
constexpr uint32_t DEFAULT_FEATURE_POINT_SIZE = 10;
constexpr double DEFAULT_BOX_THICKNESS = 10;

class AnnotationOptions {
public:
    uint32_t feature_point_size = DEFAULT_FEATURE_POINT_SIZE;
    cv::Scalar feature_point_color = cv::Scalar(DEFAULT_FEATUREPOINT_COLOR_ARRAY[0], DEFAULT_FEATUREPOINT_COLOR_ARRAY[1], DEFAULT_FEATUREPOINT_COLOR_ARRAY[2]);
    cv::Scalar bounding_box_color = cv::Scalar(DEFAULT_BOX_COLOR_ARRAY[0], DEFAULT_BOX_COLOR_ARRAY[1], DEFAULT_BOX_COLOR_ARRAY[2]);
    double bounding_box_thickness = DEFAULT_BOX_THICKNESS;
    double id_font_size = DEFAULT_ID_FONT_SIZE;
    cv::Scalar id_color = cv::Scalar(DEFAULT_ID_COLOR_ARRAY[0], DEFAULT_ID_COLOR_ARRAY[1], DEFAULT_ID_COLOR_ARRAY[2]);

    enum class IDPosition {
        TOP_LEFT,
        BOTTOM_RIGHT,
        CENTER,
        BOTTOM_CENTER,
        TOP_CENTER,
        TOP_RIGHT,
        BOTTOM_LEFT,
    };

    IDPosition id_position = IDPosition::TOP_CENTER;

    AnnotationOptions() = default;
    AnnotationOptions(const AnnotationOptions& other) = default;
    explicit AnnotationOptions(const nlohmann::json& j);

    virtual ~AnnotationOptions() = default;

    [[nodiscard]] std::string to_string() const;
};

}; // namespace laser::tracking