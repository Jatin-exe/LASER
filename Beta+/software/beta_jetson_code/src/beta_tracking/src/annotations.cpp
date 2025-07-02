#include "beta_tracking/annotations.h"

namespace laser::tracking {

AnnotationOptions::AnnotationOptions(const nlohmann::json& j)
{
    if (j.contains("feature_point_size")) {
        auto size = j["feature_point_size"].get<uint32_t>();
        if (size > 0 && size <= 100) {
            feature_point_size = size;
        } else {
            throw std::out_of_range("feature_point_size must be between 1 and 100");
        }
    }

    if (j.contains("feature_point_color")) {
        auto color = j["feature_point_color"];
        if (color.is_array() && color.size() == 3) {
            for (int i = 0; i < 3; ++i) {
                int value = color[i].get<int>();
                if (value < 0 || value > 255) {
                    throw std::out_of_range("Color values must be between 0 and 255");
                }
            }
            feature_point_color = cv::Scalar(color[0], color[1], color[2]);
        } else {
            throw std::invalid_argument("feature_point_color must be an array of 3 integers");
        }
    }

    if (j.contains("bounding_box_color")) {
        auto color = j["bounding_box_color"];
        if (color.is_array() && color.size() == 3) {
            for (int i = 0; i < 3; ++i) {
                int value = color[i].get<int>();
                if (value < 0 || value > 255) {
                    throw std::out_of_range("Color values must be between 0 and 255");
                }
            }
            bounding_box_color = cv::Scalar(color[0], color[1], color[2]);
        } else {
            throw std::invalid_argument("bounding_box_color must be an array of 3 integers");
        }
    }

    if (j.contains("bounding_box_thickness")) {
        auto thickness = j["bounding_box_thickness"].get<double>();
        if (thickness > 0 && thickness <= 100) {
            bounding_box_thickness = thickness;
        } else {
            throw std::out_of_range("bounding_box_thickness must be between 0 and 100");
        }
    }

    if (j.contains("id_font_size")) {
        auto size = j["id_font_size"].get<double>();
        if (size > 0 && size <= 100) {
            id_font_size = size;
        } else {
            throw std::out_of_range("id_font_size must be between 0 and 100");
        }
    }

    if (j.contains("id_color")) {
        auto color = j["id_color"];
        if (color.is_array() && color.size() == 3) {
            for (int i = 0; i < 3; ++i) {
                int value = color[i].get<int>();
                if (value < 0 || value > 255) {
                    throw std::out_of_range("Color values must be between 0 and 255");
                }
            }
            id_color = cv::Scalar(color[0], color[1], color[2]);
        } else {
            throw std::invalid_argument("id_color must be an array of 3 integers");
        }
    }
}

std::string AnnotationOptions::to_string() const {
    nlohmann::json j;
    j["feature_point_size"] = feature_point_size;
    j["feature_point_color"] = {feature_point_color[0], feature_point_color[1], feature_point_color[2]};
    j["bounding_box_color"] = {bounding_box_color[0], bounding_box_color[1], bounding_box_color[2]};
    j["bounding_box_thickness"] = bounding_box_thickness;
    j["id_font_size"] = id_font_size;
    j["id_color"] = {id_color[0], id_color[1], id_color[2]};
    j["id_position"] = static_cast<int>(id_position);
    return j.dump(4);
}

}; // namespace laser::tracking