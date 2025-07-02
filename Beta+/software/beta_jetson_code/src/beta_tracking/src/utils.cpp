#include "beta_tracking/utils.h"
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace laser::tracking::utils {

std::string cv_points_to_string(const std::vector<cv::Point2f> &points) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < points.size(); ++i) {
    oss << "(" << points[i].x << ", " << points[i].y << ")";
    if (i < points.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

std::string cv_points_to_string(const std::vector<cv::Point2i> &points) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < points.size(); ++i) {
    oss << "(" << points[i].x << ", " << points[i].y << ")";
    if (i < points.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

std::string cv_rect_to_string(const cv::Rect &bbox) {
  std::ostringstream oss;
  oss << "BoundingBox(x: " << bbox.x << ", y: " << bbox.y
      << ", width: " << bbox.width << ", height: " << bbox.height << ")";
  return oss.str();
}

cv::Mat extract_sub_image_with_offset(const cv::Mat &image,
                                      const cv::Point &top_left,
                                      const cv::Point &bottom_right,
                                      float offset) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty.");
  }

  if (offset < 0.0 || offset > 1.0) {
    throw std::out_of_range("Offset must be between 0.0 and 1.0.");
  }

  if (top_left.x < 0 || top_left.y < 0 || bottom_right.x < 0 ||
      bottom_right.y < 0) {
    throw std::out_of_range("Coordinates must be non-negative.");
  }
  if (offset == 0.0) {
    return image.clone();
  }

  int x1 = top_left.x;
  int y1 = top_left.y;
  int x2 = bottom_right.x;
  int y2 = bottom_right.y;

  int width = x2 - x1;
  int height = y2 - y1;

  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Invalid rectangle dimensions.");
  }

  int expand_x = static_cast<int>(width * offset / 2);
  int expand_y = static_cast<int>(height * offset / 2);

  if ((x1 - expand_x) < 0) {
    std::cerr << "Invalid offset for top left corner. Adjusting to 0."
              << std::endl;
  }
  if (y1 - expand_y < 0) {
    std::cerr << "Invalid offset for top left corner. Adjusting to 0."
              << std::endl;
  }

  x1 = std::max(0, x1 - expand_x);
  y1 = std::max(0, y1 - expand_y);

  if (x2 + expand_x > image.cols) {
    std::cerr << "Invalid offset for bottom right corner. Adjusting to the "
              << "maximum possible value." << std::endl;
  }
  if (y2 + expand_y > image.rows) {
    std::cerr << "Invalid offset for bottom right corner. Adjusting to the "
              << "maximum possible value." << std::endl;
  }
  x2 = std::min(image.cols, x2 + expand_x);
  y2 = std::min(image.rows, y2 + expand_y);
  try {
    return image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
  }catch (const cv::Exception &e) {
    std::stringstream ss;
    ss << "Error extracting sub-image: " << e.what() << std::endl;
    ss << "x1, y1, width, height: " << x1 << ", " << y1 << ", " << x2 - x1 << ", " << y2 - y1 << std::endl;
    throw std::runtime_error(ss.str());
  }
}

cv::Point2f get_rectangle_center(const cv::Rect &bbox) {
  return {static_cast<float>(bbox.x + bbox.width / 2.0f),
          static_cast<float>(bbox.y + bbox.height / 2.0f)};
}

std::vector<cv::Point2i>
extract_keypoints_from_ros(const beta_perception::msg::BoundingBox &msg) {
  cv::Point2f kp = {msg.keypoint.x, msg.keypoint.y};
  return {kp};
}

cv::Rect2i expand_rect(const cv::Rect2i &rect, double expansion_factor) {
  int width = static_cast<int>(rect.width * expansion_factor);
  int height = static_cast<int>(rect.height * expansion_factor);
  //  int x = rect.x - (width - rect.width / 2) / 2;
  //  int y = rect.y - (height - rect.height / 2) / 2;
  //make a new rectangle that has the same centroid as rect, but with the new height and width
  //what is the top left corner of the new rectangle
  auto dwidth = width - rect.width;
  auto dheight = height - rect.height;
  int x = rect.x - dwidth/2;
  int y = rect.y - dheight/2;
  return {x, y, width, height};
}

std::string rect_to_string(const cv::Rect2i &rect) {
  //To string the top left corner, the width and height amd the center
  std::ostringstream oss;
  oss << "BoundingBox(x: " << rect.x << ", y: " << rect.y
      << ", width: " << rect.width << ", height: " << rect.height
      << ", center: (" << rect.x + rect.width / 2 << ", "
      << rect.y + rect.height / 2 << "))";
  return oss.str();
}


cv::Rect2f
extract_bounding_box_from_ros(const beta_perception::msg::BoundingBox &msg) {
  return {static_cast<float>(msg.x), static_cast<float>(msg.y),
          static_cast<float>(msg.width), static_cast<float>(msg.height)};
}

double get_rectangle_iou(const cv::Rect &rect1, const cv::Rect &rect2) {
  int x_left = std::max(rect1.x, rect2.x);
  int y_top = std::max(rect1.y, rect2.y);
  int x_right = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
  int y_bottom = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

  if (x_right < x_left || y_bottom < y_top)
    return 1.0; // No overlap

  int intersection_area = (x_right - x_left) * (y_bottom - y_top);

  int rect1_area = rect1.width * rect1.height;
  int rect2_area = rect2.width * rect2.height;

  int union_area = rect1_area + rect2_area - intersection_area;

  double iou = static_cast<double>(intersection_area) / union_area;

  // Invert the result so 0 is perfect match and 1 is no overlap
  return 1.0 - iou;
}

cv::Mat extract_image_from_ros(const sensor_msgs::msg::Image &msg) {
  // use CV bridge to convert the image
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    throw std::runtime_error(std::string("cv_bridge exception: ") + e.what());
  }
  return cv_ptr->image;
}

cv::Mat extract_image_from_ros(
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg) {
  return extract_image_from_ros(*image_msg);
}

sensor_msgs::msg::Image extact_image_from_cv(const cv::Mat &image,
                                             std_msgs::msg::Header &&header) {
  // use cv bridge to convert the image from cv to sensor_msgs
  sensor_msgs::msg::Image image_msg;
  cv_bridge::CvImage cv_image(header, sensor_msgs::image_encodings::BGR8,
                              image);
  cv_image.toImageMsg(image_msg);
  return image_msg;
}

}; // namespace laser::tracking::utils