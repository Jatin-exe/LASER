#pragma once
#include <cv_bridge/cv_bridge.h>
#include <future>
#include <beta_perception/msg/bounding_box.hpp>
#include <beta_tracking/msg/tracker_output.hpp>
#include <map>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace laser::tracking::utils {

class IndexedTable {
protected:
  // first index is rows, second index is columns
  // so each cell contains a value that is mapped to a row and column index
  std::map<size_t, std::map<size_t, double>> table;

public:
  struct Element {
  public:
    const size_t row, col;
    const double value;
    Element(size_t row, size_t col, double value)
        : row(row), col(col), value(value) {}
  };

  IndexedTable() = default;
  ~IndexedTable() = default;
  [[nodiscard]] bool empty() const { return table.empty(); }
  void insert(size_t row, size_t col, double value) { table[row][col] = value; }
  [[nodiscard]] Element get(size_t row, size_t col) const {
    if (table.find(row) == table.end() ||
        table.at(row).find(col) == table.at(row).end()) {
      throw std::out_of_range("Requested index is out of range.");
    }
    return {row, col, table.at(row).at(col)};
  }
  [[nodiscard]] Element max_element() const {
    double max_value = std::numeric_limits<double>::lowest();
    size_t max_row, max_col;
    for (const auto &row : table) {
      for (const auto &cell : row.second) {
        if (cell.second > max_value) {
          max_value = cell.second;
          max_row = row.first;
          max_col = cell.first;
        }
      }
    }
    return {max_row, max_col, max_value};
  }
  [[nodiscard]] Element min_element() const {
    double min_value = std::numeric_limits<double>::max();
    size_t min_row, min_col;
    for (const auto &row : table) {
      for (const auto &cell : row.second) {
        if (cell.second < min_value) {
          min_value = cell.second;
          min_row = row.first;
          min_col = cell.first;
        }
      }
    }
    return {min_row, min_col, min_value};
  }
  void remove_column_and_row(size_t col_index, size_t row_index) {
    // Remove the column from each row first
    for (auto &row : table) {
      row.second.erase(col_index);
    }
    // Then remove the row from the table
    table.erase(row_index);
  }

  [[nodiscard]] std::string to_string() const {
    std::stringstream ss;
    ss << "IndexedTable:\n";
    for (const auto &row : table) {
      for (const auto &cell : row.second) {
        ss << "(" << row.first << ", " << cell.first << ") -> " << cell.second
           << "\n";
      }
    }
    return ss.str();
  }
};


cv::Rect2i expand_rect(const cv::Rect2i &rect, double expansion_factor);

std::string rect_to_string(const cv::Rect2i &rect);

std::string cv_points_to_string(const std::vector<cv::Point2f> &points);

std::string cv_points_to_string(const std::vector<cv::Point2i> &points);

std::string cv_rect_to_string(const cv::Rect &bbox);

cv::Mat extract_sub_image_with_offset(const cv::Mat &image,
                                      const cv::Point &top_left,
                                      const cv::Point &bottom_right,
                                      float offset = 0.0);

cv::Point2f get_rectangle_center(const cv::Rect &bbox);
double get_rectangle_iou(const cv::Rect &rect1, const cv::Rect &rect2);

std::vector<cv::Point2i>
extract_keypoints_from_ros(const beta_perception::msg::BoundingBox &msg);

cv::Rect2f
extract_bounding_box_from_ros(const beta_perception::msg::BoundingBox &msg);

cv::Mat extract_image_from_ros(const sensor_msgs::msg::Image &msg);
cv::Mat extract_image_from_ros(
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg);

sensor_msgs::msg::Image
extact_image_from_cv(const cv::Mat &image,
                     std_msgs::msg::Header &&header = std_msgs::msg::Header());

template <class T = cv::Point>
geometry_msgs::msg::Point extract_point_from_cv(const T &point) {
  geometry_msgs::msg::Point point_msg;
  point_msg.x = static_cast<double>(point.x);
  point_msg.y = static_cast<double>(point.y);
  point_msg.z = 0;
  return point_msg;
}

template <typename TYPE_CLASS> class Action {
public:
  const TYPE_CLASS type;

  explicit Action(TYPE_CLASS type) : type(type) {}
  virtual ~Action() = default;
};

// Now for requests which have a promise as well
template <typename TYPE_CLASS, typename RETURN_CLASS>
class Request : public Action<TYPE_CLASS> {
private:
  std::promise<RETURN_CLASS> promise;

public:
  explicit Request(TYPE_CLASS type) : Action<TYPE_CLASS>(type) {}
  std::future<RETURN_CLASS> get_future() { return promise.get_future(); };
  void set_result(RETURN_CLASS result) { promise.set_value(result); };
  void set_exception(std::exception &e) {
    auto ptr = std::make_exception_ptr(e);
    promise.set_exception(ptr);
  };
  virtual ~Request() = default;
};
// make void specialization when return class is void
template <typename TYPE_CLASS>
class Request<TYPE_CLASS, void> : public Action<TYPE_CLASS> {
private:
  std::promise<void> promise;

public:
  explicit Request(TYPE_CLASS type) : Action<TYPE_CLASS>(type) {}
  std::future<void> get_future() { return promise.get_future(); };
  void set_result() { promise.set_value(); };
  void set_exception(std::exception &e) {
    auto ptr = std::make_exception_ptr(e);
    promise.set_exception(ptr);
  };
  virtual ~Request() = default;
};

}; // namespace laser::tracking::utils