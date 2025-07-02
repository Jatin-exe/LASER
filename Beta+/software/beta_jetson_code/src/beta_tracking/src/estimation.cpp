#include <beta_tracking/multi_tracking.h>

namespace laser::tracking::estimation {
cv::Point2f estimate_pairwise_shift(const cv::Mat &prev_frame,
                                    const cv::Mat &curr_frame) {
  // Convert frames to grayscale
  cv::Mat prev_gray, curr_gray;
  cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

  // Parameters for Shi-Tomasi corner detection
  std::vector<cv::Point2f> prev_points;
  cv::goodFeaturesToTrack(prev_gray, prev_points, 100, 0.01, 10);

  // Calculate optical flow using Lucas-Kanade method
  std::vector<cv::Point2f> curr_points;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, curr_points,
                           status, err);

  // Calculate the average shift
  cv::Point2f shift(0, 0);
  int count = 0;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i]) {
      shift += curr_points[i] - prev_points[i];
      ++count;
    }
  }
  if (count > 0) {
    shift.x /= count;
    shift.y /= count;
  }
  return shift;
}

} // namespace laser::tracking::estimation