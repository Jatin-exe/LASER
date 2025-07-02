#include "keyframe.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <random>

using std::placeholders::_1;

KeyframeSaver::KeyframeSaver(const rclcpp::NodeOptions & options)
: Node("keyframe_saver", options),
  orb_(cv::ORB::create()),
  matcher_(cv::NORM_HAMMING, true), // crossCheck = true
  inlier_threshold_(5),
  refinement_mode_(false)
{
    declare_parameter<std::string>("image_topic", "/vimbax_camera_beta/image_raw");
    declare_parameter<std::string>("save_directory", "/workspaces/isaac_ros-dev/src/laser_framework/data_sets/keyframes");  // full path from launch
    declare_parameter<int>("max_keyframes", 100);  // Default max: 100

    get_parameter("max_keyframes", max_keyframes_);

    std::string image_topic;
    get_parameter("image_topic", image_topic);
    get_parameter("save_directory", save_directory_);

    // Create target save directory if it doesn't exist
    std::filesystem::create_directories(save_directory_);
    RCLCPP_INFO(this->get_logger(), "Saving keyframes to: %s", save_directory_.c_str());

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, 10, std::bind(&KeyframeSaver::imageCallback, this, _1)
    );
}

void KeyframeSaver::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    cv::Mat current_image;
    try {
        current_image = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(current_image, cv::noArray(), keypoints, descriptors);

    if (descriptors.empty()) return;

    // First keyframe
    if (last_keyframe_.empty()) {
        last_keyframe_ = current_image.clone();
        last_keypoints_ = keypoints;
        last_descriptors_ = descriptors;
        return;
    }

    // Match descriptors with last keyframe
    std::vector<cv::DMatch> matches;
    matcher_.match(descriptors, last_descriptors_, matches);

    // Filter good matches
    int good_matches = 0;
    for (const auto & match : matches) {
        if (match.distance < 25)
            ++good_matches;
    }

    if (good_matches < inlier_threshold_) {
        last_keyframe_ = current_image.clone();
        last_keypoints_ = keypoints;
        last_descriptors_ = descriptors;

        // Determine dynamic save probability
        double dynamic_save_prob = (saved_keyframes_.size() < static_cast<size_t>(max_keyframes_)) ? 0.9 : 0.2;

        // Transition log
        if (!refinement_mode_ && saved_keyframes_.size() >= static_cast<size_t>(max_keyframes_)) {
            refinement_mode_ = true;
            RCLCPP_INFO(this->get_logger(), "Switched to refinement mode: save probability = %.2f", dynamic_save_prob);
        }

        // Random decision to save
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        double sample = prob_dist(rng);

        if (sample < dynamic_save_prob) {
            this->saveKeyframeImage(current_image, msg->header.stamp);
            RCLCPP_INFO(this->get_logger(), "Keyframe saved (random decision passed). Matches: %d", good_matches);
        } else {
            RCLCPP_INFO(this->get_logger(), "Keyframe skipped (random decision failed). Matches: %d", good_matches);
        }
    }
}

void KeyframeSaver::saveKeyframeImage(const cv::Mat& image, const rclcpp::Time& timestamp)
{
    std::stringstream filename_stream;
    filename_stream << save_directory_ << "/" 
                    << std::fixed << std::setprecision(6) 
                    << timestamp.seconds() << ".jpg";
    std::string filename = filename_stream.str();

    // If limit exceeded, remove a random file
    if (saved_keyframes_.size() >= static_cast<size_t>(max_keyframes_)) {
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<size_t> dist(0, saved_keyframes_.size() - 1);
        size_t index_to_remove = dist(rng);

        std::string file_to_remove = saved_keyframes_[index_to_remove];

        if (std::filesystem::remove(file_to_remove)) {
            RCLCPP_INFO(this->get_logger(), "Deleted random keyframe: %s", file_to_remove.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "Failed to delete keyframe: %s", file_to_remove.c_str());
        }

        saved_keyframes_.erase(saved_keyframes_.begin() + index_to_remove);
    }

    // Save the new keyframe
    if (cv::imwrite(filename, image)) {
        saved_keyframes_.push_back(filename);
        RCLCPP_INFO(this->get_logger(), "Saved keyframe: %s", filename.c_str());
    } else {
        RCLCPP_WARN(this->get_logger(), "Failed to save keyframe: %s", filename.c_str());
    }
}

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KeyframeSaver>();
    rclcpp::spin(node);
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}
