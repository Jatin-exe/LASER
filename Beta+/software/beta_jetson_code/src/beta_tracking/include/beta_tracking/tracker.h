#pragma once

#include <beta_tracking/annotations.h>
#include <beta_tracking/path.h>
#include <beta_tracking/tracked_instance.h>

namespace laser::tracking {

    class TrackerOutput {
    protected:
        std::vector<std::unique_ptr<TrackedInstance>> tracked_instances;
        std::optional<cv::Mat> annotated_image;
    public:
        //Move constructor for TrackerOutput
        TrackerOutput(TrackerOutput &&other) noexcept: annotated_image(std::move(other.annotated_image)) {
            for (auto &&ti: other.tracked_instances) {
                tracked_instances.emplace_back(std::move(ti));
            }
        }

        explicit TrackerOutput(std::vector<std::unique_ptr<TrackedInstance>> &&ti,
                               const std::optional<cv::Mat> &ai = std::nullopt)
                : tracked_instances(std::move(ti)), annotated_image(ai) {}

        //getters
        [[nodiscard]] const std::vector<std::unique_ptr<TrackedInstance>> &get_tracked_instances() const {
            return tracked_instances;
        }

        [[nodiscard]] const std::optional<cv::Mat> &get_annotated_image() const {
            return annotated_image;
        }
    };

    class Tracker {
    protected:
        Tracker() = default;

    public:
        virtual ~Tracker() = default;

        virtual std::future<TrackerOutput>
        track(const cv::Mat &image,
              std::optional<std::vector<Anchor>> anchors = std::nullopt) = 0;
    };

} // namespace laser::tracking