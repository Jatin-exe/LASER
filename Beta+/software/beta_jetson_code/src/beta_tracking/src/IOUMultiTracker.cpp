#include <algorithm> // For std::remove
#include <cv_bridge/cv_bridge.h>
#include <stdexcept> // For std::runtime_error
#include <beta_tracking/logging.h>
#include <beta_tracking/multi_tracking.h>

namespace laser::tracking {

void IOUTracker::from_json_to_CSRT_params(const nlohmann::json &j,
                                          cv::TrackerCSRT::Params &p) {
  try {
    if (j.contains("use_hog") && j["use_hog"].is_boolean())
      p.use_hog = j["use_hog"];
    else
      logging::error("Error: 'use_hog' is missing or not a boolean");

    if (j.contains("use_color_names") && j["use_color_names"].is_boolean())
      p.use_color_names = j["use_color_names"];
    else
      logging::error("Error: 'use_color_names' is missing or not a boolean");

    if (j.contains("use_gray") && j["use_gray"].is_boolean())
      p.use_gray = j["use_gray"];
    else
      logging::error("Error: 'use_gray' is missing or not a boolean");

    if (j.contains("use_rgb") && j["use_rgb"].is_boolean())
      p.use_rgb = j["use_rgb"];
    else
      logging::error("Error: 'use_rgb' is missing or not a boolean");

    if (j.contains("use_channel_weights") &&
        j["use_channel_weights"].is_boolean())
      p.use_channel_weights = j["use_channel_weights"];
    else
      logging::error(
          "Error: 'use_channel_weights' is missing or not a boolean");

    if (j.contains("use_segmentation") && j["use_segmentation"].is_boolean())
      p.use_segmentation = j["use_segmentation"];
    else
      logging::error("Error: 'use_segmentation' is missing or not a boolean");

    if (j.contains("window_function") && j["window_function"].is_string())
      p.window_function = j["window_function"];
    else
      logging::error("Error: 'window_function' is missing or not a string");

    if (j.contains("kaiser_alpha") && j["kaiser_alpha"].is_number_float())
      p.kaiser_alpha = j["kaiser_alpha"];
    else
      logging::error("Error: 'kaiser_alpha' is missing or not a float");

    if (j.contains("cheb_attenuation") &&
        j["cheb_attenuation"].is_number_float())
      p.cheb_attenuation = j["cheb_attenuation"];
    else
      logging::error("Error: 'cheb_attenuation' is missing or not a float");

    if (j.contains("template_size") && j["template_size"].is_number_float())
      p.template_size = j["template_size"];
    else
      logging::error("Error: 'template_size' is missing or not a float");

    if (j.contains("gsl_sigma") && j["gsl_sigma"].is_number_float())
      p.gsl_sigma = j["gsl_sigma"];
    else
      logging::error("Error: 'gsl_sigma' is missing or not a float");

    if (j.contains("hog_orientations") &&
        j["hog_orientations"].is_number_float())
      p.hog_orientations = j["hog_orientations"];
    else
      logging::error("Error: 'hog_orientations' is missing or not a float");

    if (j.contains("hog_clip") && j["hog_clip"].is_number_float())
      p.hog_clip = j["hog_clip"];
    else
      logging::error("Error: 'hog_clip' is missing or not a float");

    if (j.contains("padding") && j["padding"].is_number_float())
      p.padding = j["padding"];
    else
      logging::error("Error: 'padding' is missing or not a float");

    if (j.contains("filter_lr") && j["filter_lr"].is_number_float())
      p.filter_lr = j["filter_lr"];
    else
      logging::error("Error: 'filter_lr' is missing or not a float");

    if (j.contains("weights_lr") && j["weights_lr"].is_number_float())
      p.weights_lr = j["weights_lr"];
    else
      logging::error("Error: 'weights_lr' is missing or not a float");

    if (j.contains("num_hog_channels_used") &&
        j["num_hog_channels_used"].is_number_integer())
      p.num_hog_channels_used = j["num_hog_channels_used"];
    else
      logging::error(
          "Error: 'num_hog_channels_used' is missing or not an integer");

    if (j.contains("admm_iterations") &&
        j["admm_iterations"].is_number_integer())
      p.admm_iterations = j["admm_iterations"];
    else
      logging::error("Error: 'admm_iterations' is missing or not an integer");

    if (j.contains("histogram_bins") && j["histogram_bins"].is_number_integer())
      p.histogram_bins = j["histogram_bins"];
    else
      logging::error("Error: 'histogram_bins' is missing or not an integer");

    if (j.contains("histogram_lr") && j["histogram_lr"].is_number_float())
      p.histogram_lr = j["histogram_lr"];
    else
      logging::error("Error: 'histogram_lr' is missing or not a float");

    if (j.contains("background_ratio") &&
        j["background_ratio"].is_number_integer())
      p.background_ratio = j["background_ratio"];
    else
      logging::error("Error: 'background_ratio' is missing or not an integer");

    if (j.contains("number_of_scales") &&
        j["number_of_scales"].is_number_integer())
      p.number_of_scales = j["number_of_scales"];
    else
      logging::error("Error: 'number_of_scales' is missing or not an integer");

    if (j.contains("scale_sigma_factor") &&
        j["scale_sigma_factor"].is_number_float())
      p.scale_sigma_factor = j["scale_sigma_factor"];
    else
      logging::error("Error: 'scale_sigma_factor' is missing or not a float");

    if (j.contains("scale_model_max_area") &&
        j["scale_model_max_area"].is_number_float())
      p.scale_model_max_area = j["scale_model_max_area"];
    else
      logging::error("Error: 'scale_model_max_area' is missing or not a float");

    if (j.contains("scale_lr") && j["scale_lr"].is_number_float())
      p.scale_lr = j["scale_lr"];
    else
      logging::error("Error: 'scale_lr' is missing or not a float");

    if (j.contains("scale_step") && j["scale_step"].is_number_float())
      p.scale_step = j["scale_step"];
    else
      logging::error("Error: 'scale_step' is missing or not a float");

    if (j.contains("psr_threshold") && j["psr_threshold"].is_number_float())
      p.psr_threshold = j["psr_threshold"];
    else
      logging::error("Error: 'psr_threshold' is missing or not a float");
  } catch (const std::exception &e) {
    throw e;
  }
}

void IOUTracker::from_json_to_KCF_params(const nlohmann::json &j,
                                         cv::TrackerKCF::Params &p) {
  try {
    if (j.contains("detect_thresh") && j["detect_thresh"].is_number_float())
      p.detect_thresh = j["detect_thresh"];
    else
      logging::error("Error: 'detect_thresh' is missing or not a float");

    if (j.contains("sigma") && j["sigma"].is_number_float())
      p.sigma = j["sigma"];
    else
      logging::error("Error: 'sigma' is missing or not a float");

    if (j.contains("lambda") && j["lambda"].is_number_float())
      p.lambda = j["lambda"];
    else
      logging::error("Error: 'lambda' is missing or not a float");

    if (j.contains("interp_factor") && j["interp_factor"].is_number_float())
      p.interp_factor = j["interp_factor"];
    else
      logging::error("Error: 'interp_factor' is missing or not a float");

    if (j.contains("output_sigma_factor") &&
        j["output_sigma_factor"].is_number_float())
      p.output_sigma_factor = j["output_sigma_factor"];
    else
      logging::error("Error: 'output_sigma_factor' is missing or not a float");

    if (j.contains("pca_learning_rate") &&
        j["pca_learning_rate"].is_number_float())
      p.pca_learning_rate = j["pca_learning_rate"];
    else
      logging::error("Error: 'pca_learning_rate' is missing or not a float");

    if (j.contains("resize") && j["resize"].is_boolean())
      p.resize = j["resize"];
    else
      logging::error("Error: 'resize' is missing or not a boolean");

    if (j.contains("max_patch_size") && j["max_patch_size"].is_number_integer())
      p.max_patch_size = j["max_patch_size"];
    else
      logging::error("Error: 'max_patch_size' is missing or not an integer");

    if (j.contains("desc_pca") && j["desc_pca"].is_string())
      p.desc_pca = j["desc_pca"];
    else
      logging::error("Error: 'desc_pca' is missing or not a string");

    if (j.contains("desc_npca") && j["desc_npca"].is_string())
      p.desc_npca = j["desc_npca"];
    else
      logging::error("Error: 'desc_npca' is missing or not a string");

    if (j.contains("compress_feature") && j["compress_feature"].is_boolean())
      p.compress_feature = j["compress_feature"];
    else
      logging::error("Error: 'compress_feature' is missing or not a boolean");

    if (j.contains("compressed_size") &&
        j["compressed_size"].is_number_integer())
      p.compressed_size = j["compressed_size"];
    else
      logging::error("Error: 'compressed_size' is missing or not an integer");
  } catch (const std::exception &e) {
    throw e;
  }
}

/**
 * @brief Factory method to build an IOUTracker with specific options.
 *
 * @param options The tracker options to configure the IOUTracker.
 * @return std::unique_ptr<Tracker> A unique pointer to the created IOUTracker.
 * @throws std::invalid_argument if the options cannot be downcasted to
 * IOUTrackerOptions.
 */
std::unique_ptr<Tracker>
IOUTracker::build(const Tracker::TrackerOptions &options) {
  // Downcast the tracker options to an IOUTrackerOptions
  logging::info("Building IOUTracker");
  const auto *iou_options = dynamic_cast<const IOUTrackerOptions *>(&options);
  if (!iou_options) {
    throw std::invalid_argument(
        "Invalid tracker options provided. Expected IOUTrackerOptions.");
  }
  // Check the range of min_overlap
  if (iou_options->min_overlap < 0.0 || iou_options->min_overlap > 1.0) {
    throw std::invalid_argument("Invalid min_overlap value provided. Expected "
                                "a value between 0.0 and 1.0.");
  }
  // Create and return the IOUTracker with the specified options
  // The default tracking algorithm is set to "default" (meaning no specific
  // algorithm is used) The default minimum overlap is set to 0.5 (meaning the
  // tracker will only track objects with an IoU greater than 0.5)
  auto iou_tracker = std::unique_ptr<IOUTracker>(new IOUTracker());
  iou_tracker->algorithm = iou_options->algorithm; // Set the tracking algorithm
  iou_tracker->min_overlap =
      iou_options->min_overlap; // Set the minimum overlap for IoU
  logging::info("IOUTracker built");
  return iou_tracker;
}

/**
 * @brief Reset the current state of the tracker.
 */
void IOUTracker::reset() {
  logging::info("Resetting IOUTracker");
  current_state = nullptr;
}

/**
 * @brief Retrieve the currently tracked instances.
 *
 * @return std::map<uint64_t, TrackedInstance> A map of tracked instances.
 */
std::map<uint64_t, TrackedInstance> IOUTracker::get_tracked_instances() const {
  if (has_state()) {
    return current_state->get_tracked_instances();
  } else {
    return {};
  }
}

/**
 * @brief Update the tracker with new input data.
 *
 * @param new_data The new input data to update the tracker with.
 * @throws std::invalid_argument if the new_data image is not provided.
 */
void IOUTracker::update(const InputRecord &new_data) {
  logging::debug("Beginning update with new data");
  // If there is no current state, create a new one
  if (!has_state()) {
    logging::debug("Creating new IOUTrackingState because it was empty");
    current_state = std::make_unique<IOUTrackingState>();
  }
  current_state->source_image = new_data.image();
  // Lets just feed forward the detections
  //  current_state->tracked_instances.clear();
  //  uint64_t id = 0;
  //  for(const auto &detection : new_data.detections()) {
  //    current_state->tracked_instances.emplace(id, TrackedInstance(detection,
  //    id)); id++;
  //  }

  std::vector<uint64_t> uids_to_remove = {}; // List of UIDs to remove

  // Update each tracker in the current state
  for (auto &tracker : current_state->trackers) {
    auto &tracker_algo = tracker.second;
    cv::Rect modified_roi;
    // Update the tracker algorithm with the new image
    bool target_locked =
        tracker_algo->update(current_state->source_image, modified_roi);

    // If the target was not locked, schedule it for removal
    if (!target_locked) {
      logging::debug(
          "Removing tracker with UID because target is not locked: {}",
          tracker.first);
      uids_to_remove.push_back(tracker.first);
    } else {
      // If the target is locked, update the detection ROI
      logging::debug("Updating tracker with UID: {}", tracker.first);
      current_state->update_instance_roi(Detection(modified_roi),
                                         tracker.first);
    }
  }
  //  // test with just one detection
  //  if (!new_data.detections().empty() && current_state->trackers.empty()) {
  //    Detection detection = new_data.detections().at(0);
  //    current_state->trackers.emplace(
  //        0, build_tracker_instance(detection, new_data.image()));
  //    current_state->tracked_instances.emplace(0, TrackedInstance(detection,
  //    0));
  //  }

  // Perform IoU on the detections and assign or create new trackers
  for (const auto &detection : new_data.detections()) {
    bool had_match = false; // Flag to check if a match was found

    // Check existing trackers for a match
    for (const auto &tracked_instance_entry :
         current_state->tracked_instances) {
      // Calculate the IoU of the new detection with the existing tracker
      double iou = utils::calculate_iou(
          detection.bounding_box,
          tracked_instance_entry.second.detection_roi.bounding_box);
      std::cout << "IOU: " << iou << std::endl;

      // If the IoU is greater than the minimum overlap, we have a match
      if (iou > min_overlap) {
        logging::debug("Matched tracker with UID: {}",
                       tracked_instance_entry.first);
        auto uid = tracked_instance_entry.first;
        current_state->trackers.erase(uid); // Remove the old tracker

        // Create a new tracker with the detection and initial state
        current_state->trackers.emplace(
            uid, build_tracker_instance(detection, new_data.image()));

        // Remove the UID from the removal list
        uids_to_remove.erase(
            std::remove(uids_to_remove.begin(), uids_to_remove.end(), uid),
            uids_to_remove.end());

        had_match = true;
        break;
      }
    }

    // If no match was found, create a new tracker for the detection
    if (!had_match) {
      auto uid = uid_counter++;
      logging::debug("Creating new tracker with UID: {}", uid);
      current_state->trackers.emplace(
          uid, build_tracker_instance(detection, new_data.image()));
      current_state->tracked_instances.emplace(uid,
                                               TrackedInstance(detection, uid));
    }
  }

  // Delete all tracks that require deletion
  for (const auto &uid : uids_to_remove) {
    logging::debug("Removing tracker with UID: {}", uid);
    current_state->trackers.erase(uid);
    current_state->tracked_instances.erase(uid);
  }
  logging::debug("Ending update with new data");
}

/**
 * @brief Build a tracker instance based on the specified algorithm.
 *
 * @param detection_roi The detection region of interest.
 * @param image The image to initialize the tracker with.
 * @return cv::Ptr<cv::Tracker> A pointer to the created tracker.
 * @throws std::runtime_error if the specified algorithm is unsupported.
 */
cv::Ptr<cv::Tracker> IOUTracker::build_tracker_instance(
    const Detection &detection_roi, const cv::Mat &image,
    std::optional<std::string> params_json) const {
  auto tracker = cv::Ptr<cv::Tracker>();
  switch (this->algorithm) {
  case Algorithm::KCF: {
    logging::info("Initializing KCF tracker");
    if (params_json.has_value()) {
      cv::TrackerKCF::Params kcf_params;
      from_json_to_KCF_params(params_json.value(), kcf_params);
      tracker = cv::TrackerKCF::create(
          kcf_params); // Create a KCF tracker with custom parameters
    } else {
      tracker = cv::TrackerKCF::create(); // Create a KCF tracker
    }
  } break;
  case Algorithm::CSRT: {
    logging::info("Initializing CSRT tracker");
    if (params_json.has_value()) {
      cv::TrackerCSRT::Params csrt_params;
      from_json_to_CSRT_params(params_json.value(), csrt_params);
      tracker = cv::TrackerCSRT::create(
          csrt_params); // Create a CSRT tracker with custom parameters
    } else {
      tracker = cv::TrackerCSRT::create(); // Create a CSRT tracker
    }
  } break;
  case Algorithm::GOTURN: {
    logging::info("Initializing GOTURN tracker");
    cv::TrackerGOTURN::Params goturn_params;
    goturn_params.modelBin = "/tmp/goturn.caffemodel";
    goturn_params.modelTxt = "/tmp/goturn.prototxt";
    tracker = cv::TrackerGOTURN::create(goturn_params); // Create a GOTURN tracker
  } break;
  default:
    throw std::runtime_error(
        "Unsupported tracker algorithm"); // Handle unsupported algorithms
  }
  tracker->init(image,
                detection_roi.bounding_box); // Initialize the tracker
  return tracker;
}

void IOUTracker::annotate_image(sensor_msgs::msg::Image &ros_image) const {
  auto cv_image = current_state->get_annotated_image();
  // Convert OpenCV image to ROS image
  cv_bridge::CvImagePtr cv_ptr = std::make_shared<cv_bridge::CvImage>(
      std_msgs::msg::Header(), "bgr8", cv_image);
  cv_ptr->toImageMsg(ros_image);
}

}; // namespace laser::tracking