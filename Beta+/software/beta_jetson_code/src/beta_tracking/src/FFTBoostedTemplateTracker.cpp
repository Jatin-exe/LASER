#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <beta_tracking/FFTBoostedTemplateTracker.h>
#include <beta_tracking/anchor.h>
#include <beta_tracking/logging.h>
#include <beta_tracking/path.h>
#include <beta_tracking/tracked_instance.h>
#include <utility>
#include <vector>

namespace laser::tracking {

uint64_t FFTBoostedTrackingID::id_counter;

cv::Point2f
FFTBoostedTracking::calculate_image_shift_fft(const cv::Mat &prev_image,
                                              const cv::Mat &curr_image,
                                              std::optional<double> scale) {
  cv::Mat prev_gray, curr_gray, prev_float, curr_float;
  // make sure prev image is not empty throw otherwsie
  if (prev_image.empty()) {
    throw std::invalid_argument("Previous image arg is empty.");
  }
  if (curr_image.empty()) {
    throw std::invalid_argument("Current image arg is empty.");
  }
  // If we are recaling the images then we need to create copies of the given
  // images, otherwise don't to save time
  cv::Mat im1, im2;
  if (scale.has_value()) {
    logging::trace("Scale has a value of {}", scale.value());
    // throw if scale is equal or less than 0 or 1 or greater
    if (scale.value() <= 0 || scale.value() > 1) {
      throw std::invalid_argument("Scale must be between 0 and 1.");
    }
    cv::resize(prev_image, im1, cv::Size(), scale.value(), scale.value());
    cv::resize(curr_image, im2, cv::Size(), scale.value(), scale.value());
  } else {
    im1 = prev_image;
    im2 = curr_image;
  }
  // Calculate the shift using the maximum correlation value and its position
  cv::cvtColor(im1, prev_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(im2, curr_gray, cv::COLOR_BGR2GRAY);
  // Convert to floating point
  prev_gray.convertTo(prev_float, CV_32F);
  curr_gray.convertTo(curr_float, CV_32F);

  cv::Point2d shift = cv::phaseCorrelate(prev_float, curr_float);
  // If scale had a value then we need to convert the shift back to the
  // coordinate scale of the original images
  if (scale.has_value()) {
    shift.x *= im1.cols / (scale.value() * im1.cols);
    shift.y *= im1.rows / (scale.value() * im1.rows);
  }
  return {static_cast<float>(shift.x), static_cast<float>(shift.y)};
}

TrackerOutput
FFTBoostedTracking::track(const cv::Mat &image,
                          std::vector<std::unique_ptr<Anchor>> &&anchors) {
  // Take the moved in anchors and convert them to FFTBoosted anchors
  logging::debug("Doing tracking with number of anchors, {}", anchors.size());
  std::vector<std::unique_ptr<FFTBoostedAnchor>> fft_anchors;
  for (auto &&anchor : anchors) {
    // Transfer ownership to the FFTBoostedAnchor
    auto ptr = dynamic_cast<FFTBoostedAnchor *>(anchor.release());
    auto fft_anchor = std::unique_ptr<FFTBoostedAnchor>(ptr);
    fft_anchors.emplace_back(std::move(fft_anchor));
    logging::trace("Converted anchor to FFTBoostedAnchor");
  }
  logging::debug(
      "Converted anchors to FFTBoosted anchors, now calling loop once");
  loop_once(image, std::move(fft_anchors));
  logging::debug("Loop once completed");
  std::vector<std::unique_ptr<TrackedInstance>> tracked_instances;
  cv::Mat annotated_image;
  if (tracker_setting.optional_annotation_options.has_value()) {
    logging::debug("Annotating image with annotations");
    annotated_image = image.clone();
  }
  // Populating output
  logging::debug("Populating output with {} tracked instances",
                 this->instances.size());
  for (auto &instance : this->instances) {
    auto cloned_instance = FFTBoostedInstance::create(*instance);
    if (tracker_setting.optional_annotation_options.has_value()) {
      logging::debug("Annotating image with detection info");
      logging::debug("Annotation box is at ({}, {}) and ({}, {})",
                     cloned_instance->get_bounding_box().tl().x,
                     cloned_instance->get_bounding_box().tl().y,
                     cloned_instance->get_bounding_box().br().x,
                     cloned_instance->get_bounding_box().br().y);
      annotated_image = cloned_instance->annotate_image(
          annotated_image, tracker_setting.optional_annotation_options.value());
    }
    tracked_instances.emplace_back(std::move(cloned_instance));
  }
  if (tracker_setting.optional_annotation_options.has_value() &&
      !annotated_image.empty()) {
    logging::debug("Returning annotated output");
    auto output =
        TrackerOutput(std::move(tracked_instances), std::move(annotated_image));
    return output;
  } else {
    logging::debug("Returning unannotated output");
    auto output = TrackerOutput(std::move(tracked_instances));
    return output;
  }
}

void FFTBoostedTracking::loop_once(
    const cv::Mat &image,
    std::vector<std::unique_ptr<FFTBoostedAnchor>> &&anchors) {
  logging::debug("Doing loop once with {} anchors", anchors.size());
  std::vector<FFTBoostedTrackingID> ids_to_remove;
  std::vector<bool> instances_to_keep(instances.size(), true);
  // check that there is something in the image, i.e. not an empty image
  if (image.empty()) {
    logging::warn("Image was empty so no tracking to do");
    return;
  } else {
    logging::debug("Input image is {}x{}x{}", image.cols, image.rows,
                   image.channels());
  }
  // Check if we have had a prior image
  if (this->images.empty()) {
    // If not add the image to the record of images
    logging::debug("Images were empty so this is the first image");
    images.push_back(image);
  } else {
    // Otherwise we can perform FFT shift and update tracks accordingly
    auto prev_image = images.back();
    images.push_back(image);
    auto shift = calculate_image_shift_fft(
        prev_image, image, tracker_setting.fft_downscale_factor);
    logging::debug("Calculated image shift via FFT: ({}, {})", shift.x,
                   shift.y);
    logging::debug("Doing refinement with {} threads", refinement_pool->get_thread_count());
    auto num_instances = instances.size();
    refinement_pool->detach_loop<size_t>(0, num_instances, [&](size_t i){
      auto& instance = instances.at(i);
      instance->translate(shift);
      if (instance->get_bounding_box().x < 0 ||
          instance->get_bounding_box().y < 0) {
        // remove the instance from the vector
        logging::debug("Removing instance with id: {}",
                       instance->get_tracking_id().to_string());
        instances_to_keep[i] = false;
      } else {
        // Otherwise do refinement
        instance->refine(image, tracker_setting.refinement_offset,
                         tracker_setting.refinement_strategy,
                         tracker_setting.refinement_mode);
      }
    });
    logging::debug("Submitted tasks...waiting for results...");
    refinement_pool->wait();
    logging::debug("Finished waiting for results");
  }
  std::vector<std::unique_ptr<FFTBoostedInstance>> instances_to_keep_ptrs;
  for(uint32_t i = 0; i < instances.size(); ++i) {
    if(instances_to_keep[i]) {
      instances_to_keep_ptrs.push_back(std::move(instances[i]));
    }
  }
  instances = std::move(instances_to_keep_ptrs);
  logging::debug("Finished processing erasure, kept {} instances", instances.size());
  // Now do detection assignment
  if (!anchors.empty()) {
    auto &anchors_to_process = anchors;
    // the indicies of anchors that are new instances
    std::vector<size_t> unique_anchors;
    // everything is unique before we assign
    for (size_t i = 0; i < anchors_to_process.size(); ++i) {
      unique_anchors.push_back(i);
    }
    logging::debug("Number of unique anchors: {}", unique_anchors.size());
    // if we have instances, then we try and assign
    if (!instances.empty()) {
      logging::debug("Assigning anchors to instances");
      // Create a table of scores between instances and anchors
      auto score_match_table = utils::IndexedTable();
      for (uint32_t instance_index = 0; instance_index < instances.size();
           ++instance_index) {
        for (uint32_t anchor_index = 0;
             anchor_index < anchors_to_process.size(); anchor_index++) {
          double score = instances.at(instance_index)
                             ->generate_matching_score(
                                 *anchors_to_process.at(anchor_index));
          score_match_table.insert(instance_index, anchor_index, score);
        }
      }
      logging::debug("Generated score match table");
      // now do assignment based on the scores, starting from the
      while (!score_match_table.empty()) {
        auto min_entry = score_match_table.min_element();
        logging::trace("Match Table Min score: {}", min_entry.value);
        if (min_entry.value > tracker_setting.detection_score) {
          // scores are above threshold, so we are done assigning
          logging::debug("Breaking out of assignment loop because min score is "
                         "above threshold: {} > {}",
                         min_entry.value, tracker_setting.detection_score);
          break;
        }
        // otherwise we do have a match, update the instance.
        instances.at(min_entry.row)
            ->assign_anchor(*anchors_to_process.at(min_entry.col));
        logging::debug(
            "Assigned anchor to instance with id: {}",
            instances.at(min_entry.row)->get_tracking_id().to_string());
        // remove the indices from the table
        score_match_table.remove_column_and_row(min_entry.col, min_entry.row);
        logging::debug("Removed indices from score match table");
        // remove the anchor from the vector of unique anchors
        unique_anchors.erase(std::remove(unique_anchors.begin(),
                                         unique_anchors.end(), min_entry.col),
                             unique_anchors.end());
        logging::debug("Removed anchor from unique anchors");
      }
    }
    // Any unique anchors are new and need to be added as new instances
    logging::debug("Adding new {} instances for unique anchors",
                   unique_anchors.size());
    for (size_t anchor_index : unique_anchors) {
      auto &a = anchors_to_process.at(anchor_index);
      auto new_instance = FFTBoostedInstance::create(*a);
      logging::debug("Created new instance with id: {}",
                     new_instance->get_tracking_id().to_string());
      // check that the id created is truly unique
      for (const auto &instance : instances) {
        if (instance->get_tracking_id() == new_instance->get_tracking_id()) {
          throw std::runtime_error("Tracking id is not unique");
        }
      }
      instances.emplace_back(std::move(new_instance));
    }
  }
}
FFTBoostedTracking::FFTBoostedTracking(FFTBoostedTracking::Options &&options)
    : tracker_setting(std::move(options)) {
  refinement_pool =
      std::make_unique<BS::thread_pool>(tracker_setting.num_worker_threads);
}

void FFTBoostedInstance::refine(const cv::Mat &new_image, float offset,
                                REFINEMENT_STRATEGY strategy,
                                cv::TemplateMatchModes mode) {
  // This is the image that will be searched
  logging::trace("Refining instance with id: {}",
                 get_tracking_id().to_string());
  logging::trace("Using offset: {}", offset);
  logging::trace("Input image size: {}x{}", new_image.cols, new_image.rows);
  auto bb_center = get_bounding_box().tl() +
                   (get_bounding_box().br() - get_bounding_box().tl()) / 2;
  logging::trace("TL of bounding box: ({}, {})", get_bounding_box().tl().x,
                 get_bounding_box().tl().y);
  logging::trace("Old width: {}, old height: {}", get_bounding_box().width,
                 get_bounding_box().height);
  auto new_bbox = utils::expand_rect(get_bounding_box(), offset + 1.0f);
  logging::trace("New width: {}, new height: {}", new_bbox.width,
                 new_bbox.height);
  auto new_bb_center = new_bbox.tl() + (new_bbox.br() - new_bbox.tl()) / 2;
  // print the center of the new bounding box
  logging::trace("New TL of bounding box: ({}, {})", new_bbox.tl().x,
                 new_bbox.tl().y);


  //Constrain the new bounding box within new_image dimensions
  new_bbox.x = std::max(0, std::min(new_bbox.x, new_image.cols - 1));
  new_bbox.y = std::max(0, std::min(new_bbox.y, new_image.rows - 1));
  new_bbox.width = std::min(new_bbox.width, new_image.cols - new_bbox.x);
  new_bbox.height = std::min(new_bbox.height, new_image.rows - new_bbox.y);
  logging::trace("Constrained new bounding box: ({}, {})", new_bbox.tl().x,
                 new_bbox.tl().y);

  // Extract the sub image from the new bounding box
  logging::trace("Extracting sub image from new bounding box");
  auto search_image = new_image(new_bbox);
  cv::Mat match_result;
  cv::Point min_loc, max_loc;
  double min_val, max_val;
  cv::Mat dynamic_template_image;

  if (strategy == REFINEMENT_STRATEGY::LAST_ANCHOR) {
    // We are going to use the last anchor as a template for refinement
    // Get the sub image from the last anchors source image and the bounding
    // box
    logging::trace("Using last anchor as template for refinement");
    dynamic_template_image =
        last_anchor->source_image(last_anchor->get_bounding_box());
  } else if (strategy == REFINEMENT_STRATEGY::PREVIOUS_DETECTION) {
    logging::trace("Using previous detection as template for refinement");
    dynamic_template_image = template_image(template_roi);
  } else {
    throw std::invalid_argument("Invalid refinement strategy");
  }
  // Perform refinement using the OpenCV template matching algorithm
  // Apply template matching
  try {
    logging::trace("Applying template matching to search image of size: {}x{}",
                   search_image.cols, search_image.rows);
    int result_cols = search_image.cols - dynamic_template_image.cols + 1;
    int result_rows = search_image.rows - dynamic_template_image.rows + 1;
    if (result_cols < 1 || result_rows < 1) {
      logging::error("Template image is larger than search image");
      logging::error("Template image size is {}x{}",
                     dynamic_template_image.cols, dynamic_template_image.rows);

      //throw std::runtime_error("Template image is larger than search image");
    }
    match_result.create(result_rows, result_cols, CV_32FC1);
    logging::trace("Created match result of size: {}x{}", match_result.cols,
                   match_result.rows);
    cv::matchTemplate(search_image, dynamic_template_image, match_result, mode);
  } catch (const cv::Exception &e) {
    logging::error("OpenCV error occurred during template matching: {}",
                   e.what());
    // Now print image sizes
    logging::error("Search image size: {}x{}", search_image.cols,
                   search_image.rows);
    logging::error("Template image size: {}x{}", dynamic_template_image.cols,
                   dynamic_template_image.rows);
    //throw std::runtime_error("Failed to perform template matching");
  }

  // Find the maximum match value
  logging::trace("Match result size: {}x{}", match_result.cols,
                 match_result.rows);
  cv::normalize(match_result, match_result, 0, 1, cv::NORM_MINMAX, -1);
  cv::minMaxLoc(match_result, &min_val, &max_val, &min_loc, &max_loc);

  cv::Point2i match_loc;
  switch (mode) {
  case cv::TM_SQDIFF:
  case cv::TM_SQDIFF_NORMED:
    match_loc = min_loc;
  default:
    match_loc = max_loc;
  }
  logging::trace("Match location: ({}, {})", match_loc.x, match_loc.y);
  // The match location is the point in the source image where the match occured
  // But we need convert it to the coordinates of new_image
  // the origin is the 0,0 point in the search image frame, so its actually the
  // search image source ROI in the new_image frame
  cv::Point2i origin = new_bbox.tl();
  logging::trace("Origin: ({}, {})", origin.x, origin.y);
  // The refined top left is translated into the new_image frame
  cv::Point2i refined_tl = origin + match_loc;
  logging::trace("Refined TL: ({}, {})", refined_tl.x, refined_tl.y);
  // Then to calculate how the features change, we find the shift in the
  // new_image frame
  cv::Point2i original_tl = this->get_bounding_box().tl();
  logging::trace("Original TL: ({}, {})", original_tl.x, original_tl.y);
  cv::Point2f shift = {static_cast<float>(refined_tl.x - original_tl.x),
                       static_cast<float>(refined_tl.y - original_tl.y)};
  logging::trace("Shift: ({}, {})", shift.x, shift.y);
  // Update the feature set
  this->features->translate(shift);
  // Update the image
  this->template_image = new_image;
  // update the ROI
  this->template_roi = this->features->bounding_box;
}

double FFTBoostedInstance::generate_matching_score(
    const FFTBoostedAnchor &anchor) const {
  //  double dist =
  //      cv::norm(utils::get_rectangle_center(anchor.get_bounding_box()) -
  //               utils::get_rectangle_center(get_bounding_box()));
  //  return dist;
  return utils::get_rectangle_iou(anchor.get_bounding_box(),
                                  get_bounding_box());
}

std::unique_ptr<FFTBoostedInstance>
FFTBoostedInstance::create(const FFTBoostedAnchor &anchor) {
  auto new_instance =
      std::unique_ptr<FFTBoostedInstance>(new FFTBoostedInstance());
  new_instance->template_image = anchor.source_image;
  new_instance->template_roi = anchor.get_bounding_box();
  auto features = std::make_unique<FFTFeatures>(anchor.features);
  new_instance->features = std::move(features);
  new_instance->id =
      std::make_unique<FFTBoostedTrackingID>(FFTBoostedTrackingID::create());
  new_instance->last_anchor = std::make_unique<FFTBoostedAnchor>(anchor);
  return new_instance;
}

cv::Mat FFTBoostedInstance::annotate_image(
    const cv::Mat &image,
    std::optional<AnnotationOptions> annotation_options) const {
  auto new_image = this->features->annotate_image(image, annotation_options);
  // Now add the ID to the top left corner
  std::stringstream ss;
  ss << this->id->to_string();
  cv::putText(new_image, ss.str(), this->features->bounding_box.tl(),
              cv::FONT_HERSHEY_PLAIN, annotation_options->id_font_size,
              annotation_options->id_color, 10);
  return new_image;
}

std::unique_ptr<FFTBoostedInstance>
FFTBoostedInstance::create(const FFTBoostedInstance &other) {
  auto new_instance =
      std::unique_ptr<FFTBoostedInstance>(new FFTBoostedInstance());
  new_instance->template_image = other.template_image;
  new_instance->template_roi = other.template_roi;
  auto features = std::make_unique<FFTFeatures>(*other.features);
  new_instance->features = std::move(features);
  new_instance->id = std::make_unique<FFTBoostedTrackingID>(*other.id);
  new_instance->last_anchor =
      std::make_unique<FFTBoostedAnchor>(*other.last_anchor);
  return new_instance;
}

cv::Mat FFTBoostedAnchor::annotate_image(
    const cv::Mat &image,
    std::optional<AnnotationOptions> annotation_options) const {
  auto annotated_image =
      this->features.annotate_image(image, annotation_options);
  return annotated_image;
}

std::unique_ptr<Anchor>
FFTBoostedAnchor::build(const beta_perception::msg::BoundingBox &msg,
                        const sensor_msgs::msg::Image &img) {
  class FFTBoostedAnchorBuilder : public FFTBoostedAnchor {
  public:
    FFTBoostedAnchorBuilder(cv::Rect2f bbox,
                            const std::vector<cv::Point2i> &keypoints,
                            const cv::Mat &image)
        : FFTBoostedAnchor(std::move(bbox), keypoints, image) {}
  };
  auto anchor = std::make_unique<FFTBoostedAnchorBuilder>(
      utils::extract_bounding_box_from_ros(msg),
      utils::extract_keypoints_from_ros(msg),
      utils::extract_image_from_ros(img));
  return anchor;
}

std::vector<std::unique_ptr<Anchor>>
FFTBoostedAnchor::build(const beta_perception::msg::DetectionArray &msg,
                        const sensor_msgs::msg::Image &img) {
  std::vector<std::unique_ptr<Anchor>> anchors;
  for (const auto &detection : msg.boxes) {
    anchors.push_back(build(detection, img));
  }
  return anchors;
}

std::unique_ptr<Anchor> FFTBoostedAnchor::build(const cv::Mat &image,
                                                const FFTFeatures &features) {
  return std::unique_ptr<Anchor>(
      new FFTBoostedAnchor(features.bounding_box, features.keypoints, image));
}

cv::Mat FFTFeatures::annotate_image(
    const cv::Mat &image,
    std::optional<AnnotationOptions> annotation_options) const {
  auto annotated_image = image.clone();
  // Draw the keypoints, make sure to use the options in annotation options.
  // the keypoint is a dot
  for (const auto &kp : keypoints) {
    cv::circle(annotated_image, kp, 3, annotation_options->feature_point_color,
               -1);
  }
  // draw the bounding box
  cv::rectangle(annotated_image, bounding_box,
                annotation_options->bounding_box_color, 2);

  return annotated_image;
}

FFTBoostedTrackingID FFTBoostedTrackingID::create(const TrackingID &other) {
  auto &casted_id = dynamic_cast<const FFTBoostedTrackingID &>(other);
  auto boosted_id = FFTBoostedTrackingID();
  boosted_id.id = casted_id.id;
  return boosted_id;
}

FFTBoostedTrackingID FFTBoostedTrackingID::create() {
  auto boosted_id = FFTBoostedTrackingID();
  boosted_id.id = FFTBoostedTrackingID::id_counter;
  FFTBoostedTrackingID::id_counter++;
  return boosted_id;
}
} // namespace laser::tracking
