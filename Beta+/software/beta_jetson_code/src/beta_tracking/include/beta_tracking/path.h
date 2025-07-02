#pragma once
#include "beta_tracking/tracked_instance.h"

namespace laser::tracking {

class Path {};

class SimpleCostedPath {
protected:
  // the list of Instances to hit
  std::vector<std::unique_ptr<TrackedInstance>> instance_list;
  [[nodiscard]] static double
  calculate_pritority(const TrackedInstance *instance,
                      std::optional<const TrackedInstance *> last_instance) {
    // Piority is maximum at 0, the priority is calculated based on the
    // instances target location to the left most side of an image AND its
    // closeness to the first value in the instance_list
    //  Lower values in the instance_list mean higher priority
    double left_side_distance = instance->get_target_coord().x;
    if (!last_instance) {
      return left_side_distance;
    }
    double first_instance_distance =
        sqrt(pow((last_instance.value()->get_target_coord().x -
                  instance->get_target_coord().x),
                 2) +
             pow((last_instance.value()->get_target_coord().y -
                  instance->get_target_coord().y),
                 2));
    return std::min(left_side_distance, first_instance_distance);
  }
  static void sort_instances(
      std::vector<std::unique_ptr<TrackedInstance>> &instances,
      std::optional<const TrackedInstance *> last_instance = std::nullopt) {
    std::sort(instances.begin(), instances.end(),
              [last_instance](const std::unique_ptr<TrackedInstance> &a,
                              const std::unique_ptr<TrackedInstance> &b) {
                return calculate_pritority(a.get(), last_instance) <
                       calculate_pritority(b.get(), last_instance);
              });
  }

  void remove_old_instances(
      const std::vector<std::unique_ptr<TrackedInstance>> &updated_instances) {
    std::vector<size_t> indices_to_delete;
    for (size_t i = 0; i < instance_list.size(); ++i) {
      bool found = false;
      for (const auto &updated_instance : updated_instances) {
        if (instance_list[i]->get_tracking_id() ==
            updated_instance->get_tracking_id()) {
          found = true;
          break;
        }
      }
      if (!found) {
        indices_to_delete.push_back(i);
      }
    }
    for (size_t i = indices_to_delete.size(); i > 0; --i) {
      instance_list.erase(instance_list.begin() + indices_to_delete[i - 1]);
    }
  }

  void add_new_instances(
      const std::vector<std::unique_ptr<TrackedInstance>> &updated_instances) {
    decltype(instance_list) instances_to_add;
    for (const auto &instance : updated_instances) {
      bool found = false;
      for (auto &existing_instance : instance_list) {
        if (instance->get_tracking_id() ==
            existing_instance->get_tracking_id()) {
          found = true;
          //Make sure to update the instance with the new data
          existing_instance->update(*instance);
          break;
        }
      }
      if (!found) {
        instances_to_add.emplace_back(instance->clone());
      }
    }
    // Now we need to first sort this list
    auto last_instance = instance_list.empty()
                             ? std::nullopt
                             : std::make_optional(instance_list.back().get());
    sort_instances(instances_to_add, last_instance);
    // now append these new instances to the existing list
    instance_list.insert(instance_list.end(),
                         std::make_move_iterator(instances_to_add.begin()),
                         std::make_move_iterator(instances_to_add.end()));
    // and we are done
  }

public:
  SimpleCostedPath() = default;
  void update_path(
      const std::vector<std::unique_ptr<TrackedInstance>> &updated_instances) {
    // We need to update the path....first we need to know which
    // instances are new, and which should be deleted We know its new if we have
    // an instance ID contained in updated_instances but is not in instance_list
    // We know its old if we have an instance ID contained in instance_list but
    // is not in updated_instances So first lets remove any instances that are
    // no longer in updated_instances
    remove_old_instances(updated_instances);
    add_new_instances(updated_instances);
  }
  [[nodiscard]] const std::vector<std::unique_ptr<TrackedInstance>> &
  get_path() const {
    return instance_list;
  }
};

}; // namespace laser::tracking