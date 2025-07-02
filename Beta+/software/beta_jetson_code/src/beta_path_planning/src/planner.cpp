#include "planner.hpp"

namespace beta_path_planning {

PlannerNode::PlannerNode()
: Node("beta_path_planner"), 
  target_start_time_(0.0),
  stay_duration_(0.2), // Default stay duration
  current_target_id_(""),
  reference_position_(std::make_tuple(968.0f, 608.0f)) {
    // Declare ROS parameter for stay duration
    this->declare_parameter<double>("dwell", 0.2); // Default: 0.2 seconds

    // Fetch the parameter value during initialization
    this->get_parameter("dwell", stay_duration_);

    // Init the ROI info
    this->roi_corners_ = this->loadROIFromJSON("/workspaces/isaac_ros-dev/src/laser_framework/config/camera_laser_calib.json");

    // Calculate the center of the ROI
    float center_x = 0.0f;
    float center_y = 0.0f;
    for (const auto& corner : roi_corners_) {
        center_x += std::get<0>(corner);
        center_y += std::get<1>(corner);
    }
    center_x /= roi_corners_.size();
    center_y /= roi_corners_.size();

    // Set the initial reference position to the center of the ROI
    reference_position_ = std::make_tuple(center_x, center_y);

    // Initialize the subscription
    tracker_sub_ = this->create_subscription<beta_tracking::msg::TrackerOutput>(
        "/laser/tracker", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile),
        std::bind(&PlannerNode::trackerCallback, this, std::placeholders::_1));

    // Initialize the publishers
    path_publisher_ = this->create_publisher<beta_path_planning::msg::LaserPath>("/laser/path_planned", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile));
    future_position_publisher_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/laser/current_target", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile));

    RCLCPP_INFO(this->get_logger(), "Planner Node initialized and subscribed to /laser/tracker.");
}


void PlannerNode::trackerCallback(const beta_tracking::msg::TrackerOutput::SharedPtr msg) {
    // Extract the current timestamp from the message header
    double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    // Update object history for all targets in the message
    for (const auto &target : msg->target_list) {
        updateObjectHistory(target.id, {target.target_point.x, target.target_point.y, current_time});
    }

    // Clean up object history for targets no longer present
    cleanupObjectHistory(msg);

    // Construct the graph of targets, including predictions
    auto graph = constructGraph(*msg, current_time);

    if (graph.empty()) {
        return;
    }

    // Validate and replan the path if necessary
    validateAndReplanPath(graph, current_time);

    // Publish the path as a custom message
    beta_path_planning::msg::LaserPath path_msg;

    // Set the header with the timestamp of the original message
    path_msg.header.stamp = msg->header.stamp;
    path_msg.header.frame_id = msg->header.frame_id; // Optional: Set a frame ID if needed

    // Populate the path data with the best sequence of target IDs
    for (const auto &id : best_sequence_) {
        path_msg.data.push_back(id);
    }

    // Publish the path message
    path_publisher_->publish(path_msg);

    // Get the next target
    auto next_target = getNextTarget(*msg);

    // If no target is selected, log and return
    if (next_target.id.empty()) {
        return;
    }

    return;
}

void PlannerNode::updateObjectHistory(const std::string &target_id, const std::tuple<float, float, double> &target_data) {
    // If the target ID is not in the history, create a new entry
    if (object_history_.find(target_id) == object_history_.end()) {
        object_history_[target_id] = {};
    }

    // Add the new data point to the history
    object_history_[target_id].emplace_back(target_data);

    // Keep only the last 3 data points for efficiency
    while (object_history_[target_id].size() > 3) {
        object_history_[target_id].pop_front();
    }
}

void PlannerNode::cleanupObjectHistory(const beta_tracking::msg::TrackerOutput::SharedPtr &tracker_msg) {
    // Collect current target IDs from the tracker message
    std::unordered_set<std::string> current_ids;
    for (const auto &target : tracker_msg->target_list) {
        current_ids.insert(target.id);
    }

    // Identify and remove target IDs no longer present in the tracker message
    std::vector<std::string> ids_to_remove;
    for (const auto &entry : object_history_) {
        if (current_ids.find(entry.first) == current_ids.end()) {
            ids_to_remove.push_back(entry.first);
        }
    }

    // Remove outdated target IDs from the object history
    for (const auto &target_id : ids_to_remove) {
        object_history_.erase(target_id);
    }
}

// Function implementation
bool PlannerNode::isTargetFeasibleDuringStay(const std::string &target_id, double start_time, double stay_duration) const {
    constexpr int num_steps = 20; // Divide duration into 20 steps
    double time_step = stay_duration / num_steps;

    for (int step = 0; step <= num_steps; ++step) {
        double t = start_time + step * time_step;

        // Predict future position
        auto future_position = predictFuturePosition(target_id, t);
        if (!future_position.has_value()) {
            return false; // Treat as infeasible
        }

        auto [future_x, future_y] = future_position.value();

        // Check if the position is within the camera frame bounds
        if ((future_x < 50.0 || future_x >= 1936.0 || future_y < 0.0 || future_y >= 1216.0) && !isInsideROI(future_x, future_y)) {
            return false;
        }
    }

    return true; // Target remains feasible throughout the duration
}

std::unordered_map<std::string, TargetNode> PlannerNode::constructGraph(
    const beta_tracking::msg::TrackerOutput &tracker_msg, double current_time) {
    std::unordered_map<std::string, TargetNode> graph;

    double cumulative_time = current_time;

    for (const auto &target : tracker_msg.target_list) {
        if (visited_targets_.find(target.id) != visited_targets_.end()) {
            continue; // Skip visited targets
        }

        // Check feasibility and ROI constraints
        if (isTargetFeasibleDuringStay(target.id, cumulative_time, (2 * stay_duration_))) {
            auto future_position = predictFuturePosition(target.id, cumulative_time + (2 * stay_duration_));
            if (future_position.has_value()) {
                auto [future_x, future_y] = future_position.value();

                // Check if the target is inside the ROI
                if (isInsideROI(future_x, future_y)) {
                    graph[target.id] = TargetNode{
                        future_position.value(), // Predicted position
                        target,                  // Target object
                        {},                      // Edges
                        1.0                      // Initial priority
                    };
                }
            }
        }
    }

    // Add edges between valid targets (same as before)
    const double max_distance_threshold = 800.0;
    std::vector<std::pair<std::string, std::tuple<float, float>>> nodes;

    for (const auto &[id, node] : graph) {
        nodes.emplace_back(id, node.position);
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto &[id1, pos1] = nodes[i];
        for (size_t j = i + 1; j < nodes.size(); ++j) {
            const auto &[id2, pos2] = nodes[j];

            double distance = std::sqrt(
                std::pow(std::get<0>(pos1) - std::get<0>(pos2), 2) +
                std::pow(std::get<1>(pos1) - std::get<1>(pos2), 2));

            if (distance <= max_distance_threshold) {
                graph[id1].edges[id2] = distance;
                graph[id2].edges[id1] = distance;
            }
        }
    }

    return graph;
}

beta_tracking::msg::Target PlannerNode::getNextTarget(
    const beta_tracking::msg::TrackerOutput &tracker_msg) {
    // Get the current time from the tracker message
    double current_time = tracker_msg.header.stamp.sec +
                          tracker_msg.header.stamp.nanosec * 1e-9;
    double ros_time = this->get_clock()->now().seconds();

    // If there is an active target, check if the stay duration is complete
    if (!current_target_id_.empty()) {
        // Calculate the elapsed time since the current target was selected
        double elapsed_time = ros_time - target_start_time_;

        // If the stay duration has not completed, stay on the current target
	if (elapsed_time < stay_duration_) {
	    // Single loop to process the current target
	    for (const auto &target : tracker_msg.target_list) {
		if (target.id == current_target_id_) {
		    // Predict the future position for the current target
		    if (auto future_position = predictFuturePosition(current_target_id_, ros_time + 0.2)) {
		        auto [future_x, future_y] = *future_position;

		        // Publish the future position
		        geometry_msgs::msg::PointStamped future_position_msg;
		        future_position_msg.header.stamp = tracker_msg.header.stamp;
		        future_position_msg.point.x = future_x;
		        future_position_msg.point.y = future_y;
		        future_position_msg.point.z = std::max(0.0, stay_duration_ - elapsed_time) * 1000; // Remaining dwell time in ms
		        future_position_publisher_->publish(future_position_msg);

		        // Return the current target
		        return target;
		    }
		}
	    }

            // If the current target is no longer available, treat it as invalid
            visited_targets_.insert(current_target_id_);
            current_target_id_.clear();
        } else {
            // Stay duration is complete, mark the current target as visited
            visited_targets_.insert(current_target_id_);
            current_target_id_.clear();
        }
    }

    // No active target or current target completed, find the next target
    auto graph = constructGraph(tracker_msg, current_time);

    if (graph.empty()) {
        return beta_tracking::msg::Target(); // Return an empty target if no valid targets exist
    }

    validateAndReplanPath(graph, current_time);

    // Process the next target in the best sequence
    while (!best_sequence_.empty()) {
        auto next_target_id = best_sequence_.front();
        best_sequence_.erase(best_sequence_.begin());

        if (graph.find(next_target_id) != graph.end()) {
            // Predict the future position for the next target
            if (auto future_position = predictFuturePosition(next_target_id, ros_time + 0.2)) {
                auto [future_x, future_y] = *future_position;

                // Ensure the future position is within the ROI
                if (isInsideROI(future_x, future_y)) {
                    // Start a new stay duration for the next target
                    current_target_id_ = next_target_id;
                    target_start_time_ = ros_time;
                    reference_position_ = std::make_tuple(future_x, future_y);

                    // Publish the future position before returning the target
                    geometry_msgs::msg::PointStamped future_position_msg;
                    future_position_msg.header.stamp = tracker_msg.header.stamp;
                    future_position_msg.point.x = future_x;
                    future_position_msg.point.y = future_y;
                    future_position_msg.point.z = stay_duration_ * 1000; // Full stay duration in milliseconds
                    future_position_publisher_->publish(future_position_msg);

                    return graph[next_target_id].target;
                }
            }
        }
    }

    return beta_tracking::msg::Target(); // Return an empty target if no valid next target
}



void PlannerNode::validateAndReplanPath(
    const std::unordered_map<std::string, TargetNode> &graph, double current_time) {
    std::vector<std::string> valid_sequence;
    double cumulative_time = current_time;

    for (const auto &target_id : best_sequence_) {
        if (graph.find(target_id) != graph.end()) {
            // Check if the target is feasible and inside the ROI
            auto [x, y] = graph.at(target_id).position;
            if (isTargetFeasibleDuringStay(target_id, cumulative_time, (2 * stay_duration_)) &&
                isInsideROI(x, y)) {
                valid_sequence.push_back(target_id);
                cumulative_time += stay_duration_;
            }
        }
    }

    // Detect new targets or changes in the path
    std::vector<std::string> new_targets;
    for (const auto &node : graph) {
        if (std::find(valid_sequence.begin(), valid_sequence.end(), node.first) == valid_sequence.end() &&
            visited_targets_.find(node.first) == visited_targets_.end()) {
            auto [x, y] = node.second.position;
            if (isInsideROI(x, y)) {
                new_targets.push_back(node.first);
            }
        }
    }

    if (valid_sequence.size() != best_sequence_.size() || !new_targets.empty()) {
        if (graph.empty()) {
            best_sequence_.clear();
            return;
        }

        std::string start_node_id = current_target_id_;
        if (graph.find(start_node_id) == graph.end()) {
            auto min_node = std::min_element(
                graph.begin(), graph.end(),
                [&](const auto &a, const auto &b) {
                    double a_dist = std::sqrt(
                        std::pow(std::get<0>(a.second.position) - std::get<0>(reference_position_), 2) +
                        std::pow(std::get<1>(a.second.position) - std::get<1>(reference_position_), 2));
                    double b_dist = std::sqrt(
                        std::pow(std::get<0>(b.second.position) - std::get<0>(reference_position_), 2) +
                        std::pow(std::get<1>(b.second.position) - std::get<1>(reference_position_), 2));
                    return a_dist < b_dist;
                });
            start_node_id = min_node->first;
        }

        best_sequence_ = solveTSP(graph, start_node_id).first;
    }
}


// Solve TSP with priority queue
std::pair<std::vector<std::string>, double> PlannerNode::solveTSP(
    const std::unordered_map<std::string, TargetNode> &graph,
    const std::string &start_node_id) {

    std::unordered_set<std::string> visited;
    std::vector<std::string> path;
    double total_distance = 0.0;

    std::priority_queue<PriorityNode> pq;
    pq.push({start_node_id, 0.0});

    while (!pq.empty() && visited.size() < graph.size()) {
        auto current = pq.top();
        pq.pop();

        if (visited.find(current.id) != visited.end()) continue;

        visited.insert(current.id);
        path.push_back(current.id);

        for (const auto &[neighbor_id, distance] : graph.at(current.id).edges) {
            if (visited.find(neighbor_id) == visited.end()) {
                double priority = graph.at(neighbor_id).priority - 0.1 * distance;
                pq.push({neighbor_id, priority});
            }
        }
    }

    return {path, total_distance};
}

std::optional<std::tuple<float, float>> PlannerNode::predictFuturePosition(
    const std::string &target_id, double future_timestamp) const {
    // Check if there is enough history for the target
    if (object_history_.find(target_id) == object_history_.end() || object_history_.at(target_id).size() < 2) {
        return std::nullopt;
    }

    // Extract x, y, and timestamps
    std::vector<double> x_values, y_values, timestamps;
    for (const auto &[x, y, timestamp] : object_history_.at(target_id)) {
        x_values.push_back(x);
        y_values.push_back(y);
        timestamps.push_back(timestamp);
    }

    // Normalize timestamps for numerical stability
    double time_offset = timestamps[0];
    for (auto &t : timestamps) {
        t -= time_offset;
    }
    future_timestamp -= time_offset;

    // Use Eigen to solve for coefficients of linear regression
    int n = timestamps.size();
    Eigen::VectorXd T(n), X(n), Y(n);
    for (int i = 0; i < n; ++i) {
        T(i) = timestamps[i];
        X(i) = x_values[i];
        Y(i) = y_values[i];
    }

    // Design matrix for linear regression (for fitting: X = mT + c)
    Eigen::MatrixXd A(n, 2);
    A.col(0) = T; // Timestamps
    A.col(1) = Eigen::VectorXd::Ones(n); // Bias (intercept term)

    // Solve for x coefficients
    Eigen::Vector2d coeffs_x = A.colPivHouseholderQr().solve(X); // [m_x, c_x]
    // Solve for y coefficients
    Eigen::Vector2d coeffs_y = A.colPivHouseholderQr().solve(Y); // [m_y, c_y]

    // Predict future x and y
    double future_x = coeffs_x(0) * future_timestamp + coeffs_x(1); // m_x * future_time + c_x
    double future_y = coeffs_y(0) * future_timestamp + coeffs_y(1); // m_y * future_time + c_y

    return std::make_tuple(static_cast<float>(future_x), static_cast<float>(future_y));
}


std::vector<std::tuple<float, float>> PlannerNode::loadROIFromJSON(const std::string &file_path)
{
    // Read JSON file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + file_path);
    }

    nlohmann::json json_data;
    file >> json_data;

    // Parse ROI corners
    auto roi_data = json_data.at("roi");
    return {
        std::make_tuple(roi_data["top_left"][0].get<float>(), roi_data["top_left"][1].get<float>()),
        std::make_tuple(roi_data["top_right"][0].get<float>(), roi_data["top_right"][1].get<float>()),
        std::make_tuple(roi_data["bottom_right"][0].get<float>(), roi_data["bottom_right"][1].get<float>()),
        std::make_tuple(roi_data["bottom_left"][0].get<float>(), roi_data["bottom_left"][1].get<float>())
    };
}

bool PlannerNode::isInsideROI(float x, float y) const {
    // Extract ROI corners
    const auto &[x1, y1] = roi_corners_[0]; // Top-left
    const auto &[x2, y2] = roi_corners_[1]; // Top-right
    const auto &[x3, y3] = roi_corners_[2]; // Bottom-right
    const auto &[x4, y4] = roi_corners_[3]; // Bottom-left

    // Helper function to calculate cross product
    auto crossProduct = [](float ax, float ay, float bx, float by) {
        return ax * by - ay * bx;
    };

    // Calculate vectors for the point and edges
    float v1 = crossProduct(x - x1, y - y1, x2 - x1, y2 - y1); // Edge 1
    float v2 = crossProduct(x - x2, y - y2, x3 - x2, y3 - y2); // Edge 2
    float v3 = crossProduct(x - x3, y - y3, x4 - x3, y4 - y3); // Edge 3
    float v4 = crossProduct(x - x4, y - y4, x1 - x4, y1 - y4); // Edge 4

    // Check if the signs of all cross products are the same
    return (v1 >= 0 && v2 >= 0 && v3 >= 0 && v4 >= 0) || (v1 <= 0 && v2 <= 0 && v3 <= 0 && v4 <= 0);
}

float PlannerNode::crossProduct(float x1, float y1, float x2, float y2) {
    return (x1 * y2 - y1 * x2);
}


} // namespace beta_path_planning

// Main function
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<beta_path_planning::PlannerNode>());
    rclcpp::shutdown();
    return 0;
}
