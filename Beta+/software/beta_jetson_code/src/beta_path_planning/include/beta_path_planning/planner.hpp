#ifndef PLANNER_HPP_
#define PLANNER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "beta_tracking/msg/tracker_output.hpp"
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <list>
#include <tuple>
#include <memory>
#include <optional>
#include <deque>
#include <queue>
#include <cmath>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include "beta_path_planning/msg/laser_path.hpp"
#include <nlohmann/json.hpp>

namespace beta_path_planning {

    // Structure representing a node in the target graph
    struct TargetNode {
    std::tuple<float, float> position; // Predicted position
    beta_tracking::msg::Target target; // Associated target message
    std::unordered_map<std::string, double> edges; // Edges with distances to other nodes
    double priority; // priority logic
    };

    // Node structure for priority queue
    struct PriorityNode {
        std::string id;
        double priority;
        bool operator<(const PriorityNode &other) const {
            return priority < other.priority; // Min-heap (higher priority first)
        }
    };


    class PlannerNode : public rclcpp::Node {
    public:
        PlannerNode(); // Constructor

    private:
        // Callback for tracker messages
        void trackerCallback(const beta_tracking::msg::TrackerOutput::SharedPtr msg);

        // Function to predict the future position of a target
        std::optional<std::tuple<float, float>> predictFuturePosition(
        const std::string &target_id, double future_timestamp) const;


        // Function to update the Object History
        void updateObjectHistory(const std::string &target_id, const std::tuple<float, float, double> &target_data);

        // Function to clean up object history
        void cleanupObjectHistory(const beta_tracking::msg::TrackerOutput::SharedPtr &tracker_msg);

        // Function to constuct the graph
        std::unordered_map<std::string, TargetNode> constructGraph(
        const beta_tracking::msg::TrackerOutput &tracker_msg, double current_time);

        // Function to replan the path
        void validateAndReplanPath(const std::unordered_map<std::string, TargetNode> &graph, double current_time);

        // Function to solve the heurctic clustering probem using a TSP
        std::pair<std::vector<std::string>, double> solveTSP(const std::unordered_map<std::string, TargetNode> &graph, const std::string &start_node_id);

        // Function to check feasiilty
        bool isTargetFeasibleDuringStay(const std::string &target_id, double start_time, double stay_duration) const;

        // Function to check inside ROI
        bool isInsideROI(float x, float y) const;

        // Function for cross product
        float crossProduct(float x1, float y1, float x2, float y2);

        // Function to load ROI from JSON file
        std::vector<std::tuple<float, float>> loadROIFromJSON(const std::string &file_path);

        // Function which determines what is the next target or stay on the current target
        beta_tracking::msg::Target getNextTarget(
        const beta_tracking::msg::TrackerOutput &tracker_msg);

        // Subscription to TrackerOutput topic
        rclcpp::Subscription<beta_tracking::msg::TrackerOutput>::SharedPtr tracker_sub_;

        // Publishers
        rclcpp::Publisher<beta_path_planning::msg::LaserPath>::SharedPtr path_publisher_; // For the path
        rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr future_position_publisher_; // For the predicted future position


        // Variables
        double target_start_time_; // Start time for the current target
        double stay_duration_; // Duration to stay on each target
        std::string current_target_id_; // Current target ID
        std::tuple<float, float> reference_position_; // Reference position (x, y)
        std::unordered_set<std::string> visited_targets_; // Set of visited target IDs
        std::unordered_map<std::string, std::deque<std::tuple<float, float, double>>> object_history_; // History of target locations
        std::vector<std::string> best_sequence_; // Best sequence of targets
        std::vector<std::tuple<float, float>> roi_corners_; // ROI for the Laser
    };

} // namespace beta_path_planning

#endif // PLANNER_HPP_
