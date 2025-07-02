// visualization.cpp

#include "visualization.hpp"

Visualization::Visualization()
    : Node("visualization_node")
{
    // Initialize message filters subscribers for image and detection array
    this->image_subscriber_.subscribe(this, "/vimbax_camera_beta/image_raw", rclcpp::QoS(100)
                    .reliability(rclcpp::ReliabilityPolicy::Reliable)
                    .durability(rclcpp::DurabilityPolicy::Volatile)
                    .get_rmw_qos_profile());
    this->detection_subscriber_.subscribe(this, "/laser/weed_detections", rclcpp::QoS(100)
                    .reliability(rclcpp::ReliabilityPolicy::Reliable)
                    .durability(rclcpp::DurabilityPolicy::Volatile)
                    .get_rmw_qos_profile());

    this->sync_.reset(new Sync(MySyncPolicy(100), image_subscriber_, detection_subscriber_));

    this->sync_->registerCallback(std::bind(&Visualization::synchronized_callback, this, std::placeholders::_1, std::placeholders::_2));

    // Initialize publisher for the annotated image
    this->annotated_image_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("/laser/perception", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile));

    RCLCPP_INFO(this->get_logger(), "Visualization Node with synchronized messages has been started.");
}


void Visualization::synchronized_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg, const beta_perception::msg::DetectionArray::ConstSharedPtr& detection_msg)
{
    try 
    {
        // Convert and display the image
        cv::Mat frame = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        cv::Scalar box_color_inside(255, 0, 0);  // Blue for valid bounding boxes
        cv::Scalar box_color_outside(0, 0, 255); // Red for out-of-bound bounding boxes
        cv::Scalar text_color(255, 0, 0);        // Blue for text
        int thickness = 5;
        int radius = 10;
        double font_scale = 0.75;

        int img_width = frame.cols;
        int img_height = frame.rows;

        // Loop through each bounding box in the detection array
        for (const auto& box : detection_msg->boxes)
        {
            // Calculate bounding box corners
            cv::Point top_left(box.x, box.y);
            cv::Point bottom_right(box.x + box.width, box.y + box.height);

            // Check if the bounding box is within image boundaries
            bool is_outside = (box.x < 0 || box.y < 0 || 
                               box.x + box.width > img_width || 
                               box.y + box.height > img_height);

            if (is_outside)
            {
                // Log and mark the bounding box as red
                RCLCPP_INFO(this->get_logger(), "Bounding box out of bounds: [x=%d, y=%d, width=%d, height=%d]", box.x, box.y, box.width, box.height);
                cv::rectangle(frame, top_left, bottom_right, box_color_outside, thickness);
            }
            else
            {
                // Draw the bounding box in blue
                cv::rectangle(frame, top_left, bottom_right, box_color_inside, thickness);
            }

            // Draw the keypoint if it exists within the bounding box
            cv::Point keypoint(box.keypoint.x, box.keypoint.y);
            cv::circle(frame, keypoint, radius, box_color_outside, -1);

            // Display the score
            std::string score_text = "Score: " + std::to_string(box.score).substr(0, 4); // Limit to 2 decimal places
            cv::putText(frame, score_text, top_left - cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness);
        }

	// Resize the image by 50% (fx = 0.5, fy = 0.5)
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(), 0.25, 0.25, cv::INTER_LINEAR);
        
        // New approach (publishing CompressedImage)
        sensor_msgs::msg::CompressedImage compressed_msg;
        compressed_msg.header = image_msg->header;  // Keep timestamp, frame_id, etc.
        compressed_msg.format = "rgb8; jpeg compressed bgr8";             // or "png", etc.

        // Encode the OpenCV image to a compressed buffer (JPEG in this case).
        std::vector<uchar> compressed_buffer;
        std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 25}; 
        // e.g., 50% JPEG quality -- adjust as needed

        if (!cv::imencode(".jpg", resized_frame, compressed_buffer, compression_params)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to encode image as JPEG.");
            return;
        }

        // Populate the CompressedImage message
        compressed_msg.data = std::move(compressed_buffer);

        // Publish
        annotated_image_publisher_->publish(compressed_msg);
    }
    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Could not convert from '%s' to 'bgr8'.", image_msg->encoding.c_str());
    }
}



int main(int argc, char *argv[])
{
    // Initialize ROS2
    rclcpp::init(argc, argv);

    // Create the node and spin
    auto node = std::make_shared<Visualization>();
    rclcpp::spin(node);

    // Shutdown ROS2
    rclcpp::shutdown();
    return 0;
}
