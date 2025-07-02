#include "perception.hpp"

PerceptionNode::PerceptionNode()
    : Node("beta_perception"), total_duration_(0.0), inference_count_(0)
{
    // Declare and get the 'crop_type' parameter
    this->declare_parameter<std::string>("crop_type", "carrot");
    this->get_parameter("crop_type", crop_type_);
    
    RCLCPP_INFO(this->get_logger(), "Crop type set to: %s", crop_type_.c_str());
    
    // Initialize Triton client
    auto status = triton::client::InferenceServerGrpcClient::Create(
        &triton_client_,
        "localhost:8001"
    );
    if (!status.IsOk()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create Triton client: %s", status.Message().c_str());
        throw std::runtime_error("Triton client initialization failed");
    }

    // Initialize QoS and subscriptions
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/vimbax_camera_beta/image_raw",
        rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile),
        std::bind(&PerceptionNode::imageCallback, this, std::placeholders::_1)
    );
    detection_pub_ = this->create_publisher<beta_perception::msg::DetectionArray>(
        "/laser/weed_detections",
        rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile)
    );

    this->inference_time_pub_ = this->create_publisher<std_msgs::msg::UInt16>("/laser/inference_time", rclcpp::QoS(1).reliability(rclcpp::ReliabilityPolicy::Reliable).durability(rclcpp::DurabilityPolicy::Volatile));
}

PerceptionNode::~PerceptionNode() {}

void PerceptionNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    start_time_ = this->now();

    // Convert ROS image to OpenCV format
    cv::Mat cv_image;
    try {
        cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Could not convert image: %s", e.what());
        return;
    }

    // Prepare the Triton input
    std::unique_ptr<triton::client::InferInput> triton_input;
    std::unique_ptr<triton::client::InferInput> model_name_input;
    try {
        triton_input.reset(createTritonInput("input_image", cv_image, "UINT8"));
        model_name_input.reset(createTritonStringInput("model_name", crop_type_));
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to prepare Triton input: %s", e.what());
        return;
    }

    std::vector<triton::client::InferInput*> inputs = {triton_input.get(), model_name_input.get()};

    std::unique_ptr<triton::client::InferRequestedOutput> boxes;
    std::unique_ptr<triton::client::InferRequestedOutput> areas;
    std::unique_ptr<triton::client::InferRequestedOutput> scores;
    std::unique_ptr<triton::client::InferRequestedOutput> keypoints;
    std::unique_ptr<triton::client::InferRequestedOutput> ids;

    // Create Triton outputs
    triton::client::InferRequestedOutput* temp = nullptr;

    triton::client::InferRequestedOutput::Create(&temp, "boxes");
    boxes.reset(temp);

    triton::client::InferRequestedOutput::Create(&temp, "areas");
    areas.reset(temp);

    triton::client::InferRequestedOutput::Create(&temp, "scores");
    scores.reset(temp);

    triton::client::InferRequestedOutput::Create(&temp, "keypoints");
    keypoints.reset(temp);

    triton::client::InferRequestedOutput::Create(&temp, "ids");
    ids.reset(temp);

    // Wrap in a vector for inference
    std::vector<const triton::client::InferRequestedOutput*> outputs = {
        boxes.get(), areas.get(), scores.get(), keypoints.get(), ids.get()};

    // Perform inference
    triton::client::InferResult* result = nullptr;
    auto status = triton_client_->Infer(
        &result,
        triton::client::InferOptions("perception_router"),
        inputs,
        outputs
    );

    if (!status.IsOk()) {
        RCLCPP_ERROR(this->get_logger(), "Inference failed: %s", status.Message().c_str());
        return;
    }

    std::unique_ptr<triton::client::InferResult> result_ptr(result);
    processInferenceResponse(result_ptr.get(), msg->header);

    // Log timing
    double duration = (this->now() - start_time_).seconds() * 1000;
    //RCLCPP_INFO(this->get_logger(), "Time: %f", duration);
    total_duration_ += duration;
    inference_count_++;
    double average_duration = total_duration_ / inference_count_;

    auto message = std_msgs::msg::UInt16();
    message.data = static_cast<uint16_t>(average_duration); // Convert to uint16_t
    this->inference_time_pub_->publish(message);
}

triton::client::InferInput* PerceptionNode::createTritonInput(
    const std::string& name, const cv::Mat& image, const std::string& datatype)
{
    std::vector<int64_t> shape = {image.rows, image.cols, image.channels()};
    triton::client::InferInput* input = nullptr;

    auto status = triton::client::InferInput::Create(&input, name, shape, datatype);
    if (!status.IsOk()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create Triton input: %s", status.Message().c_str());
        throw std::runtime_error("Triton input creation failed");
    }

    status = input->AppendRaw(image.data, image.total() * image.elemSize());
    if (!status.IsOk()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to set Triton input data: %s", status.Message().c_str());
        delete input;
        throw std::runtime_error("Failed to append raw data to Triton input");
    }

    return input;
}

triton::client::InferInput* PerceptionNode::createTritonStringInput(
    const std::string& name, const std::string& value)
{
    triton::client::InferInput* input = nullptr;

    auto status = triton::client::InferInput::Create(&input, name, {1}, "BYTES");
    if (!status.IsOk()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create Triton string input '%s': %s",
                     name.c_str(), status.Message().c_str());
        throw std::runtime_error("Triton string input creation failed");
    }

    std::string serialized = serializeString(value);

    status = input->AppendRaw(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size());
    if (!status.IsOk()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to append data to Triton string input '%s': %s",
                     name.c_str(), status.Message().c_str());
        delete input;
        throw std::runtime_error("Failed to append string data to Triton input");
    }

    return input;
}

void PerceptionNode::processInferenceResponse(
    const triton::client::InferResult* result,
    const std_msgs::msg::Header& header)
{
    try {
        // Retrieve and convert outputs
        const int32_t* boxes_raw = nullptr;
        const int32_t* areas_raw = nullptr;
        const uint16_t* scores_raw = nullptr;  // FP16 is stored as uint16_t
        const int32_t* keypoints_raw = nullptr;
        const int32_t* ids_raw = nullptr;
        size_t boxes_byte_size = 0, areas_byte_size = 0, scores_byte_size = 0;
        size_t keypoints_byte_size = 0, ids_byte_size = 0;

        auto status = result->RawData("boxes", reinterpret_cast<const uint8_t**>(&boxes_raw), &boxes_byte_size);
        if (!status.IsOk()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to retrieve boxes output: %s", status.Message().c_str());
            return;
        }

        status = result->RawData("areas", reinterpret_cast<const uint8_t**>(&areas_raw), &areas_byte_size);
        if (!status.IsOk()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to retrieve areas output: %s", status.Message().c_str());
            return;
        }

        status = result->RawData("scores", reinterpret_cast<const uint8_t**>(&scores_raw), &scores_byte_size);
        if (!status.IsOk()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to retrieve scores output: %s", status.Message().c_str());
            return;
        }

        status = result->RawData("keypoints", reinterpret_cast<const uint8_t**>(&keypoints_raw), &keypoints_byte_size);
        if (!status.IsOk()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to retrieve keypoints output: %s", status.Message().c_str());
            return;
        }

        status = result->RawData("ids", reinterpret_cast<const uint8_t**>(&ids_raw), &ids_byte_size);
        if (!status.IsOk()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to retrieve ids output: %s", status.Message().c_str());
            return;
        }

        // Convert FP16 scores to float
        std::vector<float> scores(scores_byte_size / sizeof(uint16_t));
        for (size_t i = 0; i < scores.size(); ++i) {
            scores[i] = fp16ToFloat(scores_raw[i]);
        }

        size_t num_boxes = boxes_byte_size / (4 * sizeof(int32_t));
        size_t num_scores = scores.size();
        size_t num_ids = ids_byte_size / sizeof(int32_t);

        if (num_boxes != num_scores || num_boxes != num_ids) {
            RCLCPP_ERROR(this->get_logger(), "Output size mismatch: boxes=%zu, scores=%zu, ids=%zu", num_boxes, num_scores, num_ids);
            return;
        }

        auto detection_msg = prepareDetectionResultsMessage(
            keypoints_raw,
            scores.data(), 
            boxes_raw, 
            areas_raw, 
            ids_raw, 
            header, 
            num_boxes
        );

        detection_pub_->publish(detection_msg);

    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error processing inference response: %s", e.what());
    }
}

float PerceptionNode::fp16ToFloat(uint16_t fp16)
{
    uint32_t t1 = fp16 & 0x7fff;            // Non-sign bits
    uint32_t t2 = fp16 & 0x8000;            // Sign bit
    uint32_t t3 = fp16 & 0x7c00;            // Exponent
    t1 <<= 13;                              // Align mantissa on MSB
    t2 <<= 16;                              // Shift sign bit into position
    t1 += 0x38000000;                       // Adjust bias
    t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero
    t1 |= t2;                               // Re-insert sign bit
    float f;
    memcpy(&f, &t1, sizeof(f));             // Return as float
    return f;
}

std::string PerceptionNode::serializeString(const std::string& str)
{
    std::string serialized;
    int32_t len = str.size();
    serialized.resize(sizeof(int32_t) + len);
    memcpy(&serialized[0], &len, sizeof(int32_t));
    memcpy(&serialized[sizeof(int32_t)], str.data(), len);
    return serialized;
}

beta_perception::msg::DetectionArray PerceptionNode::prepareDetectionResultsMessage(
    const int32_t* keypoints, const float* scores, const int32_t* boxes, const int32_t* areas,
    const int32_t* class_ids, const std_msgs::msg::Header& header, size_t num_detections)
{
    beta_perception::msg::DetectionArray detection_array_msg;
    detection_array_msg.header = header;

    for (size_t i = 0; i < num_detections; ++i) {
        // Only process detections with class ID 0
        if (class_ids[i] == 0) {
            beta_perception::msg::BoundingBox box_msg;

            // Assign bounding box properties
            box_msg.x = boxes[i * 4 + 0];       // x-coordinate
            box_msg.y = boxes[i * 4 + 1];       // y-coordinate
            box_msg.width = boxes[i * 4 + 2];   // width
            box_msg.height = boxes[i * 4 + 3];  // height

            // Assign additional properties
            box_msg.score = scores[i];
            box_msg.area = areas[i];

            // Assign keypoint associated with the bounding box
            beta_perception::msg::Keypoint kp_msg;
            kp_msg.x = keypoints[i * 2 + 0];
            kp_msg.y = keypoints[i * 2 + 1];
            box_msg.keypoint = kp_msg;

            // Add the bounding box to the DetectionArray
            detection_array_msg.boxes.push_back(box_msg);
        }
    }

    return detection_array_msg;
}



int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PerceptionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
