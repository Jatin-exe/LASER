/**
 * Copyright (c) 2025 Laudando & Associates LLC
 *
 * This file is part of the Alpha version of the L&Aser Software.
 * 
 * Licensed under the L&Aser Public Use License v1.0 (April 2025), based on SSPL v1.
 * You may use, modify, and redistribute this file only under the terms of that license.
 *
 * Commercial use, integration into proprietary systems, or attempts to circumvent the
 * license obligations are prohibited without an explicit commercial license.
 *
 * See the full license in the LICENSE file or contact chris@laudando.com for details.
 *
 * This license does NOT apply to any AgCeption™ branded systems or L&Aser Beta modules.
*/

#include "perception.hpp"

namespace alpha
{
    alpha::Perception::Perception() : Node("alpha_perception")
    {
        // Declare and get the model_variant parameter
        this->declare_parameter<std::string>("model_variant", "nano");
        std::string variant_full = this->get_parameter("model_variant").as_string();

        // Mapping from full name to short variant
        std::map<std::string, std::string> variant_map = {
            {"nano", "n"},
            {"small", "s"},
            {"medium", "m"},
            {"large", "l"},
            {"xlarge", "x"}
        };

        if (variant_map.find(variant_full) == variant_map.end())
        {
            RCLCPP_FATAL(this->get_logger(), "Invalid model_variant '%s'. Must be one of: nano, small, medium, large, xlarge", variant_full.c_str());
            throw std::runtime_error("Invalid model_variant");
        }

        std::string variant = variant_map[variant_full];
        std::string model_path = "/ros2_ws/src/alpha_perception/models/dfine_hgnetv2_" + variant + "_custom/model.trt";

        // Load engine and buffers
        if (!this->loadTRTEngine_(model_path) || !this->setupBuffers_())
        {
            throw std::runtime_error("Failed to load TensorRT engine or buffer failure");
        }

        this->declare_parameter<bool>("gui", false);
        this->gui_ = this->get_parameter("gui").as_bool();

        // ROS Subscriber
        this->imageDataSub_ = this->create_subscription<sensor_msgs::msg::Image>("/image_raw", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile(), std::bind(&Perception::imageCallback_, this, std::placeholders::_1));

        // ROS Publishers
        this->detectionArrayPub_ = this->create_publisher<alpha_perception::msg::DetectionArray>("/alpha_perception/detections", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile());
        this->inferenceTimePub_ = this->create_publisher<std_msgs::msg::Int16>("/alpha_perception/inference_time", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile());
    }

    alpha::Perception::~Perception()
    {
        cudaFreeHost(this->pinnedHostInput_);
        cudaFree(this->deviceInput_);
        cudaFree(this->deviceDetections_);
        cudaFree(this->deviceValidCount_);
        delete this->context_;
        delete this->engine_;
        delete this->runtime_;
        cv::destroyAllWindows();
    }

    int alpha::Perception::calculateVolume_(const nvinfer1::Dims& dims)
    {
        int vol = 1;
        for (int i = 0; i < dims.nbDims; ++i) vol *= dims.d[i];
        return vol;
    }

    bool alpha::Perception::loadTRTEngine_(const std::string& engineFilePath)
    {
        std::ifstream engineFile(engineFilePath, std::ios::binary);
        if (!engineFile)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open engine file: %s", engineFilePath.c_str());
            return false;
        }
        engineFile.seekg(0, std::ifstream::end);
        size_t fileSize = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);
        std::vector<char> engineData(fileSize);
        engineFile.read(engineData.data(), fileSize);
        engineFile.close();

        // Deserialize the engine
        this->runtime_ = nvinfer1::createInferRuntime(alpha::gLogger);
        this->engine_ = this->runtime_->deserializeCudaEngine(static_cast<const void*>(engineData.data()), fileSize);

        if (!this->engine_)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to deserialize CUDA engine");
            return false;
        }

        this->context_ = this->engine_->createExecutionContext();
        if (!this->context_)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to create execution context");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Deserialized CUDA Engine %s", engineFilePath.c_str());
        return true;
    }

    bool alpha::Perception::setupBuffers_()
    {
        // Create CUDA stream
        if (cudaStreamCreate(&(this->stream_)) != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA stream");
            return false;
        }

        int numTensors = this->engine_->getNbIOTensors();
        for (int i = 0; i < numTensors; ++i)
        {
            const char* name = this->engine_->getIOTensorName(i);
            nvinfer1::DataType dtype = this->engine_->getTensorDataType(name);
            nvinfer1::Dims shape = this->engine_->getTensorShape(name);

            // Calculate buffer size in elements and bytes
            int numElements = this->calculateVolume_(shape);
            size_t byteSize = 0;
            void* devicePtr = nullptr;

            // Allocate based on data type
            if (dtype == nvinfer1::DataType::kHALF)
            {
                byteSize = numElements * sizeof(__half);
                this->hostFP16_[name].resize(numElements);
            }
            else if (dtype == nvinfer1::DataType::kINT64)
            {
                byteSize = numElements * sizeof(int64_t);
                this->hostInt64_[name].resize(numElements);
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Unsupported data type for tensor: %s", name);
                return false;
            }

            // Allocate device memory
            if (cudaMalloc(&devicePtr, byteSize) != cudaSuccess)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to allocate device memory for tensor: %s", name);
                return false;
            }

            // Store buffer and bind to execution context
            this->deviceBuffers_[name] = devicePtr;
            this->context_->setTensorAddress(name, devicePtr);
        }

        RCLCPP_INFO(this->get_logger(), "All dynamic buffers allocated and bound.");
        
        cudaHostAlloc(&(this->pinnedHostInput_), this->maxInputSize_ * sizeof(uchar3), cudaHostAllocDefault);
        cudaMalloc(&(this->deviceInput_), this->maxInputSize_ * sizeof(uchar3));

        // Estimate max number of detections (safe upper bound from output tensor size)
        int maxDetections = this->hostInt64_["labels"].size();

        // Allocate GPU buffer for detections
        if (cudaMalloc(&(this->deviceDetections_), maxDetections * sizeof(alpha::perception_kernels::Detection)) != cudaSuccess)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate deviceDetections_");
            return false;
        }

        // Allocate GPU counter
        if (cudaMalloc(&(this->deviceValidCount_), sizeof(int)) != cudaSuccess)
         {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate deviceValidCount_");
            return false;
        }

        // Preallocate host-side buffer (CPU)
        this->hostDetections_.resize(maxDetections);
        
        return true;
    }

    std::tuple<float, int, int> alpha::Perception::preprocessImage_(const cv::Mat& input)
    {
        const int targetSize = this->engine_->getTensorShape("images").d[2];
        int orig_h = input.rows;
        int orig_w = input.cols;

        float ratio = std::min(static_cast<float>(targetSize) / orig_w, static_cast<float>(targetSize) / orig_h);
        int new_w = static_cast<int>(orig_w * ratio);
        int new_h = static_cast<int>(orig_h * ratio);
        int pad_w = (targetSize - new_w) / 2;
        int pad_h = (targetSize - new_h) / 2;

        // Sanity check to avoid overflow (optional)
        if (orig_h * orig_w > static_cast<int>(this->maxInputSize_)) {
            RCLCPP_ERROR(this->get_logger(), "Input size exceeds maxInputSize_: %dx%d > %zu", orig_w, orig_h, this->maxInputSize_);
            throw std::runtime_error("Image too large for preallocated buffers");
        }

        // Copy input image to pinned host buffer
        std::memcpy(this->pinnedHostInput_, input.ptr<uchar3>(), orig_h * orig_w * sizeof(uchar3));

        // Async copy to GPU
        cudaMemcpyAsync(this->deviceInput_, this->pinnedHostInput_, orig_h * orig_w * sizeof(uchar3), cudaMemcpyHostToDevice, this->stream_);

        // Launch CUDA preprocessing kernel
        __half* deviceOutput = reinterpret_cast<__half*>(this->deviceBuffers_["images"]);
        alpha::perception_kernels::launch_preprocess_kernel(
            this->deviceInput_,
            deviceOutput,
            orig_w,
            orig_h,
            targetSize,
            ratio,
            pad_w,
            pad_h,
            this->stream_
        );

        return std::make_tuple(ratio, pad_w, pad_h);
    }

    void alpha::Perception::runInference_()
    {
        if (!this->context_) {
            RCLCPP_ERROR(this->get_logger(), "Execution context is not initialized.");
            return;
        }

        // --- 1. Upload input: orig_target_sizes ---
        std::vector<int64_t>& sizeHost = this->hostInt64_["orig_target_sizes"];
        sizeHost[0] = this->engine_->getTensorShape("images").d[2];
        sizeHost[1] = this->engine_->getTensorShape("images").d[3];

        cudaMemcpyAsync(this->deviceBuffers_["orig_target_sizes"], sizeHost.data(), sizeHost.size() * sizeof(int64_t), cudaMemcpyHostToDevice, this->stream_);

        // --- 2. Set input shapes ---
        bool shapeOk = true;
        shapeOk &= this->context_->setInputShape("images", this->engine_->getTensorShape("images"));
        shapeOk &= this->context_->setInputShape("orig_target_sizes", this->engine_->getTensorShape("orig_target_sizes"));
        if (!shapeOk) 
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to set input shapes.");
            return;
        }

        // --- 3. Run inference ---
        if (!this->context_->enqueueV3(this->stream_)) 
        {
            RCLCPP_ERROR(this->get_logger(), "TensorRT inference failed.");
            return;
        }
    }

    std::tuple<std::vector<cv::Rect2f>, std::vector<float>, std::vector<int64_t>, std::vector<cv::Point2f>, std::vector<float>> alpha::Perception::postprocessDetections_(float ratio, int pad_w, int pad_h, float conf_thresh)
    {
        int num_detections = this->hostInt64_["labels"].size();  // Same as before

        // Reset counter on GPU
        cudaMemsetAsync(this->deviceValidCount_, 0, sizeof(int), this->stream_);

        // Launch GPU postprocessing kernel
        alpha::perception_kernels::launch_postprocess_kernel(
            reinterpret_cast<__half*>(this->deviceBuffers_["boxes"]),
            reinterpret_cast<__half*>(this->deviceBuffers_["scores"]),
            reinterpret_cast<int64_t*>(this->deviceBuffers_["labels"]),
            num_detections,
            conf_thresh,
            ratio,
            pad_w,
            pad_h,
            this->deviceDetections_,
            this->deviceValidCount_,
            this->stream_
        );

        // Copy back the number of valid detections
        int valid_count = 0;
        cudaMemcpyAsync(&valid_count, this->deviceValidCount_, sizeof(int), cudaMemcpyDeviceToHost, this->stream_);
        cudaStreamSynchronize(this->stream_);

        // Resize host buffer just once (if needed)
        this->hostDetections_.resize(valid_count);

        // Copy back final detections from device to host
        cudaMemcpy(this->hostDetections_.data(), this->deviceDetections_, valid_count * sizeof(alpha::perception_kernels::Detection), cudaMemcpyDeviceToHost);

        // Prepare output containers
        std::vector<cv::Rect2f> boxes;
        std::vector<float> scores;
        std::vector<int64_t> labels;
        std::vector<cv::Point2f> centers;
        std::vector<float> areas;

        boxes.reserve(valid_count);
        scores.reserve(valid_count);
        labels.reserve(valid_count);
        centers.reserve(valid_count);
        areas.reserve(valid_count);

        for (const alpha::perception_kernels::Detection& det : this->hostDetections_)
        {
            boxes.emplace_back(det.x, det.y, det.w, det.h);
            scores.push_back(det.score);
            labels.push_back(det.label);
            centers.emplace_back(det.cx, det.cy);
            areas.push_back(det.area);
        }

        return std::make_tuple(boxes, scores, labels, centers, areas);
    }

    std::shared_ptr<alpha_perception::msg::DetectionArray> alpha::Perception::prepareDetectionResultsMessage_(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, const std::vector<int64_t>& labels, const std::vector<cv::Point2f>& centers, const std::vector<float>& areas, const std_msgs::msg::Header& header)
    {
        std::shared_ptr<alpha_perception::msg::DetectionArray> detection_array_msg = std::make_shared<alpha_perception::msg::DetectionArray>();
        detection_array_msg->header = header;

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (labels[i] != 1) continue;

            alpha_perception::msg::BoundingBox box_msg;

            box_msg.x = boxes[i].x;
            box_msg.y = boxes[i].y;
            box_msg.width = boxes[i].width;
            box_msg.height = boxes[i].height;
            box_msg.score = scores[i];
            box_msg.area = areas[i];

            // Keypoint (center)
            alpha_perception::msg::Keypoint kp_msg;
            kp_msg.x = centers[i].x;
            kp_msg.y = centers[i].y;
            box_msg.keypoint = kp_msg;

            detection_array_msg->boxes.push_back(std::move(box_msg));
        }
        return detection_array_msg;
    }
    
    std::shared_ptr<std_msgs::msg::Int16> alpha::Perception::prepareInferenceTime_(const int & inferceTime)
    {
        std::shared_ptr<std_msgs::msg::Int16> out_msg = std::make_shared<std_msgs::msg::Int16>();
        out_msg->data = inferceTime;

        return out_msg;
    }

    void alpha::Perception::imageCallback_(const sensor_msgs::msg::Image::UniquePtr & msg)
    {
        try
        {
            //Timer start
            std::chrono::_V2::system_clock::time_point start = std::chrono::high_resolution_clock::now();

            cv::Mat frame = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::RGB8)->image;

            std::tuple<float, int, int> meta = this->preprocessImage_(frame);

            this->runInference_();

            std::tuple<std::vector<cv::Rect2f>, std::vector<float>, std::vector<int64_t>, std::vector<cv::Point2f>, std::vector<float>> outputs = this->postprocessDetections_(std::get<0>(meta), std::get<1>(meta), std::get<2>(meta));

            //Timer stop plus durations in ms
            std::chrono::_V2::system_clock::time_point stop = std::chrono::high_resolution_clock::now();
            std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

            this->detectionArrayPub_->publish(*(this->prepareDetectionResultsMessage_(std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs), std::get<3>(outputs), std::get<4>(outputs), msg->header)));
            this->inferenceTimePub_->publish(*(this->prepareInferenceTime_(duration.count())));

            if (this->gui_)
            {
                cv::Mat bgrImage;
                cv::cvtColor(frame, bgrImage, cv::COLOR_RGB2BGR);

                for (size_t i = 0; i < std::get<0>(outputs).size(); ++i)
                {
                    if (std::get<2>(outputs)[i] != 1) continue;

                    const cv::Rect2f& box = std::get<0>(outputs)[i];
                    const cv::Point2f& center = std::get<3>(outputs)[i];
                    float score = std::get<1>(outputs)[i];

                    // Draw bounding box
                    cv::rectangle(bgrImage, box, cv::Scalar(255, 0, 0), 5);

                    // Draw keypoint (center)
                    cv::circle(bgrImage, center, 8, cv::Scalar(0, 0, 255), -1);

                    // Draw score text
                    std::string text = cv::format("%.2f", score);
                    cv::putText(bgrImage, text, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 2);
                }

                // --- Draw FPS in top-right ---
                float fps = 1000.0f / duration.count();  // duration is in milliseconds
                std::string fps_text = cv::format("FPS: %.1f", fps);
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 1.5, 2, &baseline);
                cv::Point textOrigin(bgrImage.cols - textSize.width - 20, textSize.height + 20);
                cv::putText(bgrImage, fps_text, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 0), 2);

                // Resize and show image
                cv::Mat resizedDisplay;
                cv::resize(bgrImage, resizedDisplay, cv::Size(), 0.5, 0.5);
                cv::imshow("Alpha Perception", resizedDisplay);
                cv::waitKey(1);
            }
        }
        catch(cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Cv Bridge error: %s", e.what());
        }
    }
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<alpha::Perception>());
    rclcpp::shutdown();

    return 0;
}
