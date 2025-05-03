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

    Perception::Perception() : Node("alpha_perception")
    {
        // Initialize CUDA stream
        cudaStreamCreate(&(this->stream_));

        if (!this->loadTRTEngine_("/path/to/your/trt/engine")) 
        {
            throw std::runtime_error("Failed to load TensorRT engine");
        }

        // Initialize input and output buffers
        this->imageHostBuffer_.resize(this->imageSize_ / sizeof(float));
        this->outputHostBuffer_.resize(this->outputSize_ / sizeof(float));

        // Allocate GPU memory for input and output
        cudaMalloc(&(this->imageDeviceBuffer_), this->imageSize_);
        cudaMalloc(&(this->outputDeviceBuffer_), this->outputSize_);

        // Create CUDA stream
        cudaStreamCreate(&(this->stream_));

        // Model Warm up
        //this->warmUpModel("/path/to/test.png(or)jpg", 65);

        // ROS Subscriber
        this->imageDataSub_ = this->create_subscription<sensor_msgs::msg::Image>("/your/camera/topic", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile(), std::bind(&Perception::imageCallback_, this, std::placeholders::_1));
        this->detectionArrayPub_ = this->create_publisher<alpha_perception::msg::DetectionArray>("/alpha_perception/detections", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile());
        this->networkImagePub_ = this->create_publisher<sensor_msgs::msg::Image>("/alpha_perception/detection_image", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile());
        this->inferenceTimePub_ = this->create_publisher<std_msgs::msg::Int16>("/alpha_perception/inference_time", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile());
    
    }

    Perception::~Perception()
    {
        // Free GPU memory
        cudaFree(this->imageDeviceBuffer_);
        cudaFree(this->outputDeviceBuffer_);

        // Destroy TensorRT objects
        delete this->context_;
        delete this->engine_;
        delete this->runtime_;
    }

    void Perception::warmUpModel(const std::string& imagePath, int warmUpIterations)
    {
        // Load the image from the specified path
        cv::Mat image = cv::imread(imagePath);

        // Resize the image to the input size expected by the neural network
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(640, 384));

        // Convert the image to float and normalize
        cv::Mat floatImage;
        resizedImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

        // Convert to blob format for neural network
        cv::Mat blob = cv::dnn::blobFromImage(floatImage, 1.0, cv::Size(), cv::Scalar(), false, false);

        for (int i = 0; i < warmUpIterations; ++i) auto _ = this->performInference_(blob);

        RCLCPP_INFO(this->get_logger(), "Warm-up completed.");
    }

    bool Perception::loadTRTEngine_(const std::string& engineFilePath)
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
        this->runtime_ = nvinfer1::createInferRuntime(gLogger);
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

        // Get input and output tensor names
        const char* inputName = this->engine_->getIOTensorName(0);
        const char* outputName = this->engine_->getIOTensorName(1);

        // Get tensor shapes
        nvinfer1::Dims inputDims = this->engine_->getTensorShape(inputName);
        nvinfer1::Dims outputDims = this->engine_->getTensorShape(outputName);

        // Calculate buffer sizes (assumes float32)
        this->imageSize_ = this->calculateVolume_(inputDims) * sizeof(float);
        this->outputSize_ = this->calculateVolume_(outputDims) * sizeof(float);

        // Resize host buffers
        this->imageHostBuffer_.resize(this->imageSize_ / sizeof(float));
        this->outputHostBuffer_.resize(this->outputSize_ / sizeof(float));

        // Allocate device memory
        if (cudaMalloc(&this->imageDeviceBuffer_, this->imageSize_) != cudaSuccess ||
            cudaMalloc(&this->outputDeviceBuffer_, this->outputSize_) != cudaSuccess)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate device memory");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Loaded Engine file %s", engineFilePath.c_str());
        return true;
    }

    int Perception::calculateVolume_(const nvinfer1::Dims& d)
    {
        int vol = 1;
        for (int i = 0; i < d.nbDims; ++i) vol *= d.d[i];
        return vol;
    }

    std::vector<float> Perception::matToVector_(const cv::Mat& mat)
    {
        std::vector<float> vec;
        vec.assign((float*)mat.datastart, (float*)mat.dataend);
        return vec;
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> Perception::performInference_(const cv::Mat& inputMat)
    {
        // Creating the input
        std::vector<float> inputImage = this->matToVector_(inputMat);
        
        // Copy input from Mat to the input device buffer
        cudaMemcpyAsync(
                this->imageDeviceBuffer_, 
                inputImage.data(), 
                this->imageSize_, 
                cudaMemcpyHostToDevice, 
                this->stream_
        );

        // Set Tensor Addresses (REQUIRED for enqueueV3)
        const char* inputName  = this->engine_->getIOTensorName(0);
        const char* outputName = this->engine_->getIOTensorName(1);
    
        this->context_->setTensorAddress(inputName, this->imageDeviceBuffer_);
        this->context_->setTensorAddress(outputName, this->outputDeviceBuffer_);
    
        // Run inference
        if (!this->context_->enqueueV3(this->stream_))
        {
            RCLCPP_ERROR(this->get_logger(), "enqueueV3() failed");
            throw std::runtime_error("TensorRT inference failed");
        }
        
        // Copy the output from the output device buffer to the output host buffer
        cudaMemcpyAsync(
            this->outputHostBuffer_.data(), 
            this->outputDeviceBuffer_, 
            this->outputSize_, 
            cudaMemcpyDeviceToHost, 
            this->stream_
        );

        // Wait for all CUDA operations to finish
        cudaStreamSynchronize(this->stream_);

        // Map the vector to a 2D Eigen Matrix
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> outputMatrix(this->outputHostBuffer_.data(), 7, 5040); // For just one single class in this case weeds
        Eigen::MatrixXf matrixXfOutput = outputMatrix;
        std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> outputFromNetwork = this->postProcess_(matrixXfOutput);
        
        return outputFromNetwork;
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> Perception::postProcess_(Eigen::MatrixXf& data, float conf_thres, float iou_thres)
    {
        // Transpose the matrix for easier row-wise access
        Eigen::MatrixXf transposedData = data.transpose();

        // Prepare containers for filtered data
        std::vector<cv::Rect> cvRects;
        std::vector<float> filteredScores;
        std::vector<int> classIds;
        std::vector<float> areas;
        int numData = transposedData.rows();
        cvRects.reserve(numData);
        filteredScores.reserve(numData);
        classIds.reserve(numData);
        areas.reserve(numData);

        Eigen::MatrixXf filteredKpts(numData, 2);

        int count = 0;

        // Single loop for data processing
        for (int i = 0; i < numData; ++i)
        {
            float centerX = transposedData(i, 0);
            float centerY = transposedData(i, 1);
            float width = transposedData(i, 2);
            float height = transposedData(i, 3);
            int x = static_cast<int>(centerX - 0.5 * width);
            int y = static_cast<int>(centerY - 0.5 * height);

            float weed_scores = transposedData(i, 4);
            //float crop_scores = transposedData(i, 5);

            if (weed_scores > conf_thres) 
            {
                cvRects.emplace_back(x, y, static_cast<int>(width), static_cast<int>(height));
                areas.push_back(width * height);
                filteredScores.push_back(weed_scores);
                classIds.push_back(0); // Class ID for weed
                filteredKpts.row(count++) = transposedData.block<1, 2>(i, 5);
            }
            /*
            else if (crop_scores > conf_thres) 
            {
                cvRects.emplace_back(x, y, static_cast<int>(width), static_cast<int>(height));
                areas.push_back(width * height);
                filteredScores.push_back(crop_scores);
                classIds.push_back(1); // Class ID for crop
                filteredKpts.row(count++) = transposedData.block<1, 2>(i, 6);
            }
            */
        }

        // Resize keypoints matrix to the actual count
        filteredKpts.conservativeResize(count, Eigen::NoChange);

        // Perform Non-Maximum Suppression (NMS)
        std::vector<int> nmsIndices;
        if (!cvRects.empty()) cv::dnn::NMSBoxes(cvRects, filteredScores, conf_thres, iou_thres, nmsIndices);

        // Allocate memory for the final results
        Eigen::MatrixXf finalKpts(nmsIndices.size(), 2);
        std::vector<cv::Rect> finalBoxes;
        std::vector<float> finalScores;
        std::vector<float> finalAreas;
        std::vector<int> finalClassIds;

        finalBoxes.reserve(nmsIndices.size());
        finalScores.reserve(nmsIndices.size());
        finalAreas.reserve(nmsIndices.size());
        finalClassIds.reserve(nmsIndices.size());

        for (int idx : nmsIndices)
        {
            finalBoxes.push_back(cvRects[idx]);
            finalScores.push_back(filteredScores[idx]);
            finalAreas.push_back(areas[idx]);
            finalKpts.row(finalBoxes.size() - 1) = filteredKpts.row(idx);
            finalClassIds.push_back(classIds[idx]);
        }

        return std::make_tuple(finalBoxes, finalScores, finalKpts, finalAreas, finalClassIds);
    }

    std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> Perception::scaleInferenceOutputs(const std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>>& outputs, float scaleX, float scaleY)
    {
        // Explicitly unpack the outputs tuple
        const std::vector<cv::Rect>& boxes = std::get<0>(outputs);
        const std::vector<float>& scores = std::get<1>(outputs);
        Eigen::MatrixXf kps = std::get<2>(outputs);
        std::vector<float> areas = std::get<3>(outputs);
        std::vector<int> classIds = std::get<4>(outputs);

        // Containers for scaled outputs
        std::vector<cv::Rect> scaledBoxes;
        Eigen::MatrixXf scaledKps(kps.rows(), kps.cols());
        std::vector<float> scaledAreas;

        // Scale bounding boxes
        for (size_t i = 0; i < boxes.size(); ++i) 
        {
            int x = static_cast<int>(boxes[i].x * scaleX);
            int y = static_cast<int>(boxes[i].y * scaleY);
            int width = static_cast<int>(boxes[i].width * scaleX);
            int height = static_cast<int>(boxes[i].height * scaleY);
            scaledBoxes.emplace_back(cv::Rect(x, y, width, height));
        }

        // Scale keypoints
        for (int i = 0; i < kps.rows(); ++i) 
        {
            scaledKps(i, 0) = kps(i, 0) * scaleX;  // Scale X coordinate
            scaledKps(i, 1) = kps(i, 1) * scaleY;  // Scale Y coordinate
        }

        // Scale areas - assuming uniform scaling for simplicity
        scaledAreas.reserve(areas.size());
        for (size_t i = 0; i < areas.size(); ++i) scaledAreas.push_back(areas[i] * (scaleX * scaleY));

        return std::make_tuple(scaledBoxes, scores, scaledKps, scaledAreas, classIds);
    }

    cv::Mat Perception::annotateFrame(const cv::Mat& frame, const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, const Eigen::MatrixXf& kps, const std::vector<int>& classIds)
    {
        // Create a copy of the frame to draw on
        cv::Mat annotatedFrame = frame.clone();

        for (size_t i = 0; i < boxes.size(); ++i) 
        {
            const cv::Rect& box = boxes[i];
            float score = scores[i]; // Assuming you have scores ready to use

            // Drawing the bounding box
            cv::Scalar boxColor(0, 0, 255);
            int boxThickness = 3;
            cv::rectangle(annotatedFrame, box, boxColor, boxThickness);

            // Drawing keypoints associated with this bounding box
            if (i < static_cast<size_t>(kps.rows())) cv::circle(annotatedFrame, cv::Point(static_cast<int>(kps(i, 0)), static_cast<int>(kps(i, 1))), 5, cv::Scalar(0, 0, 255), -1);

            // Preparing text to display (score and area)
            std::stringstream ss;
            if (classIds[i] == 0) ss << std::fixed << std::setprecision(2) << "Weed Score:" << score;
            else if (classIds[i] == 1) ss << std::fixed << std::setprecision(2) << "Crop Score:" << score;
            std::string text = ss.str();

            // Calculating text size to position it inside or above the box
            int baseLine;
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            // Drawing text background for better contrast
            cv::rectangle(annotatedFrame, cv::Point(box.x, box.y - textSize.height - baseLine - 3), cv::Point(box.x + textSize.width, box.y), cv::Scalar(0, 0, 0), -1);

            // Displaying the text
            cv::putText(annotatedFrame, text, cv::Point(box.x, box.y - baseLine), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        return annotatedFrame;
    }

    std::shared_ptr<alpha_perception::msg::DetectionArray> Perception::prepareDetectionResultsMessage(const Eigen::MatrixXf& kps, const std::vector<float>& scores, const std::vector<cv::Rect>& boxes, const std::vector<float>& areas, const std::vector<int>& classIds, const std_msgs::msg::Header& header)
    {
        std::shared_ptr<alpha_perception::msg::DetectionArray> detection_array_msg = std::make_shared<alpha_perception::msg::DetectionArray>();

        // Set the message header
        detection_array_msg->header = header;  // Set the passed header

        for (size_t i = 0; i < boxes.size(); ++i) 
        {
            if (classIds[i] == 0)
            {
                alpha_perception::msg::BoundingBox box_msg;
                
                box_msg.x = boxes[i].x;
                box_msg.y = boxes[i].y;
                box_msg.width = boxes[i].width;
                box_msg.height = boxes[i].height;

                box_msg.score = scores[i];
                box_msg.area = areas[i];

                // Add only one keypoint per object
                alpha_perception::msg::Keypoint kp_msg;
                kp_msg.x = kps(i, 0);
                kp_msg.y = kps(i, 1);
                box_msg.keypoint = kp_msg;

                detection_array_msg->boxes.push_back(box_msg);
            }
        }

        return detection_array_msg;
    }

    std::shared_ptr<sensor_msgs::msg::Image> Perception::prepareNetworkImage(const cv::Mat & annotatedImg, const std_msgs::msg::Header& header)
    {
        cv::Mat outImg;
        cv::resize(annotatedImg, outImg, cv::Size(), 0.8, 0.8);
        cv_bridge::CvImage cvImage(header, sensor_msgs::image_encodings::RGB8, outImg);

        std::shared_ptr<sensor_msgs::msg::Image> out_msg = std::make_shared<sensor_msgs::msg::Image>();
        cvImage.toImageMsg(*out_msg);

        return out_msg;
    }

    std::shared_ptr<std_msgs::msg::Int16> Perception::prepareInferenceTime(const int & inferceTime)
    {
        std::shared_ptr<std_msgs::msg::Int16> out_msg = std::make_shared<std_msgs::msg::Int16>();
        out_msg->data = inferceTime;

        return out_msg;
    }


    void Perception::imageCallback_(const sensor_msgs::msg::Image & msg)
    {
        try
        {
            //Timer start
            std::chrono::_V2::system_clock::time_point start = std::chrono::high_resolution_clock::now();
            
            // Reading Image
            cv::Mat frame = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8)->image;
            // Original dimensions before resizing
            int originalWidth = frame.cols;
            int originalHeight = frame.rows;

            // Dimensions after resizing for processing
            int processedWidth = 640;
            int processedHeight = 384;

            // Calculate scale factors
            double scaleX = (double)originalWidth / processedWidth;
            double scaleY = (double)originalHeight / processedHeight;

            cv::Mat resizedFrame;
            cv::resize(frame, resizedFrame, cv::Size(640, 384));

            // Convert to float and normalize (Need this step to increase speed though it makes no sense cause doing it inside blob function, makes the preporcessing slower)
            cv::Mat floatFrame;
            resizedFrame.convertTo(floatFrame, CV_32FC3, 1.0 / 255.0);

            // Convert to blob format for neural network
            cv::Mat blob = cv::dnn::blobFromImage(floatFrame, 1.0, cv::Size(), cv::Scalar(), false, false);

            // Inference and outputs
            std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> outputs = this->scaleInferenceOutputs(this->performInference_(blob), scaleX, scaleY);
            Eigen::MatrixXf kps = std::get<2>(outputs);
            std::vector<float> scores = std::get<1>(outputs);
            std::vector<cv::Rect> boxes = std::get<0>(outputs);
            std::vector<float> areas = std::get<3>(outputs);
            std::vector<int> classIds = std::get<4>(outputs);

            //Timer stop plus durations in ms
            std::chrono::_V2::system_clock::time_point stop = std::chrono::high_resolution_clock::now();
            std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

            this->detectionArrayPub_->publish(*(this->prepareDetectionResultsMessage(kps, scores, boxes, areas, classIds, msg.header)));

            //Publish ROS message
            this->networkImagePub_->publish(*(this->prepareNetworkImage(this->annotateFrame(frame, boxes, scores, kps, classIds), msg.header)));
            this->inferenceTimePub_->publish(*(this->prepareInferenceTime(duration.count())));
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
