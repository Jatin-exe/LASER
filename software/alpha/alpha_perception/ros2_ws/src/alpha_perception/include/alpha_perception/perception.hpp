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

#pragma once

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>

// Header Msg
#include <std_msgs/msg/header.hpp>

// Int 16 Msg
#include <std_msgs/msg/int16.hpp>

// Sensor Msgs
#include <sensor_msgs/msg/image.hpp>

// Custom Message Includes
#include "alpha_perception/msg/detection_array.hpp"

// CUDA TensorRT Includes
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <cuda_fp16.h>

// Eigen3 Includes
#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif

// OpenCV
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/eigen.hpp>

// Includes
#include <iostream>
#include <fstream> 
#include <memory>

// Kernels
# include "perception_kernels.cuh"

namespace alpha 
{
    class Logger : public nvinfer1::ILogger
    {
    private:
        rclcpp::Logger rosLogger_;
    public:
        Logger() : rosLogger_(rclcpp::get_logger("alpha_perception")) {}
        void log(Severity severity, const char* msg) noexcept override 
        {
            if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) 
            {
                RCLCPP_ERROR(this->rosLogger_, "NvInfer error: %s", msg);
            }
        }
    } gLogger;

    class Perception : public rclcpp::Node
    {
        private:
            // ROS Functions
            rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageDataSub_;
            rclcpp::Publisher<alpha_perception::msg::DetectionArray>::SharedPtr detectionArrayPub_;
            rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr inferenceTimePub_;
            void imageCallback_(const sensor_msgs::msg::Image::UniquePtr & msg);

            // TensorRT and CUDA members
            nvinfer1::IRuntime* runtime_ = nullptr;
            nvinfer1::ICudaEngine* engine_ = nullptr;
            nvinfer1::IExecutionContext* context_ = nullptr;

            // TensorRT loader
            bool loadTRTEngine_(const std::string& engineFilePath);

            // CUDA stream
            cudaStream_t stream_ = nullptr;

            // Dynamic buffers
            std::unordered_map<std::string, void*> deviceBuffers_;
            std::unordered_map<std::string, std::vector<__half>> hostFP16_;
            std::unordered_map<std::string, std::vector<int64_t>> hostInt64_;

            // Pinned Memory
            uchar3* pinnedHostInput_ = nullptr;
            uchar3* deviceInput_ = nullptr;
            size_t maxInputSize_ = 1920 * 1088;

            // Persistent GPU memory for postprocessing
            alpha::perception_kernels::Detection* deviceDetections_ = nullptr;
            int* deviceValidCount_ = nullptr;

            // Persistent CPU memory to avoid per-frame allocation
            std::vector<alpha::perception_kernels::Detection> hostDetections_;

            // Setup function
            bool setupBuffers_();

            // Calcuate for dynamic buffers
            int calculateVolume_(const nvinfer1::Dims& dims);

            // Preprocess
            std::tuple<float, int, int> preprocessImage_(const cv::Mat& input);

            // Forward pass
            void runInference_();

            // PostProcess
            std::tuple<std::vector<cv::Rect2f>, std::vector<float>, std::vector<int64_t>, std::vector<cv::Point2f>, std::vector<float>> postprocessDetections_(float ratio, int pad_w, int pad_h, float conf_thresh = 0.5f);

            // Custom message 
            std::shared_ptr<alpha_perception::msg::DetectionArray> prepareDetectionResultsMessage_(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, const std::vector<int64_t>& labels, const std::vector<cv::Point2f>& centers, const std::vector<float>& areas, const std_msgs::msg::Header& header);

            // Inference Time
            std::shared_ptr<std_msgs::msg::Int16> prepareInferenceTime_(const int & inferceTime);

            // Vis
            bool gui_ = false;
        public:
            Perception();
            ~Perception();
    };
}