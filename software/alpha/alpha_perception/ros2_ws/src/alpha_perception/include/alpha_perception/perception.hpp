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


#ifndef PERCEPTION_HPP_
#define PERCEPTION_HPP_

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>

// Header Msg
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/int16.hpp>

// Sensor Msgs
#include <sensor_msgs/msg/image.hpp>

// Custom Message Includes
#include "alpha_perception/msg/detection_array.hpp"

// CUDA TensorRT Includes
#include <cuda_runtime_api.h>
#include <NvInfer.h>

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
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr networkImagePub_;
        rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr inferenceTimePub_;
        void imageCallback_(const sensor_msgs::msg::Image & msg);

        // TensorRT and CUDA members
        nvinfer1::IRuntime* runtime_;
        nvinfer1::ICudaEngine* engine_;
        nvinfer1::IExecutionContext* context_;
        cudaStream_t stream_;

        // Host and device buffers
        std::vector<float> imageHostBuffer_;
        std::vector<float> outputHostBuffer_;
        void* imageDeviceBuffer_;
        void* outputDeviceBuffer_;

        // Buffer sizes
        int imageSize_;
        int outputSize_;

        // TensorRT functions
        bool loadTRTEngine_(const std::string& engineFilePath);
        std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> performInference_(const cv::Mat& inputMat);

        // Helper Functions
        int calculateVolume_(const nvinfer1::Dims& d);
        std::vector<float> matToVector_(const cv::Mat& mat);
        std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> postProcess_(Eigen::MatrixXf& data, float conf_thres = 0.25, float iou_thres = 0.15);
        std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>> scaleInferenceOutputs(const std::tuple<std::vector<cv::Rect>, std::vector<float>, Eigen::MatrixXf, std::vector<float>, std::vector<int>>& outputs, float scaleX, float scaleY);

        //GUI 
        cv::Mat annotateFrame(const cv::Mat& frame, const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, const Eigen::MatrixXf& kps, const std::vector<int>& classIds);
        void warmUpModel(const std::string& imagePath, int warmUpIterations);

        // Publisher for custom message
        std::shared_ptr<alpha_perception::msg::DetectionArray> prepareDetectionResultsMessage(const Eigen::MatrixXf& kps, const std::vector<float>& scores, const std::vector<cv::Rect>& boxes, const std::vector<float>& areas, const std::vector<int>& classIds,const std_msgs::msg::Header& header);
        std::shared_ptr<sensor_msgs::msg::Image> prepareNetworkImage(const cv::Mat & annotatedImg, const std_msgs::msg::Header& header);
        std::shared_ptr<std_msgs::msg::Int16> prepareInferenceTime(const int & inferceTime);

    public:
        Perception();
        ~Perception();
    };
}

#endif
