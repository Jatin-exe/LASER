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

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace alpha 
{
    namespace perception_kernels 
    {
        // Structure to store final GPU-processed detection
        struct Detection {
            float x, y, w, h;
            float score;
            int64_t label;
            float cx, cy;
            float area;
        };

        /**
         * @brief CUDA kernel to preprocess an RGB image into a normalized NCHW __half tensor.
         *
         * This kernel performs the following operations:
         * - Resizes the input RGB image using bilinear interpolation
         * - Pads it to the center of a square canvas of size (target_size x target_size)
         * - Normalizes pixel values from [0, 255] to [0.0, 1.0]
         * - Converts the image layout from HWC (RGB) to NCHW (__half) for inference
         *
         * @param input         Input image in RGB format as uchar3*, with dimensions (orig_h x orig_w)
         * @param output        Output NCHW tensor as __half*, with shape (3 x target_size x target_size)
         * @param orig_w        Original image width
         * @param orig_h        Original image height
         * @param target_size   Final square output resolution (e.g., 640)
         * @param ratio         Resize ratio (computed on host as min(target_size / orig_w, target_size / orig_h))
         * @param pad_w         Horizontal offset after padding (left side)
         * @param pad_h         Vertical offset after padding (top side)
         */
        __global__ void preprocess_kernel(
            const uchar3* __restrict__ input,
            __half* __restrict__ output,
            int orig_w,
            int orig_h,
            int target_size,
            float ratio,
            int pad_w,
            int pad_h
        );

        /**
         * @brief Host wrapper to launch the CUDA preprocess kernel from C++ code.
         *
         * @param input         Device pointer to uchar3 input image (RGB)
         * @param output        Device pointer to __half NCHW output buffer
         * @param orig_w        Original image width
         * @param orig_h        Original image height
         * @param target_size   Target canvas size (e.g., 640)
         * @param ratio         Resize ratio
         * @param pad_w         Padding in width
         * @param pad_h         Padding in height
         * @param stream        CUDA stream to launch the kernel on
         */
        void launch_preprocess_kernel(
            const uchar3* input,
            __half* output,
            int orig_w,
            int orig_h,
            int target_size,
            float ratio,
            int pad_w,
            int pad_h,
            cudaStream_t stream
        );

        /**
         * @brief CUDA kernel to convert raw model outputs into final scaled detections.
         *
         * This does threshold filtering and coordinate unpadding/rescaling.
         *
         * @param boxes         [num_detections * 4] box coords (x1, y1, x2, y2) in __half
         * @param scores        [num_detections] scores in __half
         * @param labels        [num_detections] labels in int64_t
         * @param num_detections Total number of candidate detections
         * @param conf_thresh   Minimum score to keep
         * @param ratio         Resize ratio from original image
         * @param pad_w         Horizontal padding
         * @param pad_h         Vertical padding
         * @param output        Final filtered Detection buffer (output)
         * @param valid_count   Output: number of detections written (atomic counter)
         */
        __global__ void postprocess_kernel(
            const __half* __restrict__ boxes,
            const __half* __restrict__ scores,
            const int64_t* __restrict__ labels,
            int num_detections,
            float conf_thresh,
            float ratio,
            int pad_w,
            int pad_h,
            Detection* __restrict__ output,
            int* __restrict__ valid_count
        );

        /**
         * @brief Host wrapper to launch the CUDA postprocess kernel from C++ code.
         *
         * @param boxes         Device pointer to __half* boxes [num_detections * 4]
         * @param scores        Device pointer to __half* scores [num_detections]
         * @param labels        Device pointer to int64_t* labels [num_detections]
         * @param num_detections Total number of candidates to process
         * @param conf_thresh   Minimum confidence threshold to keep
         * @param ratio         Resize ratio from original image
         * @param pad_w         Horizontal pad (left)
         * @param pad_h         Vertical pad (top)
         * @param output        Device pointer to output Detection* buffer
         * @param valid_count   Device pointer to int (atomic counter initialized to 0)
         * @param stream        CUDA stream
         */
        void launch_postprocess_kernel(
            const __half* boxes,
            const __half* scores,
            const int64_t* labels,
            int num_detections,
            float conf_thresh,
            float ratio,
            int pad_w,
            int pad_h,
            Detection* output,
            int* valid_count,
            cudaStream_t stream
        );
    }
}
