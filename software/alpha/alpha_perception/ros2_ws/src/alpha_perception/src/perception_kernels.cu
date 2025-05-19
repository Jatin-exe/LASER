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

#include "perception_kernels.cuh"

namespace alpha 
{
    namespace perception_kernels 
    {
        // --- Kernel definition ---
        __global__ void preprocess_kernel(
            const uchar3* __restrict__ input,
            __half* __restrict__ output,
            int orig_w,
            int orig_h,
            int target_size,
            float ratio,
            int pad_w,
            int pad_h
        ) 
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= target_size || y >= target_size) return;

            // Map to original image space
            float in_x = (x - pad_w + 0.5f) / ratio - 0.5f;
            float in_y = (y - pad_h + 0.5f) / ratio - 0.5f;

            uchar3 pixel = {0, 0, 0};  // default black

            if (in_x >= 0 && in_x < orig_w - 1 && in_y >= 0 && in_y < orig_h - 1) 
            {
                int x0 = static_cast<int>(floorf(in_x));
                int x1 = x0 + 1;
                int y0 = static_cast<int>(floorf(in_y));
                int y1 = y0 + 1;

                float dx = in_x - x0;
                float dy = in_y - y0;

                uchar3 p00 = input[y0 * orig_w + x0];
                uchar3 p01 = input[y0 * orig_w + x1];
                uchar3 p10 = input[y1 * orig_w + x0];
                uchar3 p11 = input[y1 * orig_w + x1];

                // Bilinear interpolation per channel
                float r = (1 - dx) * (1 - dy) * p00.x + dx * (1 - dy) * p01.x +
                        (1 - dx) * dy * p10.x + dx * dy * p11.x;

                float g = (1 - dx) * (1 - dy) * p00.y + dx * (1 - dy) * p01.y +
                        (1 - dx) * dy * p10.y + dx * dy * p11.y;

                float b = (1 - dx) * (1 - dy) * p00.z + dx * (1 - dy) * p01.z +
                        (1 - dx) * dy * p10.z + dx * dy * p11.z;

                pixel = {
                    static_cast<unsigned char>(r),
                    static_cast<unsigned char>(g),
                    static_cast<unsigned char>(b)
                };
            }

            // Normalize to [0, 1]
            float r = static_cast<float>(pixel.x) / 255.0f;
            float g = static_cast<float>(pixel.y) / 255.0f;
            float b = static_cast<float>(pixel.z) / 255.0f;

            // NCHW layout
            int idx = y * target_size + x;
            output[0 * target_size * target_size + idx] = __float2half(r);  // R
            output[1 * target_size * target_size + idx] = __float2half(g);  // G
            output[2 * target_size * target_size + idx] = __float2half(b);  // B
        }

        // --- Host-callable wrapper function ---
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
        ) 
        {
            dim3 block(16, 16);
            dim3 grid((target_size + block.x - 1) / block.x, (target_size + block.y - 1) / block.y);

            preprocess_kernel<<<grid, block, 0, stream>>>(
                input,
                output,
                orig_w,
                orig_h,
                target_size,
                ratio,
                pad_w,
                pad_h
            );
        }

        __global__ void postprocess_kernel(
            const __half* __restrict__ boxes,    // [num * 4]
            const __half* __restrict__ scores,   // [num]
            const int64_t* __restrict__ labels,  // [num]
            int num_detections,
            float conf_thresh,
            float ratio,
            int pad_w,
            int pad_h,
            Detection* __restrict__ output,      // [num]
            int* __restrict__ valid_count        // output counter (single int)
        )
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_detections) return;

            float score = __half2float(scores[idx]);
            if (score < conf_thresh) return;

            // Load and convert bbox (x1, y1, x2, y2)
            float x1 = __half2float(boxes[idx * 4 + 0]);
            float y1 = __half2float(boxes[idx * 4 + 1]);
            float x2 = __half2float(boxes[idx * 4 + 2]);
            float y2 = __half2float(boxes[idx * 4 + 3]);

            // Undo padding and resizing
            float x = (x1 - pad_w) / ratio;
            float y = (y1 - pad_h) / ratio;
            float w = (x2 - x1) / ratio;
            float h = (y2 - y1) / ratio;

            if (w <= 0.f || h <= 0.f) return;

            // Calculate additional metadata
            float cx = x + w * 0.5f;
            float cy = y + h * 0.5f;
            float area = w * h;

            int out_idx = atomicAdd(valid_count, 1);
            output[out_idx] = Detection{ x, y, w, h, score, labels[idx], cx, cy, area};
        }

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
        )
        {
            dim3 block(256);
            dim3 grid((num_detections + block.x - 1) / block.x);

            postprocess_kernel<<<grid, block, 0, stream>>>(
                boxes,
                scores,
                labels,
                num_detections,
                conf_thresh,
                ratio,
                pad_w,
                pad_h,
                output,
                valid_count
            );
        }
    }
}
