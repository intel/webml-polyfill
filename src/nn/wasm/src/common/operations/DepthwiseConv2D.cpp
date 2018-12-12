/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Operations.h"
#include "CpuOperationUtils.h"

#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/depthwiseconv_float.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h"

namespace nn {
#define ANDROID_NN_DEPTHWISE_CONV_PARAMETERS                                    \
    uint32_t height       = getSizeOfDimension(inputShape, 1);                  \
    uint32_t width        = getSizeOfDimension(inputShape, 2);                  \
    uint32_t filterHeight = getSizeOfDimension(filterShape, 1);                 \
    uint32_t filterWidth  = getSizeOfDimension(filterShape, 2);                 \
    uint32_t outHeight    = getSizeOfDimension(outputShape, 1);                 \
    uint32_t outWidth     = getSizeOfDimension(outputShape, 2);                 \
                                                                                \
    uint32_t paddingHeight = (uint32_t)padding_top;                             \
    uint32_t paddingWidth = (uint32_t)padding_left;

bool depthwiseConvFloat32(const float* inputData, const Shape& inputShape,
                          const float* filterData, const Shape& filterShape,
                          const float* biasData, const Shape& biasShape,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          int32_t depth_multiplier, int32_t activation,
                          float* outputData, const Shape& outputShape) {

    ANDROID_NN_DEPTHWISE_CONV_PARAMETERS

    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    int32_t dilationWidthFactor = 1, dilationHeightFactor = 1;

    tflite::optimized_ops::DepthwiseConv(
            inputData, convertShapeToDims(inputShape),
            filterData, convertShapeToDims(filterShape),
            biasData, convertShapeToDims(biasShape),
            stride_width, stride_height,
            dilationWidthFactor, dilationHeightFactor,
            paddingWidth, paddingHeight, depth_multiplier,
            output_activation_min, output_activation_max,
            outputData, convertShapeToDims(outputShape));

    return true;
}

#undef ANDROID_NN_DEPTHWISE_CONV_PARAMETERS
} // nn