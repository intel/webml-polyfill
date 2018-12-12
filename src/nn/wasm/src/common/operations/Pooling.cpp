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

#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace nn {
#define ANDROID_NN_POOLING_PARAMETERS                                      \
    tflite::PoolParams op_params;                                          \
    op_params.stride_height = stride_height;                               \
    op_params.stride_width = stride_width;                                 \
    op_params.filter_height = filter_height;                               \
    op_params.filter_width = filter_width;                                 \
    op_params.padding_values.height = padding_top;                         \
    op_params.padding_values.width = padding_left;



bool averagePoolFloat32(const float* inputData, const Shape& inputShape,
                        int32_t padding_left, int32_t padding_right,
                        int32_t padding_top, int32_t padding_bottom,
                        int32_t stride_width, int32_t stride_height,
                        int32_t filter_width, int32_t filter_height, int32_t activation,
                        float* outputData, const Shape& outputShape) {

    ANDROID_NN_POOLING_PARAMETERS

    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;

    tflite::optimized_ops::AveragePool(
            op_params,
            convertShapeToTflshape(inputShape), inputData,
            convertShapeToTflshape(outputShape), outputData);

    return true;
}

bool averagePoolQuant8(const uint8_t* inputData, const Shape& inputShape,
                       int32_t padding_left, int32_t padding_right,
                       int32_t padding_top, int32_t padding_bottom,
                       int32_t stride_width, int32_t stride_height,
                       int32_t filter_width, int32_t filter_height, int32_t activation,
                       uint8_t* outputData, const Shape& outputShape) {

    ANDROID_NN_POOLING_PARAMETERS

    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;

    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

    op_params.quantized_activation_min = output_activation_min;
    op_params.quantized_activation_max = output_activation_max;

    tflite::optimized_ops::AveragePool(
            op_params,
            convertShapeToTflshape(inputShape), inputData,
            convertShapeToTflshape(outputShape), outputData);

    return true;
}

bool l2PoolFloat32(const float* inputData, const Shape& inputShape,
                   int32_t padding_left, int32_t padding_right,
                   int32_t padding_top, int32_t padding_bottom,
                   int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
                   float* outputData, const Shape& outputShape) {

    ANDROID_NN_POOLING_PARAMETERS

    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;

    tflite::optimized_ops::L2Pool(
            op_params,
            convertShapeToTflshape(inputShape), inputData,
            convertShapeToTflshape(outputShape), outputData);

    return true;
}

bool maxPoolFloat32(const float* inputData, const Shape& inputShape,
                    int32_t padding_left, int32_t padding_right,
                    int32_t padding_top, int32_t padding_bottom,
                    int32_t stride_width, int32_t stride_height,
                    int32_t filter_width, int32_t filter_height, int32_t activation,
                    float* outputData, const Shape& outputShape) {

    ANDROID_NN_POOLING_PARAMETERS

    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    op_params.float_activation_min = output_activation_min;
    op_params.float_activation_max = output_activation_max;

    tflite::optimized_ops::MaxPool(
            op_params,
            convertShapeToTflshape(inputShape), inputData,
            convertShapeToTflshape(outputShape), outputData);

    return true;
}

bool maxPoolQuant8(const uint8_t* inputData, const Shape& inputShape,
                   int32_t padding_left, int32_t padding_right,
                   int32_t padding_top, int32_t padding_bottom,
                   int32_t stride_width, int32_t stride_height,
                   int32_t filter_width, int32_t filter_height, int32_t activation,
                   uint8_t* outputData, const Shape& outputShape) {

    ANDROID_NN_POOLING_PARAMETERS

    int32_t output_activation_min = 0;
    int32_t output_activation_max = 0;

    CalculateActivationRangeUint8(activation, outputShape,
                                  &output_activation_min,
                                  &output_activation_max);

    op_params.quantized_activation_min = output_activation_min;
    op_params.quantized_activation_max = output_activation_max;

    tflite::optimized_ops::MaxPool(
            op_params,
            convertShapeToTflshape(inputShape), inputData,
            convertShapeToTflshape(outputShape), outputData);

    return true;
}

#undef ANDROID_NN_POOLING_PARAMETERS
} // nn
