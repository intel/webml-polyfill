// /*
//  * Copyright (C) 2017 The Android Open Source Project
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *      http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#include "Operations.h"
#include "CpuOperationUtils.h"

#include <algorithm>
#include <cmath>
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace nn {

inline bool l2normFloat32Impl(const float* inputData, const Shape& inputShape, int32_t axis,
                              float* outputData, const Shape& outputShape) {
    const uint32_t outerSize = getNumberOfElements(inputShape, 0, axis);
    const uint32_t axisSize = getSizeOfDimension(inputShape, axis);
    const uint32_t innerSize =
            getNumberOfElements(inputShape, axis + 1, getNumberOfDimensions(inputShape));
    for (uint32_t outer = 0; outer < outerSize; ++outer) {
        const float* inputBeg = inputData + outer * axisSize * innerSize;
        const float* inputEnd = inputBeg + axisSize * innerSize;
        float* outputBeg = outputData + outer * axisSize * innerSize;
        for (uint32_t inner = 0; inner < innerSize; ++inner, ++inputBeg, ++inputEnd, ++outputBeg) {
            float sum = 0.0f;
            for (const float* p = inputBeg; p < inputEnd; p += innerSize) {
                float val = *p;
                sum += val * val;
            }
            float l2_norm = std::sqrt(sum);
            float* pOut = outputBeg;
            for (const float* p = inputBeg; p < inputEnd; p += innerSize, pOut += innerSize) {
                *pOut = *p / l2_norm;
            }
        }
    }
    return true;
}

bool l2normFloat32(const float* inputData, const Shape& inputShape, int32_t axis, float* outputData,
                   const Shape& outputShape) {
    int32_t ndim = getNumberOfDimensions(inputShape);
    // TFLite optimized implementation only supports computation along the last axis
    if (axis == ndim - 1) {
        tflite::L2NormalizationParams param = {.input_zero_point = 0};
        tflite::optimized_ops::L2Normalization(param, convertShapeToTflshape(inputShape), inputData,
                                               convertShapeToTflshape(outputShape), outputData);
        return true;
    } else {
        return l2normFloat32Impl(inputData, inputShape, axis, outputData, outputShape);
    }
}

inline bool localResponseNormFloat32Impl(const float* inputData, const Shape& inputShape,
                                         int32_t radius, float bias, float alpha, float beta,
                                         int32_t axis, float* outputData,
                                         const Shape& outputShape) {
    const uint32_t outerSize = getNumberOfElements(inputShape, 0, axis);
    const uint32_t axisSize = getSizeOfDimension(inputShape, axis);
    const uint32_t innerSize =
            getNumberOfElements(inputShape, axis + 1, getNumberOfDimensions(inputShape));
    for (uint32_t outer = 0; outer < outerSize; ++outer) {
        const float* inputBase = inputData + outer * axisSize * innerSize;
        float* outputBase = outputData + outer * axisSize * innerSize;
        for (uint32_t inner = 0; inner < innerSize; ++inner, ++inputBase, ++outputBase) {
            for (int32_t i = 0; i < axisSize; i++) {
                const int32_t dBegin = std::max(0, i - radius);
                // Add 1 on dEnd to comply with optimized_ops in TFLite
                const int32_t dEnd = std::min(static_cast<int32_t>(axisSize), i + radius + 1);
                float sum = 0.0f;
                for (int32_t d = dBegin; d < dEnd; d++) {
                    float val = inputBase[d * innerSize];
                    sum += val * val;
                }
                float multiplier = std::pow(bias + alpha * sum, -beta);
                outputBase[i * innerSize] = inputBase[i * innerSize] * multiplier;
            }
        }
    }
    return true;
}

bool localResponseNormFloat32(const float* inputData, const Shape& inputShape, int32_t radius,
                              float bias, float alpha, float beta, int32_t axis, float* outputData,
                              const Shape& outputShape) {
    int32_t ndim = getNumberOfDimensions(inputShape);
    NN_CHECK(handleNegativeAxis(inputShape, &axis));
    // TFLite optimized implementation only supports computation along the last axis
    if (axis == ndim - 1) {
        tflite::LocalResponseNormalizationParams param = {
                .range = radius, .bias = bias, .alpha = alpha, .beta = beta};
        tflite::optimized_ops::LocalResponseNormalization(
                param, convertShapeToTflshape(inputShape), inputData,
                convertShapeToTflshape(outputShape), outputData);
        return true;
    } else {
        return localResponseNormFloat32Impl(inputData, inputShape, radius, bias, alpha, beta, axis,
                                            outputData, outputShape);
    }
}
}  // namespace nn
