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
#include "float.h"

#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace nn {
bool reluFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::max(0.f, *inputData);
    }
    return true;
}

bool relu1Float32(const float* inputData, const Shape& inputShape,
                  float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::min(std::max(-1.f, *inputData), 1.f);
    }
    return true;
}

bool relu6Float32(const float* inputData, const Shape& inputShape,
                  float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::min(std::max(0.f, *inputData), 6.f);
    }
    return true;
}

bool tanhFloat32(const float* inputData, const Shape& inputShape,
                 float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = std::tanh(*inputData);
    }
    return true;
}

bool logisticFloat32(const float* inputData, const Shape& inputShape,
                     float* outputData, const Shape& outputShape) {
    int numElements = getNumberOfElements(inputShape);
    for (int i=0; i<numElements; i++, inputData++, outputData++) {
        *outputData = 1.f / (1.f + std::exp(-*inputData));
    }
    return true;
}

inline bool softmaxFloat32Impl(const float* inputData, const Shape& inputShape, const float beta,
                               int32_t axis, float* outputData, const Shape& outputShape) {
    const uint32_t outerSize = getNumberOfElements(inputShape, 0, axis);
    const uint32_t axisSize = getSizeOfDimension(inputShape, axis);
    const uint32_t innerSize =
            getNumberOfElements(inputShape, axis + 1, getNumberOfDimensions(inputShape));
    for (uint32_t outer = 0; outer < outerSize; ++outer) {
        const float* inputBeg = inputData + outer * axisSize * innerSize;
        const float* inputEnd = inputBeg + axisSize * innerSize;
        float* outputBeg = outputData + outer * axisSize * innerSize;
        for (uint32_t inner = 0; inner < innerSize; ++inner, ++inputBeg, ++inputEnd, ++outputBeg) {
            // Find max
            float maxValue = -FLT_MAX;
            for (const float* p = inputBeg; p < inputEnd; p += innerSize) {
                maxValue = std::max(maxValue, *p);
            }
            // Compute sum
            float sum = 0.0f;
            for (const float* p = inputBeg; p < inputEnd; p += innerSize) {
                sum += std::exp((*p - maxValue) * beta);
            }
            // Compute result
            float* pOut = outputBeg;
            for (const float* p = inputBeg; p < inputEnd; p += innerSize, pOut += innerSize) {
                *pOut = std::exp((*p - maxValue) * beta) / sum;
            }
        }
    }
    return true;
}

bool softmaxFloat32(const float* inputData, const Shape& inputShape, const float beta,
                    float* outputData, const Shape& outputShape) {
    int32_t ndim = getNumberOfDimensions(inputShape);
    int32_t axis = ndim - 1;
    NN_CHECK(handleNegativeAxis(inputShape, &axis));
    // TFLite optimized implementation only supports computation along the last axis
    if (axis == ndim - 1) {
        tflite::SoftmaxParams param = {.beta = beta};
        tflite::optimized_ops::Softmax(param, convertShapeToTflshape(inputShape), inputData,
                                       convertShapeToTflshape(outputShape), outputData);
        return true;
    } else {
        return softmaxFloat32Impl(inputData, inputShape, beta, axis, outputData, outputShape);
    }
}
} // nn