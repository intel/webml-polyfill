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

// Contains the implementation of the operations.

#define LOG_TAG "Operations"

#include "Operations.h"
#include "CpuOperationUtils.h"

#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/reference/legacy_reference_ops.h"

namespace nn {
inline void get4DShape(const Shape& shapeIn, uint32_t* shapeOut) {
    const int32_t kNumDims = 4;
    int32_t numDims = static_cast<int32_t>(getNumberOfDimensions(shapeIn));
    for (int32_t i = 0; i < numDims; i++) {
        shapeOut[i + kNumDims - numDims] = getSizeOfDimension(shapeIn, i);
    }
    for (int32_t i = 0; i < kNumDims - numDims; i++) {
        shapeOut[i] = 1;
    }
}

inline void getBroadcastStrides(const Shape& shapeIn, uint32_t* stridesOut) {
    const int32_t kNumDims = 4;
    uint32_t dims[kNumDims];
    get4DShape(shapeIn, dims);
    stridesOut[kNumDims - 1] = 1;
    for (int32_t i = kNumDims - 2; i >= 0; i--) {
        stridesOut[i] = stridesOut[i + 1] * dims[i + 1];
    }
    for (int32_t i = kNumDims - 1; i >= 0; i--) {
        if (dims[i] == 1) {
            stridesOut[i] = 0;
        }
    }
}

template <typename T>
inline bool broadcastOpBase(const T* in1, const Shape& shape1, const T* in2, const Shape& shape2,
                            T* out, const Shape& shapeOut,
                            std::function<T(const T&, const T&)> mathKernel) {
    // extend to 4D
    uint32_t outputDims[4], in1Strides[4], in2Strides[4];
    get4DShape(shapeOut, outputDims);
    getBroadcastStrides(shape1, in1Strides);
    getBroadcastStrides(shape2, in2Strides);

    T* outPtr = out;
    for (uint32_t b = 0; b < outputDims[0]; b++) {
        for (uint32_t h = 0; h < outputDims[1]; h++) {
            for (uint32_t w = 0; w < outputDims[2]; w++) {
                for (uint32_t c = 0; c < outputDims[3]; c++) {
                    T in1Val = in1[b * in1Strides[0] + h * in1Strides[1] + w * in1Strides[2] +
                                   c * in1Strides[3]];
                    T in2Val = in2[b * in2Strides[0] + h * in2Strides[1] + w * in2Strides[2] +
                                   c * in2Strides[3]];
                    *outPtr++ = mathKernel(in1Val, in2Val);
                }
            }
        }
    }
    return true;
}

#define ANDROID_NN_MACRO_DISPATCH(macro)                                    \
    switch (activation) {                                                   \
        case (int32_t) FusedActivationFunc::NONE:                           \
            macro(kNone);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU:                           \
            macro(kRelu);                                                   \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU1:                          \
            macro(kRelu1);                                                  \
            break;                                                          \
        case (int32_t) FusedActivationFunc::RELU6:                          \
            macro(kRelu6);                                                  \
            break;                                                          \
        default:                                                            \
            LOG(ERROR) << "Unsupported fused activation function type";     \
            return false;                                                   \
    }

bool addFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    bool needBroadcast = !SameShape(shape1, shape2);
    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    if (needBroadcast) {
        #define ANDROID_NN_BROADCAST_ADD(activation)                                              \
            tflite::optimized_ops::BroadcastAdd<tflite::FusedActivationFunctionType::activation>( \
                    in1, convertShapeToDims(shape1),                                              \
                    in2, convertShapeToDims(shape2),                                              \
                    out, convertShapeToDims(shapeOut))

        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_BROADCAST_ADD)
        #undef ANDROID_NN_BROADCAST_ADD
    } else {
        #define ANDROID_NN_ADD(activation)                                               \
            tflite::optimized_ops::Add<tflite::FusedActivationFunctionType::activation>( \
                    in1, convertShapeToDims(shape1),                                     \
                    in2, convertShapeToDims(shape2),                                     \
                    out, convertShapeToDims(shapeOut))

        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_ADD)
        #undef ANDROID_NN_ADD
    }

    return true;
}

bool mulFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    bool needBroadcast = !SameShape(shape1, shape2);

    if (needBroadcast) {
        #define ANDROID_NN_BROADCAST_MUL(activation)                                          \
        tflite::optimized_ops::BroadcastMul<tflite::FusedActivationFunctionType::activation>( \
                in1, convertShapeToDims(shape1),                                              \
                in2, convertShapeToDims(shape2),                                              \
                out, convertShapeToDims(shapeOut))

        ANDROID_NN_MACRO_DISPATCH(ANDROID_NN_BROADCAST_MUL)
        #undef ANDROID_NN_BROADCAST_MUL
    } else {
        float output_activation_min, output_activation_max;
        CalculateActivationRangeFloat(activation, &output_activation_min,
                                      &output_activation_max);

        tflite::optimized_ops::Mul(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
    }

    return true;
}

bool floorFloat32(const float* inputData,
                  float* outputData,
                  const Shape& shape) {
    tflite::Dims<4> dim = convertShapeToDims(shape);
    tflite::optimized_ops::Floor(inputData, dim, outputData, dim);
    return true;
}

bool subFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    tflite::optimized_ops::Sub(
            in1, convertShapeToDims(shape1),
            in2, convertShapeToDims(shape2),
            out, convertShapeToDims(shapeOut));
    return true;
}

bool divFloat32(const float* in1, const Shape& shape1,
                const float* in2, const Shape& shape2,
                int32_t activation,
                float* out, const Shape& shapeOut) {
    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    bool needBroadcast = !SameShape(shape1, shape2);
    if (needBroadcast) {
        tflite::optimized_ops::BroadcastDiv(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
    } else {
        tflite::optimized_ops::Div(
                in1, convertShapeToDims(shape1),
                in2, convertShapeToDims(shape2),
                output_activation_min, output_activation_max,
                out, convertShapeToDims(shapeOut));
    }
    return true;
}
} // nn
