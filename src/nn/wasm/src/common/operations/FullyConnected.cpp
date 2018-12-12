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
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/reference/reference_ops.h"

namespace nn {
bool fullyConnectedFloat32(const float* inputData, const Shape& inputShape,
                           const float* weightsData, const Shape& weightsShape,
                           const float* biasData, const Shape& biasShape,
                           int32_t activation,
                           float* outputData, const Shape& outputShape) {
    float output_activation_min, output_activation_max;
    CalculateActivationRangeFloat(activation, &output_activation_min,
                                  &output_activation_max);

    // b/80425683, optimized implementation produces incorrect results when the
    // number of input elements is the squre of batch_size.
    uint32_t batch_size = getSizeOfDimension(outputShape, 0);
    uint32_t input_n_elements = getNumberOfElements(inputShape);
    if (batch_size * batch_size == input_n_elements) {
        tflite::reference_ops::FullyConnected(
                inputData, convertShapeToDims(inputShape),
                weightsData, convertShapeToDims(weightsShape),
                biasData, convertShapeToDims(biasShape),
                output_activation_min, output_activation_max,
                outputData, convertShapeToDims(outputShape));
    } else {
        tflite::optimized_ops::FullyConnected(
                inputData, convertShapeToDims(inputShape),
                weightsData, convertShapeToDims(weightsShape),
                biasData, convertShapeToDims(biasShape),
                output_activation_min, output_activation_max,
                outputData, convertShapeToDims(outputShape));
    }
    return true;
}
} // nn