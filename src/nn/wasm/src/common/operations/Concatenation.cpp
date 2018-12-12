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

#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h"

namespace nn {
bool concatenationFloat32(const std::vector<const float*>& inputDataPtrs,
                          const std::vector<Shape>& inputShapes, int32_t axis,
                          float* outputData, const Shape& outputShape) {
    int num_inputs = inputShapes.size();
    std::vector<tflite::Dims<4>*> inputDimsPtr(num_inputs);
    std::vector<tflite::Dims<4> > inputDims(num_inputs);
    for (int i=0; i<num_inputs; i++) {
        inputDims[i] = convertShapeToDims(inputShapes[i]);
        inputDimsPtr[i] = &inputDims[i];
    }

    tflite::optimized_ops::Concatenation<tflite::FusedActivationFunctionType::kNone, float>(
            getNumberOfDimensions(outputShape) - axis - 1,
            inputDataPtrs.data(), inputDimsPtr.data(), num_inputs,
            outputData, convertShapeToDims(outputShape));

    return true;
}
} // nn