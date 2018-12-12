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

#ifndef ANDROID_ML_NN_COMMON_CPU_OPERATION_UTILS_H
#define ANDROID_ML_NN_COMMON_CPU_OPERATION_UTILS_H

#include "OperationsUtils.h"

#include "external/tensorflow/tensorflow/lite/kernels/internal/types.h"

namespace nn {

// The implementations in tflite/kernels/internal/ take a Dims<4> object
// even if the original tensors were not 4D.
inline tflite::Dims<4> convertShapeToDims(const Shape& shape) {
  // nnAssert(shape.dimensions.size() <= 4);
  tflite::Dims<4> dims;

  // The dimensions are reversed in Dims<4>.
  for (int i = 0; i < 4; ++i) {
    int src = static_cast<int>(shape.dimensions.size()) - i - 1;
    if (src >= 0) {
      dims.sizes[i] = static_cast<int>(getSizeOfDimension(shape, src));
    } else {
      dims.sizes[i] = 1;
    }
  }

  dims.strides[0] = 1;
  for (int i = 1; i<4; i++) {
    dims.strides[i] = dims.strides[i-1] * dims.sizes[i-1];
  }
  return dims;
}

inline tflite::RuntimeShape convertShapeToTflshape(const Shape& shape) {
  // nnAssert(shape.dimensions.size() <= 4);

  std::vector<int32_t> tflShapeDim(shape.dimensions.begin(), shape.dimensions.end());
  return tflite::RuntimeShape(tflShapeDim.size(), tflShapeDim.data());
}
} // nn

#endif // ANDROID_ML_NN_COMMON_CPU_OPERATION_UTILS_H
