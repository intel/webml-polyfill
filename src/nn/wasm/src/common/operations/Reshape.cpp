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
#include "external/tensorflow/tensorflow/lite/kernels/internal/reference/reference_ops.h"

namespace nn {
bool reshapeGeneric(const void* inputData, const Shape& inputShape,
                    void* outputData, const Shape& outputShape) {
    size_t count = sizeOfData(inputShape.type, inputShape.dimensions);
    memcpy(outputData, inputData, count);
    return true;
}

bool resizeBilinearFloat32(const float* inputData, const Shape& inputShape,
                           float* outputData, const Shape& outputShape) {
    int32_t height = (int32_t) getSizeOfDimension(outputShape, 1);
    int32_t width  = (int32_t) getSizeOfDimension(outputShape, 2);

    int32_t outDimData[2] = {height, width};
    // We have to fake a tensor here, to satisfy ResizeBilinear().
    Shape outDimShape;
    outDimShape.dimensions = {1, 1, 1, 2};

    tflite::optimized_ops::ResizeBilinear(
            inputData, convertShapeToDims(inputShape),
            outDimData, convertShapeToDims(outDimShape),
            outputData, convertShapeToDims(outputShape));
    return true;
}

bool depthToSpaceGeneric(const uint8_t* inputData, const Shape& inputShape,
                         int32_t blockSize,
                         uint8_t* outputData, const Shape& outputShape) {
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::optimized_ops::DepthToSpace(
                 reinterpret_cast<const float*>(inputData),
                 convertShapeToDims(inputShape),
                 blockSize,
                 reinterpret_cast<float*>(outputData),
                 convertShapeToDims(outputShape));
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::optimized_ops::DepthToSpace(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
    } else {
        LOG(ERROR) << "Unsupported data type";
        return false;
    }
    return true;
}

bool spaceToDepthGeneric(const uint8_t* inputData, const Shape& inputShape,
                         int32_t blockSize,
                         uint8_t* outputData, const Shape& outputShape) {
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::optimized_ops::SpaceToDepth(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::optimized_ops::SpaceToDepth(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
    } else {
        LOG(ERROR) << "Unsupported data type";
        return false;
    }
    return true;
}

template <typename T>
static bool padGeneric(const T* inputData, const Shape& inputShape, const int32_t* paddings,
                       T padValue, T* outputData, const Shape& outputShape) {

    // Based on
    // http://google3/third_party/tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h?l=6194&rcl=213557260

    // TFLite runtime calls are currently fixed at 4 dimensions. Copy inputs so
    // we can pad them to 4 dims (yes, we are "padding the padding").
    int32_t numInputDims = static_cast<int32_t>(getNumberOfDimensions(inputShape));
    NN_OPS_CHECK(numInputDims <= 4);
    std::vector<int> leftPaddings(4 - numInputDims, 0);
    std::vector<int> rightPaddings(4 - numInputDims, 0);
    for (int32_t i = 0; i < numInputDims; ++i) {
        leftPaddings.push_back(paddings[i * 2]);
        rightPaddings.push_back(paddings[i * 2 + 1]);
    }
    const int leftBPadding = leftPaddings[0];
    const int leftHPadding = leftPaddings[1];
    const int leftWPadding = leftPaddings[2];
    const int leftDPadding = leftPaddings[3];
    const int rightBPadding = rightPaddings[0];
    const int rightHPadding = rightPaddings[1];
    const int rightWPadding = rightPaddings[2];
    const int rightDPadding = rightPaddings[3];

    const auto extInputShape =
            tflite::RuntimeShape::ExtendedShape(4, convertShapeToTflshape(inputShape));
    const auto extOutputShape =
            tflite::RuntimeShape::ExtendedShape(4, convertShapeToTflshape(outputShape));

    const int outputBatch = extOutputShape.Dims(0);
    const int outputHeight = extOutputShape.Dims(1);
    const int outputWidth = extOutputShape.Dims(2);
    const int outputDepth = extOutputShape.Dims(3);

    const int inputDepth = extInputShape.Dims(3);

    if (leftBPadding != 0) {
        tflite::optimized_ops::TypedMemset<T>(
                outputData, padValue, leftBPadding * outputHeight * outputWidth * outputDepth);
    }
    for (int outB = leftBPadding; outB < outputBatch - rightBPadding; ++outB) {
        if (leftHPadding != 0) {
            tflite::optimized_ops::TypedMemset<T>(
                    outputData + tflite::Offset(extOutputShape, outB, 0, 0, 0), padValue,
                    leftHPadding * outputWidth * outputDepth);
        }
        for (int outH = leftHPadding; outH < outputHeight - rightHPadding; ++outH) {
            if (leftWPadding != 0) {
                tflite::optimized_ops::TypedMemset<T>(
                        outputData + tflite::Offset(extOutputShape, outB, outH, 0, 0), padValue,
                        leftWPadding * outputDepth);
            }
            for (int outW = leftWPadding; outW < outputWidth - rightWPadding; ++outW) {
                if (leftDPadding != 0) {
                    tflite::optimized_ops::TypedMemset<T>(
                            outputData + tflite::Offset(extOutputShape, outB, outH, outW, 0),
                            padValue, leftDPadding);
                }

                T* out =
                        outputData + tflite::Offset(extOutputShape, outB, outH, outW, leftDPadding);
                const T* in =
                        inputData + tflite::Offset(extInputShape, outB - leftBPadding,
                                                   outH - leftHPadding, outW - leftWPadding, 0);
                memcpy(out, in, inputDepth * sizeof(T));

                if (rightDPadding != 0) {
                    tflite::optimized_ops::TypedMemset<T>(
                            outputData + tflite::Offset(extOutputShape, outB, outH, outW,
                                                        outputDepth - rightDPadding),
                            padValue, rightDPadding);
                }
            }
            if (rightWPadding != 0) {
                tflite::optimized_ops::TypedMemset<T>(
                        outputData + tflite::Offset(extOutputShape, outB, outH,
                                                    outputWidth - rightWPadding, 0),
                        padValue, rightWPadding * outputDepth);
            }
        }
        if (rightHPadding != 0) {
            tflite::optimized_ops::TypedMemset<T>(
                    outputData + tflite::Offset(extOutputShape, outB, outputHeight - rightHPadding,
                                                0, 0),
                    padValue, rightHPadding * outputWidth * outputDepth);
        }
    }
    if (rightBPadding != 0) {
        tflite::optimized_ops::TypedMemset<T>(
                outputData + tflite::Offset(extOutputShape, outputBatch - rightBPadding, 0, 0, 0),
                padValue, rightBPadding * outputHeight * outputWidth * outputDepth);
    }

    return true;
}

bool padFloat32(const float* inputData, const Shape& inputShape, const int32_t* paddings,
                float padValue, float* outputData, const Shape& outputShape) {
    return padGeneric(inputData, inputShape, paddings, padValue, outputData, outputShape);
}

bool padQuant8(const uint8_t* inputData, const Shape& inputShape, const int32_t* paddings,
               uint8_t padValue, uint8_t* outputData, const Shape& outputShape) {
    return padGeneric(inputData, inputShape, paddings, padValue, outputData, outputShape);
}

bool batchToSpaceGeneric(const uint8_t* inputData, const Shape& inputShape,
                         const int32_t* blockSize,
                         uint8_t* outputData, const Shape& outputShape) {
    // Needed by low level implementation, but not really used.
    tflite::Dims<4> blockSizeDim, cropsDim;
    const int32 crops[4] = {0, 0, 0, 0};
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::optimized_ops::BatchToSpaceND(
                 reinterpret_cast<const float*>(inputData),
                 convertShapeToDims(inputShape),
                 blockSize, blockSizeDim,
                 crops, cropsDim,
                 reinterpret_cast<float*>(outputData),
                 convertShapeToDims(outputShape));
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::optimized_ops::BatchToSpaceND(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                crops, cropsDim,
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
    } else {
        LOG(ERROR) << "Unsupported data type";
        return false;
    }
    return true;
}

bool spaceToBatchGeneric(const uint8_t* inputData, const Shape& inputShape,
                         const int32_t* blockSize,
                         const int32_t* padding, const Shape& paddingShape,
                         uint8_t* outputData, const Shape& outputShape) {
    // Needed by low level implementation, but not really used.
    tflite::Dims<4> blockSizeDim;
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::optimized_ops::SpaceToBatchND(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                padding, convertShapeToDims(paddingShape),
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape));
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::optimized_ops::SpaceToBatchND(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                blockSize, blockSizeDim,
                padding, convertShapeToDims(paddingShape),
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape));
    } else {
        LOG(ERROR) << "Unsupported data type";
        return false;
    }
    return true;
}

bool squeezeGeneric(const void* inputData, const Shape& inputShape,
                    void* outputData, const Shape& outputShape) {
    size_t count = sizeOfData(inputShape.type, inputShape.dimensions);
    memcpy(outputData, inputData, count);
    return true;
}

bool transposeGeneric(const uint8_t* inputData, const Shape& inputShape,
                      const int32_t* perm, const Shape& permShape,
                      uint8_t* outputData, const Shape& outputShape) {
    // Reverse the permuted axes and convert to 4D due to the way Dims are
    // constructed.
    const int32_t kOutputDimensionNum = 4;

    int32_t permSize = static_cast<int32_t>(getSizeOfDimension(permShape, 0));
    int32_t reversed_perm[kOutputDimensionNum];
    for (int32_t output_k = 0, input_k = permSize - 1; output_k < permSize;
             ++output_k, --input_k) {
        reversed_perm[output_k] = permSize - perm[input_k] - 1;
    }
    for (int32_t k = permSize; k < kOutputDimensionNum; ++k) {
        reversed_perm[k] = k;
    }
    if (inputShape.type == OperandType::TENSOR_FLOAT32) {
        tflite::reference_ops::Transpose(
                reinterpret_cast<const float*>(inputData),
                convertShapeToDims(inputShape),
                reinterpret_cast<float*>(outputData),
                convertShapeToDims(outputShape),
                reversed_perm);
    } else if (inputShape.type == OperandType::TENSOR_QUANT8_ASYMM) {
        tflite::reference_ops::Transpose(
                reinterpret_cast<const uint8_t*>(inputData),
                convertShapeToDims(inputShape),
                reinterpret_cast<uint8_t*>(outputData),
                convertShapeToDims(outputShape),
                reversed_perm);
    } else {
        LOG(ERROR) << "Unsupported data type";
        return false;
    }
    return true;
}
} // nn