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

#define LOG_TAG "Utils"

#include "Utils.h"


namespace nn {

namespace {

template <typename EntryType, uint32_t entryCount, uint32_t entryCountOEM>
EntryType tableLookup(const EntryType (&table)[entryCount],
                      const EntryType (&tableOEM)[entryCountOEM],
                      uint32_t code) {
    if (code < entryCount) {
        return table[code];
    } else if (code >= kOEMCodeBase && (code - kOEMCodeBase) < entryCountOEM) {
        return tableOEM[code - kOEMCodeBase];
    } else {
        nnAssert(!"tableLookup: bad code");
        return EntryType();
    }
}

};  // anonymous namespace

#define COUNT(X) (sizeof(X) / sizeof(X[0]))

const char* kTypeNames[kNumberOfDataTypes] = {
        "FLOAT32",        "INT32",
        "UINT32",         "TENSOR_FLOAT32",
        "TENSOR_INT32",   "TENSOR_QUANT8_ASYMM",
        "BOOL",           "TENSOR_QUANT16_ASYMM",
        "TENSOR_FLOAT16",
};

static_assert(COUNT(kTypeNames) == kNumberOfDataTypes, "kTypeNames is incorrect");

const char* kTypeNamesOEM[kNumberOfDataTypesOEM] = {
        "OEM",            "TENSOR_OEM_BYTE",
};

static_assert(COUNT(kTypeNamesOEM) == kNumberOfDataTypesOEM, "kTypeNamesOEM is incorrect");

const char* getOperandTypeName(OperandType type) {
    uint32_t n = static_cast<uint32_t>(type);
    return tableLookup(kTypeNames, kTypeNamesOEM, n);
}

// TODO Check if this useful
const char* kErrorNames[] = {
        "NO_ERROR", "OUT_OF_MEMORY", "INCOMPLETE", "NULL", "BAD_DATA",
};

const char* kOperationNames[kNumberOfOperationTypes] = {
        "ADD",
        "AVERAGE_POOL",
        "CONCATENATION",
        "CONV",
        "DEPTHWISE_CONV",
        "DEPTH_TO_SPACE",
        "DEQUANTIZE",
        "EMBEDDING_LOOKUP",
        "FLOOR",
        "FULLY_CONNECTED",
        "HASHTABLE_LOOKUP",
        "L2_NORMALIZATION",
        "L2_POOL",
        "LOCAL_RESPONSE_NORMALIZATION",
        "LOGISTIC",
        "LSH_PROJECTION",
        "LSTM",
        "MAX_POOL",
        "MUL",
        "RELU",
        "RELU1",
        "RELU6",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RNN",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SVDF",
        "TANH",
        "BATCH_TO_SPACE_ND",
        "DIV",
        "MEAN",
        "PAD",
        "SPACE_TO_BATCH_ND",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "TRANSPOSE",
        "ARGMAX",
        "ARGMIN",
        "PAD_V2",
        "AXIS_ALIGNED_BBOX_TRANSFORM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_RNN",
        "BOX_WITH_NMS_LIMIT",
        "CAST",
        "CHANNEL_SHUFFLE",
        "DETECTION_OUTPUT",
        "EMBEDDING_LOOKUP_SPARSE",
        "EXP",
        "EXPAND_DIMS",
        "GATHER",
        "GENERATE_PROPOSALS",
        "GREATER",
        "GREATER_EQUAL",
        "GROUPED_CONV_2D",
        "HEATMAP_MAX_KEYPOINT",
        "LESS",
        "LESS_EQUAL",
        "LOG",
        "LOGICAL_AND",
        "LOGICAL_NOT",
        "LOGICAL_OR",
        "LOG_SOFTMAX",
        "MAXIMUM",
        "MINIMUM",
        "NEG",
        "POW",
        "PRELU",
        "PRIOR_BOX",
        "QUANTIZE",
        "QUANTIZED_16BIT_LSTM",
        "RANDOM_MULTINOMIAL",
        "REDUCE",
        "ROI_ALIGN",
        "RSQRT",
        "SELECT",
        "SIN",
        "SLICE",
        "SPARSE_TO_DENSE",
        "SPLIT",
        "SQRT",
        "TILE",
        "TOPK_V2",
        "TRANSPOSE_CONV_2D",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "ROTATED_BBOX_TRANSFORM",
        "ABS",
        "ROI_POOLING",
};

static_assert(COUNT(kOperationNames) == kNumberOfOperationTypes, "kOperationNames is incorrect");

const char* kOperationNamesOEM[kNumberOfOperationTypesOEM] = {
        "OEM_OPERATION",
};

static_assert(COUNT(kOperationNamesOEM) == kNumberOfOperationTypesOEM,
              "kOperationNamesOEM is incorrect");

const char* getOperationName(OperationType type) {
    uint32_t n = static_cast<uint32_t>(type);
    return tableLookup(kOperationNames, kOperationNamesOEM, n);
}

const uint32_t kSizeOfDataType[]{
        4,  // ANEURALNETWORKS_FLOAT32
        4,  // ANEURALNETWORKS_INT32
        4,  // ANEURALNETWORKS_UINT32
        4,  // ANEURALNETWORKS_TENSOR_FLOAT32
        4,  // ANEURALNETWORKS_TENSOR_INT32
        1,  // ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8
        1,  // ANEURALNETWORKS_BOOL
        2,  // ANEURALNETWORKS_TENSOR_QUANT16_ASYMM
        2,  // ANEURALNETWORKS_TENSOR_FLOAT16
};

static_assert(COUNT(kSizeOfDataType) == kNumberOfDataTypes, "kSizeOfDataType is incorrect");

const bool kScalarDataType[]{
        true,   // ANEURALNETWORKS_FLOAT32
        true,   // ANEURALNETWORKS_INT32
        true,   // ANEURALNETWORKS_UINT32
        false,  // ANEURALNETWORKS_TENSOR_FLOAT32
        false,  // ANEURALNETWORKS_TENSOR_INT32
        false,  // ANEURALNETWORKS_TENSOR_SYMMETRICAL_QUANT8
        true,   // ANEURALNETWORKS_BOOL
        false,  // ANEURALNETWORKS_TENSOR_QUANT16_ASYMM
        false,  // ANEURALNETWORKS_TENSOR_FLOAT16
};

static_assert(COUNT(kScalarDataType) == kNumberOfDataTypes, "kScalarDataType is incorrect");

const uint32_t kSizeOfDataTypeOEM[]{
        0, // ANEURALNETWORKS_OEM
        1, // ANEURALNETWORKS_TENSOR_OEM_BYTE
};

static_assert(COUNT(kSizeOfDataTypeOEM) == kNumberOfDataTypesOEM,
              "kSizeOfDataTypeOEM is incorrect");

const bool kScalarDataTypeOEM[]{
        true,  // ANEURALNETWORKS_OEM
        false, // ANEURALNETWORKS_TENSOR_OEM_BYTE
};

static_assert(COUNT(kScalarDataTypeOEM) == kNumberOfDataTypesOEM,
              "kScalarDataTypeOEM is incorrect");

uint32_t sizeOfData(OperandType type, const std::vector<uint32_t>& dimensions) {
    int n = static_cast<int>(type);

    uint32_t size = tableLookup(kSizeOfDataType, kSizeOfDataTypeOEM, n);

    if (tableLookup(kScalarDataType, kScalarDataTypeOEM, n) == true) {
        return size;
    }

    for (auto d : dimensions) {
        size *= d;
    }
    return size;
}

uint32_t alignBytesNeeded(uint32_t index, size_t length) {
    uint32_t pattern;
    if (length < 2) {
        pattern = 0; // No alignment necessary
    } else if (length < 4) {
        pattern = 1; // Align on 2-byte boundary
    } else {
        pattern = 3; // Align on 4-byte boundary
    }
    uint32_t extra = (~(index - 1)) & pattern;
    return extra;
}

} // namespace nn
