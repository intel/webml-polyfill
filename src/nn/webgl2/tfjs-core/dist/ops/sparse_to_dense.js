"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var sparse_to_dense = require("../ops/sparse_to_dense_util");
var tensor_util_env_1 = require("../tensor_util_env");
var operation_1 = require("./operation");
function sparseToDense_(sparseIndices, sparseValues, outputShape, defaultValue) {
    var $sparseIndices = tensor_util_env_1.convertToTensor(sparseIndices, 'sparseIndices', 'sparseToDense', 'int32');
    var $sparseValues = tensor_util_env_1.convertToTensor(sparseValues, 'sparseValues', 'sparseToDense');
    var $defaultValue = tensor_util_env_1.convertToTensor(defaultValue, 'defaultValue', 'sparseToDense', $sparseValues.dtype);
    sparse_to_dense.validateInput($sparseIndices, $sparseValues, outputShape, $defaultValue);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.sparseToDense($sparseIndices, $sparseValues, outputShape, $defaultValue); }, { $sparseIndices: $sparseIndices, $sparseValues: $sparseValues, $defaultValue: $defaultValue });
}
exports.sparseToDense = operation_1.op({ sparseToDense_: sparseToDense_ });
//# sourceMappingURL=sparse_to_dense.js.map