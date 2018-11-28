"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_ops_1 = require("../ops/tensor_ops");
var tensor_1 = require("../tensor");
var util_1 = require("../util");
function castTensor(x, dtype, backend) {
    if (dtype === 'complex64') {
        if (x.dtype === 'complex64') {
            return x.clone();
        }
        var zerosTensor = tensor_ops_1.zeros(x.shape);
        var floatX = x.toFloat();
        var result = backend.complex(floatX, zerosTensor);
        zerosTensor.dispose();
        floatX.dispose();
        return result;
    }
    if (!util_1.hasEncodingLoss(x.dtype, dtype)) {
        return tensor_1.Tensor.make(x.shape, { dataId: x.dataId }, dtype);
    }
    if (x.dtype === 'complex64') {
        var real = backend.real(x);
        var result = real.cast(dtype);
        real.dispose();
        return result;
    }
    if (dtype === 'int32') {
        return backend.int(x);
    }
    else if (dtype === 'bool') {
        var zero = tensor_ops_1.scalar(0, x.dtype);
        var result = backend.notEqual(x, zero);
        zero.dispose();
        return result;
    }
    else {
        throw new Error("Error in Cast: unknown dtype argument (" + dtype + ")");
    }
}
exports.castTensor = castTensor;
function reshapeTensor(x, shape) {
    return tensor_1.Tensor.make(shape, { dataId: x.dataId }, x.dtype);
}
exports.reshapeTensor = reshapeTensor;
//# sourceMappingURL=backend_util.js.map