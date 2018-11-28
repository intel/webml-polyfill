"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var complex_ops_1 = require("../ops/complex_ops");
var operation_1 = require("../ops/operation");
var util_1 = require("../util");
function fft_(input) {
    util_1.assert(input.dtype === 'complex64', "The dtype for tf.spectral.fft() must be complex64 " +
        ("but got " + input.dtype + "."));
    var innerDimensionSize = input.shape[input.shape.length - 1];
    var batch = input.size / innerDimensionSize;
    var input2D = input.as2D(batch, innerDimensionSize);
    var ret = environment_1.ENV.engine.runKernel(function (backend) { return backend.fft(input2D); }, { input: input });
    return ret.reshape(input.shape);
}
function ifft_(input) {
    util_1.assert(input.dtype === 'complex64', "The dtype for tf.spectral.ifft() must be complex64 " +
        ("but got " + input.dtype + "."));
    var innerDimensionSize = input.shape[input.shape.length - 1];
    var batch = input.size / innerDimensionSize;
    var input2D = input.as2D(batch, innerDimensionSize);
    var ret = environment_1.ENV.engine.runKernel(function (backend) { return backend.ifft(input2D); }, { input: input });
    return ret.reshape(input.shape);
}
function rfft_(input) {
    util_1.assert(input.dtype === 'float32', "The dtype for rfft() must be real value but\n    got " + input.dtype);
    var innerDimensionSize = input.shape[input.shape.length - 1];
    var batch = input.size / innerDimensionSize;
    var zeros = input.zerosLike();
    var complexInput = complex_ops_1.complex(input, zeros).as2D(batch, innerDimensionSize);
    var ret = environment_1.ENV.engine.runKernel(function (backend) { return backend.fft(complexInput); }, { complexInput: complexInput });
    var half = Math.floor(innerDimensionSize / 2) + 1;
    var realValues = complex_ops_1.real(ret);
    var imagValues = complex_ops_1.imag(ret);
    var realComplexConjugate = realValues.split([half, innerDimensionSize - half], realValues.shape.length - 1);
    var imagComplexConjugate = imagValues.split([half, innerDimensionSize - half], imagValues.shape.length - 1);
    var outputShape = input.shape.slice();
    outputShape[input.shape.length - 1] = half;
    return complex_ops_1.complex(realComplexConjugate[0], imagComplexConjugate[0])
        .reshape(outputShape);
}
exports.fft = operation_1.op({ fft_: fft_ });
exports.ifft = operation_1.op({ ifft_: ifft_ });
exports.rfft = operation_1.op({ rfft_: rfft_ });
//# sourceMappingURL=spectral_ops.js.map