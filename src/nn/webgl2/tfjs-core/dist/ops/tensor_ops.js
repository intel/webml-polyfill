"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_1 = require("../tensor");
var tensor_util_env_1 = require("../tensor_util_env");
var tensor_util_env_2 = require("../tensor_util_env");
var util_1 = require("../util");
var complex_ops_1 = require("./complex_ops");
var operation_1 = require("./operation");
function tensor(values, shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    if (dtype === 'complex64') {
        throw new Error("Cannot construct a complex64 tensor directly. " +
            "Please use tf.complex(real, imag).");
    }
    if (!util_1.isTypedArray(values) && !Array.isArray(values) &&
        typeof values !== 'number' && typeof values !== 'boolean') {
        throw new Error('values passed to tensor(values) must be an ' +
            'array of numbers or booleans, or a TypedArray');
    }
    var inferredShape = tensor_util_env_2.inferShape(values);
    if (shape != null && inferredShape.length !== 1) {
        util_1.assertShapesMatch(shape, inferredShape, "Error creating a new Tensor. " +
            ("Inferred shape (" + inferredShape + ") does not match the ") +
            ("provided shape (" + shape + "). "));
    }
    if (!util_1.isTypedArray(values) && !Array.isArray(values)) {
        values = [values];
    }
    shape = shape || inferredShape;
    return tensor_1.Tensor.make(shape, {
        values: util_1.toTypedArray(values, dtype, environment_1.ENV.get('DEBUG'))
    }, dtype);
}
exports.tensor = tensor;
function scalar(value, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    if ((util_1.isTypedArray(value) || Array.isArray(value)) && dtype !== 'complex64') {
        throw new Error('Error creating a new Scalar: value must be a primitive ' +
            '(number|boolean)');
    }
    return tensor(value, [], dtype);
}
exports.scalar = scalar;
function tensor1d(values, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    util_1.assertNonNull(values);
    var inferredShape = tensor_util_env_2.inferShape(values);
    if (inferredShape.length !== 1) {
        throw new Error('tensor1d() requires values to be a flat/TypedArray');
    }
    return tensor(values, inferredShape, dtype);
}
exports.tensor1d = tensor1d;
function tensor2d(values, shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    util_1.assertNonNull(values);
    if (shape != null && shape.length !== 2) {
        throw new Error('tensor2d() requires shape to have two numbers');
    }
    var inferredShape = tensor_util_env_2.inferShape(values);
    if (inferredShape.length !== 2 && inferredShape.length !== 1) {
        throw new Error('tensor2d() requires values to be number[][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor2d() requires shape to be provided when `values` ' +
            'are a flat/TypedArray');
    }
    shape = shape || inferredShape;
    return tensor(values, shape, dtype);
}
exports.tensor2d = tensor2d;
function tensor3d(values, shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    util_1.assertNonNull(values);
    if (shape != null && shape.length !== 3) {
        throw new Error('tensor3d() requires shape to have three numbers');
    }
    var inferredShape = tensor_util_env_2.inferShape(values);
    if (inferredShape.length !== 3 && inferredShape.length !== 1) {
        throw new Error('tensor3d() requires values to be number[][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor3d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
    shape = shape || inferredShape;
    return tensor(values, shape, dtype);
}
exports.tensor3d = tensor3d;
function tensor4d(values, shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    util_1.assertNonNull(values);
    if (shape != null && shape.length !== 4) {
        throw new Error('tensor4d() requires shape to have four numbers');
    }
    var inferredShape = tensor_util_env_2.inferShape(values);
    if (inferredShape.length !== 4 && inferredShape.length !== 1) {
        throw new Error('tensor4d() requires values to be number[][][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor4d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
    shape = shape || inferredShape;
    return tensor(values, shape, dtype);
}
exports.tensor4d = tensor4d;
function tensor5d(values, shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    util_1.assertNonNull(values);
    if (shape != null && shape.length !== 5) {
        throw new Error('tensor5d() requires shape to have five numbers');
    }
    var inferredShape = tensor_util_env_2.inferShape(values);
    if (inferredShape.length !== 5 && inferredShape.length !== 1) {
        throw new Error('tensor5d() requires values to be ' +
            'number[][][][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor5d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
    shape = shape || inferredShape;
    return tensor(values, shape, dtype);
}
exports.tensor5d = tensor5d;
function tensor6d(values, shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    util_1.assertNonNull(values);
    if (shape != null && shape.length !== 6) {
        throw new Error('tensor6d() requires shape to have six numbers');
    }
    var inferredShape = tensor_util_env_2.inferShape(values);
    if (inferredShape.length !== 6 && inferredShape.length !== 1) {
        throw new Error('tensor6d() requires values to be number[][][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor6d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
    shape = shape ||
        inferredShape;
    return tensor(values, shape, dtype);
}
exports.tensor6d = tensor6d;
function ones(shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    if (dtype === 'complex64') {
        var real = ones(shape, 'float32');
        var imag = ones(shape, 'float32');
        return complex_ops_1.complex(real, imag);
    }
    var values = util_1.makeOnesTypedArray(util_1.sizeFromShape(shape), dtype);
    return tensor_1.Tensor.make(shape, { values: values }, dtype);
}
exports.ones = ones;
function zeros(shape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    if (dtype === 'complex64') {
        var real = zeros(shape, 'float32');
        var imag = zeros(shape, 'float32');
        return complex_ops_1.complex(real, imag);
    }
    var values = util_1.makeZerosTypedArray(util_1.sizeFromShape(shape), dtype);
    return tensor_1.Tensor.make(shape, { values: values }, dtype);
}
exports.zeros = zeros;
function fill(shape, value, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    var values = util_1.getTypedArrayFromDType(dtype, util_1.sizeFromShape(shape));
    values.fill(value);
    return tensor_1.Tensor.make(shape, { values: values }, dtype);
}
exports.fill = fill;
function onesLike_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'onesLike');
    return ones($x.shape, $x.dtype);
}
function zerosLike_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'zerosLike');
    return zeros($x.shape, $x.dtype);
}
function linspace(start, stop, num) {
    if (num === 0) {
        throw new Error('Cannot request zero samples');
    }
    var step = (stop - start) / (num - 1);
    var values = util_1.makeZerosTypedArray(num, 'float32');
    values[0] = start;
    for (var i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return tensor1d(values, 'float32');
}
exports.linspace = linspace;
function range(start, stop, step, dtype) {
    if (step === void 0) { step = 1; }
    if (dtype === void 0) { dtype = 'float32'; }
    if (step === 0) {
        throw new Error('Cannot have a step of zero');
    }
    var sameStartStop = start === stop;
    var increasingRangeNegativeStep = start < stop && step < 0;
    var decreasingRangePositiveStep = stop < start && step > 1;
    if (sameStartStop || increasingRangeNegativeStep ||
        decreasingRangePositiveStep) {
        return zeros([0], dtype);
    }
    var numElements = Math.abs(Math.ceil((stop - start) / step));
    var values = util_1.makeZerosTypedArray(numElements, dtype);
    if (stop < start && step === 1) {
        step = -1;
    }
    values[0] = start;
    for (var i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return tensor1d(values, dtype);
}
exports.range = range;
exports.onesLike = operation_1.op({ onesLike_: onesLike_ });
exports.zerosLike = operation_1.op({ zerosLike_: zerosLike_ });
//# sourceMappingURL=tensor_ops.js.map