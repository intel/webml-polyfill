"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util_1 = require("../util");
var axis_util_1 = require("./axis_util");
var concat_util_1 = require("./concat_util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function concat1d_(tensors) {
    return exports.concat(tensors, 0);
}
function concat2d_(tensors, axis) {
    return exports.concat(tensors, axis);
}
function concat3d_(tensors, axis) {
    return exports.concat(tensors, axis);
}
function concat4d_(tensors, axis) {
    return exports.concat(tensors, axis);
}
function concat_(tensors, axis) {
    if (axis === void 0) { axis = 0; }
    util_1.assert(tensors.length >= 1, 'Pass at least one tensor to concat');
    var $tensors = tensor_util_env_1.convertToTensorArray(tensors, 'tensors', 'concat');
    axis = axis_util_1.parseAxisParam(axis, $tensors[0].shape)[0];
    var outShape = concat_util_1.computeOutShape($tensors.map(function (t) { return t.shape; }), axis);
    if (util_1.sizeFromShape(outShape) === 0) {
        return tensor_ops_1.tensor([], outShape);
    }
    $tensors = $tensors.filter(function (t) { return t.size > 0; });
    if ($tensors.length === 1) {
        return $tensors[0];
    }
    var shapes = $tensors.map(function (t) { return t.shape; });
    concat_util_1.assertParamsConsistent(shapes, axis);
    var der = function (dy) {
        var sizeSplits = shapes.map(function (s) { return s[axis]; });
        var derTensors = exports.split(dy, sizeSplits, axis);
        return derTensors.map(function (t) { return function () { return t; }; });
    };
    var inputs = $tensors;
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.concat($tensors, axis); }, inputs, der);
}
function split_(x, numOrSizeSplits, axis) {
    if (axis === void 0) { axis = 0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'split');
    axis = axis_util_1.parseAxisParam(axis, $x.shape)[0];
    var splitSizes;
    if (typeof (numOrSizeSplits) === 'number') {
        util_1.assert($x.shape[axis] % numOrSizeSplits === 0, 'Number of splits must evenly divide the axis.');
        splitSizes = Array(numOrSizeSplits).fill($x.shape[axis] / numOrSizeSplits);
    }
    else {
        util_1.assert($x.shape[axis] === numOrSizeSplits.reduce(function (a, b) { return a + b; }), 'The sum of sizes must match the size of the axis dimension.');
        splitSizes = numOrSizeSplits;
    }
    var der = function (dy) { return ({ $x: function () { return exports.concat(dy, axis); } }); };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.split($x, splitSizes, axis); }, { $x: $x }, der);
}
exports.concat = operation_1.op({ concat_: concat_ });
exports.concat1d = operation_1.op({ concat1d_: concat1d_ });
exports.concat2d = operation_1.op({ concat2d_: concat2d_ });
exports.concat3d = operation_1.op({ concat3d_: concat3d_ });
exports.concat4d = operation_1.op({ concat4d_: concat4d_ });
exports.split = operation_1.op({ split_: split_ });
//# sourceMappingURL=concat_split.js.map