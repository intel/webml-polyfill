"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util_1 = require("../util");
var array_ops_1 = require("./array_ops");
var axis_util_1 = require("./axis_util");
var binary_ops_1 = require("./binary_ops");
var compare_1 = require("./compare");
var logical_ops_1 = require("./logical_ops");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function unsortedSegmentSum_(x, segmentIds, numSegments) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'unsortedSegmentSum');
    var $segmentIds = tensor_util_env_1.convertToTensor(segmentIds, 'segmentIds', 'unsortedSegmentSum', 'int32');
    util_1.assert($segmentIds.dtype === 'int32', 'segmentIds must be of dtype `int32`');
    util_1.assert(util_1.isInt(numSegments), 'numSegments must be of dtype int');
    var gradFunc = function (dy) {
        var derX = function () {
            return gatherDropNegatives(dy, $segmentIds);
        };
        return { $x: derX };
    };
    return environment_1.ENV.engine.runKernel(function (backend) {
        return backend.unsortedSegmentSum($x, $segmentIds, numSegments);
    }, { $x: $x }, gradFunc);
}
function gather_(x, indices, axis) {
    if (axis === void 0) { axis = 0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'gather');
    var $indices = tensor_util_env_1.convertToTensor(indices, 'indices', 'gather', 'int32');
    util_1.assert($indices.dtype === 'int32', 'Indices must be of dtype `int32`');
    axis = axis_util_1.parseAxisParam(axis, $x.shape)[0];
    var grad = function (dy) {
        var derX = function () {
            if (axis === 0) {
                return exports.unsortedSegmentSum(dy, $indices, $x.shape[axis]);
            }
            var paramsShape = $x.shape;
            var indicesSize = $indices.size;
            var outerShape = paramsShape.slice(0, axis);
            var outerDims = outerShape.length;
            var innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
            var innerDims = innerShape.length;
            var outerAxesIndices = arrayRange(0, outerDims);
            var innerAxesIndices = arrayRange(outerDims + 1, outerDims + 1 + innerDims);
            var valuesShape = arrayConcat([outerShape, [indicesSize], innerShape]);
            var values = dy.reshape(valuesShape);
            var reshapedIndices = $indices.reshape([indicesSize]);
            var transposeDims = arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
            var valuesTranspose = values.transpose(transposeDims);
            var paramsGrad = exports.unsortedSegmentSum(valuesTranspose, reshapedIndices, $x.shape[axis]);
            var invertTransposeDims = axis_util_1.getUndoAxesPermutation(transposeDims);
            paramsGrad = paramsGrad.transpose(invertTransposeDims);
            return paramsGrad;
        };
        return { $x: derX };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.gather($x, $indices, axis); }, { $x: $x }, grad);
}
function arrayRange(start, stop) {
    var result = [];
    for (var i = start; i < stop; ++i) {
        result.push(i);
    }
    return result;
}
function arrayConcat(arrays) {
    var result = [];
    for (var i = 0; i < arrays.length; ++i) {
        for (var j = 0; j < arrays[i].length; ++j) {
            result.push(arrays[i][j]);
        }
    }
    return result;
}
function gatherDropNegatives(x, indices) {
    var zeroClippedIndices = binary_ops_1.maximum(indices, tensor_ops_1.zerosLike(indices));
    var gathered = exports.gather(x, zeroClippedIndices);
    var isPositive = compare_1.greaterEqual(indices, tensor_ops_1.scalar(0, 'int32'));
    var numIters = gathered.rank - isPositive.rank;
    for (var i = 0; i < numIters; ++i) {
        isPositive = array_ops_1.expandDims(isPositive, i + 1);
    }
    isPositive = logical_ops_1.logicalAnd(isPositive, tensor_ops_1.ones(gathered.shape, 'bool'));
    var zeroSlice = tensor_ops_1.zerosLike(gathered);
    return logical_ops_1.where(isPositive, gathered, zeroSlice);
}
exports.gather = operation_1.op({ gather_: gather_ });
exports.unsortedSegmentSum = operation_1.op({ unsortedSegmentSum_: unsortedSegmentSum_ });
//# sourceMappingURL=segment_ops.js.map