"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var globals_1 = require("../globals");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var axis_util = require("./axis_util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function logSumExp_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'logSumExp');
    var axes = axis_util.parseAxisParam(axis, $x.shape);
    var xMax = $x.max(axes, true);
    var a = $x.sub(xMax);
    var b = a.exp();
    var c = b.sum(axes);
    var d = c.log();
    var res = xMax.reshape(d.shape).add(d);
    if (keepDims) {
        var newShape = axis_util.expandShapeToKeepDim(res.shape, axes);
        return res.reshape(newShape);
    }
    return res;
}
function sum_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'sum');
    if ($x.dtype === 'bool') {
        $x = $x.toInt();
    }
    var axes = axis_util.parseAxisParam(axis, $x.shape);
    var customOp = globals_1.customGrad(function (x) {
        var permutation = axis_util.getAxesPermutation(axes, x.rank);
        var reductionAxes = axes;
        var permutedX = x;
        if (permutation != null) {
            permutedX = x.transpose(permutation);
            reductionAxes = axis_util.getInnerMostAxes(reductionAxes.length, x.rank);
        }
        var value = environment_1.ENV.engine.runKernel(function (backend) { return backend.sum(permutedX, reductionAxes); }, { permutedX: permutedX });
        if (keepDims) {
            var newShape = axis_util.expandShapeToKeepDim(value.shape, axes);
            value = value.reshape(newShape);
        }
        var gradFunc = function (dy) {
            var expandedDyShape = x.shape.slice();
            axes.forEach(function (axis) {
                expandedDyShape[axis] = 1;
            });
            var expandedDy = dy.reshape(expandedDyShape);
            var derX = expandedDy.mul(tensor_ops_1.ones(x.shape, 'float32'));
            return derX;
        };
        return { value: value, gradFunc: gradFunc };
    });
    return customOp($x);
}
function prod_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'prod');
    if ($x.dtype === 'bool') {
        $x = $x.toInt();
    }
    var axes = axis_util.parseAxisParam(axis, $x.shape);
    var permutation = axis_util.getAxesPermutation(axes, $x.rank);
    var reductionAxes = axes;
    var permutedX = $x;
    if (permutation != null) {
        permutedX = $x.transpose(permutation);
        reductionAxes = axis_util.getInnerMostAxes(reductionAxes.length, $x.rank);
    }
    var value = environment_1.ENV.engine.runKernel(function (backend) { return backend.prod(permutedX, reductionAxes); }, { permutedX: permutedX });
    if (keepDims) {
        var newShape = axis_util.expandShapeToKeepDim(value.shape, axes);
        value = value.reshape(newShape);
    }
    return value;
}
function mean_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'mean');
    var axes = axis_util.parseAxisParam(axis, $x.shape);
    var shapes = axis_util.computeOutAndReduceShapes($x.shape, axes);
    var reduceShape = shapes[1];
    var reduceSize = util.sizeFromShape(reduceShape);
    var customOp = globals_1.customGrad(function (x) {
        var reduceSizeScalar = tensor_ops_1.scalar(reduceSize);
        var xReduce = reduceSizeScalar.dtype === x.dtype ? x : x.cast(reduceSizeScalar.dtype);
        var res = xReduce.div(reduceSizeScalar);
        var value = res.sum(axis, keepDims);
        var gradFunc = function (dy) {
            var expandedDyShape = x.shape.slice();
            axes.forEach(function (axis) {
                expandedDyShape[axis] = 1;
            });
            var expandedDy = dy.reshape(expandedDyShape);
            var derX = expandedDy.mul(tensor_ops_1.ones(x.shape, 'float32')).div(reduceSizeScalar);
            return derX;
        };
        return { value: value, gradFunc: gradFunc };
    });
    return customOp($x);
}
function gradForMinAndMax(dy, saved, xOrig, origAxes, permutedAxes) {
    var y = saved[0];
    if (y.rank < xOrig.rank) {
        y = y.reshape(axis_util.expandShapeToKeepDim(y.shape, origAxes));
    }
    if (dy.rank < xOrig.rank) {
        dy = dy.reshape(axis_util.expandShapeToKeepDim(dy.shape, origAxes));
    }
    return {
        $x: function () {
            var dx = dy.mul(xOrig.equal(y).cast(dy.dtype));
            return permutedAxes == null ? dx : dx.transpose(permutedAxes);
        }
    };
}
function min_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'min');
    var xOrig = $x;
    var origAxes = axis_util.parseAxisParam(axis, $x.shape);
    var axes = origAxes;
    var permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
        $x = $x.transpose(permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
    }
    var grad = function (dy, saved) {
        return gradForMinAndMax(dy, saved, xOrig, origAxes, permutedAxes);
    };
    var res = environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.min($x, axes)); }, { $x: $x }, grad);
    if (keepDims) {
        var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        res = res.reshape(newShape);
    }
    return res;
}
function max_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'max');
    var xOrig = $x;
    var origAxes = axis_util.parseAxisParam(axis, $x.shape);
    var axes = origAxes;
    var permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
        $x = $x.transpose(permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
    }
    var grad = function (dy, saved) {
        return gradForMinAndMax(dy, saved, xOrig, origAxes, permutedAxes);
    };
    var res = environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.max($x, axes)); }, { $x: $x }, grad);
    if (keepDims) {
        var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        res = res.reshape(newShape);
    }
    return res;
}
function argMin_(x, axis) {
    if (axis === void 0) { axis = 0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'argMin');
    if (axis == null) {
        axis = 0;
    }
    var axes = axis_util.parseAxisParam(axis, $x.shape);
    var permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
        $x = $x.transpose(permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
    }
    var grad = function (dy) {
        return { $x: function () { return tensor_ops_1.zerosLike($x); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.argMin($x, axes[0]); }, { $x: $x }, grad);
}
function argMax_(x, axis) {
    if (axis === void 0) { axis = 0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'argMax');
    if (axis == null) {
        axis = 0;
    }
    var axes = axis_util.parseAxisParam(axis, $x.shape);
    var permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
        $x = $x.transpose(permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
    }
    var grad = function (dy) {
        return { $x: function () { return tensor_ops_1.zerosLike($x); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.argMax($x, axes[0]); }, { $x: $x }, grad);
}
function all_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'all', 'bool');
    util.assert($x.dtype === 'bool', "Error Tensor must be of type bool. Got: " + $x.dtype);
    var origAxes = axis_util.parseAxisParam(axis, $x.shape);
    var axes = origAxes;
    var permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
        $x = $x.transpose(permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
    }
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.all($x, axes); }, { $x: $x });
    if (keepDims) {
        var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
    }
    return res;
}
function any_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'any', 'bool');
    util.assert($x.dtype === 'bool', "Error Tensor must be of type bool. Got: " + $x.dtype);
    var origAxes = axis_util.parseAxisParam(axis, $x.shape);
    var axes = origAxes;
    var permutedAxes = axis_util.getAxesPermutation(axes, $x.rank);
    if (permutedAxes != null) {
        $x = $x.transpose(permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, $x.rank);
    }
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.any($x, axes); }, { $x: $x });
    if (keepDims) {
        var newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
    }
    return res;
}
function moments_(x, axis, keepDims) {
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    x = tensor_util_env_1.convertToTensor(x, 'x', 'moments');
    var axes = axis_util.parseAxisParam(axis, x.shape);
    var mean = x.mean(axes, keepDims);
    var keepDimsShape = mean.shape;
    if (!keepDims) {
        keepDimsShape = axis_util.expandShapeToKeepDim(mean.shape, axes);
    }
    var devSquared = x.toFloat().sub(mean.reshape(keepDimsShape)).square();
    var variance = devSquared.mean(axes, keepDims);
    return { mean: mean, variance: variance };
}
exports.all = operation_1.op({ all_: all_ });
exports.any = operation_1.op({ any_: any_ });
exports.argMax = operation_1.op({ argMax_: argMax_ });
exports.argMin = operation_1.op({ argMin_: argMin_ });
exports.logSumExp = operation_1.op({ logSumExp_: logSumExp_ });
exports.max = operation_1.op({ max_: max_ });
exports.mean = operation_1.op({ mean_: mean_ });
exports.min = operation_1.op({ min_: min_ });
exports.moments = operation_1.op({ moments_: moments_ });
exports.sum = operation_1.op({ sum_: sum_ });
exports.prod = operation_1.op({ prod_: prod_ });
//# sourceMappingURL=reduction_ops.js.map