"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var array_ops_1 = require("./array_ops");
var conv_util = require("./conv_util");
var operation_1 = require("./operation");
function maxPoolImpl_(x, filterSize, strides, dilations, pad, dimRoundingMode) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'maxPool');
    var x4D = $x;
    var reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
    }
    if (dilations == null) {
        dilations = [1, 1];
    }
    util.assert(x4D.rank === 4, "Error in maxPool: input must be rank 4 but got rank " + x4D.rank + ".");
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), 'Error in maxPool: Either strides or dilations must be 1. ' +
        ("Got strides " + strides + " and dilations '" + dilations + "'"));
    if (dimRoundingMode != null) {
        util.assert(util.isInt(pad), "Error in maxPool: pad must be an integer when using, " +
            ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
    }
    var convInfo = conv_util.computePool2DInfo(x4D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    var grad = function (dy, saved) {
        var y4D = saved[0];
        return {
            x: function () { return maxPoolBackprop(dy, x4D, y4D, filterSize, strides, dilations, pad); }
        };
    };
    var res = environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.maxPool(x4D, convInfo)); }, { x: x4D }, grad);
    if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    }
    return res;
}
function maxPool_(x, filterSize, strides, pad, dimRoundingMode) {
    return maxPoolImpl_(x, filterSize, strides, 1, pad, dimRoundingMode);
}
function avgPoolImpl_(x, filterSize, strides, dilations, pad, dimRoundingMode) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'avgPool');
    util.assert($x.dtype === 'float32', 'The input dtype to avgPool must be float32');
    if (dilations == null) {
        dilations = [1, 1];
    }
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), 'Error in avgPool: Either strides or dilations must be 1. ' +
        ("Got strides " + strides + " and dilations '" + dilations + "'"));
    var x4D = $x;
    var reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
    }
    util.assert(x4D.rank === 4, "Error in avgPool: x must be rank 4 but got rank " + x4D.rank + ".");
    if (dimRoundingMode != null) {
        util.assert(util.isInt(pad), "Error in avgPool: pad must be an integer when using, " +
            ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
    }
    var convInfo = conv_util.computePool2DInfo(x4D.shape, filterSize, strides, dilations, pad);
    var grad = function (dy) {
        return {
            x: function () { return avgPoolBackprop(dy, x4D, filterSize, strides, dilations, pad); }
        };
    };
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.avgPool(x4D, convInfo); }, { x: x4D }, grad);
    res = res.cast($x.dtype);
    if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    }
    return res;
}
function avgPool_(x, filterSize, strides, pad, dimRoundingMode) {
    return avgPoolImpl_(x, filterSize, strides, 1, pad, dimRoundingMode);
}
function pool_(input, windowShape, poolingType, pad, dilations, strides) {
    if (dilations == null) {
        dilations = [1, 1];
    }
    if (strides == null) {
        strides = 1;
    }
    if (pad === 0) {
        pad = 'valid';
    }
    var $x = tensor_util_env_1.convertToTensor(input, 'x', 'maxPool');
    var x4D = $x;
    var reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
    }
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), 'Error in pool: Either strides or dilations must be 1. ' +
        ("Got strides " + strides + " and dilations '" + dilations + "'"));
    var convInfo = conv_util.computePool2DInfo(x4D.shape, windowShape, strides, dilations, pad);
    var dilation = [convInfo.dilationHeight, convInfo.dilationWidth];
    var basePadding;
    if (pad === 'same') {
        basePadding = withSpaceToBatchBasePaddings([convInfo.filterHeight, convInfo.filterWidth], dilation);
    }
    else {
        basePadding = [[0, 0], [0, 0]];
    }
    var isDilationOne = dilation[0] === 1 && dilation[1] === 1;
    var _a = requiredSpaceToBatchPaddings([convInfo.inHeight, convInfo.inWidth], dilation, basePadding), adjustedPadding = _a[0], adjustedCrops = _a[1];
    var convertedPad = isDilationOne ? pad : 'valid';
    var convertedX = isDilationOne ? x4D : array_ops_1.spaceToBatchND(x4D, dilation, adjustedPadding);
    var forwardOp = poolingType === 'avg' ?
        function () { return avgPoolImpl_(convertedX, windowShape, strides, 1, convertedPad); } :
        function () { return maxPoolImpl_(convertedX, windowShape, strides, 1, convertedPad); };
    var y = forwardOp();
    var res = isDilationOne ? y : array_ops_1.batchToSpaceND(y, dilation, adjustedCrops);
    if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    }
    return res;
}
function maxPoolBackprop(dy, input, output, filterSize, strides, dilations, pad, dimRoundingMode) {
    var $dy = tensor_util_env_1.convertToTensor(dy, 'dy', 'maxPoolBackprop');
    var $input = tensor_util_env_1.convertToTensor(input, 'input', 'maxPoolBackprop');
    var $output = tensor_util_env_1.convertToTensor(output, 'output', 'maxPoolBackprop');
    util.assert($input.rank === $dy.rank, "Rank of input (" + $input.rank + ") does not match rank of dy (" + $dy.rank + ")");
    if (dilations == null) {
        dilations = [1, 1];
    }
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), 'Error in maxPoolBackProp: Either strides or dilations must be 1. ' +
        ("Got strides " + strides + " and dilations '" + dilations + "'"));
    util.assert($dy.rank === 4, "Error in maxPoolBackprop: dy must be rank 4 but got rank " +
        ($dy.rank + "."));
    util.assert($input.rank === 4, "Error in maxPoolBackprop: input must be rank 4 but got rank " +
        ($input.rank + "."));
    if (dimRoundingMode != null) {
        util.assert(util.isInt(pad), "Error in maxPoolBackprop: pad must be an integer when using, " +
            ("dimRoundingMode " + dimRoundingMode + " but got pad " + pad + "."));
    }
    var convInfo = conv_util.computePool2DInfo($input.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.maxPoolBackprop($dy, $input, $output, convInfo); }, { $dy: $dy, $input: $input });
    return res;
}
function avgPoolBackprop(dy, input, filterSize, strides, dilations, pad) {
    var $dy = tensor_util_env_1.convertToTensor(dy, 'dy', 'avgPoolBackprop');
    var $input = tensor_util_env_1.convertToTensor(input, 'input', 'avgPoolBackprop');
    util.assert($input.rank === $dy.rank, "Rank of input (" + $input.rank + ") does not match rank of dy (" + $dy.rank + ")");
    if (dilations == null) {
        dilations = [1, 1];
    }
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), 'Error in avgPoolBackprop: Either strides or dilations must be 1. ' +
        ("Got strides " + strides + " and dilations '" + dilations + "'"));
    var input4D = $input;
    var dy4D = $dy;
    var reshapedTo4D = false;
    if ($input.rank === 3) {
        reshapedTo4D = true;
        input4D = $input.as4D(1, $input.shape[0], $input.shape[1], $input.shape[2]);
        dy4D = $dy.as4D(1, $dy.shape[0], $dy.shape[1], $dy.shape[2]);
    }
    util.assert(dy4D.rank === 4, "Error in avgPoolBackprop: dy must be rank 4 but got rank " +
        (dy4D.rank + "."));
    util.assert(input4D.rank === 4, "Error in avgPoolBackprop: input must be rank 4 but got rank " +
        (input4D.rank + "."));
    var convInfo = conv_util.computePool2DInfo(input4D.shape, filterSize, strides, dilations, pad);
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.avgPoolBackprop(dy4D, input4D, convInfo); }, { dy4D: dy4D, input4D: input4D });
    if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    }
    return res;
}
function requiredSpaceToBatchPaddings(inputShape, blockShape, basePadding) {
    var padStart = basePadding.map(function (b) { return b[0]; });
    var origPadEnd = basePadding.map(function (b) { return b[1]; });
    var fullInputShape = inputShape.concat(padStart, origPadEnd);
    var padEndExtra = blockShape.map(function (b, i) { return (b - fullInputShape[i] % b) % b; });
    var padEnd = origPadEnd.map(function (s, i) { return s + padEndExtra[i]; });
    var paddings = blockShape.map(function (_, i) { return [padStart[i], padEnd[i]]; });
    var crops = blockShape.map(function (_, i) { return [0, padEndExtra[i]]; });
    return [paddings, crops];
}
function withSpaceToBatchBasePaddings(filterShape, dilation) {
    var dilatedFilterShape = filterShape.map(function (s, i) {
        return s + (s - 1) * (dilation[i] - 1);
    });
    var padExtraShape = dilatedFilterShape.map(function (s) { return s - 1; });
    var padExtraStart = padExtraShape.map(function (s) { return Math.floor(s / 2); });
    var padExtraEnd = padExtraShape.map(function (s, i) { return s - padExtraStart[i]; });
    return padExtraShape.map(function (_, i) {
        return [padExtraStart[i], padExtraEnd[i]];
    });
}
exports.maxPool = operation_1.op({ maxPool_: maxPool_ });
exports.avgPool = operation_1.op({ avgPool_: avgPool_ });
exports.pool = operation_1.op({ pool_: pool_ });
//# sourceMappingURL=pool.js.map