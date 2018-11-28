"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gradients_1 = require("../gradients");
var tensor_util_env_1 = require("../tensor_util_env");
var operation_1 = require("./operation");
function softmax_(logits, dim) {
    if (dim === void 0) { dim = -1; }
    var $logits = tensor_util_env_1.convertToTensor(logits, 'logits', 'softmax');
    if (dim === -1) {
        dim = $logits.rank - 1;
    }
    if (dim !== $logits.rank - 1) {
        throw Error('Softmax along a non-last dimension is not yet supported. ' +
            ("Logits was rank " + $logits.rank + " and dim was " + dim));
    }
    var customOp = gradients_1.customGrad(function (logits) {
        var keepDims = true;
        var lse = logits.logSumExp([dim], keepDims);
        var logResult = logits.toFloat().sub(lse);
        var y = logResult.exp();
        var gradFunc = function (dy) {
            var dyTimesY = dy.mul(y);
            var keepDims = true;
            return dyTimesY.sub(dyTimesY.sum([dim], keepDims).mul(y));
        };
        return { value: y, gradFunc: gradFunc };
    });
    return customOp($logits);
}
function logSoftmax_(logits, axis) {
    if (axis === void 0) { axis = -1; }
    var $logits = tensor_util_env_1.convertToTensor(logits, 'logits', 'logSoftmax');
    if (axis === -1) {
        axis = $logits.rank - 1;
    }
    if (axis !== $logits.rank - 1) {
        throw Error('Log Softmax along a non-last dimension is not yet supported. ' +
            ("Logits was rank " + $logits.rank + " and axis was " + axis));
    }
    var customOp = gradients_1.customGrad(function (logits) {
        var keepDims = true;
        var xMax = logits.max(axis, true);
        var shifted = logits.sub(xMax);
        var value = shifted.toFloat().sub(shifted.exp().sum(axis, keepDims).log());
        var gradFunc = function (dy) {
            var softmax = value.exp();
            return dy.sub(dy.sum(axis, keepDims).mul(softmax));
        };
        return { value: value, gradFunc: gradFunc };
    });
    return customOp($logits);
}
exports.softmax = operation_1.op({ softmax_: softmax_ });
exports.logSoftmax = operation_1.op({ logSoftmax_: logSoftmax_ });
//# sourceMappingURL=softmax.js.map