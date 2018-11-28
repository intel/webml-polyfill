"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var binary_ops_1 = require("./binary_ops");
var logical_ops_1 = require("./logical_ops");
var operation_1 = require("./operation");
var selu_util_1 = require("./selu_util");
var tensor_ops_1 = require("./tensor_ops");
function relu_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'relu');
    if ($x.dtype === 'bool') {
        return $x.toInt();
    }
    var grad = function (dy) {
        var stepRes = $x.step();
        return { $x: function () { return dy.mulStrict(stepRes.toFloat()); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.relu($x); }, { $x: $x }, grad);
}
function elu_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'elu');
    var grad = function (dy, saved) {
        var y = saved[0];
        return {
            $x: function () {
                return environment_1.ENV.engine.runKernel(function (backend) { return backend.eluDer(dy, y); }, { dy: dy, y: y });
            }
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.elu($x)); }, { $x: $x }, grad);
}
function selu_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'selu');
    var grad = function (dy) {
        return {
            $x: function () {
                var mask = $x.greater(tensor_ops_1.scalar(0));
                var scaleAlpha = tensor_ops_1.scalar(selu_util_1.SELU_SCALEALPHA);
                var scale = tensor_ops_1.scalar(selu_util_1.SELU_SCALE);
                var greaterThanZeroDer = dy.mul(scale);
                var lessEqualZeroDer = dy.mul(scaleAlpha).mul($x.toFloat().exp());
                return logical_ops_1.where(mask, greaterThanZeroDer, lessEqualZeroDer);
            }
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.selu($x); }, { $x: $x }, grad);
}
function leakyRelu_(x, alpha) {
    if (alpha === void 0) { alpha = 0.2; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'leakyRelu');
    return binary_ops_1.maximum(tensor_ops_1.scalar(alpha).mul($x), $x);
}
function prelu_(x, alpha) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'prelu');
    var $alpha = tensor_util_env_1.convertToTensor(alpha, 'alpha', 'prelu');
    var zero = tensor_ops_1.scalar(0);
    return binary_ops_1.maximum(zero, $x).add($alpha.mul(binary_ops_1.minimum(zero, $x)));
}
exports.elu = operation_1.op({ elu_: elu_ });
exports.leakyRelu = operation_1.op({ leakyRelu_: leakyRelu_ });
exports.prelu = operation_1.op({ prelu_: prelu_ });
exports.relu = operation_1.op({ relu_: relu_ });
exports.selu = operation_1.op({ selu_: selu_ });
//# sourceMappingURL=relu_ops.js.map