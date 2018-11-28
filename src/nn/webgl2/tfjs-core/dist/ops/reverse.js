"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var axis_util_1 = require("./axis_util");
var operation_1 = require("./operation");
function reverse1d_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'reverse');
    util.assert($x.rank === 1, "Error in reverse1D: x must be rank 1 but got\n             rank " + $x.rank + ".");
    return exports.reverse($x, 0);
}
function reverse2d_(x, axis) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'reverse');
    util.assert($x.rank === 2, "Error in reverse2D: x must be rank 2 but got\n             rank " + $x.rank + ".");
    return exports.reverse($x, axis);
}
function reverse3d_(x, axis) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'reverse');
    util.assert($x.rank === 3, "Error in reverse3D: x must be rank 3 but got\n             rank " + $x.rank + ".");
    return exports.reverse($x, axis);
}
function reverse4d_(x, axis) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'reverse');
    util.assert($x.rank === 4, "Error in reverse4D: x must be rank 4 but got\n             rank " + $x.rank + ".");
    return exports.reverse($x, axis);
}
function reverse_(x, axis) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'reverse');
    if ($x.rank === 0) {
        return $x.clone();
    }
    var axes = axis_util_1.parseAxisParam(axis, $x.shape);
    var grad = function (dy) {
        return { $x: function () { return dy.reverse(axes); } };
    };
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.reverse($x, axes); }, { $x: $x }, grad);
    return res.reshapeAs($x);
}
exports.reverse = operation_1.op({ reverse_: reverse_ });
exports.reverse1d = operation_1.op({ reverse1d_: reverse1d_ });
exports.reverse2d = operation_1.op({ reverse2d_: reverse2d_ });
exports.reverse3d = operation_1.op({ reverse3d_: reverse3d_ });
exports.reverse4d = operation_1.op({ reverse4d_: reverse4d_ });
//# sourceMappingURL=reverse.js.map