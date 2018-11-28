"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_1 = require("../tensor_util");
var tensor_util_env_1 = require("../tensor_util_env");
var types_1 = require("../types");
var util = require("../util");
var broadcast_util = require("./broadcast_util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
var unary_ops_1 = require("./unary_ops");
function add_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'add');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'add');
    tensor_util_1.assertTypesMatch($a, $b);
    var outShape = broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () {
            var res = dy;
            var reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.reshape($a.shape);
        };
        var derB = function () {
            var res = dy;
            var reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.reshape($b.shape);
        };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.add($a, $b); }, { $a: $a, $b: $b }, der);
}
function addN_(tensors) {
    util.assert(Array.isArray(tensors), function () { return 'The argument passed to tf.addN() must be a list of tensors'; });
    util.assert(tensors.length >= 1, function () { return "Must pass at least one tensor to tf.addN(), but got " +
        ("" + tensors.length); });
    var $tensors = tensors.map(function (t, i) { return tensor_util_env_1.convertToTensor(t, "tensors" + i, 'addN'); });
    var firstTensor = $tensors[0];
    $tensors.forEach(function (t) {
        if (t.dtype !== firstTensor.dtype) {
            throw new Error('All tensors passed to tf.addN() must have the same dtype');
        }
    });
    $tensors.forEach(function (t) {
        if (!util.arraysEqual(t.shape, firstTensor.shape)) {
            throw new Error('All tensors passed to tf.addN() must have the same shape');
        }
    });
    var der = function (dy) {
        var ders = {};
        $tensors.forEach(function (t, i) {
            ders[i] = function () { return dy.clone(); };
        });
        return ders;
    };
    var inputs = $tensors;
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.addN($tensors); }, inputs, der);
}
function addStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in addStrict: ');
    return a.add(b);
}
function sub_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'sub');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'sub');
    tensor_util_1.assertTypesMatch($a, $b);
    var outShape = broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () {
            var res = dy;
            var reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.reshape($a.shape);
        };
        var derB = function () {
            var res = dy;
            var reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.neg().reshape($b.shape);
        };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.subtract($a, $b); }, { $a: $a, $b: $b }, der);
}
function subStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
    return a.sub(b);
}
function pow_(base, exp) {
    var $base = tensor_util_env_1.convertToTensor(base, 'base', 'pow');
    var $exp = tensor_util_env_1.convertToTensor(exp, 'exp', 'pow');
    var outShape = broadcast_util.assertAndGetBroadcastShape($base.shape, $exp.shape);
    base = $base.cast(types_1.upcastType($base.dtype, $exp.dtype));
    exp = $exp.cast(types_1.upcastType($base.dtype, $exp.dtype));
    var grad = function (dy, saved) {
        var y = saved[0];
        var derBase = function () {
            var expFloat = $exp.toFloat();
            var res = dy.mul(expFloat.mul($base.pow(expFloat.sub(tensor_ops_1.scalar(1)))));
            var reduceAxes = broadcast_util.getReductionAxes($base.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.reshape($base.shape);
        };
        var derExp = function () {
            var res = dy.mul(y.mul($base.log()).toFloat());
            var reduceAxes = broadcast_util.getReductionAxes($exp.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.reshape($exp.shape);
        };
        return { $base: derBase, $exp: derExp };
    };
    return environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.pow($base, $exp)); }, { $base: $base, $exp: $exp }, grad);
}
function powStrict_(base, exp) {
    util.assertShapesMatch(base.shape, exp.shape, 'Error in powStrict: ');
    return base.pow(exp);
}
function mul_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'mul');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'mul');
    tensor_util_1.assertTypesMatch($a, $b);
    var outShape = broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () {
            var res = dy.mul($b.toFloat());
            var reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
            if (reduceAxes.length > 0) {
                return res.sum(reduceAxes).reshape($a.shape);
            }
            return res;
        };
        var derB = function () {
            var res = dy.mul($a.toFloat());
            var reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
            if (reduceAxes.length > 0) {
                return res.sum(reduceAxes).reshape($b.shape);
            }
            return res;
        };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.multiply($a, $b); }, { $a: $a, $b: $b }, der);
}
function mulStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in multiplyStrict: ');
    return a.mul(b);
}
function div_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'div');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'div');
    tensor_util_1.assertTypesMatch($a, $b);
    var forwardFunc;
    if ($a.dtype === 'int32' && $b.dtype === 'int32') {
        return exports.floorDiv($a, $b);
    }
    else {
        forwardFunc = function (backend) { return backend.realDivide($a, $b); };
    }
    var outShape = broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () {
            var res = dy.div($b.toFloat());
            var reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
            if (reduceAxes.length > 0) {
                return res.sum(reduceAxes).reshape($a.shape);
            }
            return res;
        };
        var derB = function () {
            var res = dy.mul($a.toFloat());
            var reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes).reshape($b.shape);
            }
            var tmp = $b.square();
            return res.div(tmp.toFloat()).neg();
        };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(forwardFunc, { $a: $a, $b: $b }, der);
}
function floorDiv_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'floorDiv');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'floorDiv');
    tensor_util_1.assertTypesMatch($a, $b);
    var forwardFunc = function (backend) { return backend.floorDiv($a, $b); };
    var outShape = broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () {
            var res = dy.div($b.toFloat());
            var reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
            if (reduceAxes.length > 0) {
                return res.sum(reduceAxes).reshape($a.shape);
            }
            return res;
        };
        var derB = function () {
            var res = dy.mul($a.toFloat());
            var reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes).reshape($b.shape);
            }
            var tmp = $b.square();
            return res.div(tmp.toFloat()).neg();
        };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(forwardFunc, { $a: $a, $b: $b }, der);
}
function divStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in divideStrict: ');
    return a.div(b);
}
function mod_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'mod');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'mod');
    tensor_util_1.assertTypesMatch($a, $b);
    var outShape = broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () {
            var reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
            if (reduceAxes.length > 0) {
                return dy.sum(reduceAxes).reshape($a.shape);
            }
            return dy;
        };
        var derB = function () {
            var res = dy.mul($a.div($b).floor().neg());
            var reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
            if (reduceAxes.length > 0) {
                return res.sum(reduceAxes).reshape($b.shape);
            }
            return res;
        };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.mod($a, $b); }, { $a: $a, $b: $b }, der);
}
function modStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in modStrict: ');
    return a.mod(b);
}
function minimum_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'minimum');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'minimum');
    tensor_util_1.assertTypesMatch($a, $b);
    if ($a.dtype === 'bool') {
        $a = $a.toInt();
    }
    if ($b.dtype === 'bool') {
        $b = $b.toInt();
    }
    broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () { return dy.mul($a.lessEqual($b).toFloat()); };
        var derB = function () { return dy.mul($a.greater($b).toFloat()); };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.minimum($a, $b); }, { $a: $a, $b: $b }, der);
}
function minimumStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in minimumStrict: ');
    return a.minimum(b);
}
function maximum_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'maximum');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'maximum');
    tensor_util_1.assertTypesMatch($a, $b);
    if ($a.dtype === 'bool') {
        $a = $a.toInt();
    }
    if ($b.dtype === 'bool') {
        $b = $b.toInt();
    }
    broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () { return dy.mul($a.greaterEqual($b).toFloat()); };
        var derB = function () { return dy.mul($a.less($b).toFloat()); };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.maximum($a, $b); }, { $a: $a, $b: $b }, der);
}
function maximumStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in maximumStrict: ');
    return a.maximum(b);
}
function squaredDifference_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'squaredDifference');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'squaredDifference');
    tensor_util_1.assertTypesMatch($a, $b);
    broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var two = tensor_ops_1.scalar(2);
        var derA = function () { return dy.mul($a.sub($b).mul(two)); };
        var derB = function () { return dy.mul($b.sub($a).mul(two)); };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.squaredDifference($a, $b); }, { $a: $a, $b: $b }, der);
}
function squaredDifferenceStrict_(a, b) {
    util.assertShapesMatch(a.shape, b.shape, 'Error in squaredDifferenceStrict: ');
    return a.squaredDifference(b);
}
function atan2_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'atan2');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'atan2');
    tensor_util_1.assertTypesMatch($a, $b);
    var outShape = broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
    var der = function (dy) {
        var derA = function () {
            var d = exports.add($a.square(), $b.square());
            var res = dy.mul($b.div(d));
            var reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.reshape($a.shape);
        };
        var derB = function () {
            var d = exports.add($a.square(), $b.square());
            var res = unary_ops_1.neg(dy.mul($a.div(d)));
            var reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
            if (reduceAxes.length > 0) {
                res = res.sum(reduceAxes);
            }
            return res.reshape($b.shape);
        };
        return { $a: derA, $b: derB };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.atan2($a, $b); }, { $a: $a, $b: $b }, der);
}
exports.add = operation_1.op({ add_: add_ });
exports.addN = operation_1.op({ addN_: addN_ });
exports.addStrict = operation_1.op({ addStrict_: addStrict_ });
exports.atan2 = operation_1.op({ atan2_: atan2_ });
exports.div = operation_1.op({ div_: div_ });
exports.divStrict = operation_1.op({ divStrict_: divStrict_ });
exports.floorDiv = operation_1.op({ floorDiv_: floorDiv_ });
exports.maximum = operation_1.op({ maximum_: maximum_ });
exports.maximumStrict = operation_1.op({ maximumStrict_: maximumStrict_ });
exports.minimum = operation_1.op({ minimum_: minimum_ });
exports.minimumStrict = operation_1.op({ minimumStrict_: minimumStrict_ });
exports.mod = operation_1.op({ mod_: mod_ });
exports.modStrict = operation_1.op({ modStrict_: modStrict_ });
exports.mul = operation_1.op({ mul_: mul_ });
exports.mulStrict = operation_1.op({ mulStrict_: mulStrict_ });
exports.pow = operation_1.op({ pow_: pow_ });
exports.powStrict = operation_1.op({ powStrict_: powStrict_ });
exports.squaredDifference = operation_1.op({ squaredDifference_: squaredDifference_ });
exports.squaredDifferenceStrict = operation_1.op({ squaredDifferenceStrict_: squaredDifferenceStrict_ });
exports.sub = operation_1.op({ sub_: sub_ });
exports.subStrict = operation_1.op({ subStrict_: subStrict_ });
//# sourceMappingURL=binary_ops.js.map