"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function neg_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'neg');
    var grad = function (dy) {
        return { $x: function () { return dy.neg(); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.neg($x); }, { $x: $x }, grad);
}
function ceil_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'ceil');
    var grad = function (dy) {
        return { $x: function () { return tensor_ops_1.zerosLike(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.ceil($x); }, { $x: $x }, grad);
}
function floor_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'floor');
    var grad = function (dy) {
        return { $x: function () { return tensor_ops_1.zerosLike(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.floor($x); }, { $x: $x }, grad);
}
function sign_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'sign');
    var grad = function (dy) {
        return { $x: function () { return tensor_ops_1.zerosLike(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.sign($x); }, { $x: $x }, grad);
}
function round_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'round');
    var grad = function (dy) {
        return { $x: function () { return tensor_ops_1.zerosLike(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.round($x); }, { $x: $x }, grad);
}
function exp_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'exp');
    var bck = function (dy, saved) {
        var y = saved[0];
        return { $x: function () { return dy.mulStrict(y); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.exp($x)); }, { $x: $x }, bck);
}
function expm1_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'expm1');
    var grad = function (dy) {
        return { $x: function () { return dy.mulStrict($x.exp()); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.expm1($x); }, { $x: $x }, grad);
}
function log_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'log');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict($x.toFloat()); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.log($x); }, { $x: $x }, grad);
}
function log1p_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'log1p');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict($x.add(tensor_ops_1.scalar(1))); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.log1p($x); }, { $x: $x }, grad);
}
function sqrt_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'sqrt');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict($x.toFloat().sqrt().mul(tensor_ops_1.scalar(2))); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.sqrt($x); }, { $x: $x }, grad);
}
function rsqrt_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'rsqrt');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict($x.pow(tensor_ops_1.scalar(1.5)).mul(tensor_ops_1.scalar(2))).neg(); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.rsqrt($x); }, { $x: $x }, grad);
}
function square_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'square');
    var grad = function (dy) {
        return { $x: function () { return dy.mulStrict($x.toFloat().mul(tensor_ops_1.scalar(2))); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.square($x); }, { $x: $x }, grad);
}
function reciprocal_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'reciprocal');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict($x.square().neg()); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.reciprocal($x); }, { $x: $x }, grad);
}
function abs_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'abs');
    if ($x.dtype === 'complex64') {
        return environment_1.ENV.engine.runKernel(function (backend) { return backend.complexAbs($x); }, { $x: $x });
    }
    var grad = function (dy) {
        return { $x: function () { return dy.mulStrict($x.toFloat().step(-1)); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.abs($x); }, { $x: $x }, grad);
}
function clipByValue_(x, clipValueMin, clipValueMax) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'clipByValue');
    util.assert((clipValueMin <= clipValueMax), "Error in clip: min (" + clipValueMin + ") must be " +
        ("less than or equal to max (" + clipValueMax + ")."));
    var grad = function (dy) {
        return {
            $x: function () { return dy.where($x.greaterEqual(tensor_ops_1.scalar(clipValueMin))
                .logicalAnd($x.lessEqual(tensor_ops_1.scalar(clipValueMax))), tensor_ops_1.zerosLike(dy)); },
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.clip($x, clipValueMin, clipValueMax); }, { $x: $x }, grad);
}
function sigmoid_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'sigmoid');
    var grad = function (dy, saved) {
        var y = saved[0];
        return { $x: function () { return dy.mulStrict(y.mul(tensor_ops_1.scalar(1).sub(y))); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.sigmoid($x)); }, { $x: $x }, grad);
}
function logSigmoid_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'logSigmoid');
    var grad = function (dy) {
        return { $x: function () { return dy.mulStrict($x.neg().sigmoid()); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.softplus($x.neg()).neg(); }, { $x: $x }, grad);
}
function softplus_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'softplus');
    var grad = function (dy) {
        return { $x: function () { return dy.mulStrict($x.sigmoid()); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.softplus($x); }, { $x: $x }, grad);
}
function sin_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'sin');
    var grad = function (dy) {
        return { $x: function () { return $x.toFloat().cos().mulStrict(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.sin($x); }, { $x: $x }, grad);
}
function cos_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'cos');
    var grad = function (dy) {
        return { $x: function () { return $x.toFloat().sin().neg().mulStrict(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.cos($x); }, { $x: $x }, grad);
}
function tan_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'tan');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict($x.cos().square()); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.tan($x); }, { $x: $x }, grad);
}
function asin_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'asin');
    var grad = function (dy) {
        return {
            $x: function () { return dy.divStrict(tensor_ops_1.scalar(1).sub($x.toFloat().square()).sqrt()); }
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.asin($x); }, { $x: $x }, grad);
}
function acos_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'acos');
    var grad = function (dy) {
        return {
            $x: function () {
                return dy.divStrict(tensor_ops_1.scalar(1).sub($x.toFloat().square()).sqrt()).neg();
            }
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.acos($x); }, { $x: $x }, grad);
}
function atan_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'atan');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict(tensor_ops_1.scalar(1).add($x.toFloat().square())); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.atan($x); }, { $x: $x }, grad);
}
function sinh_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'sinh');
    var grad = function (dy) {
        return { $x: function () { return $x.toFloat().cosh().mulStrict(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.sinh($x); }, { $x: $x }, grad);
}
function cosh_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'cosh');
    var grad = function (dy) {
        return { $x: function () { return $x.toFloat().sinh().mulStrict(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.cosh($x); }, { $x: $x }, grad);
}
function tanh_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'tanh');
    var grad = function (dy, saved) {
        var y = saved[0];
        return { $x: function () { return tensor_ops_1.scalar(1).sub(y.square()).mulStrict(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.tanh($x)); }, { $x: $x }, grad);
}
function asinh_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'asinh');
    var grad = function (dy) {
        return {
            $x: function () { return dy.divStrict(tensor_ops_1.scalar(1).add($x.toFloat().square()).sqrt()); }
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.asinh($x); }, { $x: $x }, grad);
}
function acosh_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'acosh');
    var grad = function (dy) {
        return {
            $x: function () { return dy.divStrict($x.toFloat().square().sub(tensor_ops_1.scalar(1)).sqrt()); }
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.acosh($x); }, { $x: $x }, grad);
}
function atanh_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'atanh');
    var grad = function (dy) {
        return { $x: function () { return dy.divStrict(tensor_ops_1.scalar(1).sub($x.toFloat().square())); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.atanh($x); }, { $x: $x }, grad);
}
function erf_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'erf');
    util.assert($x.dtype === 'int32' || $x.dtype === 'float32', 'Input dtype must be `int32` or `float32`.');
    if ($x.dtype === 'int32') {
        $x = $x.toFloat();
    }
    var grad = function (dy) {
        return {
            $x: function () { return dy.mulStrict(tensor_ops_1.scalar(2 / Math.sqrt(Math.PI)).mul($x.square().neg().exp())); }
        };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.erf($x); }, { $x: $x }, grad);
}
function step_(x, alpha) {
    if (alpha === void 0) { alpha = 0.0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'step');
    var grad = function (dy) {
        return { $x: function () { return tensor_ops_1.zerosLike(dy); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.step($x, alpha); }, { $x: $x }, grad);
}
exports.abs = operation_1.op({ abs_: abs_ });
exports.acos = operation_1.op({ acos_: acos_ });
exports.acosh = operation_1.op({ acosh_: acosh_ });
exports.asin = operation_1.op({ asin_: asin_ });
exports.asinh = operation_1.op({ asinh_: asinh_ });
exports.atan = operation_1.op({ atan_: atan_ });
exports.atanh = operation_1.op({ atanh_: atanh_ });
exports.ceil = operation_1.op({ ceil_: ceil_ });
exports.clipByValue = operation_1.op({ clipByValue_: clipByValue_ });
exports.cos = operation_1.op({ cos_: cos_ });
exports.cosh = operation_1.op({ cosh_: cosh_ });
exports.erf = operation_1.op({ erf_: erf_ });
exports.exp = operation_1.op({ exp_: exp_ });
exports.expm1 = operation_1.op({ expm1_: expm1_ });
exports.floor = operation_1.op({ floor_: floor_ });
exports.log = operation_1.op({ log_: log_ });
exports.log1p = operation_1.op({ log1p_: log1p_ });
exports.logSigmoid = operation_1.op({ logSigmoid_: logSigmoid_ });
exports.neg = operation_1.op({ neg_: neg_ });
exports.reciprocal = operation_1.op({ reciprocal_: reciprocal_ });
exports.round = operation_1.op({ round_: round_ });
exports.rsqrt = operation_1.op({ rsqrt_: rsqrt_ });
exports.sigmoid = operation_1.op({ sigmoid_: sigmoid_ });
exports.sign = operation_1.op({ sign_: sign_ });
exports.sin = operation_1.op({ sin_: sin_ });
exports.sinh = operation_1.op({ sinh_: sinh_ });
exports.softplus = operation_1.op({ softplus_: softplus_ });
exports.sqrt = operation_1.op({ sqrt_: sqrt_ });
exports.square = operation_1.op({ square_: square_ });
exports.step = operation_1.op({ step_: step_ });
exports.tan = operation_1.op({ tan_: tan_ });
exports.tanh = operation_1.op({ tanh_: tanh_ });
//# sourceMappingURL=unary_ops.js.map