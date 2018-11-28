"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_1 = require("../tensor_util");
var tensor_util_env_1 = require("../tensor_util_env");
var util_1 = require("../util");
var broadcast_util_1 = require("./broadcast_util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function notEqual_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'notEqual');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'notEqual');
    tensor_util_1.assertTypesMatch($a, $b);
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.notEqual($a, $b); }, { $a: $a, $b: $b });
}
function notEqualStrict_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'notEqualStrict');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'notEqualStrict');
    util_1.assertShapesMatch($a.shape, $b.shape, 'Error in notEqualStrict: ');
    return $a.notEqual($b);
}
function less_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'less');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'less');
    tensor_util_1.assertTypesMatch($a, $b);
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.less($a, $b); }, { $a: $a, $b: $b });
}
function lessStrict_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'lessStrict');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'lessStrict');
    util_1.assertShapesMatch($a.shape, $b.shape, 'Error in lessStrict: ');
    return $a.less($b);
}
function equal_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'equal');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'equal');
    tensor_util_1.assertTypesMatch($a, $b);
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.equal($a, $b); }, { $a: $a, $b: $b });
}
function equalStrict_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'equalStrict');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'equalStrict');
    util_1.assertShapesMatch($a.shape, $b.shape, 'Error in equalStrict: ');
    return $a.equal($b);
}
function lessEqual_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'lessEqual');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'lessEqual');
    tensor_util_1.assertTypesMatch($a, $b);
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.lessEqual($a, $b); }, { $a: $a, $b: $b });
}
function lessEqualStrict_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'lessEqualStrict');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'lessEqualStrict');
    util_1.assertShapesMatch($a.shape, $b.shape, 'Error in lessEqualStrict: ');
    return $a.lessEqual($b);
}
function greater_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'greater');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'greater');
    tensor_util_1.assertTypesMatch($a, $b);
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.greater($a, $b); }, { $a: $a, $b: $b });
}
function greaterStrict_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'greaterStrict');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'greaterStrict');
    util_1.assertShapesMatch($a.shape, $b.shape, 'Error in greaterStrict: ');
    return $a.greater($b);
}
function greaterEqual_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'greaterEqual');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'greaterEqual');
    tensor_util_1.assertTypesMatch($a, $b);
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    var grad = function (dy) {
        return { $a: function () { return tensor_ops_1.zerosLike($a); }, $b: function () { return tensor_ops_1.zerosLike($b); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.greaterEqual($a, $b); }, { $a: $a, $b: $b }, grad);
}
function greaterEqualStrict_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'greaterEqualStrict');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'greaterEqualStrict');
    util_1.assertShapesMatch($a.shape, $b.shape, 'Error in greaterEqualStrict: ');
    return $a.greaterEqual($b);
}
exports.equal = operation_1.op({ equal_: equal_ });
exports.equalStrict = operation_1.op({ equalStrict_: equalStrict_ });
exports.greater = operation_1.op({ greater_: greater_ });
exports.greaterEqual = operation_1.op({ greaterEqual_: greaterEqual_ });
exports.greaterEqualStrict = operation_1.op({ greaterEqualStrict_: greaterEqualStrict_ });
exports.greaterStrict = operation_1.op({ greaterStrict_: greaterStrict_ });
exports.less = operation_1.op({ less_: less_ });
exports.lessEqual = operation_1.op({ lessEqual_: lessEqual_ });
exports.lessEqualStrict = operation_1.op({ lessEqualStrict_: lessEqualStrict_ });
exports.lessStrict = operation_1.op({ lessStrict_: lessStrict_ });
exports.notEqual = operation_1.op({ notEqual_: notEqual_ });
exports.notEqualStrict = operation_1.op({ notEqualStrict_: notEqualStrict_ });
//# sourceMappingURL=compare.js.map