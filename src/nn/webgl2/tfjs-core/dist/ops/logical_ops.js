"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var where_impl_1 = require("../kernels/where_impl");
var tensor_util_env_1 = require("../tensor_util_env");
var util_1 = require("../util");
var broadcast_util_1 = require("./broadcast_util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function logicalNot_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'logicalNot', 'bool');
    util_1.assert($x.dtype === 'bool', 'Error Array must be of type bool.');
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.logicalNot($x); }, { $x: $x });
}
function logicalAnd_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'logicalAnd', 'bool');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'logicalAnd', 'bool');
    util_1.assert($a.dtype === 'bool' && $b.dtype === 'bool', 'Error Array must be of type bool.');
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.logicalAnd($a, $b); }, { $a: $a, $b: $b });
}
function logicalOr_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'logicalOr', 'bool');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'logicalOr', 'bool');
    util_1.assert($a.dtype === 'bool' && $b.dtype === 'bool', 'Error Array must be of type bool.');
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.logicalOr($a, $b); }, { $a: $a, $b: $b });
}
function logicalXor_(a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'logicalXor', 'bool');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'logicalXor', 'bool');
    util_1.assert($a.dtype === 'bool' && $b.dtype === 'bool', 'Error Array must be of type bool.');
    broadcast_util_1.assertAndGetBroadcastShape($a.shape, $b.shape);
    return exports.logicalOr(a, b).logicalAnd(exports.logicalAnd(a, b).logicalNot());
}
function where_(condition, a, b) {
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'where');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'where');
    var $condition = tensor_util_env_1.convertToTensor(condition, 'condition', 'where', 'bool');
    util_1.assert($condition.dtype === 'bool', 'Error Condition must be of type bool.');
    util_1.assertShapesMatch($a.shape, $b.shape, 'Error in where: ');
    if ($condition.rank === 1) {
        util_1.assert($condition.shape[0] === $a.shape[0], 'The first dimension of `a` must match the size of `condition`.');
    }
    else {
        util_1.assertShapesMatch($condition.shape, $b.shape, 'Error in where: ');
    }
    var grad = function (dy) { return ({
        $condition: function () { return tensor_ops_1.zerosLike($condition); },
        $a: function () { return dy.mul($condition.cast($a.dtype)); },
        $b: function () { return dy.mul($condition.logicalNot().cast($b.dtype)); }
    }); };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.select($condition, $a, $b); }, { $condition: $condition, $a: $a, $b: $b }, grad);
}
function whereAsync_(condition) {
    return __awaiter(this, void 0, void 0, function () {
        var $condition, vals, res;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    $condition = tensor_util_env_1.convertToTensor(condition, 'condition', 'where', 'bool');
                    util_1.assert($condition.dtype === 'bool', 'Condition must be of type bool.');
                    return [4, $condition.data()];
                case 1:
                    vals = _a.sent();
                    res = where_impl_1.whereImpl($condition.shape, vals);
                    if (condition !== $condition) {
                        $condition.dispose();
                    }
                    return [2, res];
            }
        });
    });
}
exports.logicalAnd = operation_1.op({ logicalAnd_: logicalAnd_ });
exports.logicalNot = operation_1.op({ logicalNot_: logicalNot_ });
exports.logicalOr = operation_1.op({ logicalOr_: logicalOr_ });
exports.logicalXor = operation_1.op({ logicalXor_: logicalXor_ });
exports.where = operation_1.op({ where_: where_ });
exports.whereAsync = whereAsync_;
//# sourceMappingURL=logical_ops.js.map