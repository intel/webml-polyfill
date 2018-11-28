"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_util_env_1 = require("../tensor_util_env");
var axis_util = require("./axis_util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function norm_(x, ord, axis, keepDims) {
    if (ord === void 0) { ord = 'euclidean'; }
    if (axis === void 0) { axis = null; }
    if (keepDims === void 0) { keepDims = false; }
    x = tensor_util_env_1.convertToTensor(x, 'x', 'norm');
    var norm = normImpl(x, ord, axis);
    var keepDimsShape = norm.shape;
    if (keepDims) {
        var axes = axis_util.parseAxisParam(axis, x.shape);
        keepDimsShape = axis_util.expandShapeToKeepDim(norm.shape, axes);
    }
    return norm.reshape(keepDimsShape);
}
function normImpl(x, p, axis) {
    if (axis === void 0) { axis = null; }
    if (x.rank === 0) {
        return x.abs();
    }
    if (x.rank !== 1 && axis === null) {
        return normImpl(x.reshape([-1]), p, axis);
    }
    if (x.rank === 1 || typeof axis === 'number' ||
        axis instanceof Array && axis.length === 1) {
        if (p === 1) {
            return x.abs().sum(axis);
        }
        if (p === Infinity) {
            return x.abs().max(axis);
        }
        if (p === -Infinity) {
            return x.abs().min(axis);
        }
        if (p === 'euclidean' || p === 2) {
            return x.abs().pow(tensor_ops_1.scalar(2, 'int32')).sum(axis).sqrt();
        }
        throw new Error("Error in norm: invalid ord value: " + p);
    }
    if (axis instanceof Array && axis.length === 2) {
        if (p === 1) {
            return x.abs().sum(axis[0]).max(axis[1] - 1);
        }
        if (p === Infinity) {
            return x.abs().sum(axis[1]).max(axis[0]);
        }
        if (p === -Infinity) {
            return x.abs().sum(axis[1]).min(axis[0]);
        }
        if (p === 'fro' || p === 'euclidean') {
            return x.square().sum(axis).sqrt();
        }
        throw new Error("Error in norm: invalid ord value: " + p);
    }
    throw new Error("Error in norm: invalid axis: " + axis);
}
exports.norm = operation_1.op({ norm_: norm_ });
//# sourceMappingURL=norm.js.map