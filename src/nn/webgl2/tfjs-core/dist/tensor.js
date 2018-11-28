"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
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
var tensor_format_1 = require("./tensor_format");
var util = require("./util");
var util_1 = require("./util");
var TensorBuffer = (function () {
    function TensorBuffer(shape, dtype, values) {
        this.dtype = dtype;
        this.shape = shape.slice();
        this.size = util.sizeFromShape(shape);
        if (values != null) {
            var n = values.length;
            util.assert(n === this.size, "Length of values '" + n + "' does not match the size " +
                ("inferred by the shape '" + this.size + "'."));
        }
        if (dtype === 'complex64') {
            throw new Error("complex64 dtype TensorBuffers are not supported. Please create " +
                "a TensorBuffer for the real and imaginary parts separately and " +
                "call tf.complex(real, imag).");
        }
        this.values = values ||
            util.getTypedArrayFromDType(dtype, util.sizeFromShape(this.shape));
        this.strides = util_1.computeStrides(shape);
    }
    TensorBuffer.prototype.set = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        if (locs.length === 0) {
            locs = [0];
        }
        util.assert(locs.length === this.rank, "The number of provided coordinates (" + locs.length + ") must " +
            ("match the rank (" + this.rank + ")"));
        var index = this.locToIndex(locs);
        this.values[index] = value;
    };
    TensorBuffer.prototype.get = function () {
        var locs = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            locs[_i] = arguments[_i];
        }
        if (locs.length === 0) {
            locs = [0];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.values[index];
    };
    TensorBuffer.prototype.locToIndex = function (locs) {
        if (this.rank === 0) {
            return 0;
        }
        else if (this.rank === 1) {
            return locs[0];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    };
    TensorBuffer.prototype.indexToLoc = function (index) {
        if (this.rank === 0) {
            return [];
        }
        else if (this.rank === 1) {
            return [index];
        }
        var locs = new Array(this.shape.length);
        for (var i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    };
    Object.defineProperty(TensorBuffer.prototype, "rank", {
        get: function () {
            return this.shape.length;
        },
        enumerable: true,
        configurable: true
    });
    TensorBuffer.prototype.toTensor = function () {
        return Tensor.make(this.shape, { values: this.values }, this.dtype);
    };
    return TensorBuffer;
}());
exports.TensorBuffer = TensorBuffer;
var trackerFn = null;
var opHandler = null;
function setTensorTracker(fn) {
    trackerFn = fn;
}
exports.setTensorTracker = setTensorTracker;
function setOpHandler(handler) {
    opHandler = handler;
}
exports.setOpHandler = setOpHandler;
var Tensor = (function () {
    function Tensor(shape, dtype, values, dataId) {
        this.isDisposedInternal = false;
        this.shape = shape.slice();
        this.dtype = dtype || 'float32';
        this.size = util.sizeFromShape(shape);
        if (values != null) {
            util.assert(this.size === values.length, "Based on the provided shape, [" + shape + "], and dtype " +
                (this.dtype + ", the tensor should have ") +
                (this.size + " values but has " + values.length));
        }
        this.strides = util_1.computeStrides(shape);
        this.dataId = dataId != null ? dataId : {};
        this.id = trackerFn().nextTensorId();
        this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
        trackerFn().registerTensor(this);
        if (values != null) {
            trackerFn().write(this.dataId, values);
        }
    }
    Tensor.make = function (shape, data, dtype) {
        return new Tensor(shape, dtype, data.values, data.dataId);
    };
    Tensor.prototype.flatten = function () {
        this.throwIfDisposed();
        return this.as1D();
    };
    Tensor.prototype.asScalar = function () {
        this.throwIfDisposed();
        util.assert(this.size === 1, 'The array must have only 1 element.');
        return this.reshape([]);
    };
    Tensor.prototype.as1D = function () {
        this.throwIfDisposed();
        return this.reshape([this.size]);
    };
    Tensor.prototype.as2D = function (rows, columns) {
        this.throwIfDisposed();
        return this.reshape([rows, columns]);
    };
    Tensor.prototype.as3D = function (rows, columns, depth) {
        this.throwIfDisposed();
        return this.reshape([rows, columns, depth]);
    };
    Tensor.prototype.as4D = function (rows, columns, depth, depth2) {
        this.throwIfDisposed();
        return this.reshape([rows, columns, depth, depth2]);
    };
    Tensor.prototype.asType = function (dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    };
    Object.defineProperty(Tensor.prototype, "rank", {
        get: function () {
            return this.shape.length;
        },
        enumerable: true,
        configurable: true
    });
    Tensor.prototype.get = function () {
        var locs = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            locs[_i] = arguments[_i];
        }
        util.assert(locs.length === this.rank, 'Number of coordinates in get() must match the rank of the tensor');
        util.assert(this.dtype !== 'complex64', 'Tensor.get() is not supported for complex64 tensors yet.');
        this.throwIfDisposed();
        if (locs.length === 0) {
            locs = [0];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.dataSync()[index];
    };
    Tensor.prototype.buffer = function () {
        return opHandler.buffer(this.shape, this.dtype, this.dataSync());
    };
    Tensor.prototype.data = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                this.throwIfDisposed();
                return [2, trackerFn().read(this.dataId)];
            });
        });
    };
    Tensor.prototype.dataSync = function () {
        this.throwIfDisposed();
        return trackerFn().readSync(this.dataId);
    };
    Tensor.prototype.dispose = function () {
        if (this.isDisposed) {
            return;
        }
        trackerFn().disposeTensor(this);
        this.isDisposedInternal = true;
    };
    Object.defineProperty(Tensor.prototype, "isDisposed", {
        get: function () {
            return this.isDisposedInternal;
        },
        enumerable: true,
        configurable: true
    });
    Tensor.prototype.throwIfDisposed = function () {
        if (this.isDisposed) {
            throw new Error("Tensor is disposed.");
        }
    };
    Tensor.prototype.toFloat = function () {
        return this.asType('float32');
    };
    Tensor.prototype.toInt = function () {
        return this.asType('int32');
    };
    Tensor.prototype.toBool = function () {
        return this.asType('bool');
    };
    Tensor.prototype.print = function (verbose) {
        if (verbose === void 0) { verbose = false; }
        return opHandler.print(this, verbose);
    };
    Tensor.prototype.reshape = function (newShape) {
        this.throwIfDisposed();
        return opHandler.reshape(this, newShape);
    };
    Tensor.prototype.reshapeAs = function (x) {
        this.throwIfDisposed();
        return this.reshape(x.shape);
    };
    Tensor.prototype.expandDims = function (axis) {
        if (axis === void 0) { axis = 0; }
        return opHandler.expandDims(this, axis);
    };
    Tensor.prototype.cumsum = function (axis, exclusive, reverse) {
        if (axis === void 0) { axis = 0; }
        if (exclusive === void 0) { exclusive = false; }
        if (reverse === void 0) { reverse = false; }
        return opHandler.cumsum(this, axis, exclusive, reverse);
    };
    Tensor.prototype.squeeze = function (axis) {
        this.throwIfDisposed();
        return opHandler.squeeze(this, axis);
    };
    Tensor.prototype.clone = function () {
        this.throwIfDisposed();
        return opHandler.clone(this);
    };
    Tensor.prototype.toString = function (verbose) {
        if (verbose === void 0) { verbose = false; }
        var vals = this.dataSync();
        return tensor_format_1.tensorToString(vals, this.shape, this.dtype, verbose);
    };
    Tensor.prototype.tile = function (reps) {
        this.throwIfDisposed();
        return opHandler.tile(this, reps);
    };
    Tensor.prototype.gather = function (indices, axis) {
        if (axis === void 0) { axis = 0; }
        this.throwIfDisposed();
        return opHandler.gather(this, indices, axis);
    };
    Tensor.prototype.matMul = function (b, transposeA, transposeB) {
        if (transposeA === void 0) { transposeA = false; }
        if (transposeB === void 0) { transposeB = false; }
        this.throwIfDisposed();
        return opHandler.matMul(this, b, transposeA, transposeB);
    };
    Tensor.prototype.dot = function (b) {
        this.throwIfDisposed();
        return opHandler.dot(this, b);
    };
    Tensor.prototype.norm = function (ord, axis, keepDims) {
        if (ord === void 0) { ord = 'euclidean'; }
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.norm(this, ord, axis, keepDims);
    };
    Tensor.prototype.slice = function (begin, size) {
        this.throwIfDisposed();
        return opHandler.slice(this, begin, size);
    };
    Tensor.prototype.reverse = function (axis) {
        this.throwIfDisposed();
        return opHandler.reverse(this, axis);
    };
    Tensor.prototype.concat = function (x, axis) {
        if (axis === void 0) { axis = 0; }
        this.throwIfDisposed();
        return opHandler.concat([this, x], axis);
    };
    Tensor.prototype.split = function (numOrSizeSplits, axis) {
        if (axis === void 0) { axis = 0; }
        this.throwIfDisposed();
        return opHandler.split(this, numOrSizeSplits, axis);
    };
    Tensor.prototype.stack = function (x, axis) {
        if (axis === void 0) { axis = 0; }
        return opHandler.stack([this, x], axis);
    };
    Tensor.prototype.unstack = function (x, axis) {
        if (axis === void 0) { axis = 0; }
        return opHandler.unstack(this, axis);
    };
    Tensor.prototype.pad = function (paddings, constantValue) {
        if (constantValue === void 0) { constantValue = 0; }
        return opHandler.pad(this, paddings, constantValue);
    };
    Tensor.prototype.batchNormalization = function (mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        this.throwIfDisposed();
        return opHandler.batchNormalization(this, mean, variance, varianceEpsilon, scale, offset);
    };
    Tensor.prototype.all = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.all(this, axis, keepDims);
    };
    Tensor.prototype.any = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.any(this, axis, keepDims);
    };
    Tensor.prototype.logSumExp = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.logSumExp(this, axis, keepDims);
    };
    Tensor.prototype.sum = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.sum(this, axis, keepDims);
    };
    Tensor.prototype.prod = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.prod(this, axis, keepDims);
    };
    Tensor.prototype.mean = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.mean(this, axis, keepDims);
    };
    Tensor.prototype.min = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.min(this, axis, keepDims);
    };
    Tensor.prototype.max = function (axis, keepDims) {
        if (axis === void 0) { axis = null; }
        if (keepDims === void 0) { keepDims = false; }
        this.throwIfDisposed();
        return opHandler.max(this, axis, keepDims);
    };
    Tensor.prototype.argMin = function (axis) {
        if (axis === void 0) { axis = null; }
        this.throwIfDisposed();
        return opHandler.argMin(this, axis);
    };
    Tensor.prototype.argMax = function (axis) {
        if (axis === void 0) { axis = null; }
        this.throwIfDisposed();
        return opHandler.argMax(this, axis);
    };
    Tensor.prototype.cast = function (dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    };
    Tensor.prototype.add = function (x) {
        this.throwIfDisposed();
        return opHandler.add(this, x);
    };
    Tensor.prototype.addStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.addStrict(this, x);
    };
    Tensor.prototype.atan2 = function (x) {
        this.throwIfDisposed();
        return opHandler.atan2(this, x);
    };
    Tensor.prototype.sub = function (x) {
        this.throwIfDisposed();
        return opHandler.sub(this, x);
    };
    Tensor.prototype.subStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.subStrict(this, x);
    };
    Tensor.prototype.pow = function (exp) {
        this.throwIfDisposed();
        return opHandler.pow(this, exp);
    };
    Tensor.prototype.powStrict = function (exp) {
        this.throwIfDisposed();
        return opHandler.powStrict(this, exp);
    };
    Tensor.prototype.mul = function (x) {
        this.throwIfDisposed();
        return opHandler.mul(this, x);
    };
    Tensor.prototype.mulStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.mulStrict(this, x);
    };
    Tensor.prototype.div = function (x) {
        this.throwIfDisposed();
        return opHandler.div(this, x);
    };
    Tensor.prototype.floorDiv = function (x) {
        this.throwIfDisposed();
        return opHandler.floorDiv(this, x);
    };
    Tensor.prototype.divStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.divStrict(this, x);
    };
    Tensor.prototype.minimum = function (x) {
        this.throwIfDisposed();
        return opHandler.minimum(this, x);
    };
    Tensor.prototype.minimumStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.minimumStrict(this, x);
    };
    Tensor.prototype.maximum = function (x) {
        this.throwIfDisposed();
        return opHandler.maximum(this, x);
    };
    Tensor.prototype.maximumStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.maximumStrict(this, x);
    };
    Tensor.prototype.mod = function (x) {
        this.throwIfDisposed();
        return opHandler.mod(this, x);
    };
    Tensor.prototype.modStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.modStrict(this, x);
    };
    Tensor.prototype.squaredDifference = function (x) {
        this.throwIfDisposed();
        return opHandler.squaredDifference(this, x);
    };
    Tensor.prototype.squaredDifferenceStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.squaredDifferenceStrict(this, x);
    };
    Tensor.prototype.transpose = function (perm) {
        this.throwIfDisposed();
        return opHandler.transpose(this, perm);
    };
    Tensor.prototype.notEqual = function (x) {
        this.throwIfDisposed();
        return opHandler.notEqual(this, x);
    };
    Tensor.prototype.notEqualStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.notEqualStrict(this, x);
    };
    Tensor.prototype.less = function (x) {
        this.throwIfDisposed();
        return opHandler.less(this, x);
    };
    Tensor.prototype.lessStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.lessStrict(this, x);
    };
    Tensor.prototype.equal = function (x) {
        this.throwIfDisposed();
        return opHandler.equal(this, x);
    };
    Tensor.prototype.equalStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.equalStrict(this, x);
    };
    Tensor.prototype.lessEqual = function (x) {
        this.throwIfDisposed();
        return opHandler.lessEqual(this, x);
    };
    Tensor.prototype.lessEqualStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.lessEqualStrict(this, x);
    };
    Tensor.prototype.greater = function (x) {
        this.throwIfDisposed();
        return opHandler.greater(this, x);
    };
    Tensor.prototype.greaterStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.greaterStrict(this, x);
    };
    Tensor.prototype.greaterEqual = function (x) {
        this.throwIfDisposed();
        return opHandler.greaterEqual(this, x);
    };
    Tensor.prototype.greaterEqualStrict = function (x) {
        this.throwIfDisposed();
        return opHandler.greaterEqualStrict(this, x);
    };
    Tensor.prototype.logicalAnd = function (x) {
        this.throwIfDisposed();
        return opHandler.logicalAnd(this, x);
    };
    Tensor.prototype.logicalOr = function (x) {
        this.throwIfDisposed();
        return opHandler.logicalOr(this, x);
    };
    Tensor.prototype.logicalNot = function () {
        this.throwIfDisposed();
        return opHandler.logicalNot(this);
    };
    Tensor.prototype.logicalXor = function (x) {
        this.throwIfDisposed();
        return opHandler.logicalXor(this, x);
    };
    Tensor.prototype.where = function (condition, x) {
        this.throwIfDisposed();
        return opHandler.where(condition, this, x);
    };
    Tensor.prototype.neg = function () {
        this.throwIfDisposed();
        return opHandler.neg(this);
    };
    Tensor.prototype.ceil = function () {
        this.throwIfDisposed();
        return opHandler.ceil(this);
    };
    Tensor.prototype.floor = function () {
        this.throwIfDisposed();
        return opHandler.floor(this);
    };
    Tensor.prototype.sign = function () {
        this.throwIfDisposed();
        return opHandler.sign(this);
    };
    Tensor.prototype.exp = function () {
        this.throwIfDisposed();
        return opHandler.exp(this);
    };
    Tensor.prototype.expm1 = function () {
        this.throwIfDisposed();
        return opHandler.expm1(this);
    };
    Tensor.prototype.log = function () {
        this.throwIfDisposed();
        return opHandler.log(this);
    };
    Tensor.prototype.log1p = function () {
        this.throwIfDisposed();
        return opHandler.log1p(this);
    };
    Tensor.prototype.sqrt = function () {
        this.throwIfDisposed();
        return opHandler.sqrt(this);
    };
    Tensor.prototype.rsqrt = function () {
        this.throwIfDisposed();
        return opHandler.rsqrt(this);
    };
    Tensor.prototype.square = function () {
        this.throwIfDisposed();
        return opHandler.square(this);
    };
    Tensor.prototype.reciprocal = function () {
        this.throwIfDisposed();
        return opHandler.reciprocal(this);
    };
    Tensor.prototype.abs = function () {
        this.throwIfDisposed();
        return opHandler.abs(this);
    };
    Tensor.prototype.clipByValue = function (min, max) {
        this.throwIfDisposed();
        return opHandler.clipByValue(this, min, max);
    };
    Tensor.prototype.relu = function () {
        this.throwIfDisposed();
        return opHandler.relu(this);
    };
    Tensor.prototype.elu = function () {
        this.throwIfDisposed();
        return opHandler.elu(this);
    };
    Tensor.prototype.selu = function () {
        this.throwIfDisposed();
        return opHandler.selu(this);
    };
    Tensor.prototype.leakyRelu = function (alpha) {
        if (alpha === void 0) { alpha = 0.2; }
        this.throwIfDisposed();
        return opHandler.leakyRelu(this, alpha);
    };
    Tensor.prototype.prelu = function (alpha) {
        this.throwIfDisposed();
        return opHandler.prelu(this, alpha);
    };
    Tensor.prototype.sigmoid = function () {
        this.throwIfDisposed();
        return opHandler.sigmoid(this);
    };
    Tensor.prototype.logSigmoid = function () {
        this.throwIfDisposed();
        return opHandler.logSigmoid(this);
    };
    Tensor.prototype.softplus = function () {
        this.throwIfDisposed();
        return opHandler.softplus(this);
    };
    Tensor.prototype.zerosLike = function () {
        this.throwIfDisposed();
        return opHandler.zerosLike(this);
    };
    Tensor.prototype.onesLike = function () {
        this.throwIfDisposed();
        return opHandler.onesLike(this);
    };
    Tensor.prototype.sin = function () {
        this.throwIfDisposed();
        return opHandler.sin(this);
    };
    Tensor.prototype.cos = function () {
        this.throwIfDisposed();
        return opHandler.cos(this);
    };
    Tensor.prototype.tan = function () {
        this.throwIfDisposed();
        return opHandler.tan(this);
    };
    Tensor.prototype.asin = function () {
        this.throwIfDisposed();
        return opHandler.asin(this);
    };
    Tensor.prototype.acos = function () {
        this.throwIfDisposed();
        return opHandler.acos(this);
    };
    Tensor.prototype.atan = function () {
        this.throwIfDisposed();
        return opHandler.atan(this);
    };
    Tensor.prototype.sinh = function () {
        this.throwIfDisposed();
        return opHandler.sinh(this);
    };
    Tensor.prototype.cosh = function () {
        this.throwIfDisposed();
        return opHandler.cosh(this);
    };
    Tensor.prototype.tanh = function () {
        this.throwIfDisposed();
        return opHandler.tanh(this);
    };
    Tensor.prototype.asinh = function () {
        this.throwIfDisposed();
        return opHandler.asinh(this);
    };
    Tensor.prototype.acosh = function () {
        this.throwIfDisposed();
        return opHandler.acosh(this);
    };
    Tensor.prototype.atanh = function () {
        this.throwIfDisposed();
        return opHandler.atanh(this);
    };
    Tensor.prototype.erf = function () {
        this.throwIfDisposed();
        return opHandler.erf(this);
    };
    Tensor.prototype.round = function () {
        this.throwIfDisposed();
        return opHandler.round(this);
    };
    Tensor.prototype.step = function (alpha) {
        if (alpha === void 0) { alpha = 0.0; }
        this.throwIfDisposed();
        return opHandler.step(this, alpha);
    };
    Tensor.prototype.softmax = function (dim) {
        if (dim === void 0) { dim = -1; }
        this.throwIfDisposed();
        return opHandler.softmax(this, dim);
    };
    Tensor.prototype.logSoftmax = function (axis) {
        if (axis === void 0) { axis = -1; }
        this.throwIfDisposed();
        return opHandler.logSoftmax(this, axis);
    };
    Tensor.prototype.resizeBilinear = function (newShape2D, alignCorners) {
        if (alignCorners === void 0) { alignCorners = false; }
        this.throwIfDisposed();
        return opHandler.image.resizeBilinear(this, newShape2D, alignCorners);
    };
    Tensor.prototype.resizeNearestNeighbor = function (newShape2D, alignCorners) {
        if (alignCorners === void 0) { alignCorners = false; }
        this.throwIfDisposed();
        return opHandler.image.resizeNearestNeighbor(this, newShape2D, alignCorners);
    };
    Tensor.prototype.conv1d = function (filter, stride, pad, dataFormat, dilation, dimRoundingMode) {
        if (dataFormat === void 0) { dataFormat = 'NWC'; }
        if (dilation === void 0) { dilation = 1; }
        this.throwIfDisposed();
        return opHandler.conv1d(this, filter, stride, pad, dataFormat, dilation, dimRoundingMode);
    };
    Tensor.prototype.conv2d = function (filter, strides, pad, dataFormat, dilations, dimRoundingMode) {
        if (dataFormat === void 0) { dataFormat = 'NHWC'; }
        if (dilations === void 0) { dilations = [1, 1]; }
        this.throwIfDisposed();
        return opHandler.conv2d(this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
    };
    Tensor.prototype.conv2dTranspose = function (filter, outputShape, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return opHandler.conv2dTranspose(this, filter, outputShape, strides, pad, dimRoundingMode);
    };
    Tensor.prototype.depthwiseConv2D = function (filter, strides, pad, dataFormat, dilations, dimRoundingMode) {
        if (dataFormat === void 0) { dataFormat = 'NHWC'; }
        if (dilations === void 0) { dilations = [1, 1]; }
        this.throwIfDisposed();
        return opHandler.depthwiseConv2d(this, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
    };
    Tensor.prototype.separableConv2d = function (depthwiseFilter, pointwiseFilter, strides, pad, dilation, dataFormat) {
        if (dilation === void 0) { dilation = [1, 1]; }
        if (dataFormat === void 0) { dataFormat = 'NHWC'; }
        this.throwIfDisposed();
        return opHandler.separableConv2d(this, depthwiseFilter, pointwiseFilter, strides, pad, dilation, dataFormat);
    };
    Tensor.prototype.avgPool = function (filterSize, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return opHandler.avgPool(this, filterSize, strides, pad, dimRoundingMode);
    };
    Tensor.prototype.maxPool = function (filterSize, strides, pad, dimRoundingMode) {
        this.throwIfDisposed();
        return opHandler.maxPool(this, filterSize, strides, pad, dimRoundingMode);
    };
    Tensor.prototype.localResponseNormalization = function (radius, bias, alpha, beta) {
        if (radius === void 0) { radius = 5; }
        if (bias === void 0) { bias = 1; }
        if (alpha === void 0) { alpha = 1; }
        if (beta === void 0) { beta = 0.5; }
        return opHandler.localResponseNormalization(this, radius, bias, alpha, beta);
    };
    Tensor.prototype.pool = function (windowShape, poolingType, padding, dilationRate, strides) {
        this.throwIfDisposed();
        return opHandler.pool(this, windowShape, poolingType, padding, dilationRate, strides);
    };
    Tensor.prototype.variable = function (trainable, name, dtype) {
        if (trainable === void 0) { trainable = true; }
        this.throwIfDisposed();
        return Variable.variable(this, trainable, name, dtype);
    };
    Tensor.prototype.unsortedSegmentSum = function (segmentIds, numSegments) {
        this.throwIfDisposed();
        return opHandler.unsortedSegmentSum(this, segmentIds, numSegments);
    };
    Tensor.prototype.batchToSpaceND = function (blockShape, crops) {
        this.throwIfDisposed();
        return opHandler.batchToSpaceND(this, blockShape, crops);
    };
    Tensor.prototype.spaceToBatchND = function (blockShape, paddings) {
        this.throwIfDisposed();
        return opHandler.spaceToBatchND(this, blockShape, paddings);
    };
    Tensor.prototype.topk = function (k, sorted) {
        if (k === void 0) { k = 1; }
        if (sorted === void 0) { sorted = true; }
        this.throwIfDisposed();
        return opHandler.topk(this, k, sorted);
    };
    Tensor.prototype.stridedSlice = function (begin, end, strides, beginMask, endMask) {
        if (beginMask === void 0) { beginMask = 0; }
        if (endMask === void 0) { endMask = 0; }
        this.throwIfDisposed();
        return opHandler.stridedSlice(this, begin, end, strides, beginMask, endMask);
    };
    Tensor.prototype.depthToSpace = function (blockSize, dataFormat) {
        this.throwIfDisposed();
        return opHandler.depthToSpace(this, blockSize, dataFormat);
    };
    Tensor.prototype.fft = function () {
        this.throwIfDisposed();
        return opHandler.spectral.fft(this);
    };
    Tensor.prototype.ifft = function () {
        this.throwIfDisposed();
        return opHandler.spectral.ifft(this);
    };
    return Tensor;
}());
exports.Tensor = Tensor;
Object.defineProperty(Tensor, Symbol.hasInstance, {
    value: function (instance) {
        return !!instance && instance.shape != null && instance.dtype != null;
    }
});
var Variable = (function (_super) {
    __extends(Variable, _super);
    function Variable(initialValue, trainable, name) {
        if (trainable === void 0) { trainable = true; }
        var _this = _super.call(this, initialValue.shape, initialValue.dtype, null, initialValue.dataId) || this;
        _this.trainable = trainable;
        _this.name = name;
        if (_this.name == null) {
            _this.name = trackerFn().nextVariableId().toString();
        }
        try {
            trackerFn().registerVariable(_this);
        }
        catch (ex) {
            trackerFn().disposeTensor(_this);
            throw ex;
        }
        return _this;
    }
    Variable.variable = function (initialValue, trainable, name, dtype) {
        if (trainable === void 0) { trainable = true; }
        if (dtype != null && dtype !== initialValue.dtype) {
            initialValue = initialValue.asType(dtype);
        }
        return new Variable(initialValue, trainable, name);
    };
    Variable.prototype.assign = function (newValue) {
        if (newValue.dtype !== this.dtype) {
            throw new Error("dtype of the new value (" + newValue.dtype + ") and " +
                ("previous value (" + this.dtype + ") must match"));
        }
        if (!util.arraysEqual(newValue.shape, this.shape)) {
            throw new Error("shape of the new value (" + newValue.shape + ") and " +
                ("previous value (" + this.shape + ") must match"));
        }
        trackerFn().disposeTensor(this);
        this.dataId = newValue.dataId;
        trackerFn().registerTensor(this);
    };
    return Variable;
}(Tensor));
exports.Variable = Variable;
Object.defineProperty(Variable, Symbol.hasInstance, {
    value: function (instance) {
        return instance instanceof Tensor && instance.assign != null &&
            instance.assign instanceof Function;
    }
});
var variable = Variable.variable;
exports.variable = variable;
//# sourceMappingURL=tensor.js.map