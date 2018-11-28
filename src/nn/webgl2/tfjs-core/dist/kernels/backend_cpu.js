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
var seedrandom = require("seedrandom");
var environment_1 = require("../environment");
var log_1 = require("../log");
var array_ops_util = require("../ops/array_ops_util");
var axis_util = require("../ops/axis_util");
var broadcast_util = require("../ops/broadcast_util");
var concat_util = require("../ops/concat_util");
var erf_util = require("../ops/erf_util");
var gather_nd_util = require("../ops/gather_nd_util");
var ops = require("../ops/ops");
var ops_1 = require("../ops/ops");
var scatter_nd_util = require("../ops/scatter_nd_util");
var selu_util = require("../ops/selu_util");
var slice_util_1 = require("../ops/slice_util");
var tensor_1 = require("../tensor");
var types_1 = require("../types");
var util = require("../util");
var util_1 = require("../util");
var backend_1 = require("./backend");
var backend_util = require("./backend_util");
var complex_util = require("./complex_util");
var non_max_suppression_impl_1 = require("./non_max_suppression_impl");
var split_shared_1 = require("./split_shared");
var topk_impl_1 = require("./topk_impl");
var where_impl_1 = require("./where_impl");
var MathBackendCPU = (function () {
    function MathBackendCPU() {
        this.blockSize = 48;
        this.firstUse = true;
        if (environment_1.ENV.get('IS_BROWSER')) {
            this.fromPixels2DContext =
                document.createElement('canvas').getContext('2d');
        }
    }
    MathBackendCPU.prototype.setDataMover = function (dataMover) {
        this.data = new backend_1.DataStorage(dataMover);
    };
    MathBackendCPU.prototype.register = function (dataId, shape, dtype) {
        if (this.firstUse) {
            this.firstUse = false;
            if (environment_1.ENV.get('IS_NODE')) {
                log_1.warn('\n============================\n' +
                    'Hi there 👋. Looks like you are running TensorFlow.js in ' +
                    'Node.js. To speed things up dramatically, install our node ' +
                    'backend, which binds to TensorFlow C++, by running ' +
                    'npm i @tensorflow/tfjs-node, ' +
                    'or npm i @tensorflow/tfjs-node-gpu if you have CUDA. ' +
                    'Then call require(\'@tensorflow/tfjs-node\'); (-gpu ' +
                    'suffix for CUDA) at the start of your program. ' +
                    'Visit https://github.com/tensorflow/tfjs-node for more details.' +
                    '\n============================\n');
            }
        }
        if (this.data.has(dataId)) {
            throw new Error("Data buffer is already registered");
        }
        this.data.set(dataId, { dtype: dtype });
    };
    MathBackendCPU.prototype.write = function (dataId, values) {
        if (values == null) {
            throw new Error('MathBackendCPU.write(): values can not be null');
        }
        this.data.get(dataId).values = values;
    };
    MathBackendCPU.prototype.fromPixels = function (pixels, numChannels) {
        if (pixels == null) {
            throw new Error('pixels passed to tf.fromPixels() can not be null');
        }
        var vals;
        if (environment_1.ENV.get('IS_NODE') && pixels.getContext == null) {
            throw new Error('When running in node, pixels must be an HTMLCanvasElement ' +
                'like the one returned by the `canvas` npm package');
        }
        if (pixels.getContext != null) {
            vals = pixels
                .getContext('2d')
                .getImageData(0, 0, pixels.width, pixels.height)
                .data;
        }
        else if (pixels instanceof ImageData) {
            vals = pixels.data;
        }
        else if (pixels instanceof HTMLImageElement ||
            pixels instanceof HTMLVideoElement) {
            if (this.fromPixels2DContext == null) {
                throw new Error('Can\'t read pixels from HTMLImageElement outside ' +
                    'the browser.');
            }
            this.fromPixels2DContext.canvas.width = pixels.width;
            this.fromPixels2DContext.canvas.height = pixels.height;
            this.fromPixels2DContext.drawImage(pixels, 0, 0, pixels.width, pixels.height);
            vals = this.fromPixels2DContext
                .getImageData(0, 0, pixels.width, pixels.height)
                .data;
        }
        else {
            throw new Error('pixels passed to tf.fromPixels() must be either an ' +
                "HTMLVideoElement, HTMLImageElement, HTMLCanvasElement or " +
                ("ImageData, but was " + pixels.constructor.name));
        }
        var values;
        if (numChannels === 4) {
            values = new Int32Array(vals);
        }
        else {
            var numPixels = pixels.width * pixels.height;
            values = new Int32Array(numPixels * numChannels);
            for (var i = 0; i < numPixels; i++) {
                for (var channel = 0; channel < numChannels; ++channel) {
                    values[i * numChannels + channel] = vals[i * 4 + channel];
                }
            }
        }
        var outShape = [pixels.height, pixels.width, numChannels];
        return ops_1.tensor3d(values, outShape, 'int32');
    };
    MathBackendCPU.prototype.read = function (dataId) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2, this.readSync(dataId)];
            });
        });
    };
    MathBackendCPU.prototype.readSync = function (dataId) {
        var _a = this.data.get(dataId), dtype = _a.dtype, complexTensors = _a.complexTensors;
        if (dtype === 'complex64') {
            var realValues = complexTensors.real.dataSync();
            var imagValues = complexTensors.imag.dataSync();
            return complex_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        return this.data.get(dataId).values;
    };
    MathBackendCPU.prototype.disposeData = function (dataId) {
        if (this.data.has(dataId)) {
            var complexTensors = this.data.get(dataId).complexTensors;
            if (complexTensors != null) {
                complexTensors.real.dispose();
                complexTensors.imag.dispose();
            }
            this.data.delete(dataId);
        }
    };
    MathBackendCPU.prototype.time = function (f) {
        return __awaiter(this, void 0, void 0, function () {
            var start, kernelMs;
            return __generator(this, function (_a) {
                start = util_1.now();
                f();
                kernelMs = util_1.now() - start;
                return [2, { kernelMs: kernelMs }];
            });
        });
    };
    MathBackendCPU.prototype.memory = function () {
        return {
            unreliable: true
        };
    };
    MathBackendCPU.prototype.complex = function (real, imag) {
        var result = tensor_1.Tensor.make(real.shape, {}, 'complex64');
        var resultData = this.data.get(result.dataId);
        resultData.complexTensors = {
            real: environment_1.ENV.engine.keep(real.clone()),
            imag: environment_1.ENV.engine.keep(imag.clone())
        };
        return result;
    };
    MathBackendCPU.prototype.real = function (input) {
        var resultData = this.data.get(input.dataId);
        return resultData.complexTensors.real.clone();
    };
    MathBackendCPU.prototype.imag = function (input) {
        var resultData = this.data.get(input.dataId);
        return resultData.complexTensors.imag.clone();
    };
    MathBackendCPU.prototype.assertNotComplex = function (tensor, opName) {
        if (!Array.isArray(tensor)) {
            tensor = [tensor];
        }
        tensor.forEach(function (t) {
            if (t != null) {
                util.assert(t.dtype !== 'complex64', opName + " does not support complex64 tensors.");
            }
        });
    };
    MathBackendCPU.prototype.slice = function (x, begin, size) {
        this.assertNotComplex(x, 'slice');
        var buffer = ops.buffer(size, x.dtype);
        for (var i = 0; i < buffer.size; ++i) {
            var loc = buffer.indexToLoc(i);
            var xLoc = loc.map(function (idx, j) { return idx + begin[j]; });
            buffer.set.apply(buffer, [x.get.apply(x, xLoc)].concat(loc));
        }
        return buffer.toTensor();
    };
    MathBackendCPU.prototype.stridedSlice = function (x, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask) {
        this.assertNotComplex(x, 'stridedSlice');
        var _a = slice_util_1.getStridedSlicedInfo(x.shape, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask), beginIndex = _a[0], size = _a[1], shrinkAxis = _a[2];
        var shape = size.filter(function (v, index) { return shrinkAxis.indexOf(index) === -1; });
        if (shape.some(function (axis) { return axis === 0; })) {
            return ops.tensor([], shape);
        }
        var buffer = ops.buffer(size, x.dtype);
        for (var i = 0; i < buffer.size; i++) {
            var loc = buffer.indexToLoc(i);
            var newLoc = new Array(loc.length);
            for (var j = 0; j < newLoc.length; j++) {
                newLoc[j] = loc[j] * strides[j] + beginIndex[j];
            }
            buffer.set.apply(buffer, [x.get.apply(x, newLoc)].concat(loc));
        }
        return buffer.toTensor().reshape(shape);
    };
    MathBackendCPU.prototype.reverse = function (x, axis) {
        this.assertNotComplex(x, 'reverse');
        var buffer = ops.buffer(x.shape, x.dtype);
        var xBuffer = x.buffer();
        var _loop_1 = function (i) {
            var outLoc = buffer.indexToLoc(i);
            var inLoc = outLoc.slice();
            axis.forEach(function (ax) { return inLoc[ax] = x.shape[ax] - 1 - inLoc[ax]; });
            buffer.set.apply(buffer, [xBuffer.get.apply(xBuffer, inLoc)].concat(outLoc));
        };
        for (var i = 0; i < buffer.size; i++) {
            _loop_1(i);
        }
        return buffer.toTensor();
    };
    MathBackendCPU.prototype.concat = function (tensors, axis) {
        this.assertNotComplex(tensors, 'concat');
        var tensors2D = tensors.map(function (t) {
            var innerSize = util.sizeFromShape(t.shape.slice(axis));
            return t.as2D(-1, innerSize);
        });
        var outShape = concat_util.computeOutShape(tensors2D.map(function (t) { return t.shape; }), 1);
        var values = ops.buffer(outShape, tensors[0].dtype)
            .values;
        if (tensors2D[0].shape[0] === 1) {
            var offset_1 = 0;
            tensors2D.forEach(function (t) {
                values.set(t.dataSync(), offset_1);
                offset_1 += t.size;
            });
        }
        else {
            var colOffset_1 = 0;
            tensors2D.forEach(function (t) {
                var tVals = t.dataSync();
                var tIdx = 0;
                for (var row = 0; row < t.shape[0]; ++row) {
                    var resIdx = row * outShape[1] + colOffset_1;
                    for (var col = 0; col < t.shape[1]; ++col) {
                        values[resIdx + col] = tVals[tIdx++];
                    }
                }
                colOffset_1 += t.shape[1];
            });
        }
        var finalOutShape = concat_util.computeOutShape(tensors.map(function (t) { return t.shape; }), axis);
        return ops_1.tensor(values, finalOutShape, tensors[0].dtype);
    };
    MathBackendCPU.prototype.neg = function (x) {
        this.assertNotComplex(x, 'neg');
        return this.multiply(ops.scalar(-1), x);
    };
    MathBackendCPU.prototype.add = function (a, b) {
        if (a.dtype === 'complex64' || b.dtype === 'complex64') {
            return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), function (aReal, aImag, bReal, bImag) {
                return { real: aReal + bReal, imag: aImag + bImag };
            });
        }
        return this.broadcastedBinaryOp(a, b, types_1.upcastType(a.dtype, b.dtype), function (aValue, bValue) { return aValue + bValue; });
    };
    MathBackendCPU.prototype.addN = function (tensors) {
        this.assertNotComplex(tensors, 'addN');
        var vals = tensors.map(function (t) { return t.dataSync(); });
        var result = ops.buffer(tensors[0].shape, tensors[0].dtype);
        var resultVals = result.values;
        for (var i = 0; i < tensors.length; i++) {
            var currVals = vals[i];
            for (var j = 0; j < resultVals.length; j++) {
                resultVals[j] += currVals[j];
            }
        }
        return result.toTensor();
    };
    MathBackendCPU.prototype.subtract = function (a, b) {
        if (a.dtype === 'complex64' || b.dtype === 'complex64') {
            return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), function (aReal, aImag, bReal, bImag) {
                return { real: aReal - bReal, imag: aImag - bImag };
            });
        }
        return this.broadcastedBinaryOp(a, b, types_1.upcastType(a.dtype, b.dtype), function (aValue, bValue) { return aValue - bValue; });
    };
    MathBackendCPU.prototype.pow = function (a, b) {
        this.assertNotComplex([a, b], 'pow');
        return this.broadcastedBinaryOp(a, b, a.dtype, function (aValue, bValue) { return Math.pow(aValue, bValue); });
    };
    MathBackendCPU.prototype.batchMatMul = function (a, b, transposeA, transposeB) {
        this.assertNotComplex([a, b], 'matMul');
        var sharedDim = transposeA ? a.shape[1] : a.shape[2];
        var leftDim = transposeA ? a.shape[2] : a.shape[1];
        var rightDim = transposeB ? b.shape[1] : b.shape[2];
        var batchDim = a.shape[0];
        var aValues = a.dataSync();
        var bValues = b.dataSync();
        var _a = transposeA ?
            [a.strides[0], 1, a.strides[1]] :
            [a.strides[0], a.strides[1], 1], aBatch = _a[0], aOuterStep = _a[1], aInnerStep = _a[2];
        var _b = transposeB ?
            [1, b.strides[1], b.strides[0]] :
            [b.strides[1], 1, b.strides[0]], bInnerStep = _b[0], bOuterStep = _b[1], bBatch = _b[2];
        var size = leftDim * rightDim;
        var result = new Float32Array(batchDim * size);
        var blockSize = this.blockSize;
        for (var b_1 = 0; b_1 < batchDim; b_1++) {
            for (var i0 = 0; i0 < leftDim; i0 += blockSize) {
                for (var j0 = 0; j0 < rightDim; j0 += blockSize) {
                    for (var k0 = 0; k0 < sharedDim; k0 += blockSize) {
                        var iBlock = Math.min(i0 + blockSize, leftDim);
                        var jBlock = Math.min(j0 + blockSize, rightDim);
                        var kBlock = Math.min(k0 + blockSize, sharedDim);
                        for (var i = i0; i < iBlock; i++) {
                            for (var j = j0; j < jBlock; j++) {
                                var sum = 0.0;
                                for (var k = k0; k < kBlock; k++) {
                                    sum += aValues[b_1 * aBatch + i * aOuterStep + k * aInnerStep] *
                                        bValues[k * bInnerStep + j * bOuterStep + b_1 * bBatch];
                                }
                                result[b_1 * size + (i * rightDim + j)] += sum;
                            }
                        }
                    }
                }
            }
        }
        return ops.tensor3d(result, [batchDim, leftDim, rightDim]);
    };
    MathBackendCPU.prototype.multiply = function (a, b) {
        if (a.dtype === 'complex64' || b.dtype === 'complex64') {
            return this.broadcastedBinaryComplexOp(a.cast('complex64'), b.cast('complex64'), function (aReal, aImag, bReal, bImag) {
                return {
                    real: aReal * bReal - aImag * bImag,
                    imag: aReal * bImag + aImag * bReal
                };
            });
        }
        return this.broadcastedBinaryOp(a, b, types_1.upcastType(a.dtype, b.dtype), function (aValue, bValue) { return aValue * bValue; });
    };
    MathBackendCPU.prototype.realDivide = function (a, b) {
        this.assertNotComplex([a, b], 'realDivide');
        var op = function (a, b) { return a / b; };
        var outputDtype = 'float32';
        return this.broadcastedBinaryOp(a, b, outputDtype, op);
    };
    MathBackendCPU.prototype.floorDiv = function (a, b) {
        this.assertNotComplex([a, b], 'floorDiv');
        var op = function (a, b) { return Math.floor(a / b); };
        var outputDtype = 'int32';
        return this.broadcastedBinaryOp(a, b, outputDtype, op);
    };
    MathBackendCPU.prototype.sum = function (x, axes) {
        this.assertNotComplex(x, 'sum');
        axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var resultDtype = types_1.upcastType(x.dtype, 'int32');
        var result = ops.zeros(outShape, resultDtype);
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var sum = 0;
            for (var j = 0; j < reduceSize; ++j) {
                sum += aVals[offset + j];
            }
            vals[i] = sum;
        }
        return result;
    };
    MathBackendCPU.prototype.prod = function (x, axes) {
        this.assertNotComplex(x, 'sum');
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var resultDtype = types_1.upcastType(x.dtype, 'int32');
        var result = ops.zeros(outShape, resultDtype);
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var prod = 1;
            for (var j = 0; j < reduceSize; ++j) {
                prod *= aVals[offset + j];
            }
            vals[i] = prod;
        }
        return result;
    };
    MathBackendCPU.prototype.unsortedSegmentSum = function (x, segmentIds, numSegments) {
        this.assertNotComplex(x, 'unsortedSegmentSum');
        var res = [];
        var numIters = x.rank - segmentIds.rank;
        for (var i = 0; i < numIters; ++i) {
            segmentIds = segmentIds.expandDims(i + 1);
        }
        for (var i = 0; i < numSegments; ++i) {
            var segmentId = ops.scalar(i, 'int32');
            var mask = ops.equal(segmentId, segmentIds).asType('float32');
            var sum = mask.mul(x).sum(0);
            res.push(sum);
        }
        return ops.stack(res);
    };
    MathBackendCPU.prototype.argMin = function (x, axis) {
        this.assertNotComplex(x, 'argMin');
        var axes = [axis];
        axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var result = ops.zeros(outShape, 'int32');
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var min = aVals[offset];
            var minIndex = 0;
            for (var j = 0; j < reduceSize; ++j) {
                var value = aVals[offset + j];
                if (value < min) {
                    min = value;
                    minIndex = j;
                }
            }
            vals[i] = minIndex;
        }
        return result;
    };
    MathBackendCPU.prototype.argMax = function (x, axis) {
        this.assertNotComplex(x, 'argMax');
        var axes = [axis];
        axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var result = ops.zeros(outShape, 'int32');
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var max = aVals[offset];
            var maxIndex = 0;
            for (var j = 0; j < reduceSize; ++j) {
                var value = aVals[offset + j];
                if (value > max) {
                    max = value;
                    maxIndex = j;
                }
            }
            vals[i] = maxIndex;
        }
        return result;
    };
    MathBackendCPU.prototype.cumsum = function (x, axis, exclusive, reverse) {
        this.assertNotComplex(x, 'cumsum');
        if (axis !== x.rank - 1) {
            throw new Error("backend.cumsum in CPU expects an inner-most axis=" + (x.rank - 1) + " " +
                ("but got axis=" + axis));
        }
        var resultDtype = types_1.upcastType(x.dtype, 'int32');
        var result = ops.zeros(x.shape, resultDtype);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        var finalDim = x.shape[x.rank - 1];
        var indexAdjuster = reverse ?
            function (i, j) { return i + finalDim - j - 1; } :
            function (i, j) { return i + j; };
        for (var i = 0; i < aVals.length; i += finalDim) {
            for (var j = 0; j < finalDim; j++) {
                var idx = indexAdjuster(i, j);
                if (j === 0) {
                    vals[idx] = exclusive ? 0 : aVals[idx];
                }
                else {
                    var prevIdx = indexAdjuster(i, j - 1);
                    vals[idx] = exclusive ? aVals[prevIdx] + vals[prevIdx] :
                        aVals[idx] + vals[prevIdx];
                }
            }
        }
        return result;
    };
    MathBackendCPU.prototype.equal = function (a, b) {
        this.assertNotComplex([a, b], 'equal');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return (aVal === bVal) ? 1 : 0;
        });
    };
    MathBackendCPU.prototype.notEqual = function (a, b) {
        this.assertNotComplex([a, b], 'notEqual');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return (aVal !== bVal) ? 1 : 0;
        });
    };
    MathBackendCPU.prototype.less = function (a, b) {
        this.assertNotComplex([a, b], 'less');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return (aVal < bVal) ? 1 : 0;
        });
    };
    MathBackendCPU.prototype.lessEqual = function (a, b) {
        this.assertNotComplex([a, b], 'lessEqual');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return (aVal <= bVal) ? 1 : 0;
        });
    };
    MathBackendCPU.prototype.greater = function (a, b) {
        this.assertNotComplex([a, b], 'greater');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return (aVal > bVal) ? 1 : 0;
        });
    };
    MathBackendCPU.prototype.greaterEqual = function (a, b) {
        this.assertNotComplex([a, b], 'greaterEqual');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return (aVal >= bVal) ? 1 : 0;
        });
    };
    MathBackendCPU.prototype.logicalNot = function (x) {
        this.assertNotComplex(x, 'logicalNot');
        var values = x.dataSync();
        var newValues = new Int32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = values[i] ? 0 : 1;
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues }, 'bool');
    };
    MathBackendCPU.prototype.logicalAnd = function (a, b) {
        this.assertNotComplex([a, b], 'logicalAnd');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return aVal && bVal;
        });
    };
    MathBackendCPU.prototype.logicalOr = function (a, b) {
        this.assertNotComplex([a, b], 'logicalOr');
        return this.broadcastedBinaryOp(a, b, 'bool', function (aVal, bVal) {
            return aVal || bVal;
        });
    };
    MathBackendCPU.prototype.select = function (condition, a, b) {
        this.assertNotComplex([condition, a, b], 'select');
        var values = condition.dataSync();
        var aValues = a.dataSync();
        var bValues = b.dataSync();
        var result = ops.zeros(a.shape, types_1.upcastType(a.dtype, b.dtype));
        var newValues = result.dataSync();
        var index = 0;
        var offset = condition.rank === 0 || condition.rank > 1 || a.rank === 1 ?
            1 :
            a.shape[1];
        for (var i = 0; i < values.length; i++) {
            for (var j = 0; j < offset; j++) {
                if (values[i] === 1) {
                    newValues[index++] = aValues[i];
                }
                else {
                    newValues[index++] = bValues[i];
                }
            }
        }
        return result;
    };
    MathBackendCPU.prototype.where = function (condition) {
        this.assertNotComplex([condition], 'where');
        var condVals = condition.dataSync();
        return where_impl_1.whereImpl(condition.shape, condVals);
    };
    MathBackendCPU.prototype.topk = function (x, k, sorted) {
        this.assertNotComplex(x, 'topk');
        var xVals = x.dataSync();
        return topk_impl_1.topkImpl(xVals, x.shape, x.dtype, k, sorted);
    };
    MathBackendCPU.prototype.min = function (x, axes) {
        this.assertNotComplex(x, 'min');
        axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var result = ops.zeros(outShape, x.dtype);
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var min = aVals[offset];
            for (var j = 0; j < reduceSize; ++j) {
                var value = aVals[offset + j];
                if (value < min) {
                    min = value;
                }
            }
            vals[i] = min;
        }
        return result;
    };
    MathBackendCPU.prototype.minimum = function (a, b) {
        this.assertNotComplex([a, b], 'minimum');
        return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) { return Math.min(aVal, bVal); });
    };
    MathBackendCPU.prototype.mod = function (a, b) {
        this.assertNotComplex([a, b], 'mod');
        return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) {
            var rem = aVal % bVal;
            if ((aVal < 0 && bVal < 0) || (aVal >= 0 && bVal >= 0)) {
                return rem;
            }
            else {
                return (rem + bVal) % bVal;
            }
        });
    };
    MathBackendCPU.prototype.max = function (x, axes) {
        this.assertNotComplex(x, 'max');
        axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var result = ops.zeros(outShape, x.dtype);
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var max = aVals[offset];
            for (var j = 0; j < reduceSize; ++j) {
                var value = aVals[offset + j];
                if (value > max) {
                    max = value;
                }
            }
            vals[i] = max;
        }
        return result;
    };
    MathBackendCPU.prototype.maximum = function (a, b) {
        this.assertNotComplex([a, b], 'maximum');
        return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) { return Math.max(aVal, bVal); });
    };
    MathBackendCPU.prototype.all = function (x, axes) {
        this.assertNotComplex(x, 'all');
        axis_util.assertAxesAreInnerMostDims('all', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var result = ops.zeros(outShape, x.dtype);
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var all = aVals[offset];
            for (var j = 0; j < reduceSize; ++j) {
                var value = aVals[offset + j];
                all = all && value;
            }
            vals[i] = all;
        }
        return result;
    };
    MathBackendCPU.prototype.any = function (x, axes) {
        this.assertNotComplex(x, 'any');
        axis_util.assertAxesAreInnerMostDims('any', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var result = ops.zeros(outShape, x.dtype);
        var reduceSize = util.sizeFromShape(reduceShape);
        var vals = result.dataSync();
        var aVals = x.dataSync();
        for (var i = 0; i < vals.length; ++i) {
            var offset = i * reduceSize;
            var anyVal = aVals[offset];
            for (var j = 0; j < reduceSize; ++j) {
                var value = aVals[offset + j];
                anyVal = anyVal || value;
            }
            vals[i] = anyVal;
        }
        return result;
    };
    MathBackendCPU.prototype.squaredDifference = function (a, b) {
        this.assertNotComplex([a, b], 'squaredDifference');
        return this.broadcastedBinaryOp(a, b, a.dtype, function (aVal, bVal) {
            var diff = aVal - bVal;
            return diff * diff;
        });
    };
    MathBackendCPU.prototype.ceil = function (x) {
        this.assertNotComplex(x, 'ceil');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = Math.ceil(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.floor = function (x) {
        this.assertNotComplex(x, 'floor');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = Math.floor(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.sign = function (x) {
        this.assertNotComplex(x, 'x');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            if (values[i] < 0) {
                newValues[i] = -1;
            }
            else if (values[i] > 0) {
                newValues[i] = 1;
            }
            else {
                newValues[i] = 0;
            }
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.round = function (x) {
        this.assertNotComplex(x, 'round');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var base = Math.floor(values[i]);
            if (values[i] - base < 0.5) {
                newValues[i] = Math.floor(values[i]);
            }
            else if (values[i] - base > 0.5) {
                newValues[i] = Math.ceil(values[i]);
            }
            else {
                if (base % 2.0 === 0.0) {
                    newValues[i] = base;
                }
                else {
                    newValues[i] = base + 1.0;
                }
            }
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.exp = function (x) {
        this.assertNotComplex(x, 'exp');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = Math.exp(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.expm1 = function (x) {
        this.assertNotComplex(x, 'expm1');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = Math.expm1(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.log = function (x) {
        this.assertNotComplex(x, 'log');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = Math.log(value);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.log1p = function (x) {
        this.assertNotComplex(x, 'log1p');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = Math.log1p(value);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.sqrt = function (x) {
        this.assertNotComplex(x, 'sqrt');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = Math.sqrt(value);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.rsqrt = function (x) {
        this.assertNotComplex(x, 'rsqrt');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = 1 / Math.sqrt(value);
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.square = function (x) {
        this.assertNotComplex(x, 'square');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = value * value;
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.reciprocal = function (x) {
        this.assertNotComplex(x, 'reciprocal');
        var values = x.dataSync();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = 1 / values[i];
        }
        return tensor_1.Tensor.make(x.shape, { values: newValues });
    };
    MathBackendCPU.prototype.relu = function (x) {
        this.assertNotComplex(x, 'relu');
        var res = ops.zeros(x.shape, x.dtype);
        var resVals = res.dataSync();
        var inVals = x.dataSync();
        for (var i = 0; i < inVals.length; ++i) {
            resVals[i] = Math.max(0, inVals[i]);
        }
        return res;
    };
    MathBackendCPU.prototype.elu = function (x) {
        this.assertNotComplex(x, 'elu');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            var v = values[i];
            if (v >= 0) {
                resultValues[i] = v;
            }
            else {
                resultValues[i] = (Math.exp(v) - 1);
            }
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.eluDer = function (dy, y) {
        this.assertNotComplex([dy, y], 'eluDer');
        var resultValues = new Float32Array(y.size);
        var values = y.dataSync();
        var dyValues = dy.dataSync();
        for (var i = 0; i < values.length; ++i) {
            var v = values[i];
            if (v >= 1) {
                resultValues[i] = dyValues[i];
            }
            else {
                resultValues[i] = dyValues[i] * (v + 1);
            }
        }
        return tensor_1.Tensor.make(y.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.selu = function (x) {
        this.assertNotComplex(x, 'selu');
        var scaleAlpha = selu_util.SELU_SCALEALPHA;
        var scale = selu_util.SELU_SCALE;
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            var v = values[i];
            if (v >= 0) {
                resultValues[i] = scale * v;
            }
            else {
                resultValues[i] = scaleAlpha * (Math.exp(v) - 1);
            }
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.clip = function (x, min, max) {
        this.assertNotComplex(x, 'clip');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            var v = values[i];
            resultValues[i] = v > max ? max : (v < min ? min : v);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.abs = function (x) {
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.abs(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.complexAbs = function (x) {
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < x.size; ++i) {
            var real = values[i * 2];
            var imag = values[i * 2 + 1];
            resultValues[i] = Math.sqrt(real * real + imag * imag);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.int = function (x) {
        this.assertNotComplex(x, 'int');
        var resultValues = new Int32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = values[i];
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues }, 'int32');
    };
    MathBackendCPU.prototype.sigmoid = function (x) {
        this.assertNotComplex(x, 'sigmoid');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = 1 / (1 + Math.exp(-values[i]));
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.softplus = function (x) {
        this.assertNotComplex(x, 'softplus');
        var epsilon = 1.1920928955078125e-7;
        var threshold = Math.log(epsilon) + 2.0;
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            var tooLarge = values[i] > -threshold;
            var tooSmall = values[i] < threshold;
            var expX = Math.exp(values[i]);
            var result = void 0;
            if (tooSmall) {
                result = expX;
            }
            else if (tooLarge) {
                result = values[i];
            }
            else {
                result = Math.log(1.0 + expX);
            }
            resultValues[i] = result;
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.sin = function (x) {
        this.assertNotComplex(x, 'sin');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.sin(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.cos = function (x) {
        this.assertNotComplex(x, 'cos');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.cos(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.tan = function (x) {
        this.assertNotComplex(x, 'tan');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.tan(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.asin = function (x) {
        this.assertNotComplex(x, 'asin');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.asin(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.acos = function (x) {
        this.assertNotComplex(x, 'acos');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.acos(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.atan = function (x) {
        this.assertNotComplex(x, 'atan');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.atan(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.atan2 = function (a, b) {
        this.assertNotComplex([a, b], 'atan2');
        return this.broadcastedBinaryOp(a, b, a.dtype, function (aValue, bValue) { return Math.atan2(aValue, bValue); });
    };
    MathBackendCPU.prototype.sinh = function (x) {
        this.assertNotComplex(x, 'sinh');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.sinh(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.cosh = function (x) {
        this.assertNotComplex(x, 'cosh');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.cosh(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.tanh = function (x) {
        this.assertNotComplex(x, 'tanh');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = util.tanh(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.asinh = function (x) {
        this.assertNotComplex(x, 'asinh');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.asinh(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.acosh = function (x) {
        this.assertNotComplex(x, 'acosh');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.acosh(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.atanh = function (x) {
        this.assertNotComplex(x, 'atanh');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.atanh(values[i]);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.erf = function (x) {
        this.assertNotComplex(x, 'erf');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        var p = erf_util.ERF_P;
        var a1 = erf_util.ERF_A1;
        var a2 = erf_util.ERF_A2;
        var a3 = erf_util.ERF_A3;
        var a4 = erf_util.ERF_A4;
        var a5 = erf_util.ERF_A5;
        for (var i = 0; i < values.length; ++i) {
            var v = values[i];
            var t = 1.0 / (1.0 + p * v);
            resultValues[i] = 1.0 -
                (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                    Math.exp(-v * v);
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.step = function (x, alpha) {
        if (alpha === void 0) { alpha = 0; }
        this.assertNotComplex(x, 'step');
        var resultValues = new Float32Array(x.size);
        var values = x.dataSync();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                resultValues[i] = NaN;
            }
            else {
                resultValues[i] = value > 0 ? 1 : alpha;
            }
        }
        return tensor_1.Tensor.make(x.shape, { values: resultValues });
    };
    MathBackendCPU.prototype.conv2d = function (x, filter, convInfo) {
        this.assertNotComplex([x, filter], 'conv2d');
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var dilationHeight = convInfo.dilationHeight;
        var dilationWidth = convInfo.dilationWidth;
        var padLeft = convInfo.padInfo.left;
        var padTop = convInfo.padInfo.top;
        var y = ops.buffer(convInfo.outShape, x.dtype);
        var xVals = x.dataSync();
        var wVals = filter.dataSync();
        var yVals = y.values;
        for (var b = 0; b < convInfo.batchSize; ++b) {
            var xOffset1 = b * x.strides[0];
            var yOffset1 = b * y.strides[0];
            for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                var yOffset2 = yOffset1 + yR * y.strides[1];
                var xRCorner = yR * convInfo.strideHeight - padLeft;
                for (var wR = 0; wR < filterHeight; wR++) {
                    var xR = xRCorner + wR * dilationHeight;
                    if (xR < 0 || xR >= convInfo.inHeight) {
                        continue;
                    }
                    var wOffset1 = wR * filter.strides[0];
                    var xOffset2 = xOffset1 + xR * x.strides[1];
                    for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                        var yOffset3 = yOffset2 + yC * convInfo.outChannels;
                        var xCCorner = yC * convInfo.strideWidth - padTop;
                        for (var wC = 0; wC < filterWidth; wC++) {
                            var xC = xCCorner + wC * dilationWidth;
                            if (xC < 0 || xC >= convInfo.inWidth) {
                                continue;
                            }
                            var wOffset2 = wOffset1 + wC * filter.strides[1];
                            var xOffset3 = xOffset2 + xC * convInfo.inChannels;
                            var wOffset3 = wOffset2;
                            for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                var xVal = xVals[xOffset3 + d1];
                                for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                                    yVals[yOffset3 + d2] += xVal * wVals[wOffset3 + d2];
                                }
                                wOffset3 += convInfo.outChannels;
                            }
                        }
                    }
                }
            }
        }
        return y.toTensor();
    };
    MathBackendCPU.prototype.conv2dDerInput = function (dy, filter, convInfo) {
        this.assertNotComplex([dy, filter], 'conv2dDerInput');
        var dx = ops.buffer(convInfo.inShape, 'float32');
        var dxValues = dx.values;
        var _a = dx.strides, dxS0 = _a[0], dxS1 = _a[1], dxS2 = _a[2];
        var dyValues = dy.dataSync();
        var _b = dy.strides, dyS0 = _b[0], dyS1 = _b[1], dyS2 = _b[2];
        var fltValues = filter.dataSync();
        var _c = filter.strides, fltS0 = _c[0], fltS1 = _c[1], fltS2 = _c[2];
        var batchSize = convInfo.batchSize, filterHeight = convInfo.filterHeight, filterWidth = convInfo.filterWidth, inChannels = convInfo.inChannels, inHeight = convInfo.inHeight, inWidth = convInfo.inWidth, outChannels = convInfo.outChannels, outHeight = convInfo.outHeight, outWidth = convInfo.outWidth, strideHeight = convInfo.strideHeight, strideWidth = convInfo.strideWidth;
        var topPad = filterHeight - 1 - convInfo.padInfo.top;
        var leftPad = filterWidth - 1 - convInfo.padInfo.left;
        for (var b = 0; b < batchSize; ++b) {
            for (var d1 = 0; d1 < inChannels; ++d1) {
                for (var xR = 0; xR < inHeight; ++xR) {
                    var xRCorner = xR - topPad;
                    var xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                    var yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                    for (var xC = 0; xC < inWidth; ++xC) {
                        var xCCorner = xC - leftPad;
                        var xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                        var yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                        var dotProd = 0;
                        for (var yR = xRMin; yR < yRMax; ++yR) {
                            var wR = yR * strideHeight - xRCorner;
                            for (var yC = xCMin; yC < yCMax; ++yC) {
                                var wC = yC * strideWidth - xCCorner;
                                var dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                                var fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                    fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                                for (var d2 = 0; d2 < outChannels; ++d2) {
                                    var pixel = dyValues[dyOffset + d2];
                                    var weight = fltValues[fltOffset + d2];
                                    dotProd += pixel * weight;
                                }
                            }
                        }
                        dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
                    }
                }
            }
        }
        return dx.toTensor();
    };
    MathBackendCPU.prototype.conv2dDerFilter = function (x, dy, convInfo) {
        this.assertNotComplex([x, dy], 'conv2dDerFilter');
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var dW = ops.buffer(convInfo.filterShape, 'float32');
        var leftPad = convInfo.padInfo.left;
        var topPad = convInfo.padInfo.top;
        for (var wR = 0; wR < filterHeight; ++wR) {
            var yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
            var yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
            for (var wC = 0; wC < filterWidth; ++wC) {
                var yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                var yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                    for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                        var dotProd = 0;
                        for (var b = 0; b < convInfo.batchSize; ++b) {
                            for (var yR = yRMin; yR < yRMax; ++yR) {
                                var xR = wR + yR * strideHeight - topPad;
                                for (var yC = yCMin; yC < yCMax; ++yC) {
                                    var xC = wC + yC * strideWidth - leftPad;
                                    dotProd += x.get(b, xR, xC, d1) * dy.get(b, yR, yC, d2);
                                }
                            }
                        }
                        dW.set(dotProd, wR, wC, d1, d2);
                    }
                }
            }
        }
        return dW.toTensor();
    };
    MathBackendCPU.prototype.depthwiseConv2D = function (x, filter, convInfo) {
        this.assertNotComplex([x, filter], 'depthwiseConv2D');
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var dilationHeight = convInfo.dilationHeight;
        var dilationWidth = convInfo.dilationWidth;
        var padLeft = convInfo.padInfo.left;
        var padTop = convInfo.padInfo.top;
        var chMul = convInfo.outChannels / convInfo.inChannels;
        var y = ops.buffer(convInfo.outShape, x.dtype);
        var xVals = x.dataSync();
        var wVals = filter.dataSync();
        var yVals = y.values;
        for (var b = 0; b < convInfo.batchSize; ++b) {
            var xOffset1 = b * x.strides[0];
            var yOffset1 = b * y.strides[0];
            for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                var yOffset2 = yOffset1 + yR * y.strides[1];
                var xRCorner = yR * convInfo.strideHeight - padLeft;
                for (var wR = 0; wR < filterHeight; ++wR) {
                    var xR = xRCorner + wR * dilationHeight;
                    if (xR < 0 || xR >= convInfo.inHeight) {
                        continue;
                    }
                    var wOffset1 = wR * filter.strides[0];
                    var xOffset2 = xOffset1 + xR * x.strides[1];
                    for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                        var yOffset3 = yOffset2 + yC * y.strides[2];
                        var xCCorner = yC * convInfo.strideWidth - padTop;
                        for (var wC = 0; wC < filterWidth; ++wC) {
                            var xC = xCCorner + wC * dilationWidth;
                            if (xC < 0 || xC >= convInfo.inWidth) {
                                continue;
                            }
                            var wOffset2 = wOffset1 + wC * filter.strides[1];
                            var xOffset3 = xOffset2 + xC * convInfo.inChannels;
                            var yOffset4 = yOffset3;
                            var wOffset3 = wOffset2;
                            for (var d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                var xVal = xVals[xOffset3 + d1];
                                for (var q = 0; q < chMul; ++q) {
                                    yVals[yOffset4 + q] += xVal * wVals[wOffset3 + q];
                                }
                                yOffset4 += chMul;
                                wOffset3 += chMul;
                            }
                        }
                    }
                }
            }
        }
        return y.toTensor();
    };
    MathBackendCPU.prototype.depthwiseConv2DDerInput = function (dy, filter, convInfo) {
        this.assertNotComplex([dy, filter], 'depthwiseConv2DDerInput');
        var dx = ops.buffer(convInfo.inShape, 'float32');
        var dxValues = dx.values;
        var _a = dx.strides, dxS0 = _a[0], dxS1 = _a[1], dxS2 = _a[2];
        var dyValues = dy.dataSync();
        var _b = dy.strides, dyS0 = _b[0], dyS1 = _b[1], dyS2 = _b[2];
        var fltValues = filter.dataSync();
        var _c = filter.strides, fltS0 = _c[0], fltS1 = _c[1], fltS2 = _c[2];
        var batchSize = convInfo.batchSize, filterHeight = convInfo.filterHeight, filterWidth = convInfo.filterWidth, inChannels = convInfo.inChannels, inHeight = convInfo.inHeight, inWidth = convInfo.inWidth, outChannels = convInfo.outChannels, outHeight = convInfo.outHeight, outWidth = convInfo.outWidth, strideHeight = convInfo.strideHeight, strideWidth = convInfo.strideWidth;
        var topPad = filterHeight - 1 - convInfo.padInfo.top;
        var leftPad = filterWidth - 1 - convInfo.padInfo.left;
        var chMul = outChannels / inChannels;
        for (var b = 0; b < batchSize; ++b) {
            for (var d1 = 0; d1 < inChannels; ++d1) {
                for (var xR = 0; xR < inHeight; ++xR) {
                    var xRCorner = xR - topPad;
                    var xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                    var yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                    for (var xC = 0; xC < inWidth; ++xC) {
                        var xCCorner = xC - leftPad;
                        var xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                        var yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                        var dotProd = 0;
                        for (var yR = xRMin; yR < yRMax; ++yR) {
                            var wR = yR * strideHeight - xRCorner;
                            for (var yC = xCMin; yC < yCMax; ++yC) {
                                var wC = yC * strideWidth - xCCorner;
                                var dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                                var fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                    fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                                for (var dm = 0; dm < chMul; ++dm) {
                                    var d2 = d1 * chMul + dm;
                                    var pixel = dyValues[dyOffset + d2];
                                    var weight = fltValues[fltOffset + dm];
                                    dotProd += pixel * weight;
                                }
                            }
                        }
                        dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
                    }
                }
            }
        }
        return dx.toTensor();
    };
    MathBackendCPU.prototype.depthwiseConv2DDerFilter = function (x, dy, convInfo) {
        this.assertNotComplex([x, dy], 'depthwiseConv2DDerFilter');
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var dW = ops.buffer(convInfo.filterShape, 'float32');
        var leftPad = convInfo.padInfo.left;
        var topPad = convInfo.padInfo.top;
        var chMul = convInfo.outChannels / convInfo.inChannels;
        for (var wR = 0; wR < filterHeight; ++wR) {
            var yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
            var yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
            for (var wC = 0; wC < filterWidth; ++wC) {
                var yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                var yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                for (var d2 = 0; d2 < convInfo.outChannels; ++d2) {
                    var d1 = Math.trunc(d2 / chMul);
                    var dm = d2 % chMul;
                    var dotProd = 0;
                    for (var b = 0; b < convInfo.batchSize; ++b) {
                        for (var yR = yRMin; yR < yRMax; ++yR) {
                            var xR = wR + yR * strideHeight - topPad;
                            for (var yC = yCMin; yC < yCMax; ++yC) {
                                var xC = wC + yC * strideWidth - leftPad;
                                dotProd += x.get(b, xR, xC, d1) * dy.get(b, yR, yC, d2);
                            }
                        }
                    }
                    dW.set(dotProd, wR, wC, d1, dm);
                }
            }
        }
        return dW.toTensor();
    };
    MathBackendCPU.prototype.tile = function (x, reps) {
        this.assertNotComplex(x, 'tile');
        var newShape = new Array(x.rank);
        for (var i = 0; i < newShape.length; i++) {
            newShape[i] = x.shape[i] * reps[i];
        }
        var result = ops.buffer(newShape, x.dtype);
        var xBuf = x.buffer();
        for (var i = 0; i < result.values.length; ++i) {
            var newLoc = result.indexToLoc(i);
            var originalLoc = new Array(x.rank);
            for (var i_1 = 0; i_1 < originalLoc.length; i_1++) {
                originalLoc[i_1] = newLoc[i_1] % x.shape[i_1];
            }
            var originalIndex = xBuf.locToIndex(originalLoc);
            result.values[i] = xBuf.values[originalIndex];
        }
        return result.toTensor();
    };
    MathBackendCPU.prototype.pad = function (x, paddings, constantValue) {
        this.assertNotComplex(x, 'pad');
        var outShape = paddings.map(function (p, i) { return p[0] + x.shape[i] + p[1]; });
        var start = paddings.map(function (p) { return p[0]; });
        var xBuffer = x.buffer();
        var buffer = ops.buffer(outShape, x.dtype);
        if (constantValue !== 0) {
            buffer.values.fill(constantValue);
        }
        for (var i = 0; i < x.size; i++) {
            var coords = xBuffer.indexToLoc(i);
            var outCoords = coords.map(function (c, i) { return c + start[i]; });
            buffer.set.apply(buffer, [x.get.apply(x, coords)].concat(outCoords));
        }
        return buffer.toTensor();
    };
    MathBackendCPU.prototype.transpose = function (x, perm) {
        this.assertNotComplex(x, 'transpose');
        var newShape = new Array(x.rank);
        for (var i = 0; i < newShape.length; i++) {
            newShape[i] = x.shape[perm[i]];
        }
        var values = x.dataSync();
        var result = ops_1.buffer(newShape, x.dtype);
        var xBuf = x.buffer();
        for (var i = 0; i < x.size; ++i) {
            var loc = xBuf.indexToLoc(i);
            var newLoc = new Array(loc.length);
            for (var i_2 = 0; i_2 < newLoc.length; i_2++) {
                newLoc[i_2] = loc[perm[i_2]];
            }
            var newIndex = result.locToIndex(newLoc);
            result.values[newIndex] = values[i];
        }
        return result.toTensor();
    };
    MathBackendCPU.prototype.gather = function (x, indices, axis) {
        this.assertNotComplex([x, indices], 'gather');
        var newShape = x.shape.slice();
        var indicesValues = indices.dataSync();
        newShape[axis] = indicesValues.length;
        var result = ops_1.buffer(newShape, x.dtype);
        var xBuf = x.buffer();
        for (var i = 0; i < result.size; ++i) {
            var newLoc = result.indexToLoc(i);
            var originalLoc = newLoc.slice();
            originalLoc[axis] = indicesValues[newLoc[axis]];
            var originalIndex = xBuf.locToIndex(originalLoc);
            result.values[i] = xBuf.values[originalIndex];
        }
        return result.toTensor();
    };
    MathBackendCPU.prototype.batchToSpaceND = function (x, blockShape, crops) {
        this.assertNotComplex([x], 'batchToSpaceND');
        var prod = blockShape.reduce(function (a, b) { return a * b; });
        var reshaped = array_ops_util.getReshaped(x.shape, blockShape, prod);
        var permuted = array_ops_util.getPermuted(reshaped.length, blockShape.length);
        var reshapedPermuted = array_ops_util.getReshapedPermuted(x.shape, blockShape, prod);
        var sliceBeginCoords = array_ops_util.getSliceBeginCoords(crops, blockShape.length);
        var sliceSize = array_ops_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
        return x.reshape(reshaped)
            .transpose(permuted)
            .reshape(reshapedPermuted)
            .slice(sliceBeginCoords, sliceSize);
    };
    MathBackendCPU.prototype.spaceToBatchND = function (x, blockShape, paddings) {
        this.assertNotComplex([x], 'spaceToBatchND');
        var prod = blockShape.reduce(function (a, b) { return a * b; });
        var completePaddings = [[0, 0]];
        completePaddings.push.apply(completePaddings, paddings);
        for (var i = 1 + blockShape.length; i < x.shape.length; ++i) {
            completePaddings.push([0, 0]);
        }
        var paddedX = x.pad(completePaddings);
        var reshapedPaddedShape = array_ops_util.getReshaped(paddedX.shape, blockShape, prod, false);
        var permutedReshapedPaddedPermutation = array_ops_util.getPermuted(reshapedPaddedShape.length, blockShape.length, false);
        var flattenShape = array_ops_util.getReshapedPermuted(paddedX.shape, blockShape, prod, false);
        return paddedX.reshape(reshapedPaddedShape)
            .transpose(permutedReshapedPaddedPermutation)
            .reshape(flattenShape);
    };
    MathBackendCPU.prototype.pool = function (x, convInfo, poolType) {
        this.assertNotComplex(x, 'pool');
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var dilationHeight = convInfo.dilationHeight;
        var dilationWidth = convInfo.dilationWidth;
        var effectiveFilterHeight = convInfo.effectiveFilterHeight;
        var effectiveFilterWidth = convInfo.effectiveFilterWidth;
        var padTop = convInfo.padInfo.top;
        var padLeft = convInfo.padInfo.left;
        var initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
            Number.POSITIVE_INFINITY);
        var xValues = x.dataSync();
        var output = ops.buffer(convInfo.outShape, x.dtype);
        var outputVals = output.values;
        var outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
        var outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
        var outputColStrides = convInfo.outShape[3];
        for (var b = 0; b < convInfo.batchSize; ++b) {
            var outputBatchOffset = b * outputBatchStrides;
            var inputBatchOffset = b * x.strides[0];
            for (var d = 0; d < convInfo.inChannels; ++d) {
                for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                    var xRCorner = yR * strideHeight - padTop;
                    var xRMin = Math.max(0, xRCorner);
                    var xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                    var outputRowOffset = outputBatchOffset + yR * outputRowStrides;
                    for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                        var xCCorner = yC * strideWidth - padLeft;
                        var xCMin = Math.max(0, xCCorner);
                        var xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                        var minMaxValue = initialValue;
                        var avgValue = 0;
                        var count = 0;
                        for (var xR = xRMin; xR < xRMax; xR += dilationHeight) {
                            var xROffset = inputBatchOffset + xR * x.strides[1];
                            for (var xC = xCMin; xC < xCMax; xC += dilationWidth) {
                                var xCOffset = xROffset + xC * x.strides[2];
                                var pixel = xValues[xCOffset + d];
                                if ((poolType === 'max' && pixel > minMaxValue)) {
                                    minMaxValue = pixel;
                                }
                                else if (poolType === 'avg') {
                                    avgValue += pixel;
                                    count++;
                                }
                            }
                            if (isNaN(minMaxValue)) {
                                break;
                            }
                        }
                        var outputOffset = outputRowOffset + yC * outputColStrides + d;
                        outputVals[outputOffset] =
                            poolType === 'avg' ? avgValue / count : minMaxValue;
                    }
                }
            }
        }
        return output.toTensor();
    };
    MathBackendCPU.prototype.maxPool = function (x, convInfo) {
        return this.pool(x, convInfo, 'max');
    };
    MathBackendCPU.prototype.maxPoolPositions = function (x, convInfo) {
        var maxPositions = ops.buffer(convInfo.outShape, 'int32');
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var dilationHeight = convInfo.dilationHeight;
        var dilationWidth = convInfo.dilationWidth;
        var effectiveFilterHeight = convInfo.effectiveFilterHeight;
        var effectiveFilterWidth = convInfo.effectiveFilterWidth;
        var padTop = convInfo.padInfo.top;
        var padLeft = convInfo.padInfo.left;
        for (var b = 0; b < convInfo.batchSize; ++b) {
            for (var d = 0; d < convInfo.inChannels; ++d) {
                for (var yR = 0; yR < convInfo.outHeight; ++yR) {
                    var xRCorner = yR * strideHeight - padTop;
                    var xRMin = xRCorner;
                    while (xRMin < 0) {
                        xRMin += dilationHeight;
                    }
                    var xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                    for (var yC = 0; yC < convInfo.outWidth; ++yC) {
                        var xCCorner = yC * strideWidth - padLeft;
                        var xCMin = xCCorner;
                        while (xCMin < 0) {
                            xCMin += dilationWidth;
                        }
                        var xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                        var maxValue = Number.NEGATIVE_INFINITY;
                        var maxPosition = -1;
                        for (var xR = xRMin; xR < xRMax; xR += dilationHeight) {
                            var wR = xR - xRCorner;
                            for (var xC = xCMin; xC < xCMax; xC += dilationWidth) {
                                var wC = xC - xCCorner;
                                var pixel = x.get(b, xR, xC, d);
                                if (pixel > maxValue) {
                                    maxValue = pixel;
                                    maxPosition = wR * effectiveFilterWidth + wC;
                                }
                            }
                        }
                        maxPositions.set(maxPosition, b, yR, yC, d);
                    }
                }
            }
        }
        return maxPositions.toTensor();
    };
    MathBackendCPU.prototype.maxPoolBackprop = function (dy, x, y, convInfo) {
        this.assertNotComplex([x, y], 'maxPoolBackprop');
        var maxPositions = this.maxPoolPositions(x, convInfo);
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var dilationHeight = convInfo.dilationHeight;
        var dilationWidth = convInfo.dilationWidth;
        var effectiveFilterHeight = convInfo.effectiveFilterHeight;
        var effectiveFilterWidth = convInfo.effectiveFilterWidth;
        var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
        var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
        var dx = ops.buffer(x.shape, 'float32');
        for (var b = 0; b < convInfo.batchSize; ++b) {
            for (var d = 0; d < convInfo.inChannels; ++d) {
                for (var dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                    for (var dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                        var dyRCorner = dxR - padTop;
                        var dyCCorner = dxC - padLeft;
                        var dotProd = 0;
                        for (var wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                            var dyR = (dyRCorner + wR) / strideHeight;
                            if (dyR < 0 || dyR >= convInfo.outHeight ||
                                Math.floor(dyR) !== dyR) {
                                continue;
                            }
                            for (var wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                                var dyC = (dyCCorner + wC) / strideWidth;
                                if (dyC < 0 || dyC >= convInfo.outWidth ||
                                    Math.floor(dyC) !== dyC) {
                                    continue;
                                }
                                var maxPos = effectiveFilterHeight * effectiveFilterWidth -
                                    1 - maxPositions.get(b, dyR, dyC, d);
                                var curPos = wR * effectiveFilterWidth + wC;
                                var mask = maxPos === curPos ? 1 : 0;
                                if (mask === 0) {
                                    continue;
                                }
                                var pixel = dy.get(b, dyR, dyC, d);
                                dotProd += pixel * mask;
                            }
                        }
                        dx.set(dotProd, b, dxR, dxC, d);
                    }
                }
            }
        }
        return dx.toTensor();
    };
    MathBackendCPU.prototype.avgPoolBackprop = function (dy, x, convInfo) {
        this.assertNotComplex([dy, x], 'avgPoolBackprop');
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var dilationHeight = convInfo.dilationHeight;
        var dilationWidth = convInfo.dilationWidth;
        var effectiveFilterHeight = convInfo.effectiveFilterHeight;
        var effectiveFilterWidth = convInfo.effectiveFilterWidth;
        var padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
        var padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
        var dx = ops.buffer(x.shape, 'float32');
        var avgMultiplier = 1 / (filterHeight * filterWidth);
        for (var b = 0; b < convInfo.batchSize; ++b) {
            for (var d = 0; d < convInfo.inChannels; ++d) {
                for (var dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                    for (var dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                        var dyRCorner = dxR - padTop;
                        var dyCCorner = dxC - padLeft;
                        var dotProd = 0;
                        for (var wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                            var dyR = (dyRCorner + wR) / strideHeight;
                            if (dyR < 0 || dyR >= convInfo.outHeight ||
                                Math.floor(dyR) !== dyR) {
                                continue;
                            }
                            for (var wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                                var dyC = (dyCCorner + wC) / strideWidth;
                                if (dyC < 0 || dyC >= convInfo.outWidth ||
                                    Math.floor(dyC) !== dyC) {
                                    continue;
                                }
                                var pixel = dy.get(b, dyR, dyC, d);
                                dotProd += pixel;
                            }
                        }
                        dx.set(dotProd * avgMultiplier, b, dxR, dxC, d);
                    }
                }
            }
        }
        return dx.toTensor();
    };
    MathBackendCPU.prototype.cast = function (x, dtype) {
        return backend_util.castTensor(x, dtype, this);
    };
    MathBackendCPU.prototype.reshape = function (x, shape) {
        return backend_util.reshapeTensor(x, shape);
    };
    MathBackendCPU.prototype.avgPool = function (x, convInfo) {
        this.assertNotComplex(x, 'avgPool');
        return this.pool(x, convInfo, 'avg').toFloat();
    };
    MathBackendCPU.prototype.resizeBilinear = function (x, newHeight, newWidth, alignCorners) {
        this.assertNotComplex(x, 'resizeBilinear');
        var _a = x.shape, batch = _a[0], oldHeight = _a[1], oldWidth = _a[2], numChannels = _a[3];
        var xValues = x.dataSync();
        var result = new Float32Array(util.sizeFromShape([batch, newHeight, newWidth, numChannels]));
        var effectiveInputSize = [
            (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
            (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
        ];
        var effectiveOutputSize = [
            (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
            (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
        ];
        var outputIdx = 0;
        var effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
        var effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
        for (var b = 0; b < batch; b++) {
            for (var r = 0; r < newHeight; r++) {
                var sourceFracRow = effectiveRowSizeRatio * r;
                var sourceRowFloor = Math.floor(sourceFracRow);
                var rowFrac = sourceFracRow - sourceRowFloor;
                var sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
                var topRowOffset = b * x.strides[0] + sourceRowFloor * x.strides[1];
                var botRowOffset = b * x.strides[0] + sourceRowCeil * x.strides[1];
                for (var c = 0; c < newWidth; c++) {
                    var sourceFracCol = effectiveColSizeRatio * c;
                    var sourceColFloor = Math.floor(sourceFracCol);
                    var colFrac = sourceFracCol - sourceColFloor;
                    var sourceColCeil = Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
                    var topLeftOffest = topRowOffset + sourceColFloor * x.strides[2];
                    var botLeftOffset = botRowOffset + sourceColFloor * x.strides[2];
                    var topRightOffset = topRowOffset + +sourceColCeil * x.strides[2];
                    var botRightOffest = botRowOffset + sourceColCeil * x.strides[2];
                    for (var d = 0; d < numChannels; d++) {
                        var topLeft = xValues[topLeftOffest + d];
                        var bottomLeft = xValues[botLeftOffset + d];
                        var topRight = xValues[topRightOffset + d];
                        var bottomRight = xValues[botRightOffest + d];
                        var top_1 = topLeft + (topRight - topLeft) * colFrac;
                        var bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                        var newValue = top_1 + (bottom - top_1) * rowFrac;
                        result[outputIdx++] = newValue;
                    }
                }
            }
        }
        return ops.tensor(result, [batch, newHeight, newWidth, numChannels]);
    };
    MathBackendCPU.prototype.resizeBilinearBackprop = function (dy, x, alignCorners) {
        this.assertNotComplex([dy, x], 'resizeBilinearBackprop');
        var _a = x.shape, batch = _a[0], xHeight = _a[1], xWidth = _a[2], depth = _a[3];
        var _b = dy.shape, yHeight = _b[1], yWidth = _b[2];
        var output = new Float32Array(batch * xHeight * xWidth * depth);
        var effectiveXSize = [
            (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
            (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
        ];
        var effectiveYSize = [
            (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
            (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
        ];
        var heightScale = effectiveXSize[0] / effectiveYSize[0];
        var widthScale = effectiveXSize[1] / effectiveYSize[1];
        var dyValues = dy.dataSync();
        var offset = 0;
        for (var b = 0; b < batch; b++) {
            var bOffset = b * x.strides[0];
            for (var r = 0; r < yHeight; r++) {
                var dxR = r * heightScale;
                var topDxRIndex = Math.floor(dxR);
                var bottomDxRIndex = Math.min(Math.ceil(dxR), xHeight - 1);
                var topDxROffset = bOffset + topDxRIndex * x.strides[1];
                var bottomDxROffset = bOffset + bottomDxRIndex * x.strides[1];
                var dxRLerp = dxR - topDxRIndex;
                var inverseDxRLerp = 1.0 - dxRLerp;
                for (var c = 0; c < yWidth; c++) {
                    var dxC = c * widthScale;
                    var leftDxCIndex = Math.floor(dxC);
                    var rightDxCIndex = Math.min(Math.ceil(dxC), xWidth - 1);
                    var dxCLerp = dxC - leftDxCIndex;
                    var inverseDxCLerp = 1.0 - dxCLerp;
                    var topLeftRCOffset = topDxROffset + leftDxCIndex * x.strides[2];
                    var topRightRCOffset = topDxROffset + rightDxCIndex * x.strides[2];
                    var bottomLeftRCOffset = bottomDxROffset + leftDxCIndex * x.strides[2];
                    var bottomRightRCOffset = bottomDxROffset + rightDxCIndex * x.strides[2];
                    var inverseDxRLerpTimesInverseDxCLerp = inverseDxRLerp * inverseDxCLerp;
                    var inverseDxRLerpTimesDxCLerp = inverseDxRLerp * dxCLerp;
                    var dxRLerpTimesInverseDxCLerp = dxRLerp * inverseDxCLerp;
                    var dxRLerpTimesDxCLerp = dxRLerp * dxCLerp;
                    for (var d = 0; d < depth; d++) {
                        var dyVal = dyValues[offset++];
                        output[topLeftRCOffset + d] +=
                            dyVal * inverseDxRLerpTimesInverseDxCLerp;
                        output[topRightRCOffset + d] += dyVal * inverseDxRLerpTimesDxCLerp;
                        output[bottomLeftRCOffset + d] +=
                            dyVal * dxRLerpTimesInverseDxCLerp;
                        output[bottomRightRCOffset + d] += dyVal * dxRLerpTimesDxCLerp;
                    }
                }
            }
        }
        return ops.tensor4d(output, [batch, xWidth, xHeight, depth], x.dtype);
    };
    MathBackendCPU.prototype.resizeNearestNeighbor = function (x, newHeight, newWidth, alignCorners) {
        this.assertNotComplex(x, 'resizeNearestNeighbor');
        var _a = x.shape, batch = _a[0], oldHeight = _a[1], oldWidth = _a[2], numChannels = _a[3];
        var xValues = x.dataSync();
        var output = new Float32Array(batch * newHeight * newWidth * numChannels);
        var effectiveInputSize = [
            (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
            (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
        ];
        var effectiveOutputSize = [
            (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
            (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
        ];
        var effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
        var effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
        var outputOffset = 0;
        for (var b = 0; b < batch; b++) {
            var batchOffset = b * x.strides[0];
            for (var r = 0; r < newHeight; r++) {
                var sourceFracRow = effectiveRowSizeRatio * r;
                var sourceNearestRow = Math.min(oldHeight - 1, alignCorners ? Math.round(sourceFracRow) :
                    Math.floor(sourceFracRow));
                var rowOffset = batchOffset + sourceNearestRow * x.strides[1];
                for (var c = 0; c < newWidth; c++) {
                    var sourceFracCol = effectiveColSizeRatio * c;
                    var sourceNearestCol = Math.min(oldWidth - 1, alignCorners ? Math.round(sourceFracCol) :
                        Math.floor(sourceFracCol));
                    var colOffset = rowOffset + sourceNearestCol * x.strides[2];
                    for (var d = 0; d < numChannels; d++) {
                        var newVal = xValues[colOffset + d];
                        output[outputOffset++] = newVal;
                    }
                }
            }
        }
        return ops.tensor(output, [batch, newHeight, newWidth, numChannels], x.dtype);
    };
    MathBackendCPU.prototype.resizeNearestNeighborBackprop = function (dy, x, alignCorners) {
        this.assertNotComplex([dy, x], 'resizeNearestNeighborBackprop');
        var _a = x.shape, batch = _a[0], xHeight = _a[1], xWidth = _a[2], depth = _a[3];
        var _b = dy.shape, yHeight = _b[1], yWidth = _b[2];
        var output = new Float32Array(batch * xHeight * xWidth * depth);
        var dyValues = dy.dataSync();
        var effectiveXSize = [
            (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
            (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
        ];
        var effectiveYSize = [
            (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
            (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
        ];
        var heightScale = effectiveXSize[0] / effectiveYSize[0];
        var widthScale = effectiveXSize[1] / effectiveYSize[1];
        var invHeightScale = 1 / heightScale;
        var invWidthScale = 1 / widthScale;
        var winHeight = (Math.ceil(invHeightScale) * 2) + 2;
        var winWidth = (Math.ceil(invWidthScale) * 2) + 2;
        for (var b = 0; b < batch; b++) {
            var batchOffset = b * x.strides[0];
            for (var r = 0; r < xHeight; r++) {
                var rowOffset = batchOffset + r * x.strides[1];
                var startRLerp = Math.floor(r * invHeightScale);
                var startDyR = Math.floor(startRLerp - (winHeight / 2));
                for (var c = 0; c < xWidth; c++) {
                    var colOffset = rowOffset + c * x.strides[2];
                    var startCLerp = Math.floor(c * invWidthScale);
                    var startDyC = Math.floor(startCLerp - (winWidth / 2));
                    for (var d = 0; d < depth; d++) {
                        var accum = 0;
                        for (var dyRIndex = 0; dyRIndex < winHeight; dyRIndex++) {
                            var dyR = dyRIndex + startDyR;
                            if (dyR < 0 || dyR >= yHeight) {
                                continue;
                            }
                            var dyROffset = batchOffset + dyR * dy.strides[1];
                            var sourceFracRow = dyR * heightScale;
                            var sourceNearestRow = Math.min(xHeight - 1, alignCorners ? Math.round(sourceFracRow) :
                                Math.floor(sourceFracRow));
                            if (r !== sourceNearestRow) {
                                continue;
                            }
                            for (var dyCIndex = 0; dyCIndex < winWidth; dyCIndex++) {
                                var dyC = dyCIndex + startDyC;
                                if (dyC < 0 || dyC >= yWidth) {
                                    continue;
                                }
                                var dyCOffset = dyROffset + dyC * dy.strides[2];
                                var sourceFracCol = dyC * widthScale;
                                var sourceNearestCol = Math.min(xWidth - 1, alignCorners ? Math.round(sourceFracCol) :
                                    Math.floor(sourceFracCol));
                                if (c === sourceNearestCol) {
                                    accum += dyValues[dyCOffset + d];
                                }
                            }
                        }
                        output[colOffset + d] = accum;
                    }
                }
            }
        }
        return ops.tensor4d(output, x.shape, x.dtype);
    };
    MathBackendCPU.prototype.batchNormalization = function (x, mean, variance, varianceEpsilon, scale, offset) {
        this.assertNotComplex([x, mean, variance, scale, offset], 'batchNormalization');
        var xVals = x.dataSync();
        var mVals = mean.dataSync();
        var varVals = variance.dataSync();
        var sVals = scale ? scale.dataSync() : new Float32Array([1]);
        var offVals = offset ? offset.dataSync() : new Float32Array([0]);
        var outVals = new Float32Array(xVals.length);
        var offValsLength = offVals.length;
        var sValsLength = sVals.length;
        var varValsLength = varVals.length;
        var mValsLength = mVals.length;
        var offi = 0;
        var mi = 0;
        var si = 0;
        var vi = 0;
        for (var i = 0; i < xVals.length; ++i) {
            outVals[i] = offVals[offi++] +
                (xVals[i] - mVals[mi++]) * sVals[si++] /
                    Math.sqrt(varVals[vi++] + varianceEpsilon);
            if (offi >= offValsLength) {
                offi = 0;
            }
            if (mi >= mValsLength) {
                mi = 0;
            }
            if (si >= sValsLength) {
                si = 0;
            }
            if (vi >= varValsLength) {
                vi = 0;
            }
        }
        return ops_1.tensor4d(outVals, x.shape);
    };
    MathBackendCPU.prototype.localResponseNormalization4D = function (x, depthRadius, bias, alpha, beta) {
        this.assertNotComplex(x, 'localResponseNormalization4D');
        var channels = x.shape[3];
        var maxD = channels - 1;
        var xValues = x.dataSync();
        var size = util.sizeFromShape(x.shape);
        var result = new Float32Array(size);
        function sumAcrossChannels(offset) {
            var currentChannel = offset % channels;
            var beginSumOffset = offset - currentChannel + Math.max(0, currentChannel - depthRadius);
            var endSumOffset = offset - currentChannel +
                Math.min(currentChannel + depthRadius, maxD);
            var sum = 0.0;
            for (; beginSumOffset <= endSumOffset; beginSumOffset++) {
                var z = xValues[beginSumOffset];
                sum += z * z;
            }
            return sum;
        }
        for (var offset = 0; offset < size; offset++) {
            var sum = sumAcrossChannels(offset);
            var val = xValues[offset] * Math.pow(bias + alpha * sum, -beta);
            result[offset] = val;
        }
        return ops.tensor4d(result, x.shape);
    };
    MathBackendCPU.prototype.LRNGrad = function (dy, inputImage, outputImage, depthRadius, bias, alpha, beta) {
        this.assertNotComplex(dy, 'LRNGrad');
        var channels = dy.shape[3];
        var dyValues = dy.dataSync();
        var inputImageValues = inputImage.dataSync();
        var outputImageValues = outputImage.dataSync();
        var result = new Float32Array(util.sizeFromShape(dy.shape));
        var size = util.sizeFromShape(dy.shape);
        for (var offset = 0; offset < size; offset++) {
            var currentChannel = offset % channels;
            var depthBegin = (offset - currentChannel) + Math.max(0, currentChannel - depthRadius);
            var depthEnd = (offset - currentChannel) +
                Math.min(channels, currentChannel + depthRadius + 1);
            var norm = 0;
            for (var k = depthBegin; k < depthEnd; k++) {
                norm += Math.pow(inputImageValues[k], 2);
            }
            norm = alpha * norm + bias;
            for (var k = depthBegin; k < depthEnd; k++) {
                var dyi = -2 * alpha * beta * inputImageValues[k] *
                    outputImageValues[offset] / norm;
                if (offset === k) {
                    dyi += Math.pow(norm, -beta);
                }
                dyi *= dyValues[offset];
                result[k] += dyi;
            }
        }
        return ops.tensor4d(result, dy.shape);
    };
    MathBackendCPU.prototype.multinomial = function (logits, normalized, numSamples, seed) {
        this.assertNotComplex(logits, 'multinomial');
        var probabilities = normalized ? logits : ops.softmax(logits);
        var batchSize = probabilities.shape[0];
        var numEvents = probabilities.shape[1];
        var res = ops.zeros([batchSize, numSamples], 'int32');
        var resVals = res.dataSync();
        var probVals = probabilities.dataSync();
        for (var b = 0; b < batchSize; ++b) {
            var offset = b * numEvents;
            var cdf = new Float32Array(numEvents - 1);
            cdf[0] = probVals[offset];
            for (var event_1 = 1; event_1 < cdf.length; ++event_1) {
                cdf[event_1] = cdf[event_1 - 1] + probVals[offset + event_1];
            }
            var random = seedrandom.alea(seed.toString());
            var outOffset = b * numSamples;
            for (var sampleId = 0; sampleId < numSamples; ++sampleId) {
                var r = random();
                resVals[outOffset + sampleId] = cdf.length;
                for (var event_2 = 0; event_2 < cdf.length; event_2++) {
                    if (r < cdf[event_2]) {
                        resVals[outOffset + sampleId] = event_2;
                        break;
                    }
                }
            }
        }
        return res;
    };
    MathBackendCPU.prototype.oneHot = function (indices, depth, onValue, offValue) {
        this.assertNotComplex(indices, 'oneHot');
        var res = new Float32Array(indices.size * depth);
        res.fill(offValue);
        for (var event_3 = 0; event_3 < indices.size; ++event_3) {
            if (indices.get(event_3) >= 0 && indices.get(event_3) < depth) {
                res[event_3 * depth + indices.get(event_3)] = onValue;
            }
        }
        return ops.tensor2d(res, [indices.size, depth], 'int32');
    };
    MathBackendCPU.prototype.nonMaxSuppression = function (boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
        this.assertNotComplex(boxes, 'nonMaxSuppression');
        var boxesVals = boxes.dataSync();
        var scoresVals = scores.dataSync();
        return non_max_suppression_impl_1.nonMaxSuppressionImpl(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
    };
    MathBackendCPU.prototype.fft = function (x) {
        if (x.shape[0] !== 1) {
            throw new Error("tf.fft() on CPU only supports vectors.");
        }
        var inverse = false;
        return this.fftImpl(x, inverse);
    };
    MathBackendCPU.prototype.ifft = function (x) {
        if (x.shape[0] !== 1) {
            throw new Error("tf.ifft() on CPU only supports vectors.");
        }
        var inverse = true;
        return this.fftImpl(x, inverse);
    };
    MathBackendCPU.prototype.fftImpl = function (x, inverse) {
        var x1D = x.as1D();
        var n = x1D.size;
        if (this.isExponentOf2(n)) {
            var result = this.fftRadix2(x1D, n, inverse).as2D(x.shape[0], x.shape[1]);
            if (inverse) {
                result = ops.complex(ops.real(result).div(ops_1.scalar(n)), ops.imag(result).div(ops_1.scalar(n)));
            }
            return result;
        }
        else {
            var data = x.dataSync();
            var rawOutput = this.fourierTransformByMatmul(data, n, inverse);
            var output = complex_util.splitRealAndImagArrays(rawOutput);
            return ops.complex(output.real, output.imag).as2D(x.shape[0], x.shape[1]);
        }
    };
    MathBackendCPU.prototype.isExponentOf2 = function (size) {
        return (size & size - 1) === 0;
    };
    MathBackendCPU.prototype.fftRadix2 = function (input, size, inverse) {
        if (size === 1) {
            return input;
        }
        var data = input.dataSync();
        var half = size / 2;
        var evenComplex = complex_util.complexWithEvenIndex(data);
        var evenTensor = ops.complex(evenComplex.real, evenComplex.imag).as1D();
        var oddComplex = complex_util.complexWithOddIndex(data);
        var oddTensor = ops.complex(oddComplex.real, oddComplex.imag).as1D();
        evenTensor = this.fftRadix2(evenTensor, half, inverse);
        oddTensor = this.fftRadix2(oddTensor, half, inverse);
        var e = complex_util.exponents(size, inverse);
        var exponent = ops.complex(e.real, e.imag).mul(oddTensor);
        var addPart = evenTensor.add(exponent);
        var subPart = evenTensor.sub(exponent);
        var realTensor = ops.real(addPart).concat(ops.real(subPart));
        var imagTensor = ops.imag(addPart).concat(ops.imag(subPart));
        return ops.complex(realTensor, imagTensor).as1D();
    };
    MathBackendCPU.prototype.fourierTransformByMatmul = function (data, size, inverse) {
        var ret = new Float32Array(size * 2);
        for (var r = 0; r < size; r++) {
            var real = 0.0;
            var imag = 0.0;
            for (var c = 0; c < size; c++) {
                var e = complex_util.exponent(r * c, size, inverse);
                var term = complex_util.getComplexWithIndex(data, c);
                real += term.real * e.real - term.imag * e.imag;
                imag += term.real * e.imag + term.imag * e.real;
            }
            if (inverse) {
                real /= size;
                imag /= size;
            }
            complex_util.assignToTypedArray(ret, real, imag, r);
        }
        return ret;
    };
    MathBackendCPU.prototype.depthToSpace = function (x, blockSize, dataFormat) {
        util.assert(dataFormat === 'NHWC', "Only NHWC dataFormat supported on CPU for depthToSpace. Got " + dataFormat);
        util.assert(blockSize > 1, "blockSize should be > 1 for depthToSpace, but was: " + blockSize);
        var batchSize = x.shape[0];
        var inputHeight = x.shape[1];
        var inputWidth = x.shape[2];
        var inputDepth = x.shape[3];
        var outputHeight = inputHeight * blockSize;
        var outputWidth = inputWidth * blockSize;
        var outputDepth = inputDepth / (blockSize * blockSize);
        var xValues = x.dataSync();
        var result = new Float32Array(batchSize * outputHeight * outputWidth * outputDepth);
        var outputIdx = 0;
        for (var b = 0; b < batchSize; ++b) {
            for (var h = 0; h < outputHeight; ++h) {
                var inH = Math.floor(h / blockSize);
                var offsetH = (h % blockSize);
                for (var w = 0; w < outputWidth; ++w) {
                    var inW = Math.floor(w / blockSize);
                    var offsetW = (w % blockSize);
                    var offsetD = (offsetH * blockSize + offsetW) * outputDepth;
                    for (var d = 0; d < outputDepth; ++d) {
                        var inD = d + offsetD;
                        var inputIdx = inD + inputDepth * (inW + inputWidth * (inH + inputHeight * b));
                        result[outputIdx++] = xValues[inputIdx];
                    }
                }
            }
        }
        return ops.tensor4d(result, [batchSize, outputHeight, outputWidth, outputDepth]);
    };
    MathBackendCPU.prototype.broadcastedBinaryOp = function (a, b, dtype, op) {
        var newShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        var result = ops.buffer(newShape, dtype);
        var aVals = a.dataSync();
        var bVals = b.dataSync();
        var aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
        var bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);
        var resVals = result.values;
        if (aBroadcastDims.length + bBroadcastDims.length === 0) {
            for (var i = 0; i < resVals.length; ++i) {
                resVals[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
            }
        }
        else {
            var aBuf = a.buffer();
            var bBuf = b.buffer();
            var _loop_2 = function (i) {
                var loc = result.indexToLoc(i);
                var aLoc = loc.slice(-a.rank);
                aBroadcastDims.forEach(function (d) { return aLoc[d] = 0; });
                var aIndex = aBuf.locToIndex(aLoc);
                var bLoc = loc.slice(-b.rank);
                bBroadcastDims.forEach(function (d) { return bLoc[d] = 0; });
                var bIndex = bBuf.locToIndex(bLoc);
                resVals[i] = op(aVals[aIndex], bVals[bIndex]);
            };
            for (var i = 0; i < resVals.length; ++i) {
                _loop_2(i);
            }
        }
        return result.toTensor();
    };
    MathBackendCPU.prototype.broadcastedBinaryComplexOp = function (a, b, op) {
        var newShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
        var realResult = ops.buffer(newShape, 'float32');
        var imagResult = ops.buffer(newShape, 'float32');
        var aVals = a.dataSync();
        var bVals = b.dataSync();
        var aBroadcastDims = broadcast_util.getBroadcastDims(a.shape, newShape);
        var bBroadcastDims = broadcast_util.getBroadcastDims(b.shape, newShape);
        var realVals = realResult.values;
        var imagVals = imagResult.values;
        if (aBroadcastDims.length + bBroadcastDims.length === 0) {
            for (var i = 0; i < realVals.length; i++) {
                var aIdx = i % aVals.length;
                var bIdx = i % bVals.length;
                var result = op(aVals[aIdx * 2], aVals[aIdx * 2 + 1], bVals[bIdx * 2], bVals[bIdx * 2 + 1]);
                realVals[i] = result.real;
                imagVals[i] = result.imag;
            }
        }
        else {
            var aRealBuf = this.data.get(a.dataId).complexTensors.real.buffer();
            var bRealBuf = this.data.get(b.dataId).complexTensors.real.buffer();
            var _loop_3 = function (i) {
                var loc = realResult.indexToLoc(i);
                var aLoc = loc.slice(-a.rank);
                aBroadcastDims.forEach(function (d) { return aLoc[d] = 0; });
                var aIndex = aRealBuf.locToIndex(aLoc);
                var bLoc = loc.slice(-b.rank);
                bBroadcastDims.forEach(function (d) { return bLoc[d] = 0; });
                var bIndex = bRealBuf.locToIndex(bLoc);
                var opResult = op(aVals[aIndex * 2], aVals[aIndex * 2 + 1], bVals[bIndex * 2], bVals[bIndex * 2 + 1]);
                realVals[i] = opResult.real;
                imagVals[i] = opResult.imag;
            };
            for (var i = 0; i < realVals.length; i++) {
                _loop_3(i);
            }
        }
        return this.complex(realResult.toTensor(), imagResult.toTensor());
    };
    MathBackendCPU.prototype.split = function (x, sizeSplits, axis) {
        return split_shared_1.split(x, sizeSplits, axis);
    };
    MathBackendCPU.prototype.dispose = function () { };
    MathBackendCPU.prototype.floatPrecision = function () {
        return 32;
    };
    MathBackendCPU.prototype.cropAndResize = function (images, boxes, boxIndex, cropSize, method, extrapolationValue) {
        var _a = images.shape, batch = _a[0], imageHeight = _a[1], imageWidth = _a[2], numChannels = _a[3];
        var numBoxes = boxes.shape[0];
        var cropHeight = cropSize[0], cropWidth = cropSize[1];
        var output = ops.buffer([numBoxes, cropHeight, cropWidth, numChannels]);
        var boxVals = boxes.dataSync();
        var boxIndVals = boxIndex.dataSync();
        var imageVals = images.dataSync();
        var inStride = images.strides;
        var outStride = output.strides;
        for (var b = 0; b < numBoxes; b++) {
            var startInd = b * 4;
            var y1 = boxVals[startInd];
            var x1 = boxVals[startInd + 1];
            var y2 = boxVals[startInd + 2];
            var x2 = boxVals[startInd + 3];
            var bInd = boxIndVals[b];
            if (bInd >= batch) {
                continue;
            }
            var heightScale = (cropHeight > 1) ?
                (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) :
                0;
            var widthScale = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;
            for (var y = 0; y < cropHeight; y++) {
                var yInd = (cropHeight > 1) ?
                    y1 * (imageHeight - 1) + y * (heightScale) :
                    0.5 * (y1 + y2) * (imageHeight - 1);
                if (yInd < 0 || yInd > imageHeight - 1) {
                    for (var x = 0; x < cropWidth; x++) {
                        for (var c = 0; c < numChannels; c++) {
                            var ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                            output.values[ind] = extrapolationValue;
                        }
                    }
                    continue;
                }
                if (method === 'bilinear') {
                    var topInd = Math.floor(yInd);
                    var bottomInd = Math.ceil(yInd);
                    var yLerp = yInd - topInd;
                    for (var x = 0; x < cropWidth; x++) {
                        var xInd = (cropWidth > 1) ?
                            x1 * (imageWidth - 1) + x * widthScale :
                            0.5 * (x1 + x2) * (imageWidth - 1);
                        if (xInd < 0 || xInd > imageWidth - 1) {
                            for (var c = 0; c < numChannels; c++) {
                                var ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                                output.values[ind] = extrapolationValue;
                            }
                            continue;
                        }
                        var leftInd = Math.floor(xInd);
                        var rightInd = Math.ceil(xInd);
                        var xLerp = xInd - leftInd;
                        for (var c = 0; c < numChannels; c++) {
                            var ind = c + leftInd * inStride[2] + topInd * inStride[1] +
                                bInd * inStride[0];
                            var topLeft = imageVals[ind];
                            ind = c + rightInd * inStride[2] + topInd * inStride[1] +
                                bInd * inStride[0];
                            var topRight = imageVals[ind];
                            ind = c + leftInd * inStride[2] + bottomInd * inStride[1] +
                                bInd * inStride[0];
                            var bottomLeft = imageVals[ind];
                            ind = c + rightInd * inStride[2] + bottomInd * inStride[1] +
                                bInd * inStride[0];
                            var bottomRight = imageVals[ind];
                            var top_2 = topLeft + (topRight - topLeft) * xLerp;
                            var bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
                            ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                            output.values[ind] = top_2 + ((bottom - top_2) * yLerp);
                        }
                    }
                }
                else {
                    for (var x = 0; x < cropWidth; ++x) {
                        var xInd = (cropWidth > 1) ?
                            x1 * (imageWidth - 1) + x * widthScale :
                            0.5 * (x1 + x2) * (imageWidth - 1);
                        if (xInd < 0 || xInd > imageWidth - 1) {
                            for (var c = 0; c < numChannels; c++) {
                                var ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                                output.values[ind] = extrapolationValue;
                            }
                            continue;
                        }
                        var closestX = Math.round(xInd);
                        var closestY = Math.round(yInd);
                        for (var c = 0; c < numChannels; c++) {
                            var inInd = c + closestX * inStride[2] +
                                closestY * inStride[1] + bInd * inStride[0];
                            var outInd = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                            output.values[outInd] = imageVals[inInd];
                        }
                    }
                }
            }
        }
        return output.toTensor();
    };
    MathBackendCPU.prototype.sparseToDense = function (sparseIndices, sparseValues, outputShape, defaultValue) {
        var _a = scatter_nd_util.calculateShapes(sparseValues, sparseIndices, outputShape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, sliceSize = _a.sliceSize, strides = _a.strides, outputSize = _a.outputSize;
        var sumDupeIndices = false;
        return this.scatter(sparseIndices, sparseValues, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
    };
    MathBackendCPU.prototype.gatherND = function (x, indices) {
        var indicesShape = indices.shape;
        var sliceRank = indicesShape[indicesShape.length - 1];
        var _a = gather_nd_util.prepareAndValidate(x, indices), resultShape = _a[0], numSlices = _a[1], sliceSize = _a[2], strides = _a[3];
        if (numSlices === 0) {
            return ops_1.tensor([], resultShape, x.dtype);
        }
        var buffer = new tensor_1.TensorBuffer([numSlices, sliceSize], x.dtype);
        var indicesData = indices.dataSync();
        var xData = x.dataSync();
        for (var i = 0; i < numSlices; i++) {
            var index = [];
            var flattenIndex = 0;
            for (var j = 0; j < sliceRank; j++) {
                var dim = indicesData[i * sliceRank + j];
                flattenIndex += dim * strides[j];
                index.push(dim);
            }
            if (flattenIndex < 0 || flattenIndex >= x.size / sliceSize) {
                throw new Error("Invalid indices: " + index + " does not index into " + x.shape);
            }
            for (var k = 0; k < sliceSize; k++) {
                buffer.values[i * sliceSize + k] = xData[flattenIndex * sliceSize + k];
            }
        }
        return buffer.toTensor().reshape(resultShape);
    };
    MathBackendCPU.prototype.scatterND = function (indices, updates, shape) {
        var _a = scatter_nd_util.calculateShapes(updates, indices, shape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, sliceSize = _a.sliceSize, strides = _a.strides, outputSize = _a.outputSize;
        var defaultValue = ops_1.scalar(0);
        var sumDupeIndices = true;
        return this.scatter(indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices);
    };
    MathBackendCPU.prototype.scatter = function (indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices) {
        var flattenShape = [outputSize / sliceSize, sliceSize];
        var indicesData = indices.dataSync();
        var updatesData = updates.dataSync();
        if (outputSize === 0) {
            return ops_1.tensor([], shape, updates.dtype);
        }
        var buffer = new tensor_1.TensorBuffer(flattenShape, updates.dtype);
        buffer.values.fill(defaultValue.dataSync()[0]);
        for (var i = 0; i < numUpdates; i++) {
            var index = [];
            var flattenIndex = 0;
            for (var j = 0; j < sliceRank; j++) {
                var dim = indicesData[i * sliceRank + j];
                index.push(dim);
                flattenIndex += dim * strides[j];
            }
            if (flattenIndex < 0 || flattenIndex >= outputSize / sliceSize) {
                throw new Error("Invalid indices: " + index + " does not index into " + shape);
            }
            for (var k = 0; k < sliceSize; k++) {
                if (sumDupeIndices) {
                    buffer.values[flattenIndex * sliceSize + k] +=
                        updatesData[i * sliceSize + k];
                }
                else {
                    buffer.values[flattenIndex * sliceSize + k] = updates.rank === 0 ?
                        updatesData[0] :
                        updatesData[i * sliceSize + k];
                }
            }
        }
        return buffer.toTensor().reshape(shape);
    };
    return MathBackendCPU;
}());
exports.MathBackendCPU = MathBackendCPU;
environment_1.ENV.registerBackend('cpu', function () { return new MathBackendCPU(); }, 1, tensor_1.setTensorTracker);
//# sourceMappingURL=backend_cpu.js.map