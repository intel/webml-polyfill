"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var DataStorage = (function () {
    function DataStorage(dataMover) {
        this.dataMover = dataMover;
        this.data = new WeakMap();
    }
    DataStorage.prototype.get = function (dataId) {
        if (!this.data.has(dataId)) {
            this.dataMover.moveData(dataId);
        }
        return this.data.get(dataId);
    };
    DataStorage.prototype.set = function (dataId, value) {
        this.data.set(dataId, value);
    };
    DataStorage.prototype.has = function (dataId) {
        return this.data.has(dataId);
    };
    DataStorage.prototype.delete = function (dataId) {
        return this.data.delete(dataId);
    };
    return DataStorage;
}());
exports.DataStorage = DataStorage;
var KernelBackend = (function () {
    function KernelBackend() {
    }
    KernelBackend.prototype.time = function (f) {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.read = function (dataId) {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.readSync = function (dataId) {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.disposeData = function (dataId) {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.write = function (dataId, values) {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.fromPixels = function (pixels, numChannels) {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.register = function (dataId, shape, dtype) {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.memory = function () {
        throw new Error('Not yet implemented.');
    };
    KernelBackend.prototype.floatPrecision = function () {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.batchMatMul = function (a, b, transposeA, transposeB) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.slice = function (x, begin, size) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.stridedSlice = function (x, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.reverse = function (a, axis) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.concat = function (tensors, axis) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.neg = function (a) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.add = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.addN = function (tensors) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.subtract = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.multiply = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.realDivide = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.floorDiv = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.sum = function (x, axes) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.prod = function (x, axes) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.unsortedSegmentSum = function (x, segmentIds, numSegments) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.argMin = function (x, axis) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.argMax = function (x, axis) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.equal = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.notEqual = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.less = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.lessEqual = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.greater = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.greaterEqual = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.logicalNot = function (a) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.logicalAnd = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.logicalOr = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.where = function (condition) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.select = function (condition, a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.topk = function (x, k, sorted) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.min = function (x, axes) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.minimum = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.mod = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.max = function (x, axes) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.maximum = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.all = function (x, axes) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.any = function (x, axes) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.squaredDifference = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.ceil = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.floor = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.round = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.sign = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.pow = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.exp = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.expm1 = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.log = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.log1p = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.sqrt = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.rsqrt = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.square = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.reciprocal = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.relu = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.elu = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.eluDer = function (dy, y) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.selu = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.int = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.clip = function (x, min, max) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.abs = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.complexAbs = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.sigmoid = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.softplus = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.sin = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.cos = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.tan = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.asin = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.acos = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.atan = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.atan2 = function (a, b) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.sinh = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.cosh = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.tanh = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.asinh = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.acosh = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.atanh = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.erf = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.step = function (x, alpha) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.conv2d = function (x, filter, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.conv2dDerInput = function (dy, filter, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.conv2dDerFilter = function (x, dY, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.depthwiseConv2D = function (input, filter, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.depthwiseConv2DDerInput = function (dy, filter, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.depthwiseConv2DDerFilter = function (x, dY, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.maxPool = function (x, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.maxPoolBackprop = function (dy, x, y, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.avgPool = function (x, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.avgPoolBackprop = function (dy, x, convInfo) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.reshape = function (x, shape) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.cast = function (x, dtype) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.tile = function (x, reps) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.pad = function (x, paddings, constantValue) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.transpose = function (x, perm) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.gather = function (x, indices, axis) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.gatherND = function (x, indices) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.scatterND = function (indices, updates, shape) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.batchToSpaceND = function (x, blockShape, crops) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.spaceToBatchND = function (x, blockShape, paddings) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.resizeBilinear = function (x, newHeight, newWidth, alignCorners) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.resizeBilinearBackprop = function (dy, x, alignCorners) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.resizeNearestNeighbor = function (x, newHEight, newWidth, alignCorners) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.resizeNearestNeighborBackprop = function (dy, x, alignCorners) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.batchNormalization = function (x, mean, variance, varianceEpsilon, scale, offset) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.localResponseNormalization4D = function (x, radius, bias, alpha, beta) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.LRNGrad = function (dy, inputImage, outputImage, radius, bias, alpha, beta) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.multinomial = function (logits, normalized, numSamples, seed) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.oneHot = function (indices, depth, onValue, offValue) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.cumsum = function (x, axis, exclusive, reverse) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.nonMaxSuppression = function (boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.fft = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.ifft = function (x) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.complex = function (real, imag) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.real = function (input) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.imag = function (input) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.cropAndResize = function (image, boxes, boxIndex, cropSize, method, extrapolationValue) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.depthToSpace = function (x, blockSize, dataFormat) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.split = function (value, sizeSplits, axis) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.sparseToDense = function (sparseIndices, sparseValues, outputShape, defaultValue) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.setDataMover = function (dataMover) {
        throw new Error('Not yet implemented');
    };
    KernelBackend.prototype.dispose = function () {
        throw new Error('Not yet implemented');
    };
    return KernelBackend;
}());
exports.KernelBackend = KernelBackend;
//# sourceMappingURL=backend.js.map