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
var tensor_1 = require("../tensor");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var axis_util_1 = require("./axis_util");
var concat_split_1 = require("./concat_split");
var operation_1 = require("./operation");
var rand_1 = require("./rand");
var tensor_ops_1 = require("./tensor_ops");
function clone_(x) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'clone');
    var der = function (dy) {
        return { $x: function () { return dy.toFloat(); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) {
        return tensor_1.Tensor.make($x.shape, { dataId: $x.dataId }, $x.dtype);
    }, { $x: $x }, der);
}
function eye_(numRows, numColumns, batchShape, dtype) {
    if (dtype === void 0) { dtype = 'float32'; }
    if (numColumns == null) {
        numColumns = numRows;
    }
    var buff = buffer([numRows, numColumns], dtype);
    var n = numRows <= numColumns ? numRows : numColumns;
    for (var i = 0; i < n; ++i) {
        buff.set(1, i, i);
    }
    var out = buff.toTensor().as2D(numRows, numColumns);
    if (batchShape == null) {
        return out;
    }
    else {
        if (batchShape.length === 1) {
            return exports.tile(exports.expandDims(out, 0), [batchShape[0], 1, 1]);
        }
        else if (batchShape.length === 2) {
            return exports.tile(exports.expandDims(exports.expandDims(out, 0), 0), [batchShape[0], batchShape[1], 1, 1]);
        }
        else if (batchShape.length === 3) {
            return exports.tile(exports.expandDims(exports.expandDims(exports.expandDims(out, 0), 0), 0), [batchShape[0], batchShape[1], batchShape[2], 1, 1]);
        }
        else {
            throw new Error("eye() currently supports only 1D and 2D " +
                ("batchShapes, but received " + batchShape.length + "D."));
        }
    }
}
function randomNormal_(shape, mean, stdDev, dtype, seed) {
    if (mean === void 0) { mean = 0; }
    if (stdDev === void 0) { stdDev = 1; }
    if (dtype != null && dtype === 'bool') {
        throw new Error("Unsupported data type " + dtype);
    }
    var randGauss = new rand_1.MPRandGauss(mean, stdDev, dtype, false, seed);
    var res = buffer(shape, dtype);
    for (var i = 0; i < res.values.length; i++) {
        res.values[i] = randGauss.nextValue();
    }
    return res.toTensor();
}
function truncatedNormal_(shape, mean, stdDev, dtype, seed) {
    if (mean === void 0) { mean = 0; }
    if (stdDev === void 0) { stdDev = 1; }
    if (dtype != null && dtype === 'bool') {
        throw new Error("Unsupported data type " + dtype);
    }
    var randGauss = new rand_1.MPRandGauss(mean, stdDev, dtype, true, seed);
    var res = buffer(shape, dtype);
    for (var i = 0; i < res.values.length; i++) {
        res.values[i] = randGauss.nextValue();
    }
    return res.toTensor();
}
function randomUniform_(shape, minval, maxval, dtype) {
    if (minval === void 0) { minval = 0; }
    if (maxval === void 0) { maxval = 1; }
    if (dtype === void 0) { dtype = 'float32'; }
    var res = buffer(shape, dtype);
    for (var i = 0; i < res.values.length; i++) {
        res.values[i] = util.randUniform(minval, maxval);
    }
    return res.toTensor();
}
function rand_(shape, randFunction, dtype) {
    var size = util.sizeFromShape(shape);
    var values = null;
    if (dtype == null || dtype === 'float32') {
        values = new Float32Array(size);
    }
    else if (dtype === 'int32') {
        values = new Int32Array(size);
    }
    else if (dtype === 'bool') {
        values = new Uint8Array(size);
    }
    else {
        throw new Error("Unknown data type " + dtype);
    }
    for (var i = 0; i < size; i++) {
        values[i] = randFunction();
    }
    return tensor_1.Tensor.make(shape, { values: values }, dtype);
}
function multinomial_(logits, numSamples, seed, normalized) {
    if (normalized === void 0) { normalized = false; }
    var $logits = tensor_util_env_1.convertToTensor(logits, 'logits', 'multinomial');
    var numOutcomes = $logits.size;
    var origRank = $logits.rank;
    if (numOutcomes < 2) {
        throw new Error("Error in multinomial: you need at least 2 outcomes, but got " +
            (numOutcomes + "."));
    }
    if (origRank > 2) {
        throw new Error("Rank of probabilities must be 1 or 2, but is " + origRank);
    }
    seed = seed || Math.random();
    var logits2D = origRank === 1 ? $logits.as2D(1, -1) : $logits;
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.multinomial(logits2D, normalized, numSamples, seed); }, { logits2D: logits2D });
    return origRank === 1 ? res.as1D() : res;
}
function oneHot_(indices, depth, onValue, offValue) {
    if (onValue === void 0) { onValue = 1; }
    if (offValue === void 0) { offValue = 0; }
    var $indices = tensor_util_env_1.convertToTensor(indices, 'indices', 'oneHot', 'int32');
    util.assert($indices.dtype === 'int32', 'Indices must be of dtype `int32`');
    if (depth < 2) {
        throw new Error("Error in oneHot: depth must be >=2, but it is " + depth);
    }
    var grad = function (dy) {
        return { $indices: function () { return tensor_ops_1.zerosLike($indices); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.oneHot($indices, depth, onValue, offValue); }, { $indices: $indices }, grad);
}
function fromPixels_(pixels, numChannels) {
    if (numChannels === void 0) { numChannels = 3; }
    if (numChannels > 4) {
        throw new Error('Cannot construct Tensor with more than 4 channels from pixels.');
    }
    return environment_1.ENV.engine.fromPixels(pixels, numChannels);
}
function toPixels(img, canvas) {
    return __awaiter(this, void 0, void 0, function () {
        var $img, _a, height, width, depth, minTensor, maxTensor, min, max, data, multiplier, bytes, i, r, g, b, a, j, ctx, imageData;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    $img = tensor_util_env_1.convertToTensor(img, 'img', 'toPixels', 'int32');
                    if ($img.rank !== 2 && $img.rank !== 3) {
                        throw new Error("toPixels only supports rank 2 or 3 tensors, got rank " + $img.rank + ".");
                    }
                    _a = $img.shape.slice(0, 2), height = _a[0], width = _a[1];
                    depth = $img.rank === 2 ? 1 : $img.shape[2];
                    if (depth > 4 || depth === 2) {
                        throw new Error("toPixels only supports depth of size " +
                            ("1, 3 or 4 but got " + depth));
                    }
                    minTensor = $img.min();
                    maxTensor = $img.max();
                    return [4, minTensor.data()];
                case 1:
                    min = (_b.sent())[0];
                    return [4, maxTensor.data()];
                case 2:
                    max = (_b.sent())[0];
                    minTensor.dispose();
                    maxTensor.dispose();
                    if ($img.dtype === 'float32') {
                        if (min < 0 || max > 1) {
                            throw new Error("Tensor values for a float32 Tensor must be in the " +
                                ("range [0 - 1] but got range [" + min + " - " + max + "]."));
                        }
                    }
                    else if ($img.dtype === 'int32') {
                        if (min < 0 || max > 255) {
                            throw new Error("Tensor values for a int32 Tensor must be in the " +
                                ("range [0 - 255] but got range [" + min + " - " + max + "]."));
                        }
                    }
                    else {
                        throw new Error("Unsupported type for toPixels: " + $img.dtype + "." +
                            " Please use float32 or int32 tensors.");
                    }
                    return [4, $img.data()];
                case 3:
                    data = _b.sent();
                    multiplier = $img.dtype === 'float32' ? 255 : 1;
                    bytes = new Uint8ClampedArray(width * height * 4);
                    for (i = 0; i < height * width; ++i) {
                        r = void 0, g = void 0, b = void 0, a = void 0;
                        if (depth === 1) {
                            r = data[i] * multiplier;
                            g = data[i] * multiplier;
                            b = data[i] * multiplier;
                            a = 255;
                        }
                        else if (depth === 3) {
                            r = data[i * 3] * multiplier;
                            g = data[i * 3 + 1] * multiplier;
                            b = data[i * 3 + 2] * multiplier;
                            a = 255;
                        }
                        else if (depth === 4) {
                            r = data[i * 4] * multiplier;
                            g = data[i * 4 + 1] * multiplier;
                            b = data[i * 4 + 2] * multiplier;
                            a = data[i * 4 + 3] * multiplier;
                        }
                        j = i * 4;
                        bytes[j + 0] = Math.round(r);
                        bytes[j + 1] = Math.round(g);
                        bytes[j + 2] = Math.round(b);
                        bytes[j + 3] = Math.round(a);
                    }
                    if (canvas != null) {
                        canvas.width = width;
                        canvas.height = height;
                        ctx = canvas.getContext('2d');
                        imageData = new ImageData(bytes, width, height);
                        ctx.putImageData(imageData, 0, 0);
                    }
                    if ($img !== img) {
                        $img.dispose();
                    }
                    return [2, bytes];
            }
        });
    });
}
exports.toPixels = toPixels;
function reshape_(x, shape) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'reshape');
    shape = util.inferFromImplicitShape(shape, $x.size);
    util.assert($x.size === util.sizeFromShape(shape), 'new shape and old shape must have the same number of elements.');
    var grad = function (dy) {
        return { $x: function () { return dy.reshape($x.shape); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.reshape($x, shape); }, { $x: $x }, grad);
}
function squeeze_(x, axis) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'squeeze');
    return exports.reshape($x, util.squeezeShape($x.shape, axis).newShape);
}
function cast_(x, dtype) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'cast');
    var grad = function (dy) {
        return { $x: function () { return dy.clone(); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.cast($x, dtype); }, { $x: $x }, grad);
}
function tile_(x, reps) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'tile');
    util.assert($x.rank === reps.length, "Error in transpose: rank of input " + $x.rank + " " +
        ("must match length of reps " + reps + "."));
    var grad = function (dy) {
        var derX = function () {
            var xGrad = tensor_ops_1.zerosLike($x);
            if ($x.rank === 1) {
                for (var i = 0; i < reps[0]; ++i) {
                    xGrad = xGrad.add(dy.slice([i * $x.shape[0]], [$x.shape[0]]));
                }
            }
            else if ($x.rank === 2) {
                for (var i = 0; i < reps[0]; ++i) {
                    for (var j = 0; j < reps[1]; ++j) {
                        xGrad = xGrad.add(dy.slice([i * $x.shape[0], j * $x.shape[1]], [$x.shape[0], $x.shape[1]]));
                    }
                }
            }
            else if ($x.rank === 3) {
                for (var i = 0; i < reps[0]; ++i) {
                    for (var j = 0; j < reps[1]; ++j) {
                        for (var k = 0; k < reps[2]; ++k) {
                            xGrad = xGrad.add(dy.slice([i * $x.shape[0], j * $x.shape[1], k * $x.shape[2]], [$x.shape[0], $x.shape[1], $x.shape[2]]));
                        }
                    }
                }
            }
            else if ($x.rank === 4) {
                for (var i = 0; i < reps[0]; ++i) {
                    for (var j = 0; j < reps[1]; ++j) {
                        for (var k = 0; k < reps[2]; ++k) {
                            for (var l = 0; l < reps[3]; ++l) {
                                xGrad = xGrad.add(dy.slice([
                                    i * $x.shape[0], j * $x.shape[1], k * $x.shape[2],
                                    l * $x.shape[3]
                                ], [$x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]));
                            }
                        }
                    }
                }
            }
            else {
                throw new Error("Gradient for tile operation is not implemented for rank-" +
                    ($x.rank + " tensors yet."));
            }
            return xGrad;
        };
        return { $x: derX };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.tile($x, reps); }, { $x: $x }, grad);
}
function pad1d_(x, paddings, constantValue) {
    if (constantValue === void 0) { constantValue = 0; }
    util.assert(paddings.length === 2, 'Invalid number of paddings. Must be length of 2.');
    return exports.pad(x, [paddings], constantValue);
}
function pad2d_(x, paddings, constantValue) {
    if (constantValue === void 0) { constantValue = 0; }
    util.assert(paddings.length === 2 && paddings[0].length === 2 &&
        paddings[1].length === 2, 'Invalid number of paddings. Must be length of 2 each.');
    return exports.pad(x, paddings, constantValue);
}
function pad3d_(x, paddings, constantValue) {
    if (constantValue === void 0) { constantValue = 0; }
    util.assert(paddings.length === 3 && paddings[0].length === 2 &&
        paddings[1].length === 2 && paddings[2].length === 2, 'Invalid number of paddings. Must be length of 2 each.');
    return exports.pad(x, paddings, constantValue);
}
function pad4d_(x, paddings, constantValue) {
    if (constantValue === void 0) { constantValue = 0; }
    util.assert(paddings.length === 4 && paddings[0].length === 2 &&
        paddings[1].length === 2 && paddings[2].length === 2 &&
        paddings[3].length === 2, 'Invalid number of paddings. Must be length of 2 each.');
    return exports.pad(x, paddings, constantValue);
}
function pad_(x, paddings, constantValue) {
    if (constantValue === void 0) { constantValue = 0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'pad');
    if ($x.rank === 0) {
        throw new Error('pad(scalar) is not defined. Pass non-scalar to pad');
    }
    var begin = paddings.map(function (p) { return p[0]; });
    var grad = function (dy) {
        return { $x: function () { return dy.slice(begin, $x.shape); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.pad($x, paddings, constantValue); }, { $x: $x }, grad);
}
function stack_(tensors, axis) {
    if (axis === void 0) { axis = 0; }
    var $tensors = tensor_util_env_1.convertToTensorArray(tensors, 'tensors', 'stack');
    util.assert($tensors.length >= 1, 'Pass at least one tensor to tf.stack');
    if ($tensors.length === 1) {
        return $tensors[0].expandDims(axis);
    }
    var rank = $tensors[0].rank;
    var shape = $tensors[0].shape;
    var dtype = $tensors[0].dtype;
    util.assert(axis <= rank, 'Axis must be <= rank of the tensor');
    $tensors.forEach(function (t) {
        util.assertShapesMatch(shape, t.shape, 'All tensors passed to stack must have matching shapes');
    });
    $tensors.forEach(function (t) {
        util.assert(dtype === t.dtype, 'All tensors passed to stack must have matching dtypes');
    });
    var expandedTensors = $tensors.map(function (t) { return t.expandDims(axis); });
    return concat_split_1.concat(expandedTensors, axis);
}
function batchToSpaceND_(x, blockShape, crops) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'batchToSpaceND');
    var prod = blockShape.reduce(function (a, b) { return a * b; });
    util.assert($x.rank >= 1 + blockShape.length, "input rank is " + $x.rank + " but should be > than blockShape.length " + blockShape.length);
    util.assert(crops.length === blockShape.length, "crops.length is " + crops.length + " but should be equal to blockShape.length  " + blockShape.length);
    util.assert($x.shape[0] % prod === 0, "input tensor batch is " + $x.shape[0] + " but is not divisible by the product of " +
        ("the elements of blockShape " + blockShape.join(' * ') + " === " + prod));
    var grad = function (dy) {
        return { $x: function () { return dy.spaceToBatchND(blockShape, crops); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.batchToSpaceND($x, blockShape, crops); }, { $x: $x }, grad);
}
function spaceToBatchND_(x, blockShape, paddings) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'spaceToBatchND');
    util.assert($x.rank >= 1 + blockShape.length, "input rank " + $x.rank + " should be > than [blockShape] " + blockShape.length);
    util.assert(paddings.length === blockShape.length, "paddings.shape[0] " + paddings.length + " must be equal to [blockShape] " + blockShape.length);
    util.assert($x.shape.reduce(function (a, b, i) {
        if (i > 0 && i <= blockShape.length) {
            return a &&
                ((b + paddings[i - 1][0] + paddings[i - 1][1]) %
                    blockShape[i - 1] ===
                    0);
        }
        return a;
    }, true), "input spatial dimensions " + $x.shape.slice(1) + " with paddings " + paddings.toString() + " must be divisible by blockShapes " + blockShape.toString());
    var grad = function (dy) {
        return { $x: function () { return dy.batchToSpaceND(blockShape, paddings); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.spaceToBatchND($x, blockShape, paddings); }, { $x: $x }, grad);
}
function unstack_(x, axis) {
    if (axis === void 0) { axis = 0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'unstack');
    var num = $x.shape[axis];
    var outputShape = Array($x.rank - 1).fill(0);
    var outIndex = 0;
    for (var i = 0; i < $x.rank; i++) {
        if (i !== axis) {
            outputShape[outIndex] = $x.shape[i];
            outIndex++;
        }
    }
    var splitSizes;
    splitSizes = Array(num).fill(1);
    var begin = Array($x.rank).fill(0);
    var size = $x.shape.slice();
    return splitSizes.map(function (s) {
        size[axis] = s;
        var slice = $x.slice(begin, size);
        begin[axis] += s;
        return slice.reshape(outputShape);
    });
}
function cumsum_(x, axis, exclusive, reverse) {
    if (axis === void 0) { axis = 0; }
    if (exclusive === void 0) { exclusive = false; }
    if (reverse === void 0) { reverse = false; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'cumsum');
    axis = axis | 0;
    var permutation = axis_util_1.getAxesPermutation([axis], $x.rank);
    var permutedX = $x;
    if (permutation != null) {
        permutedX = $x.transpose(permutation);
    }
    var permutedAxis = axis_util_1.getInnerMostAxes(1, $x.rank)[0];
    var grad = function (dy) {
        return { permutedX: function () { return dy.cumsum(axis, exclusive, !reverse); } };
    };
    var value = environment_1.ENV.engine.runKernel(function (backend) { return backend.cumsum(permutedX, permutedAxis, exclusive, reverse); }, { permutedX: permutedX }, grad);
    if (permutation != null) {
        value = value.transpose(permutation);
    }
    return value;
}
function expandDims_(x, axis) {
    if (axis === void 0) { axis = 0; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'expandDims');
    util.assert(axis <= $x.rank, 'Axis must be <= rank of the tensor');
    var newShape = $x.shape.slice();
    if (axis < 0) {
        util.assert(-($x.rank + 1) <= axis, "Axis must be in the interval [" + -($x.rank + 1) + ", " + $x.rank + "]");
        axis = $x.rank + axis + 1;
    }
    newShape.splice(axis, 0, 1);
    return exports.reshape($x, newShape);
}
function depthToSpace_(x, blockSize, dataFormat) {
    if (dataFormat === void 0) { dataFormat = 'NHWC'; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'depthToSpace');
    var inputHeight = (dataFormat === 'NHWC') ? $x.shape[1] : $x.shape[2];
    var inputWidth = (dataFormat === 'NHWC') ? $x.shape[2] : $x.shape[3];
    var inputDepth = (dataFormat === 'NHWC') ? $x.shape[3] : $x.shape[1];
    util.assert(inputHeight * blockSize >= 0, "Negative dimension size caused by overflow when multiplying\n      " + inputHeight + " and " + blockSize + "  for depthToSpace with input shape\n      " + $x.shape);
    util.assert(inputWidth * blockSize >= 0, "Negative dimension size caused by overflow when multiplying\n      " + inputWidth + " and " + blockSize + " for depthToSpace with input shape\n          " + $x.shape);
    util.assert((inputDepth % (blockSize * blockSize) === 0), "Dimension size must be evenly divisible by " + blockSize * blockSize + " but is " + inputDepth + " for depthToSpace with input shape " + $x.shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.depthToSpace($x, blockSize, dataFormat); }, { $x: $x });
}
function setdiff1dAsync_(x, y) {
    return __awaiter(this, void 0, void 0, function () {
        var $x, $y, xVals, yVals, ySet, outputSize, i, buffer, indices, i, p;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    $x = tensor_util_env_1.convertToTensor(x, 'x', 'setdiff1d');
                    $y = tensor_util_env_1.convertToTensor(y, 'y', 'setdiff1d');
                    util.assert($x.dtype === $y.dtype, "x and y should have the same dtype, but got x (" + $x.dtype + ") and y (" + $y.dtype + ").");
                    util.assert($x.rank === 1, "x should be 1D tensor, but got x (" + $x.shape + ").");
                    util.assert($y.rank === 1, "y should be 1D tensor, but got y (" + $y.shape + ").");
                    return [4, $x.data()];
                case 1:
                    xVals = _a.sent();
                    return [4, $y.data()];
                case 2:
                    yVals = _a.sent();
                    ySet = new Set(yVals);
                    outputSize = 0;
                    for (i = 0; i < xVals.length; i++) {
                        if (!ySet.has(xVals[i])) {
                            outputSize++;
                        }
                    }
                    buffer = new tensor_1.TensorBuffer([outputSize], $x.dtype);
                    indices = new tensor_1.TensorBuffer([outputSize], 'int32');
                    for (i = 0, p = 0; i < xVals.length; i++) {
                        if (!ySet.has(xVals[i])) {
                            buffer.values[p] = xVals[i];
                            indices.values[p] = i;
                            p++;
                        }
                    }
                    return [2, [buffer.toTensor(), indices.toTensor()]];
            }
        });
    });
}
function buffer(shape, dtype, values) {
    if (dtype === void 0) { dtype = 'float32'; }
    return new tensor_1.TensorBuffer(shape, dtype, values);
}
exports.buffer = buffer;
function print(x, verbose) {
    if (verbose === void 0) { verbose = false; }
    console.log(x.toString(verbose));
}
exports.print = print;
exports.batchToSpaceND = operation_1.op({ batchToSpaceND_: batchToSpaceND_ });
exports.cast = operation_1.op({ cast_: cast_ });
exports.clone = operation_1.op({ clone_: clone_ });
exports.cumsum = operation_1.op({ cumsum_: cumsum_ });
exports.depthToSpace = operation_1.op({ depthToSpace_: depthToSpace_ });
exports.expandDims = operation_1.op({ expandDims_: expandDims_ });
exports.eye = operation_1.op({ eye_: eye_ });
exports.fromPixels = operation_1.op({ fromPixels_: fromPixels_ });
exports.multinomial = operation_1.op({ multinomial_: multinomial_ });
exports.oneHot = operation_1.op({ oneHot_: oneHot_ });
exports.pad = operation_1.op({ pad_: pad_ });
exports.pad1d = operation_1.op({ pad1d_: pad1d_ });
exports.pad2d = operation_1.op({ pad2d_: pad2d_ });
exports.pad3d = operation_1.op({ pad3d_: pad3d_ });
exports.pad4d = operation_1.op({ pad4d_: pad4d_ });
exports.rand = operation_1.op({ rand_: rand_ });
exports.randomNormal = operation_1.op({ randomNormal_: randomNormal_ });
exports.randomUniform = operation_1.op({ randomUniform_: randomUniform_ });
exports.reshape = operation_1.op({ reshape_: reshape_ });
exports.spaceToBatchND = operation_1.op({ spaceToBatchND_: spaceToBatchND_ });
exports.squeeze = operation_1.op({ squeeze_: squeeze_ });
exports.stack = operation_1.op({ stack_: stack_ });
exports.tile = operation_1.op({ tile_: tile_ });
exports.truncatedNormal = operation_1.op({ truncatedNormal_: truncatedNormal_ });
exports.unstack = operation_1.op({ unstack_: unstack_ });
exports.setdiff1dAsync = setdiff1dAsync_;
//# sourceMappingURL=array_ops.js.map