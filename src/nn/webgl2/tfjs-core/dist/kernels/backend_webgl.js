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
var canvas_util_1 = require("../canvas_util");
var environment_1 = require("../environment");
var globals_1 = require("../globals");
var log_1 = require("../log");
var array_ops_util = require("../ops/array_ops_util");
var axis_util = require("../ops/axis_util");
var concat_util_1 = require("../ops/concat_util");
var gather_nd_util = require("../ops/gather_nd_util");
var reduce_util = require("../ops/reduce_util");
var scatter_nd_util = require("../ops/scatter_nd_util");
var segment_util = require("../ops/segment_util");
var slice_util_1 = require("../ops/slice_util");
var softmax_1 = require("../ops/softmax");
var tensor_ops_1 = require("../ops/tensor_ops");
var tensor_1 = require("../tensor");
var types_1 = require("../types");
var util = require("../util");
var util_1 = require("../util");
var backend_1 = require("./backend");
var backend_util = require("./backend_util");
var complex_util_1 = require("./complex_util");
var non_max_suppression_impl_1 = require("./non_max_suppression_impl");
var split_shared_1 = require("./split_shared");
var topk_impl_1 = require("./topk_impl");
var argminmax_gpu_1 = require("./webgl/argminmax_gpu");
var avg_pool_backprop_gpu_1 = require("./webgl/avg_pool_backprop_gpu");
var batchnorm_gpu_1 = require("./webgl/batchnorm_gpu");
var batchnorm_packed_gpu_1 = require("./webgl/batchnorm_packed_gpu");
var binaryop_complex_gpu = require("./webgl/binaryop_complex_gpu");
var binaryop_complex_gpu_1 = require("./webgl/binaryop_complex_gpu");
var binaryop_gpu = require("./webgl/binaryop_gpu");
var binaryop_gpu_1 = require("./webgl/binaryop_gpu");
var clip_gpu_1 = require("./webgl/clip_gpu");
var complex_abs_gpu_1 = require("./webgl/complex_abs_gpu");
var concat_gpu_1 = require("./webgl/concat_gpu");
var conv_backprop_gpu_1 = require("./webgl/conv_backprop_gpu");
var conv_backprop_gpu_depthwise_1 = require("./webgl/conv_backprop_gpu_depthwise");
var conv_gpu_1 = require("./webgl/conv_gpu");
var conv_gpu_depthwise_1 = require("./webgl/conv_gpu_depthwise");
var crop_and_resize_gpu_1 = require("./webgl/crop_and_resize_gpu");
var cumsum_gpu_1 = require("./webgl/cumsum_gpu");
var depth_to_space_gpu_1 = require("./webgl/depth_to_space_gpu");
var encode_float_gpu_1 = require("./webgl/encode_float_gpu");
var fft_gpu = require("./webgl/fft_gpu");
var fft_gpu_1 = require("./webgl/fft_gpu");
var from_pixels_gpu_1 = require("./webgl/from_pixels_gpu");
var gather_gpu_1 = require("./webgl/gather_gpu");
var gather_nd_gpu_1 = require("./webgl/gather_nd_gpu");
var gpgpu_context_1 = require("./webgl/gpgpu_context");
var gpgpu_math = require("./webgl/gpgpu_math");
var im2col_gpu_1 = require("./webgl/im2col_gpu");
var lrn_gpu_1 = require("./webgl/lrn_gpu");
var lrn_grad_gpu_1 = require("./webgl/lrn_grad_gpu");
var max_pool_backprop_gpu_1 = require("./webgl/max_pool_backprop_gpu");
var mulmat_gpu_1 = require("./webgl/mulmat_gpu");
var mulmat_packed_gpu_1 = require("./webgl/mulmat_packed_gpu");
var multinomial_gpu_1 = require("./webgl/multinomial_gpu");
var onehot_gpu_1 = require("./webgl/onehot_gpu");
var pack_gpu_1 = require("./webgl/pack_gpu");
var pad_gpu_1 = require("./webgl/pad_gpu");
var pool_gpu_1 = require("./webgl/pool_gpu");
var reduce_gpu_1 = require("./webgl/reduce_gpu");
var reshape_packed_gpu_1 = require("./webgl/reshape_packed_gpu");
var resize_bilinear_backprop_gpu_1 = require("./webgl/resize_bilinear_backprop_gpu");
var resize_bilinear_gpu_1 = require("./webgl/resize_bilinear_gpu");
var resize_nearest_neighbor_backprop_gpu_1 = require("./webgl/resize_nearest_neighbor_backprop_gpu");
var resize_nearest_neighbor_gpu_1 = require("./webgl/resize_nearest_neighbor_gpu");
var reverse_gpu_1 = require("./webgl/reverse_gpu");
var scatter_gpu_1 = require("./webgl/scatter_gpu");
var segment_gpu_1 = require("./webgl/segment_gpu");
var select_gpu_1 = require("./webgl/select_gpu");
var slice_gpu_1 = require("./webgl/slice_gpu");
var strided_slice_gpu_1 = require("./webgl/strided_slice_gpu");
var tex_util_1 = require("./webgl/tex_util");
var texture_manager_1 = require("./webgl/texture_manager");
var tile_gpu_1 = require("./webgl/tile_gpu");
var transpose_gpu_1 = require("./webgl/transpose_gpu");
var unary_op = require("./webgl/unaryop_gpu");
var unaryop_gpu_1 = require("./webgl/unaryop_gpu");
var unpack_gpu_1 = require("./webgl/unpack_gpu");
var webgl_util = require("./webgl/webgl_util");
var where_impl_1 = require("./where_impl");
var CPU_HANDOFF_SIZE_THRESHOLD = 10;
var BEFORE_PAGING_CONSTANT = 300;
exports.SIZE_UPLOAD_UNIFORM = 4;
var MATMUL_SHARED_DIM_THRESHOLD = 1000;
var MathBackendWebGL = (function () {
    function MathBackendWebGL(gpgpu, delayedStorage) {
        if (delayedStorage === void 0) { delayedStorage = true; }
        this.gpgpu = gpgpu;
        this.delayedStorage = delayedStorage;
        this.pendingRead = new WeakMap();
        this.pendingDisposal = new WeakSet();
        this.lruDataGPU = [];
        this.numBytesInGPU = 0;
        this.uploadWaitMs = 0;
        this.downloadWaitMs = 0;
        this.binaryCache = {};
        this.disposed = false;
        if (environment_1.ENV.get('WEBGL_VERSION') < 1) {
            throw new Error('WebGL is not supported on this device');
        }
        if (gpgpu == null) {
            var gl = canvas_util_1.getWebGLContext(environment_1.ENV.get('WEBGL_VERSION'));
            this.gpgpu = new gpgpu_context_1.GPGPUContext(gl);
            this.canvas = gl.canvas;
            this.gpgpuCreatedLocally = true;
        }
        else {
            this.gpgpuCreatedLocally = false;
            this.canvas = gpgpu.gl.canvas;
        }
        if (environment_1.ENV.get('WEBGL_PAGING_ENABLED')) {
            this.NUM_BYTES_BEFORE_PAGING =
                (window.screen.height * window.screen.width *
                    window.devicePixelRatio) *
                    BEFORE_PAGING_CONSTANT;
        }
        this.textureManager = new texture_manager_1.TextureManager(this.gpgpu);
    }
    MathBackendWebGL.prototype.register = function (dataId, shape, dtype) {
        if (this.texData.has(dataId)) {
            throw new Error('Data buffer is already registered');
        }
        this.texData.set(dataId, {
            shape: shape,
            dtype: dtype,
            values: null,
            texture: null,
            complexTensors: null,
            texShape: null,
            usage: tex_util_1.TextureUsage.RENDER,
            isPacked: false
        });
    };
    MathBackendWebGL.prototype.setDataMover = function (dataMover) {
        this.texData = new backend_1.DataStorage(dataMover);
    };
    MathBackendWebGL.prototype.fromPixels = function (pixels, numChannels) {
        if (pixels == null) {
            throw new Error('pixels passed to tf.fromPixels() can not be null');
        }
        var texShape = [pixels.height, pixels.width];
        var outShape = [pixels.height, pixels.width, numChannels];
        if (!(pixels instanceof HTMLVideoElement) &&
            !(pixels instanceof HTMLImageElement) &&
            !(pixels instanceof HTMLCanvasElement) &&
            !(pixels instanceof ImageData)) {
            throw new Error('pixels passed to tf.fromPixels() must be either an ' +
                "HTMLVideoElement, HTMLImageElement, HTMLCanvasElement or " +
                ("ImageData, but was " + pixels.constructor.name));
        }
        if (pixels instanceof HTMLVideoElement) {
            if (this.fromPixels2DContext == null) {
                if (!environment_1.ENV.get('IS_BROWSER')) {
                    throw new Error('Can\'t read pixels from HTMLImageElement outside the browser.');
                }
                if (document.readyState !== 'complete') {
                    throw new Error('The DOM is not ready yet. Please call tf.fromPixels() ' +
                        'once the DOM is ready. One way to do that is to add an event ' +
                        'listener for `DOMContentLoaded` on the document object');
                }
                this.fromPixels2DContext =
                    document.createElement('canvas').getContext('2d');
            }
            this.fromPixels2DContext.canvas.width = pixels.width;
            this.fromPixels2DContext.canvas.height = pixels.height;
            this.fromPixels2DContext.drawImage(pixels, 0, 0, pixels.width, pixels.height);
            pixels = this.fromPixels2DContext.canvas;
        }
        var tempPixelHandle = this.makeTensorHandle(texShape, 'int32');
        this.texData.get(tempPixelHandle.dataId).usage = tex_util_1.TextureUsage.PIXELS;
        this.gpgpu.uploadPixelDataToTexture(this.getTexture(tempPixelHandle.dataId), pixels);
        var program = new from_pixels_gpu_1.FromPixelsProgram(outShape);
        var res = this.compileAndRun(program, [tempPixelHandle]);
        this.disposeData(tempPixelHandle.dataId);
        return res;
    };
    MathBackendWebGL.prototype.makeTensorHandle = function (shape, dtype) {
        var dataId = {};
        this.register(dataId, shape, dtype);
        return { dataId: dataId, shape: shape, dtype: dtype };
    };
    MathBackendWebGL.prototype.write = function (dataId, values) {
        if (values == null) {
            throw new Error('MathBackendWebGL.write(): values can not be null');
        }
        var texData = this.texData.get(dataId);
        var texture = texData.texture, texShape = texData.texShape, usage = texData.usage, dtype = texData.dtype, isPacked = texData.isPacked;
        if (dtype === 'complex64') {
            throw new Error("Cannot write to a complex64 dtype. " +
                "Please use tf.complex(real, imag).");
        }
        if (texture != null) {
            this.releaseTexture(dataId, texture, texShape, usage, isPacked);
            texData.texture = null;
            texData.texShape = null;
        }
        texData.usage = tex_util_1.TextureUsage.UPLOAD;
        texData.values = values;
        if (!this.delayedStorage) {
            this.uploadToGPU(dataId);
        }
    };
    MathBackendWebGL.prototype.readSync = function (dataId) {
        var texData = this.texData.get(dataId);
        var values = texData.values, dtype = texData.dtype, complexTensors = texData.complexTensors;
        if (values != null) {
            return this.convertAndCacheOnCPU(dataId);
        }
        var shouldTimeProgram = this.activeTimers != null;
        var start;
        if (shouldTimeProgram) {
            start = performance.now();
        }
        var result;
        if (dtype === 'complex64') {
            var realValues = complexTensors.real.dataSync();
            var imagValues = complexTensors.imag.dataSync();
            result = complex_util_1.mergeRealAndImagArrays(realValues, imagValues);
        }
        else {
            result = this.getValuesFromTexture(dataId);
        }
        if (shouldTimeProgram) {
            this.downloadWaitMs += performance.now() - start;
        }
        return this.convertAndCacheOnCPU(dataId, result);
    };
    MathBackendWebGL.prototype.read = function (dataId) {
        return __awaiter(this, void 0, void 0, function () {
            var subscribers_1, texData, texture, values, texShape, bufferOrTexture, vals, dTypeVals, subscribers;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (this.pendingRead.has(dataId)) {
                            subscribers_1 = this.pendingRead.get(dataId);
                            return [2, new Promise(function (resolve) { return subscribers_1.push(resolve); })];
                        }
                        texData = this.texData.get(dataId);
                        texture = texData.texture, values = texData.values, texShape = texData.texShape;
                        if (values != null) {
                            return [2, this.convertAndCacheOnCPU(dataId)];
                        }
                        this.pendingRead.set(dataId, []);
                        if (!environment_1.ENV.get('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
                            environment_1.ENV.get('WEBGL_VERSION') === 2) {
                            throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and " +
                                "WEBGL_VERSION=2 not yet supported.");
                        }
                        bufferOrTexture = this.gpgpu.maybeCreateBufferFromTexture(texture, texShape[0], texShape[1]);
                        return [4, this.gpgpu.createAndWaitForFence()];
                    case 1:
                        _a.sent();
                        if (bufferOrTexture instanceof WebGLTexture) {
                            vals = this.getValuesFromTexture(dataId);
                        }
                        else {
                            vals = this.gpgpu.downloadFloat32MatrixFromBuffer(bufferOrTexture, texShape[0], texShape[1]);
                        }
                        dTypeVals = this.convertAndCacheOnCPU(dataId, vals);
                        subscribers = this.pendingRead.get(dataId);
                        this.pendingRead.delete(dataId);
                        subscribers.forEach(function (resolve) { return resolve(dTypeVals); });
                        if (this.pendingDisposal.has(dataId)) {
                            this.pendingDisposal.delete(dataId);
                            this.disposeData(dataId);
                        }
                        return [2, dTypeVals];
                }
            });
        });
    };
    MathBackendWebGL.prototype.getValuesFromTexture = function (dataId) {
        var _a = this.texData.get(dataId), shape = _a.shape, dtype = _a.dtype, texture = _a.texture, texShape = _a.texShape;
        if (environment_1.ENV.get('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
            if (this.texData.get(dataId).isPacked) {
                var batch = util.sizeFromShape(shape.slice(0, shape.length - 2));
                var rows = shape.length > 1 ? shape[shape.length - 2] : 1;
                var cols = shape[shape.length - 1];
                return this.gpgpu.downloadMatrixFromPackedTexture(texture, batch, rows, cols, texShape[0], texShape[1]);
            }
            else {
                return this.gpgpu.downloadFloat32MatrixFromOutputTexture(texture, texShape[0], texShape[1]);
            }
        }
        var tmpTarget = this.makeTensorHandle(shape, 'float32');
        tmpTarget.size = util_1.sizeFromShape(shape);
        this.texData.get(tmpTarget.dataId).usage = tex_util_1.TextureUsage.DOWNLOAD;
        var program = new encode_float_gpu_1.EncodeFloatProgram(shape);
        var pageToCpu = false;
        this.compileAndRun(program, [{ shape: shape, dtype: dtype, dataId: dataId }], tmpTarget, null, pageToCpu);
        var tmpData = this.texData.get(tmpTarget.dataId);
        var vals = this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(tmpData.texture, tmpData.texShape[0], tmpData.texShape[1]);
        this.disposeData(tmpTarget.dataId);
        return vals;
    };
    MathBackendWebGL.prototype.time = function (f) {
        return __awaiter(this, void 0, void 0, function () {
            var oldActiveTimers, newActiveTimers, outerMostTime, flattenedActiveTimerQueries, flattenedActiveTimerNames, kernelMs, res;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        oldActiveTimers = this.activeTimers;
                        newActiveTimers = [];
                        outerMostTime = false;
                        if (this.programTimersStack == null) {
                            this.programTimersStack = newActiveTimers;
                            outerMostTime = true;
                        }
                        else {
                            this.activeTimers.push(newActiveTimers);
                        }
                        this.activeTimers = newActiveTimers;
                        f();
                        flattenedActiveTimerQueries = util.flatten(this.activeTimers.map(function (d) { return d.query; }))
                            .filter(function (d) { return d != null; });
                        flattenedActiveTimerNames = util.flatten(this.activeTimers.map(function (d) { return d.name; }))
                            .filter(function (d) { return d != null; });
                        this.activeTimers = oldActiveTimers;
                        if (outerMostTime) {
                            this.programTimersStack = null;
                        }
                        return [4, Promise.all(flattenedActiveTimerQueries)];
                    case 1:
                        kernelMs = _a.sent();
                        res = {
                            uploadWaitMs: this.uploadWaitMs,
                            downloadWaitMs: this.downloadWaitMs,
                            kernelMs: util.sum(kernelMs),
                            getExtraProfileInfo: function () {
                                return kernelMs.map(function (d, i) { return ({ name: flattenedActiveTimerNames[i], ms: d }); })
                                    .map(function (d) { return d.name + ": " + d.ms; })
                                    .join(', ');
                            },
                            wallMs: null
                        };
                        this.uploadWaitMs = 0;
                        this.downloadWaitMs = 0;
                        return [2, res];
                }
            });
        });
    };
    MathBackendWebGL.prototype.memory = function () {
        return { unreliable: false, numBytesInGPU: this.numBytesInGPU };
    };
    MathBackendWebGL.prototype.startTimer = function () {
        if (environment_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
            return this.gpgpu.beginQuery();
        }
        return { startMs: performance.now(), endMs: null };
    };
    MathBackendWebGL.prototype.endTimer = function (query) {
        if (environment_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
            this.gpgpu.endQuery();
            return query;
        }
        query.endMs = performance.now();
        return query;
    };
    MathBackendWebGL.prototype.getQueryTime = function (query) {
        return __awaiter(this, void 0, void 0, function () {
            var timerQuery;
            return __generator(this, function (_a) {
                if (environment_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
                    return [2, this.gpgpu.waitForQueryAndGetTime(query)];
                }
                timerQuery = query;
                return [2, timerQuery.endMs - timerQuery.startMs];
            });
        });
    };
    MathBackendWebGL.prototype.disposeData = function (dataId) {
        if (this.pendingDisposal.has(dataId)) {
            return;
        }
        if (this.pendingRead.has(dataId)) {
            this.pendingDisposal.add(dataId);
            return;
        }
        if (this.texData.has(dataId)) {
            var _a = this.texData.get(dataId), texture = _a.texture, texShape = _a.texShape, usage = _a.usage, complexTensors = _a.complexTensors, isPacked = _a.isPacked;
            if (texture != null) {
                this.releaseTexture(dataId, texture, texShape, usage, isPacked);
            }
            if (complexTensors != null) {
                complexTensors.real.dispose();
                complexTensors.imag.dispose();
            }
            this.texData.delete(dataId);
        }
    };
    MathBackendWebGL.prototype.getTexture = function (dataId) {
        this.uploadToGPU(dataId);
        return this.texData.get(dataId).texture;
    };
    MathBackendWebGL.prototype.getCPUBackend = function () {
        if (!environment_1.ENV.get('WEBGL_CPU_FORWARD')) {
            return null;
        }
        if (this.cpuBackend == null) {
            this.cpuBackend = environment_1.ENV.findBackend('cpu');
        }
        return this.cpuBackend;
    };
    MathBackendWebGL.prototype.shouldExecuteOnCPU = function (inputs, sizeThreshold) {
        var _this = this;
        if (sizeThreshold === void 0) { sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD; }
        return this.getCPUBackend() != null &&
            inputs.every(function (input) { return _this.texData.get(input.dataId).texture == null &&
                input.size < sizeThreshold; });
    };
    MathBackendWebGL.prototype.getGPGPUContext = function () {
        return this.gpgpu;
    };
    MathBackendWebGL.prototype.getCanvas = function () {
        return this.canvas;
    };
    MathBackendWebGL.prototype.complex = function (real, imag) {
        var result = this.makeOutputArray(real.shape, 'complex64');
        var resultData = this.texData.get(result.dataId);
        resultData.complexTensors = {
            real: environment_1.ENV.engine.keep(real.clone()),
            imag: environment_1.ENV.engine.keep(imag.clone())
        };
        return result;
    };
    MathBackendWebGL.prototype.real = function (input) {
        var resultData = this.texData.get(input.dataId);
        return resultData.complexTensors.real.clone();
    };
    MathBackendWebGL.prototype.imag = function (input) {
        var resultData = this.texData.get(input.dataId);
        return resultData.complexTensors.imag.clone();
    };
    MathBackendWebGL.prototype.slice = function (x, begin, size) {
        if (this.shouldExecuteOnCPU([x])) {
            return this.cpuBackend.slice(x, begin, size);
        }
        var program = new slice_gpu_1.SliceProgram(size);
        var customSetup = program.getCustomSetupFunc(begin);
        return this.compileAndRun(program, [x], null, customSetup);
    };
    MathBackendWebGL.prototype.stridedSlice = function (x, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask) {
        if (this.shouldExecuteOnCPU([x])) {
            return this.cpuBackend.stridedSlice(x, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
        }
        var _a = slice_util_1.getStridedSlicedInfo(x.shape, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask), beginIndex = _a[0], size = _a[1], shrinkAxis = _a[2];
        var shape = size.filter(function (v, index) { return shrinkAxis.indexOf(index) === -1; });
        if (shape.some(function (axis) { return axis === 0; })) {
            return tensor_ops_1.tensor([], shape);
        }
        var program = new strided_slice_gpu_1.StridedSliceProgram(beginIndex, strides, size, shrinkAxis);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.reverse = function (x, axis) {
        var program = new reverse_gpu_1.ReverseProgram(x.shape, axis);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.concat2Tensors = function (a, b, axis) {
        var outShape = concat_util_1.computeOutShape([a.shape, b.shape], axis);
        var a2D = a.as2D(-1, util_1.sizeFromShape(a.shape.slice(axis)));
        var b2D = b.as2D(-1, util_1.sizeFromShape(b.shape.slice(axis)));
        var program = new concat_gpu_1.ConcatProgram(a2D.shape, b2D.shape);
        var res = this.compileAndRun(program, [a2D, b2D]);
        return res.reshape(outShape);
    };
    MathBackendWebGL.prototype.concat = function (tensors, axis) {
        if (this.shouldExecuteOnCPU(tensors)) {
            return this.cpuBackend.concat(tensors, axis);
        }
        if (tensors.length === 1) {
            return tensors[0];
        }
        var result = tensors[0];
        for (var i = 1; i < tensors.length; ++i) {
            result = this.concat2Tensors(result, tensors[i], axis);
        }
        return result;
    };
    MathBackendWebGL.prototype.neg = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.NEG);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.batchMatMul = function (a, b, transposeA, transposeB) {
        var outerShapeA = transposeA ? a.shape[2] : a.shape[1];
        var outerShapeB = transposeB ? b.shape[1] : b.shape[2];
        var _a = a.shape, batch = _a[0], firstDim = _a[1], sharedDim = _a[2];
        var _b = b.shape, secondDim = _b[2];
        if ((firstDim === 1 || secondDim === 1) &&
            sharedDim > MATMUL_SHARED_DIM_THRESHOLD) {
            var a3D = secondDim === 1 ? a : a.as3D(batch, sharedDim, 1);
            var axis = secondDim === 1 ? 2 : 1;
            var b3D = secondDim === 1 ? b.as3D(batch, 1, sharedDim) : b;
            return this.multiply(a3D, b3D).sum(axis, true);
        }
        if (batch === 1) {
            var aSqueezed = a.as2D(a.shape[1], a.shape[2]);
            var bSqueezed = b.as2D(b.shape[1], b.shape[2]);
            var program = new mulmat_packed_gpu_1.MatMulPackedProgram(aSqueezed.shape, bSqueezed.shape, [outerShapeA, outerShapeB], transposeA, transposeB);
            var result = this.compileAndRun(program, [aSqueezed, bSqueezed], this.makePackedTensor(program.outputShape));
            if (environment_1.ENV.get('WEBGL_LAZILY_UNPACK') === false) {
                result = this.unpackTensor(result);
            }
            return result.reshape([1, result.shape[0], result.shape[1]]);
        }
        else {
            return this.compileAndRun(new mulmat_gpu_1.MatMulProgram(a.shape, b.shape, transposeA, transposeB), [a, b]);
        }
    };
    MathBackendWebGL.prototype.multiply = function (a, b) {
        if (a.dtype === 'complex64') {
            var aData = this.texData.get(a.dataId);
            var bData = this.texData.get(b.dataId);
            var realProgram = new binaryop_complex_gpu_1.BinaryOpComplexProgram(binaryop_complex_gpu.COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
            var imagProgram = new binaryop_complex_gpu_1.BinaryOpComplexProgram(binaryop_complex_gpu.COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);
            var inputs = [
                this.makeComplexComponentTensorHandle(a, aData.complexTensors.real),
                this.makeComplexComponentTensorHandle(a, aData.complexTensors.imag),
                this.makeComplexComponentTensorHandle(b, bData.complexTensors.real),
                this.makeComplexComponentTensorHandle(b, bData.complexTensors.imag)
            ];
            var real = this.compileAndRun(realProgram, inputs);
            var imag = this.compileAndRun(imagProgram, inputs);
            var complex = this.complex(real, imag);
            real.dispose();
            imag.dispose();
            return complex;
        }
        if (this.shouldExecuteOnCPU([a, b])) {
            return this.cpuBackend.multiply(a, b);
        }
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.MUL, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, a.dtype);
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.batchNormalization = function (x, mean, variance, varianceEpsilon, scale, offset) {
        var inputs = [x, mean, variance];
        var offsetShape = null;
        if (offset != null) {
            offsetShape = offset.shape;
            inputs.push(offset);
        }
        var scaleShape = null;
        if (scale != null) {
            scaleShape = scale.shape;
            inputs.push(scale);
        }
        var output = null;
        var envSpecificBatchNormProgram = batchnorm_gpu_1.BatchNormProgram;
        if (environment_1.ENV.get('WEBGL_PACK_BATCHNORMALIZATION')) {
            output = this.makePackedTensor(x.shape);
            envSpecificBatchNormProgram = batchnorm_packed_gpu_1.BatchNormPackedProgram;
        }
        var program = new envSpecificBatchNormProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape, varianceEpsilon);
        return this.compileAndRun(program, inputs, output);
    };
    MathBackendWebGL.prototype.localResponseNormalization4D = function (x, radius, bias, alpha, beta) {
        var program = new lrn_gpu_1.LRNProgram(x.shape, radius, bias, alpha, beta);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.LRNGrad = function (dy, inputImage, outputImage, depthRadius, bias, alpha, beta) {
        var program = new lrn_grad_gpu_1.LRNGradProgram(inputImage.shape, depthRadius, bias, alpha, beta);
        return this.compileAndRun(program, [inputImage, outputImage, dy]);
    };
    MathBackendWebGL.prototype.tile = function (x, reps) {
        var program = new tile_gpu_1.TileProgram(x.shape, reps);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.pad = function (x, paddings, constantValue) {
        var program = new pad_gpu_1.PadProgram(x.shape, paddings, constantValue);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.transpose = function (x, perm) {
        var program = new transpose_gpu_1.TransposeProgram(x.shape, perm);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.gather = function (x, indices, axis) {
        var program = new gather_gpu_1.GatherProgram(x.shape, indices.size, axis);
        return this.compileAndRun(program, [x, indices]);
    };
    MathBackendWebGL.prototype.batchToSpaceND = function (x, blockShape, crops) {
        util.assert(x.rank <= 4, 'batchToSpaceND for rank > 4 with a WebGL backend not implemented yet');
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
    MathBackendWebGL.prototype.spaceToBatchND = function (x, blockShape, paddings) {
        util.assert(x.rank <= 4, 'spaceToBatchND for rank > 4 with a WebGL backend not implemented yet');
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
    MathBackendWebGL.prototype.reduce = function (x, reduceType, dtype) {
        var batchSize = x.shape[0];
        var inSize = x.shape[1];
        var windowSize = reduce_util.computeOptimalWindowSize(inSize);
        var reduceInfo = { windowSize: windowSize, inSize: inSize, batchSize: batchSize };
        var program = new reduce_gpu_1.ReduceProgram(reduceInfo, reduceType);
        var _a = program.outputShape, rows = _a[0], cols = _a[1];
        var output = this.makeOutputArray([rows, cols], dtype);
        this.compileAndRun(program, [x], output);
        if (output.shape[1] === 1) {
            return output;
        }
        return this.reduce(output, reduceType, dtype);
    };
    MathBackendWebGL.prototype.argReduce = function (x, reduceType, bestIndicesA) {
        if (bestIndicesA === void 0) { bestIndicesA = null; }
        var batchSize = x.shape[0];
        var inSize = x.shape[1];
        if (bestIndicesA != null) {
            batchSize = bestIndicesA.shape[0];
            inSize = bestIndicesA.shape[1];
        }
        var windowSize = reduce_util.computeOptimalWindowSize(inSize);
        var reduceInfo = { windowSize: windowSize, inSize: inSize, batchSize: batchSize };
        var program = new argminmax_gpu_1.ArgMinMaxProgram(reduceInfo, reduceType, bestIndicesA == null);
        var _a = program.outputShape, rows = _a[0], cols = _a[1];
        var output = this.makeOutputArray([rows, cols], 'int32');
        var inputs = [x];
        if (bestIndicesA != null) {
            inputs.push(bestIndicesA);
        }
        this.compileAndRun(program, inputs, output);
        if (output.shape[1] === 1) {
            return output;
        }
        return this.argReduce(x, reduceType, output);
    };
    MathBackendWebGL.prototype.sum = function (x, axes) {
        axis_util.assertAxesAreInnerMostDims('sum', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        var outputDType = types_1.sumOutType(x.dtype);
        return this.reduce(a2D, 'sum', outputDType).reshape(outShape);
    };
    MathBackendWebGL.prototype.prod = function (x, axes) {
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        var outputDType = types_1.sumOutType(x.dtype);
        return this.reduce(a2D, 'prod', outputDType).reshape(outShape);
    };
    MathBackendWebGL.prototype.unsortedSegmentSum = function (x, segmentIds, numSegments) {
        var axis = 0;
        var permutation = axis_util.getAxesPermutation([axis], x.rank);
        var permutedX = x;
        if (permutation != null) {
            permutedX = x.transpose(permutation);
            axis = axis_util.getInnerMostAxes(1, x.rank)[0];
        }
        var outShape = segment_util.computeOutShape(permutedX.shape, axis, numSegments);
        var inSize = util.sizeFromShape([permutedX.shape[axis]]);
        var a2D = permutedX.as2D(-1, inSize);
        var outputDType = types_1.sumOutType(x.dtype);
        var result = this.segOpCompute(a2D, 'unsortedSegmentSum', segmentIds, outputDType, numSegments)
            .reshape(outShape);
        if (permutation != null) {
            result = result.transpose(axis_util.getUndoAxesPermutation(permutation));
        }
        return result;
    };
    MathBackendWebGL.prototype.segOpCompute = function (x, segOpType, segmentIds, dtype, numSegments) {
        var batchSize = x.shape[0];
        var inSize = x.shape[1];
        var windowSize = segment_util.segOpComputeOptimalWindowSize(inSize, numSegments);
        var segOpInfo = { windowSize: windowSize, inSize: inSize, batchSize: batchSize, numSegments: numSegments };
        var program = new segment_gpu_1.SegmentOpProgram(segOpInfo, segOpType);
        var _a = program.outputShape, rows = _a[0], cols = _a[1];
        var output = this.makeOutputArray([rows, cols], dtype);
        this.compileAndRun(program, [x, segmentIds], output);
        if (output.shape[1] === numSegments) {
            return output;
        }
        segmentIds = tensor_ops_1.range(0, numSegments).tile([inSize / windowSize]);
        return this.segOpCompute(output, segOpType, segmentIds, dtype, numSegments);
    };
    MathBackendWebGL.prototype.argMin = function (x, axis) {
        var axes = [axis];
        axis_util.assertAxesAreInnerMostDims('argMin', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        return this.argReduce(a2D, 'min').reshape(outShape);
    };
    MathBackendWebGL.prototype.argMax = function (x, axis) {
        var axes = [axis];
        axis_util.assertAxesAreInnerMostDims('argMax', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        return this.argReduce(a2D, 'max').reshape(outShape);
    };
    MathBackendWebGL.prototype.cumsum = function (x, axis, exclusive, reverse) {
        if (axis !== x.rank - 1) {
            throw new Error("WebGL cumsum shader expects an inner-most axis=" + (x.rank - 1) + " " +
                ("but got axis=" + axis));
        }
        var program = new cumsum_gpu_1.CumSumProgram(x.shape, exclusive, reverse);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.equal = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.EQUAL, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.notEqual = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.NOT_EQUAL, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.less = function (a, b) {
        if (this.shouldExecuteOnCPU([a, b])) {
            return this.cpuBackend.less(a, b);
        }
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.LESS, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.lessEqual = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.LESS_EQUAL, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.greater = function (a, b) {
        if (this.shouldExecuteOnCPU([a, b])) {
            return this.cpuBackend.greater(a, b);
        }
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.GREATER, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.greaterEqual = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.GREATER_EQUAL, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.logicalNot = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.LOGICAL_NOT);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.logicalAnd = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.LOGICAL_AND, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.logicalOr = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.LOGICAL_OR, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, 'bool');
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.select = function (condition, a, b) {
        var program = new select_gpu_1.SelectProgram(condition.rank, a.shape, a.rank);
        var output = this.makeOutputArray(program.outputShape, types_1.upcastType(a.dtype, b.dtype));
        return this.compileAndRun(program, [condition, a, b], output);
    };
    MathBackendWebGL.prototype.where = function (condition) {
        log_1.warn('tf.where() in webgl locks the UI thread. ' +
            'Call tf.whereAsync() instead');
        var condVals = condition.dataSync();
        return where_impl_1.whereImpl(condition.shape, condVals);
    };
    MathBackendWebGL.prototype.topk = function (x, k, sorted) {
        var xVals = x.dataSync();
        return topk_impl_1.topkImpl(xVals, x.shape, x.dtype, k, sorted);
    };
    MathBackendWebGL.prototype.min = function (x, axes) {
        axis_util.assertAxesAreInnerMostDims('min', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        return this.reduce(a2D, 'min', a2D.dtype).reshape(outShape);
    };
    MathBackendWebGL.prototype.minimum = function (a, b) {
        if (this.shouldExecuteOnCPU([a, b])) {
            return this.cpuBackend.minimum(a, b);
        }
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.MIN, a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    MathBackendWebGL.prototype.mod = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.MOD, a.shape, b.shape);
        var customSetup = program.getCustomSetupFunc();
        return this.compileAndRun(program, [a, b], null, customSetup);
    };
    MathBackendWebGL.prototype.max = function (x, axes) {
        axis_util.assertAxesAreInnerMostDims('max', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        return this.reduce(a2D, 'max', a2D.dtype).reshape(outShape);
    };
    MathBackendWebGL.prototype.maximum = function (a, b) {
        if (this.shouldExecuteOnCPU([a, b])) {
            return this.cpuBackend.maximum(a, b);
        }
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.MAX, a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    MathBackendWebGL.prototype.all = function (x, axes) {
        axis_util.assertAxesAreInnerMostDims('all', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        return this.reduce(a2D, 'all', a2D.dtype).reshape(outShape);
    };
    MathBackendWebGL.prototype.any = function (x, axes) {
        axis_util.assertAxesAreInnerMostDims('any', axes, x.rank);
        var _a = axis_util.computeOutAndReduceShapes(x.shape, axes), outShape = _a[0], reduceShape = _a[1];
        var inSize = util.sizeFromShape(reduceShape);
        var a2D = x.as2D(-1, inSize);
        return this.reduce(a2D, 'any', a2D.dtype).reshape(outShape);
    };
    MathBackendWebGL.prototype.squaredDifference = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.SQUARED_DIFFERENCE, a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    MathBackendWebGL.prototype.realDivide = function (a, b) {
        var op = binaryop_gpu.DIV;
        var outputDtype = 'float32';
        var program = new binaryop_gpu_1.BinaryOpProgram(op, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, outputDtype);
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.floorDiv = function (a, b) {
        var op = binaryop_gpu.INT_DIV;
        var outputDtype = 'int32';
        var program = new binaryop_gpu_1.BinaryOpProgram(op, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, outputDtype);
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.add = function (a, b) {
        if (a.dtype === 'complex64' && b.dtype === 'complex64') {
            return this.complexSeparableBinaryOp(a, b, binaryop_gpu.ADD);
        }
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.ADD, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, types_1.upcastType(a.dtype, b.dtype));
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.complexSeparableBinaryOp = function (a, b, op) {
        var _this = this;
        var aData = this.texData.get(a.dataId);
        var bData = this.texData.get(b.dataId);
        var _a = [
            [aData.complexTensors.real, bData.complexTensors.real],
            [aData.complexTensors.imag, bData.complexTensors.imag]
        ].map(function (complexParts) {
            var aPart = complexParts[0], bPart = complexParts[1];
            var program = new binaryop_gpu_1.BinaryOpProgram(op, a.shape, b.shape);
            var output = _this.makeOutputArray(program.outputShape, types_1.upcastType(aPart.dtype, bPart.dtype));
            var aHandle = _this.makeComplexComponentTensorHandle(a, aPart);
            var bHandle = _this.makeComplexComponentTensorHandle(b, bPart);
            return _this.compileAndRun(program, [aHandle, bHandle], output);
        }), real = _a[0], imag = _a[1];
        var complex = this.complex(real, imag);
        real.dispose();
        imag.dispose();
        return complex;
    };
    MathBackendWebGL.prototype.makeComplexComponentTensorHandle = function (complexTensor, complexPart) {
        return {
            dataId: complexPart.dataId,
            dtype: complexPart.dtype,
            shape: complexTensor.shape
        };
    };
    MathBackendWebGL.prototype.addN = function (tensors) {
        var res = tensors[0];
        for (var i = 1; i < tensors.length; i++) {
            res = this.add(res, tensors[i]);
        }
        return res;
    };
    MathBackendWebGL.prototype.subtract = function (a, b) {
        if (a.dtype === 'complex64' && b.dtype === 'complex64') {
            return this.complexSeparableBinaryOp(a, b, binaryop_gpu.SUB);
        }
        if (this.shouldExecuteOnCPU([a, b])) {
            return this.cpuBackend.subtract(a, b);
        }
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.SUB, a.shape, b.shape);
        var output = this.makeOutputArray(program.outputShape, types_1.upcastType(a.dtype, b.dtype));
        return this.compileAndRun(program, [a, b], output);
    };
    MathBackendWebGL.prototype.pow = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.POW, a.shape, b.shape);
        var customSetup = program.getCustomSetupFunc();
        var output = this.makeOutputArray(program.outputShape, types_1.upcastType(a.dtype, b.dtype));
        return this.compileAndRun(program, [a, b], output, customSetup);
    };
    MathBackendWebGL.prototype.ceil = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.CEIL);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.floor = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.FLOOR);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.sign = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SIGN);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.round = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ROUND);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.exp = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.EXP);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.expm1 = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.EXPM1);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.log = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.LOG);
        var customSetup = program.getCustomSetupFunc();
        return this.compileAndRun(program, [x], null, customSetup);
    };
    MathBackendWebGL.prototype.log1p = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.LOG1P);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.sqrt = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SQRT);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.rsqrt = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.RSQRT);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.square = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SQUARE);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.reciprocal = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.RECIPROCAL);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.relu = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.RELU);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.elu = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ELU);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.eluDer = function (dy, y) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.ELU_DER, dy.shape, y.shape);
        return this.compileAndRun(program, [dy, y]);
    };
    MathBackendWebGL.prototype.selu = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SELU);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.int = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.TO_INT);
        var output = this.makeOutputArray(program.outputShape, 'int32');
        return this.compileAndRun(program, [x], output);
    };
    MathBackendWebGL.prototype.clip = function (x, min, max) {
        var program = new clip_gpu_1.ClipProgram(x.shape, min, max);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.abs = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ABS);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.complexAbs = function (x) {
        var xData = this.texData.get(x.dataId);
        var program = new complex_abs_gpu_1.ComplexAbsProgram(x.shape);
        var inputs = [
            this.makeComplexComponentTensorHandle(x, xData.complexTensors.real),
            this.makeComplexComponentTensorHandle(x, xData.complexTensors.imag),
        ];
        return this.compileAndRun(program, inputs);
    };
    MathBackendWebGL.prototype.sigmoid = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SIGMOID);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.softplus = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SOFTPLUS);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.sin = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SIN);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.cos = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.COS);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.tan = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.TAN);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.asin = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ASIN);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.acos = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ACOS);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.atan = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ATAN);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.atan2 = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram(binaryop_gpu.ATAN2, a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    MathBackendWebGL.prototype.sinh = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.SINH);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.cosh = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.COSH);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.tanh = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.TANH);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.asinh = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ASINH);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.acosh = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ACOSH);
        var customSetup = program.getCustomSetupFunc();
        return this.compileAndRun(program, [x], null, customSetup);
    };
    MathBackendWebGL.prototype.atanh = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ATANH);
        var customSetup = program.getCustomSetupFunc();
        return this.compileAndRun(program, [x], null, customSetup);
    };
    MathBackendWebGL.prototype.erf = function (x) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.ERF);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.step = function (x, alpha) {
        var program = new unaryop_gpu_1.UnaryOpProgram(x.shape, unary_op.STEP(alpha));
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.conv2dWithIm2Row = function (x, filter, convInfo) {
        var filterWidth = convInfo.filterWidth, filterHeight = convInfo.filterHeight, inChannels = convInfo.inChannels, outWidth = convInfo.outWidth, outHeight = convInfo.outHeight;
        var sharedDim = filterWidth * filterHeight * inChannels;
        var numCols = outHeight * outWidth;
        var x2ColShape = [sharedDim, numCols];
        var xSqueezed = x.squeeze([0]);
        var w2Row = filter.reshape([sharedDim, -1]);
        var im2ColProgram = new im2col_gpu_1.Im2ColProgram(x2ColShape, xSqueezed.shape, convInfo);
        var im2Col = this.compileAndRun(im2ColProgram, [xSqueezed], this.makePackedTensor(x2ColShape));
        var matmulProgram = new mulmat_packed_gpu_1.MatMulPackedProgram(im2Col.shape, w2Row.shape, [numCols, convInfo.outChannels], true, false);
        var product = this.compileAndRun(matmulProgram, [im2Col, w2Row], this.makePackedTensor(matmulProgram.outputShape));
        if (environment_1.ENV.get('WEBGL_LAZILY_UNPACK') === false) {
            product = this.unpackTensor(product);
        }
        return product.reshape([1, outHeight, outWidth, convInfo.outChannels]);
    };
    MathBackendWebGL.prototype.conv2d = function (x, filter, convInfo) {
        if (environment_1.ENV.get('WEBGL_CONV_IM2COL') && x.shape[0] === 1) {
            return this.conv2dWithIm2Row(x, filter, convInfo);
        }
        var program = new conv_gpu_1.Conv2DProgram(convInfo);
        return this.compileAndRun(program, [x, filter]);
    };
    MathBackendWebGL.prototype.conv2dDerInput = function (dy, filter, convInfo) {
        var program = new conv_backprop_gpu_1.Conv2DDerInputProgram(convInfo);
        return this.compileAndRun(program, [dy, filter]);
    };
    MathBackendWebGL.prototype.conv2dDerFilter = function (x, dy, convInfo) {
        var program = new conv_backprop_gpu_1.Conv2DDerFilterProgram(convInfo);
        return this.compileAndRun(program, [x, dy]);
    };
    MathBackendWebGL.prototype.depthwiseConv2D = function (x, filter, convInfo) {
        var program = new conv_gpu_depthwise_1.DepthwiseConv2DProgram(convInfo);
        return this.compileAndRun(program, [x, filter]);
    };
    MathBackendWebGL.prototype.depthwiseConv2DDerInput = function (dy, filter, convInfo) {
        var program = new conv_backprop_gpu_depthwise_1.DepthwiseConv2DDerInputProgram(convInfo);
        return this.compileAndRun(program, [dy, filter]);
    };
    MathBackendWebGL.prototype.depthwiseConv2DDerFilter = function (x, dy, convInfo) {
        var program = new conv_backprop_gpu_depthwise_1.DepthwiseConv2DDerFilterProgram(convInfo);
        return this.compileAndRun(program, [x, dy]);
    };
    MathBackendWebGL.prototype.maxPool = function (x, convInfo) {
        var program = new pool_gpu_1.Pool2DProgram(convInfo, 'max', false);
        var output = this.makeOutputArray(program.outputShape, x.dtype);
        return this.compileAndRun(program, [x], output);
    };
    MathBackendWebGL.prototype.avgPool = function (x, convInfo) {
        var program = new pool_gpu_1.Pool2DProgram(convInfo, 'avg', false);
        var output = this.makeOutputArray(program.outputShape, 'float32');
        return this.compileAndRun(program, [x], output);
    };
    MathBackendWebGL.prototype.maxPoolBackprop = function (dy, x, y, convInfo) {
        var getPositions = true;
        var maxPoolPositionsProgram = new pool_gpu_1.Pool2DProgram(convInfo, 'max', getPositions);
        var maxPoolPositions = this.compileAndRun(maxPoolPositionsProgram, [x]);
        var maxPoolBackPropProgram = new max_pool_backprop_gpu_1.MaxPool2DBackpropProgram(convInfo);
        var output = this.makeOutputArray(maxPoolBackPropProgram.outputShape, x.dtype);
        var result = this.compileAndRun(maxPoolBackPropProgram, [dy, maxPoolPositions], output);
        maxPoolPositions.dispose();
        return result;
    };
    MathBackendWebGL.prototype.avgPoolBackprop = function (dy, x, convInfo) {
        var avgPoolBackpropProgram = new avg_pool_backprop_gpu_1.AvgPool2DBackpropProgram(convInfo);
        var output = this.makeOutputArray(avgPoolBackpropProgram.outputShape, x.dtype);
        return this.compileAndRun(avgPoolBackpropProgram, [dy], output);
    };
    MathBackendWebGL.prototype.cast = function (x, dtype) {
        return backend_util.castTensor(x, dtype, this);
    };
    MathBackendWebGL.prototype.reshape = function (x, shape) {
        if (this.texData.get(x.dataId).isPacked &&
            !webgl_util.isReshapeFree(x.shape, shape)) {
            return this.packedReshape(x, shape);
        }
        return backend_util.reshapeTensor(x, shape);
    };
    MathBackendWebGL.prototype.resizeBilinear = function (x, newHeight, newWidth, alignCorners) {
        var program = new resize_bilinear_gpu_1.ResizeBilinearProgram(x.shape, newHeight, newWidth, alignCorners);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.resizeBilinearBackprop = function (dy, x, alignCorners) {
        var program = new resize_bilinear_backprop_gpu_1.ResizeBilinearBackpropProgram(dy, x, alignCorners);
        return this.compileAndRun(program, [dy]);
    };
    MathBackendWebGL.prototype.resizeNearestNeighbor = function (x, newHeight, newWidth, alignCorners) {
        var program = new resize_nearest_neighbor_gpu_1.ResizeNearestNeighborProgram(x.shape, newHeight, newWidth, alignCorners);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.resizeNearestNeighborBackprop = function (dy, x, alignCorners) {
        var program = new resize_nearest_neighbor_backprop_gpu_1.ResizeNearestNeigborBackpropProgram(dy, x, alignCorners);
        return this.compileAndRun(program, [dy]);
    };
    MathBackendWebGL.prototype.multinomial = function (logits, normalized, numSamples, seed) {
        var probs = normalized ? logits : softmax_1.softmax(logits);
        var batchSize = probs.shape[0];
        var numOutcomes = probs.shape[1];
        var program = new multinomial_gpu_1.MultinomialProgram(batchSize, numOutcomes, numSamples);
        var output = this.makeOutputArray(program.outputShape, 'int32');
        var customSetup = program.getCustomSetupFunc(seed);
        return this.compileAndRun(program, [probs], output, customSetup);
    };
    MathBackendWebGL.prototype.oneHot = function (indices, depth, onValue, offValue) {
        var program = new onehot_gpu_1.OneHotProgram(indices.size, depth, onValue, offValue);
        return this.compileAndRun(program, [indices]);
    };
    MathBackendWebGL.prototype.nonMaxSuppression = function (boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
        log_1.warn('tf.nonMaxSuppression() in webgl locks the UI thread. ' +
            'Call tf.nonMaxSuppressionAsync() instead');
        var boxesVals = boxes.dataSync();
        var scoresVals = scores.dataSync();
        return non_max_suppression_impl_1.nonMaxSuppressionImpl(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
    };
    MathBackendWebGL.prototype.cropAndResize = function (image, boxes, boxIndex, cropSize, method, extrapolationValue) {
        var program = new crop_and_resize_gpu_1.CropAndResizeProgram(image.shape, boxes.shape, cropSize, method, extrapolationValue);
        return this.compileAndRun(program, [image, boxes, boxIndex]);
    };
    MathBackendWebGL.prototype.depthToSpace = function (x, blockSize, dataFormat) {
        util.assert(blockSize > 1, "blockSize should be > 1 for depthToSpace, but was: " + blockSize);
        var batchSize = x.shape[0];
        var inputHeight = (dataFormat === 'NHWC') ? x.shape[1] : x.shape[2];
        var inputWidth = (dataFormat === 'NHWC') ? x.shape[2] : x.shape[3];
        var inputDepth = (dataFormat === 'NHWC') ? x.shape[3] : x.shape[1];
        var outputHeight = inputHeight * blockSize;
        var outputWidth = inputWidth * blockSize;
        var outputDepth = inputDepth / (blockSize * blockSize);
        var outputShape = (dataFormat === 'NHWC') ?
            [batchSize, outputHeight, outputWidth, outputDepth] :
            [batchSize, outputDepth, outputHeight, outputWidth];
        var program = new depth_to_space_gpu_1.DepthToSpaceProgram(outputShape, blockSize, dataFormat);
        return this.compileAndRun(program, [x]);
    };
    MathBackendWebGL.prototype.split = function (x, sizeSplits, axis) {
        return split_shared_1.split(x, sizeSplits, axis);
    };
    MathBackendWebGL.prototype.scatterND = function (indices, updates, shape) {
        var _a = scatter_nd_util.calculateShapes(updates, indices, shape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, sliceSize = _a.sliceSize, strides = _a.strides, outputSize = _a.outputSize;
        var flattenShape = [outputSize / sliceSize, sliceSize];
        var flattenIndices = indices.reshape([numUpdates, sliceRank]);
        var flattenX = updates.reshape([numUpdates, sliceSize]);
        if (outputSize === 0) {
            return backend_util.reshapeTensor(tensor_ops_1.tensor([]), shape);
        }
        var defaultValue = tensor_ops_1.scalar(0);
        var program = new scatter_gpu_1.ScatterProgram(numUpdates, sliceRank, flattenIndices.rank, flattenX.rank, strides, flattenShape);
        return this.compileAndRun(program, [flattenX, flattenIndices, defaultValue])
            .reshape(shape);
    };
    MathBackendWebGL.prototype.sparseToDense = function (sparseIndices, sparseValues, outputShape, defaultValue) {
        var _a = scatter_nd_util.calculateShapes(sparseValues, sparseIndices, outputShape), sliceRank = _a.sliceRank, numUpdates = _a.numUpdates, strides = _a.strides, outputSize = _a.outputSize;
        var sumDupeIndices = false;
        var program = new scatter_gpu_1.ScatterProgram(numUpdates, sliceRank, sparseIndices.rank, sparseValues.rank, strides, [outputSize, 1], sumDupeIndices);
        return this.compileAndRun(program, [sparseValues, sparseIndices, defaultValue])
            .reshape(outputShape);
    };
    MathBackendWebGL.prototype.fft = function (x) {
        var inverse = false;
        return this.fftImpl(x, inverse);
    };
    MathBackendWebGL.prototype.ifft = function (x) {
        var inverse = true;
        return this.fftImpl(x, inverse);
    };
    MathBackendWebGL.prototype.fftImpl = function (x, inverse) {
        var xData = this.texData.get(x.dataId);
        var realProgram = new fft_gpu_1.FFTProgram(fft_gpu.COMPLEX_FFT.REAL, x.shape, inverse);
        var imagProgram = new fft_gpu_1.FFTProgram(fft_gpu.COMPLEX_FFT.IMAG, x.shape, inverse);
        var inputs = [
            this.makeComplexComponentTensorHandle(x, xData.complexTensors.real),
            this.makeComplexComponentTensorHandle(x, xData.complexTensors.imag),
        ];
        var real = this.compileAndRun(realProgram, inputs);
        var imag = this.compileAndRun(imagProgram, inputs);
        var complex = this.complex(real, imag).as2D(x.shape[0], x.shape[1]);
        real.dispose();
        imag.dispose();
        return complex;
    };
    MathBackendWebGL.prototype.gatherND = function (x, indices) {
        var indicesShape = indices.shape;
        var sliceRank = indicesShape[indicesShape.length - 1];
        var _a = gather_nd_util.prepareAndValidate(x, indices), resultShape = _a[0], numSlices = _a[1], sliceSize = _a[2], strides = _a[3];
        var flattenIndices = indices.reshape([numSlices, sliceRank]);
        var flattenX = x.reshape([x.size / sliceSize, sliceSize]);
        var program = new gather_nd_gpu_1.GatherNDProgram(sliceRank, strides, [numSlices, sliceSize]);
        return this.compileAndRun(program, [flattenX, flattenIndices])
            .reshape(resultShape);
    };
    MathBackendWebGL.prototype.makeOutputArray = function (shape, dtype) {
        return tensor_1.Tensor.make(shape, {}, dtype);
    };
    MathBackendWebGL.prototype.makePackedTensor = function (shape) {
        var packedTensor = tensor_1.Tensor.make(shape, {});
        this.texData.get(packedTensor.dataId).isPacked = true;
        return packedTensor;
    };
    MathBackendWebGL.prototype.unpackTensor = function (input) {
        var program = new unpack_gpu_1.UnpackProgram(input.shape);
        return this.compileAndRun(program, [input]);
    };
    MathBackendWebGL.prototype.getBatchDim = function (shape, dimsToSkip) {
        if (dimsToSkip === void 0) { dimsToSkip = 2; }
        return util.sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
    };
    MathBackendWebGL.prototype.getRowsCols = function (shape) {
        if (shape.length === 0) {
            throw Error('Cannot get rows and columns of an empty shape array.');
        }
        return [
            shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]
        ];
    };
    MathBackendWebGL.prototype.packedReshape = function (input, afterShape) {
        var inputAs3D = input.reshape([this.getBatchDim(input.shape)].concat(this.getRowsCols(input.shape)));
        var afterShapeAs3D = [this.getBatchDim(afterShape)].concat(this.getRowsCols(afterShape));
        var program = new reshape_packed_gpu_1.ReshapePackedProgram(afterShapeAs3D, inputAs3D.shape);
        return this
            .compileAndRun(program, [inputAs3D], this.makePackedTensor(afterShapeAs3D))
            .reshape(afterShape);
    };
    MathBackendWebGL.prototype.compileAndRun = function (program, inputs, output, customSetup, pageToCpu) {
        var _this = this;
        if (pageToCpu === void 0) { pageToCpu = true; }
        if (output == null) {
            output =
                this.makeOutputArray(program.outputShape, inputs[0].dtype);
        }
        if (output.size === 0) {
            this.texData.get(output.dataId).values =
                util_1.getTypedArrayFromDType(output.dtype, 0);
            return output;
        }
        var inputsData = inputs.map(function (input) {
            if (input.dtype === 'complex64') {
                throw new Error("GPGPUProgram does not support complex64 input. For complex64 " +
                    "dtypes, please separate the program into real and imaginary " +
                    "parts.");
            }
            var texData = _this.texData.get(input.dataId);
            if (texData.texture == null &&
                !(!texData.isPacked && program.usesPackedTextures) &&
                util.sizeFromShape(input.shape) <=
                    environment_1.ENV.get('WEBGL_SIZE_UPLOAD_UNIFORM')) {
                return {
                    shape: input.shape,
                    texData: null,
                    isUniform: true,
                    uniformValues: _this.readSync(input.dataId)
                };
            }
            if (texData.isPacked !== !!program.usesPackedTextures) {
                var preProcessProgram = void 0;
                var processedInput = void 0;
                if (texData.isPacked) {
                    preProcessProgram = new unpack_gpu_1.UnpackProgram(input.shape);
                    processedInput = _this.compileAndRun(preProcessProgram, [input]);
                }
                else {
                    preProcessProgram = new pack_gpu_1.PackProgram(input.shape);
                    processedInput = _this.compileAndRun(preProcessProgram, [input], _this.makePackedTensor(input.shape));
                }
                texData = _this.texData.get(processedInput.dataId);
                input = processedInput;
            }
            _this.uploadToGPU(input.dataId);
            return { shape: input.shape, texData: texData, isUniform: false };
        });
        this.uploadToGPU(output.dataId);
        var outputData = {
            shape: output.shape,
            texData: this.texData.get(output.dataId),
            isUniform: false
        };
        var key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
        var binary = this.getAndSaveBinary(key, function () {
            return gpgpu_math.compileProgram(_this.gpgpu, program, inputsData, outputData);
        });
        var shouldTimeProgram = this.activeTimers != null;
        var query;
        if (shouldTimeProgram) {
            query = this.startTimer();
        }
        gpgpu_math.runProgram(binary, inputsData, outputData, customSetup);
        if (environment_1.ENV.get('WEBGL_PAGING_ENABLED') && pageToCpu &&
            this.numBytesInGPU > this.NUM_BYTES_BEFORE_PAGING) {
            var numBytesToPage = this.numBytesInGPU - this.NUM_BYTES_BEFORE_PAGING;
            while (numBytesToPage > 0 && this.lruDataGPU.length > 0) {
                var dataId = this.lruDataGPU.shift();
                var _a = this.texData.get(dataId), shape = _a.shape, dtype = _a.dtype;
                numBytesToPage -= this.computeBytes(shape, dtype);
                this.read(dataId);
            }
        }
        if (shouldTimeProgram) {
            query = this.endTimer(query);
            this.activeTimers.push({ name: program.constructor.name, query: this.getQueryTime(query) });
        }
        return output;
    };
    MathBackendWebGL.prototype.getAndSaveBinary = function (key, getBinary) {
        if (!(key in this.binaryCache)) {
            this.binaryCache[key] = getBinary();
        }
        return this.binaryCache[key];
    };
    MathBackendWebGL.prototype.getTextureManager = function () {
        return this.textureManager;
    };
    MathBackendWebGL.prototype.dispose = function () {
        if (this.disposed) {
            return;
        }
        for (var key in this.binaryCache) {
            this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
        }
        this.textureManager.dispose();
        this.canvas.remove();
        if (this.fromPixels2DContext != null) {
            this.fromPixels2DContext.canvas.remove();
        }
        if (this.gpgpuCreatedLocally) {
            this.gpgpu.dispose();
        }
        this.disposed = true;
    };
    MathBackendWebGL.prototype.floatPrecision = function () {
        var _this = this;
        return globals_1.tidy(function () {
            if (_this.abs(tensor_ops_1.scalar(1e-8)).get() > 0) {
                return 32;
            }
            return 16;
        });
    };
    MathBackendWebGL.prototype.uploadToGPU = function (dataId) {
        var texData = this.texData.get(dataId);
        var shape = texData.shape, values = texData.values, texture = texData.texture, dtype = texData.dtype, usage = texData.usage, isPacked = texData.isPacked;
        if (texture != null) {
            if (environment_1.ENV.get('WEBGL_PAGING_ENABLED')) {
                var index = this.lruDataGPU.indexOf(dataId);
                if (index >= 0) {
                    this.lruDataGPU.splice(this.lruDataGPU.indexOf(dataId), 1);
                    this.lruDataGPU.push(dataId);
                }
            }
            return;
        }
        var shouldTimeProgram = this.activeTimers != null;
        var start;
        if (shouldTimeProgram) {
            start = performance.now();
        }
        var texShape = webgl_util.getTextureShapeFromLogicalShape(shape, isPacked);
        texData.texShape = texShape;
        var newTexture = this.acquireTexture(dataId, texShape, usage, isPacked);
        texData.texture = newTexture;
        if (values != null) {
            if (isPacked) {
                var batch = util.sizeFromShape(shape.slice(0, shape.length - 2));
                var rows = shape.length > 1 ? shape[shape.length - 2] : 1;
                var cols = shape[shape.length - 1];
                this.gpgpu.uploadMatrixToPackedTexture(newTexture, batch, rows, cols, typedArrayToFloat32(values, dtype));
            }
            else {
                this.gpgpu.uploadMatrixToTexture(newTexture, texShape[0], texShape[1], typedArrayToFloat32(values, dtype));
            }
            texData.values = null;
            if (shouldTimeProgram) {
                this.uploadWaitMs += performance.now() - start;
            }
        }
    };
    MathBackendWebGL.prototype.convertAndCacheOnCPU = function (dataId, float32Values) {
        var dontKeepCopyOnGPU = this.delayedStorage;
        var texData = this.texData.get(dataId);
        var texture = texData.texture, texShape = texData.texShape, dtype = texData.dtype, usage = texData.usage, isPacked = texData.isPacked;
        if (dontKeepCopyOnGPU && texture != null) {
            this.releaseTexture(dataId, texture, texShape, usage, isPacked);
            texData.texture = null;
            texData.texShape = null;
        }
        texData.usage = tex_util_1.TextureUsage.UPLOAD;
        if (float32Values != null) {
            texData.values = float32ToTypedArray(float32Values, dtype);
        }
        return texData.values;
    };
    MathBackendWebGL.prototype.releaseTexture = function (dataId, texture, texShape, texType, isPacked) {
        var _a = this.texData.get(dataId), shape = _a.shape, dtype = _a.dtype;
        if (environment_1.ENV.get('WEBGL_PAGING_ENABLED')) {
            var idx = this.lruDataGPU.indexOf(dataId);
            if (idx >= 0) {
                this.lruDataGPU.splice(idx, 1);
            }
        }
        this.numBytesInGPU -= this.computeBytes(shape, dtype);
        this.textureManager.releaseTexture(texture, texShape, texType, isPacked);
    };
    MathBackendWebGL.prototype.acquireTexture = function (dataId, texShape, texType, isPacked) {
        var _a = this.texData.get(dataId), shape = _a.shape, dtype = _a.dtype;
        if (environment_1.ENV.get('WEBGL_PAGING_ENABLED')) {
            this.lruDataGPU.push(dataId);
        }
        this.numBytesInGPU += this.computeBytes(shape, dtype);
        return this.textureManager.acquireTexture(texShape, texType, isPacked);
    };
    MathBackendWebGL.prototype.computeBytes = function (shape, dtype) {
        return util.sizeFromShape(shape) * util.bytesPerElement(dtype);
    };
    return MathBackendWebGL;
}());
exports.MathBackendWebGL = MathBackendWebGL;
if (environment_1.ENV.get('IS_BROWSER')) {
    environment_1.ENV.registerBackend('webgl', function () { return new MathBackendWebGL(); }, 2, tensor_1.setTensorTracker);
}
function float32ToTypedArray(a, dtype) {
    if (dtype === 'float32' || dtype === 'complex64') {
        return a;
    }
    else if (dtype === 'int32' || dtype === 'bool') {
        var result = (dtype === 'int32') ? new Int32Array(a.length) :
            new Uint8Array(a.length);
        for (var i = 0; i < result.length; ++i) {
            result[i] = Math.round(a[i]);
        }
        return result;
    }
    else {
        throw new Error("Unknown dtype " + dtype);
    }
}
function typedArrayToFloat32(a, dtype) {
    return (a instanceof Float32Array) ? a : new Float32Array(a);
}
//# sourceMappingURL=backend_webgl.js.map