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
var tensor_ops_1 = require("../ops/tensor_ops");
var util_1 = require("../util");
var types_1 = require("./types");
function encodeWeights(tensors) {
    return __awaiter(this, void 0, void 0, function () {
        var specs, dataPromises, name_1, t, tensorValues;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    specs = [];
                    dataPromises = [];
                    for (name_1 in tensors) {
                        t = tensors[name_1];
                        if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool') {
                            throw new Error("Unsupported dtype in weight '" + name_1 + "': " + t.dtype);
                        }
                        specs.push({ name: name_1, shape: t.shape, dtype: t.dtype });
                        dataPromises.push(t.data());
                    }
                    return [4, Promise.all(dataPromises)];
                case 1:
                    tensorValues = _a.sent();
                    return [2, { data: concatenateTypedArrays(tensorValues), specs: specs }];
            }
        });
    });
}
exports.encodeWeights = encodeWeights;
function decodeWeights(buffer, specs) {
    var out = {};
    var offset = 0;
    var _loop_1 = function (spec) {
        var name_2 = spec.name;
        var dtype = spec.dtype;
        var shape = spec.shape;
        var size = util_1.sizeFromShape(shape);
        var typedArray = void 0;
        if ('quantization' in spec) {
            var quantization_1 = spec.quantization;
            if (quantization_1.dtype !== 'uint8' && quantization_1.dtype !== 'uint16') {
                throw new Error("Weight " + spec.name + " has unknown " +
                    ("quantization dtype " + quantization_1.dtype + ". ") +
                    "Supported quantization dtypes are: 'uint8' and 'uint16'.");
            }
            var quantizationSizeFactor = types_1.DTYPE_VALUE_SIZE_MAP[quantization_1.dtype];
            var byteBuffer = buffer.slice(offset, offset + size * quantizationSizeFactor);
            var quantizedArray = (quantization_1.dtype === 'uint8') ?
                new Uint8Array(byteBuffer) :
                new Uint16Array(byteBuffer);
            if (dtype === 'float32') {
                typedArray = Float32Array.from(quantizedArray, function (v) { return v * quantization_1.scale + quantization_1.min; });
            }
            else if (dtype === 'int32') {
                typedArray = Int32Array.from(quantizedArray, function (v) { return Math.round(v * quantization_1.scale + quantization_1.min); });
            }
            else {
                throw new Error("Unsupported dtype in weight '" + name_2 + "': " + dtype);
            }
            offset += size * quantizationSizeFactor;
        }
        else {
            var dtypeFactor = types_1.DTYPE_VALUE_SIZE_MAP[dtype];
            var byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
            if (dtype === 'float32') {
                typedArray = new Float32Array(byteBuffer);
            }
            else if (dtype === 'int32') {
                typedArray = new Int32Array(byteBuffer);
            }
            else if (dtype === 'bool') {
                typedArray = new Uint8Array(byteBuffer);
            }
            else {
                throw new Error("Unsupported dtype in weight '" + name_2 + "': " + dtype);
            }
            offset += size * dtypeFactor;
        }
        var value = void 0;
        if (dtype === 'float32') {
            value = tensor_ops_1.tensor(typedArray, shape, 'float32');
        }
        else if (dtype === 'int32') {
            value = tensor_ops_1.tensor(typedArray, shape, 'int32');
        }
        else if (dtype === 'bool') {
            value = tensor_ops_1.tensor(typedArray, shape, 'bool');
        }
        else {
            throw new Error("Unsupported dtype in weight '" + name_2 + "': " + dtype);
        }
        out[name_2] = value;
    };
    for (var _i = 0, specs_1 = specs; _i < specs_1.length; _i++) {
        var spec = specs_1[_i];
        _loop_1(spec);
    }
    return out;
}
exports.decodeWeights = decodeWeights;
function concatenateTypedArrays(xs) {
    if (xs === null) {
        throw new Error("Invalid input value: " + JSON.stringify(xs));
    }
    var totalByteLength = 0;
    var normalizedXs = [];
    xs.forEach(function (x) {
        totalByteLength += x.byteLength;
        normalizedXs.push(x.byteLength === x.buffer.byteLength ? x :
            new x.constructor(x));
        if (!(x instanceof Float32Array || x instanceof Int32Array ||
            x instanceof Uint8Array)) {
            throw new Error("Unsupported TypedArray subtype: " + x.constructor.name);
        }
    });
    var y = new Uint8Array(totalByteLength);
    var offset = 0;
    normalizedXs.forEach(function (x) {
        y.set(new Uint8Array(x.buffer), offset);
        offset += x.byteLength;
    });
    return y.buffer;
}
exports.concatenateTypedArrays = concatenateTypedArrays;
var useNodeBuffer = typeof Buffer !== 'undefined' &&
    (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
        typeof btoa === 'undefined');
function stringByteLength(str) {
    if (useNodeBuffer) {
        return Buffer.byteLength(str);
    }
    return new Blob([str]).size;
}
exports.stringByteLength = stringByteLength;
function arrayBufferToBase64String(buffer) {
    if (useNodeBuffer) {
        return Buffer.from(buffer).toString('base64');
    }
    return btoa(String.fromCharCode.apply(null, new Uint8Array(buffer)));
}
exports.arrayBufferToBase64String = arrayBufferToBase64String;
function base64StringToArrayBuffer(str) {
    if (useNodeBuffer) {
        var buf = Buffer.from(str, 'base64');
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    var s = atob(str);
    var buffer = new Uint8Array(s.length);
    for (var i = 0; i < s.length; ++i) {
        buffer.set([s.charCodeAt(i)], i);
    }
    return buffer.buffer;
}
exports.base64StringToArrayBuffer = base64StringToArrayBuffer;
function concatenateArrayBuffers(buffers) {
    var totalByteLength = 0;
    buffers.forEach(function (buffer) {
        totalByteLength += buffer.byteLength;
    });
    var temp = new Uint8Array(totalByteLength);
    var offset = 0;
    buffers.forEach(function (buffer) {
        temp.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    });
    return temp.buffer;
}
exports.concatenateArrayBuffers = concatenateArrayBuffers;
function basename(path) {
    var SEPARATOR = '/';
    path = path.trim();
    while (path.endsWith(SEPARATOR)) {
        path = path.slice(0, path.length - 1);
    }
    var items = path.split(SEPARATOR);
    return items[items.length - 1];
}
exports.basename = basename;
function getModelArtifactsInfoForJSON(modelArtifacts) {
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
        throw new Error('Expected JSON model topology, received ArrayBuffer.');
    }
    return {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: modelArtifacts.modelTopology == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
        weightSpecsBytes: modelArtifacts.weightSpecs == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
        weightDataBytes: modelArtifacts.weightData == null ?
            0 :
            modelArtifacts.weightData.byteLength,
    };
}
exports.getModelArtifactsInfoForJSON = getModelArtifactsInfoForJSON;
//# sourceMappingURL=io_utils.js.map