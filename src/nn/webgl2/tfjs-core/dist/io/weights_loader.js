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
var util = require("../util");
var io_utils_1 = require("./io_utils");
var types_1 = require("./types");
function loadWeightsAsArrayBuffer(fetchURLs, requestOptions) {
    return __awaiter(this, void 0, void 0, function () {
        var requests, responses, buffers;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    requests = fetchURLs.map(function (fetchURL) { return fetch(fetchURL, requestOptions); });
                    return [4, Promise.all(requests)];
                case 1:
                    responses = _a.sent();
                    return [4, Promise.all(responses.map(function (response) { return response.arrayBuffer(); }))];
                case 2:
                    buffers = _a.sent();
                    return [2, buffers];
            }
        });
    });
}
exports.loadWeightsAsArrayBuffer = loadWeightsAsArrayBuffer;
function loadWeights(manifest, filePathPrefix, weightNames, requestOptions) {
    if (filePathPrefix === void 0) { filePathPrefix = ''; }
    return __awaiter(this, void 0, void 0, function () {
        var groupIndicesToFetchMap, groupWeightsToFetch, weightsFound, allManifestWeightNames, weightsNotFound, groupIndicesToFetch, fetchUrls, buffers, weightsTensorMap, bufferIndexOffset;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    groupIndicesToFetchMap = manifest.map(function () { return false; });
                    groupWeightsToFetch = {};
                    weightsFound = weightNames != null ? weightNames.map(function () { return false; }) : [];
                    allManifestWeightNames = [];
                    manifest.forEach(function (manifestGroupConfig, groupIndex) {
                        var groupOffset = 0;
                        manifestGroupConfig.weights.forEach(function (weightsEntry) {
                            var rawDtype = ('quantization' in weightsEntry) ?
                                weightsEntry.quantization.dtype :
                                weightsEntry.dtype;
                            var weightsBytes = types_1.DTYPE_VALUE_SIZE_MAP[rawDtype] *
                                util.sizeFromShape(weightsEntry.shape);
                            var enqueueWeightsForFetchingFn = function () {
                                groupIndicesToFetchMap[groupIndex] = true;
                                if (groupWeightsToFetch[groupIndex] == null) {
                                    groupWeightsToFetch[groupIndex] = [];
                                }
                                groupWeightsToFetch[groupIndex].push({
                                    manifestEntry: weightsEntry,
                                    groupOffset: groupOffset,
                                    sizeBytes: weightsBytes
                                });
                            };
                            if (weightNames != null) {
                                weightNames.forEach(function (weightName, weightIndex) {
                                    if (weightName === weightsEntry.name) {
                                        enqueueWeightsForFetchingFn();
                                        weightsFound[weightIndex] = true;
                                    }
                                });
                            }
                            else {
                                enqueueWeightsForFetchingFn();
                            }
                            allManifestWeightNames.push(weightsEntry.name);
                            groupOffset += weightsBytes;
                        });
                    });
                    if (!weightsFound.every(function (found) { return found; })) {
                        weightsNotFound = weightNames.filter(function (weight, i) { return !weightsFound[i]; });
                        throw new Error("Could not find weights in manifest with names: " +
                            (weightsNotFound.join(', ') + ". \n") +
                            "Manifest JSON has weights with names: " +
                            (allManifestWeightNames.join(', ') + "."));
                    }
                    groupIndicesToFetch = groupIndicesToFetchMap.reduce(function (accumulator, shouldFetch, i) {
                        if (shouldFetch) {
                            accumulator.push(i);
                        }
                        return accumulator;
                    }, []);
                    fetchUrls = [];
                    groupIndicesToFetch.forEach(function (i) {
                        manifest[i].paths.forEach(function (filepath) {
                            var fetchUrl = filePathPrefix +
                                (!filePathPrefix.endsWith('/') ? '/' : '') + filepath;
                            fetchUrls.push(fetchUrl);
                        });
                    });
                    return [4, loadWeightsAsArrayBuffer(fetchUrls, requestOptions)];
                case 1:
                    buffers = _a.sent();
                    weightsTensorMap = {};
                    bufferIndexOffset = 0;
                    groupIndicesToFetch.forEach(function (i) {
                        var numBuffers = manifest[i].paths.length;
                        var groupBytes = 0;
                        for (var i_1 = 0; i_1 < numBuffers; i_1++) {
                            groupBytes += buffers[bufferIndexOffset + i_1].byteLength;
                        }
                        var groupBuffer = new ArrayBuffer(groupBytes);
                        var groupByteBuffer = new Uint8Array(groupBuffer);
                        var groupBufferOffset = 0;
                        for (var i_2 = 0; i_2 < numBuffers; i_2++) {
                            var buffer = new Uint8Array(buffers[bufferIndexOffset + i_2]);
                            groupByteBuffer.set(buffer, groupBufferOffset);
                            groupBufferOffset += buffer.byteLength;
                        }
                        var weightsEntries = groupWeightsToFetch[i];
                        weightsEntries.forEach(function (weightsEntry) {
                            var byteBuffer = groupBuffer.slice(weightsEntry.groupOffset, weightsEntry.groupOffset + weightsEntry.sizeBytes);
                            var nameToTensorMap = io_utils_1.decodeWeights(byteBuffer, [weightsEntry.manifestEntry]);
                            for (var name_1 in nameToTensorMap) {
                                weightsTensorMap[name_1] = nameToTensorMap[name_1];
                            }
                        });
                        bufferIndexOffset += numBuffers;
                    });
                    return [2, weightsTensorMap];
            }
        });
    });
}
exports.loadWeights = loadWeights;
//# sourceMappingURL=weights_loader.js.map