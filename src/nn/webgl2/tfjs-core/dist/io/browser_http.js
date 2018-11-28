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
var util_1 = require("../util");
var io_utils_1 = require("./io_utils");
var router_registry_1 = require("./router_registry");
var weights_loader_1 = require("./weights_loader");
var BrowserHTTPRequest = (function () {
    function BrowserHTTPRequest(path, requestInit, weightPathPrefix) {
        this.weightPathPrefix = weightPathPrefix;
        this.DEFAULT_METHOD = 'POST';
        if (typeof fetch === 'undefined') {
            throw new Error('browserHTTPRequest is not supported outside the web browser without a fetch polyfill.');
        }
        util_1.assert(path != null && path.length > 0, 'URL path for browserHTTPRequest must not be null, undefined or ' +
            'empty.');
        if (Array.isArray(path)) {
            util_1.assert(path.length === 2, 'URL paths for browserHTTPRequest must have a length of 2, ' +
                ("(actual length is " + path.length + ")."));
        }
        this.path = path;
        if (requestInit != null && requestInit.body != null) {
            throw new Error('requestInit is expected to have no pre-existing body, but has one.');
        }
        this.requestInit = requestInit || {};
    }
    BrowserHTTPRequest.prototype.save = function (modelArtifacts) {
        return __awaiter(this, void 0, void 0, function () {
            var init, weightsManifest, modelTopologyAndWeightManifest, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
                            throw new Error('BrowserHTTPRequest.save() does not support saving model topology ' +
                                'in binary formats yet.');
                        }
                        init = Object.assign({ method: this.DEFAULT_METHOD }, this.requestInit);
                        init.body = new FormData();
                        weightsManifest = [{
                                paths: ['./model.weights.bin'],
                                weights: modelArtifacts.weightSpecs,
                            }];
                        modelTopologyAndWeightManifest = {
                            modelTopology: modelArtifacts.modelTopology,
                            weightsManifest: weightsManifest
                        };
                        init.body.append('model.json', new Blob([JSON.stringify(modelTopologyAndWeightManifest)], { type: 'application/json' }), 'model.json');
                        if (modelArtifacts.weightData != null) {
                            init.body.append('model.weights.bin', new Blob([modelArtifacts.weightData], { type: 'application/octet-stream' }), 'model.weights.bin');
                        }
                        return [4, fetch(this.path, init)];
                    case 1:
                        response = _a.sent();
                        if (response.ok) {
                            return [2, {
                                    modelArtifactsInfo: io_utils_1.getModelArtifactsInfoForJSON(modelArtifacts),
                                    responses: [response],
                                }];
                        }
                        else {
                            throw new Error("BrowserHTTPRequest.save() failed due to HTTP response status " +
                                (response.status + "."));
                        }
                        return [2];
                }
            });
        });
    };
    BrowserHTTPRequest.prototype.load = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2, Array.isArray(this.path) ? this.loadBinaryModel() :
                        this.loadJSONModel()];
            });
        });
    };
    BrowserHTTPRequest.prototype.loadBinaryTopology = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response, error_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 3, , 4]);
                        return [4, fetch(this.path[0], this.requestInit)];
                    case 1:
                        response = _a.sent();
                        if (!response.ok) {
                            throw new Error("BrowserHTTPRequest.load() failed due to HTTP response: " + response.statusText);
                        }
                        return [4, response.arrayBuffer()];
                    case 2: return [2, _a.sent()];
                    case 3:
                        error_1 = _a.sent();
                        throw new Error(this.path[0] + " not found. " + error_1);
                    case 4: return [2];
                }
            });
        });
    };
    BrowserHTTPRequest.prototype.loadBinaryModel = function () {
        return __awaiter(this, void 0, void 0, function () {
            var graphPromise, manifestPromise, results, modelTopology, weightsManifestResponse, weightsManifest, weightSpecs, weightData, results_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        graphPromise = this.loadBinaryTopology();
                        return [4, fetch(this.path[1], this.requestInit)];
                    case 1:
                        manifestPromise = _a.sent();
                        if (!manifestPromise.ok) {
                            throw new Error("BrowserHTTPRequest.load() failed due to HTTP response: " + manifestPromise.statusText);
                        }
                        return [4, Promise.all([graphPromise, manifestPromise])];
                    case 2:
                        results = _a.sent();
                        modelTopology = results[0], weightsManifestResponse = results[1];
                        return [4, weightsManifestResponse.json()];
                    case 3:
                        weightsManifest = _a.sent();
                        if (!(weightsManifest != null)) return [3, 5];
                        return [4, this.loadWeights(weightsManifest)];
                    case 4:
                        results_1 = _a.sent();
                        weightSpecs = results_1[0], weightData = results_1[1];
                        _a.label = 5;
                    case 5: return [2, { modelTopology: modelTopology, weightSpecs: weightSpecs, weightData: weightData }];
                }
            });
        });
    };
    BrowserHTTPRequest.prototype.loadJSONModel = function () {
        return __awaiter(this, void 0, void 0, function () {
            var modelConfigRequest, modelConfig, modelTopology, weightsManifest, weightSpecs, weightData, weightsManifest_1, results;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, fetch(this.path, this.requestInit)];
                    case 1:
                        modelConfigRequest = _a.sent();
                        if (!modelConfigRequest.ok) {
                            throw new Error("BrowserHTTPRequest.load() failed due to HTTP response: " + modelConfigRequest.statusText);
                        }
                        return [4, modelConfigRequest.json()];
                    case 2:
                        modelConfig = _a.sent();
                        modelTopology = modelConfig['modelTopology'];
                        weightsManifest = modelConfig['weightsManifest'];
                        if (modelTopology == null && weightsManifest == null) {
                            throw new Error("The JSON from HTTP path " + this.path + " contains neither model " +
                                "topology or manifest for weights.");
                        }
                        if (!(weightsManifest != null)) return [3, 4];
                        weightsManifest_1 = modelConfig['weightsManifest'];
                        return [4, this.loadWeights(weightsManifest_1)];
                    case 3:
                        results = _a.sent();
                        weightSpecs = results[0], weightData = results[1];
                        _a.label = 4;
                    case 4: return [2, { modelTopology: modelTopology, weightSpecs: weightSpecs, weightData: weightData }];
                }
            });
        });
    };
    BrowserHTTPRequest.prototype.loadWeights = function (weightsManifest) {
        return __awaiter(this, void 0, void 0, function () {
            var weightPath, _a, prefix, suffix, pathPrefix, weightSpecs, _i, weightsManifest_2, entry, fetchURLs, _b, _c;
            return __generator(this, function (_d) {
                switch (_d.label) {
                    case 0:
                        weightPath = Array.isArray(this.path) ? this.path[1] : this.path;
                        _a = parseUrl(weightPath), prefix = _a[0], suffix = _a[1];
                        pathPrefix = this.weightPathPrefix || prefix;
                        weightSpecs = [];
                        for (_i = 0, weightsManifest_2 = weightsManifest; _i < weightsManifest_2.length; _i++) {
                            entry = weightsManifest_2[_i];
                            weightSpecs.push.apply(weightSpecs, entry.weights);
                        }
                        fetchURLs = [];
                        weightsManifest.forEach(function (weightsGroup) {
                            weightsGroup.paths.forEach(function (path) {
                                fetchURLs.push(pathPrefix + path + suffix);
                            });
                        });
                        _b = [weightSpecs];
                        _c = io_utils_1.concatenateArrayBuffers;
                        return [4, weights_loader_1.loadWeightsAsArrayBuffer(fetchURLs, this.requestInit)];
                    case 1: return [2, _b.concat([
                            _c.apply(void 0, [_d.sent()])
                        ])];
                }
            });
        });
    };
    BrowserHTTPRequest.URL_SCHEME_REGEX = /^https?:\/\//;
    return BrowserHTTPRequest;
}());
exports.BrowserHTTPRequest = BrowserHTTPRequest;
function parseUrl(url) {
    var lastSlash = url.lastIndexOf('/');
    var lastSearchParam = url.lastIndexOf('?');
    var prefix = url.substring(0, lastSlash);
    var suffix = lastSearchParam > lastSlash ? url.substring(lastSearchParam) : '';
    return [prefix + '/', suffix];
}
exports.parseUrl = parseUrl;
function isHTTPScheme(url) {
    return url.match(BrowserHTTPRequest.URL_SCHEME_REGEX) != null;
}
exports.httpRequestRouter = function (url) {
    if (typeof fetch === 'undefined') {
        return null;
    }
    else {
        var isHTTP = true;
        if (Array.isArray(url)) {
            isHTTP = url.every(function (urlItem) { return isHTTPScheme(urlItem); });
        }
        else {
            isHTTP = isHTTPScheme(url);
        }
        if (isHTTP) {
            return browserHTTPRequest(url);
        }
    }
    return null;
};
router_registry_1.IORouterRegistry.registerSaveRouter(exports.httpRequestRouter);
router_registry_1.IORouterRegistry.registerLoadRouter(exports.httpRequestRouter);
function browserHTTPRequest(path, requestInit, weightPathPrefix) {
    return new BrowserHTTPRequest(path, requestInit, weightPathPrefix);
}
exports.browserHTTPRequest = browserHTTPRequest;
//# sourceMappingURL=browser_http.js.map