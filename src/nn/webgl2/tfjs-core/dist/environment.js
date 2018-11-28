"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var device_util = require("./device_util");
var engine_1 = require("./engine");
var environment_util_1 = require("./environment_util");
var tensor_1 = require("./tensor");
var tensor_util_1 = require("./tensor_util");
var EPSILON_FLOAT16 = 1e-3;
var TEST_EPSILON_FLOAT16 = 1e-1;
var EPSILON_FLOAT32 = 1e-7;
var TEST_EPSILON_FLOAT32 = 1e-3;
var Environment = (function () {
    function Environment(features) {
        this.features = {};
        this.registry = {};
        if (features != null) {
            this.features = features;
        }
        if (this.get('DEBUG')) {
            console.warn('Debugging mode is ON. The output of every math call will ' +
                'be downloaded to CPU and checked for NaNs. ' +
                'This significantly impacts performance.');
        }
    }
    Environment.setBackend = function (backendName, safeMode) {
        if (safeMode === void 0) { safeMode = false; }
        if (!(backendName in exports.ENV.registry)) {
            throw new Error("Backend name '" + backendName + "' not found in registry");
        }
        exports.ENV.engine.backend = exports.ENV.findBackend(backendName);
        exports.ENV.backendName = backendName;
    };
    Environment.getBackend = function () {
        exports.ENV.initEngine();
        return exports.ENV.backendName;
    };
    Environment.disposeVariables = function () {
        exports.ENV.engine.disposeVariables();
    };
    Environment.memory = function () {
        return exports.ENV.engine.memory();
    };
    Environment.profile = function (f) {
        return exports.ENV.engine.profile(f);
    };
    Environment.tidy = function (nameOrFn, fn, gradMode) {
        if (gradMode === void 0) { gradMode = false; }
        return exports.ENV.engine.tidy(nameOrFn, fn, gradMode);
    };
    Environment.dispose = function (container) {
        var tensors = tensor_util_1.getTensorsInContainer(container);
        tensors.forEach(function (tensor) { return tensor.dispose(); });
    };
    Environment.keep = function (result) {
        return exports.ENV.engine.keep(result);
    };
    Environment.time = function (f) {
        return exports.ENV.engine.time(f);
    };
    Environment.prototype.get = function (feature) {
        if (feature in this.features) {
            return this.features[feature];
        }
        this.features[feature] = this.evaluateFeature(feature);
        return this.features[feature];
    };
    Environment.prototype.getFeatures = function () {
        return this.features;
    };
    Environment.prototype.set = function (feature, value) {
        this.features[feature] = value;
    };
    Environment.prototype.getBestBackendName = function () {
        var _this = this;
        if (Object.keys(this.registry).length === 0) {
            throw new Error('No backend found in registry.');
        }
        var sortedBackends = Object.keys(this.registry)
            .map(function (name) {
            return { name: name, entry: _this.registry[name] };
        })
            .sort(function (a, b) {
            return b.entry.priority - a.entry.priority;
        });
        return sortedBackends[0].name;
    };
    Environment.prototype.evaluateFeature = function (feature) {
        if (feature === 'DEBUG') {
            return false;
        }
        else if (feature === 'IS_BROWSER') {
            return typeof window !== 'undefined';
        }
        else if (feature === 'IS_NODE') {
            return (typeof process !== 'undefined') &&
                (typeof process.versions.node !== 'undefined');
        }
        else if (feature === 'IS_CHROME') {
            return environment_util_1.isChrome();
        }
        else if (feature === 'WEBGL_CPU_FORWARD') {
            return true;
        }
        else if (feature === 'WEBGL_PACK_BATCHNORMALIZATION') {
            return false;
        }
        else if (feature === 'WEBGL_LAZILY_UNPACK') {
            return false;
        }
        else if (feature === 'WEBGL_CONV_IM2COL') {
            return false;
        }
        else if (feature === 'WEBGL_PAGING_ENABLED') {
            return this.get('IS_BROWSER') && !this.get('PROD');
        }
        else if (feature === 'WEBGL_MAX_TEXTURE_SIZE') {
            return environment_util_1.getWebGLMaxTextureSize(this.get('WEBGL_VERSION'));
        }
        else if (feature === 'IS_TEST') {
            return false;
        }
        else if (feature === 'BACKEND') {
            return this.getBestBackendName();
        }
        else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') {
            var webGLVersion = this.get('WEBGL_VERSION');
            if (webGLVersion === 0) {
                return 0;
            }
            return environment_util_1.getWebGLDisjointQueryTimerVersion(webGLVersion);
        }
        else if (feature === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') {
            return this.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0 &&
                !device_util.isMobile();
        }
        else if (feature === 'HAS_WEBGL') {
            return this.get('WEBGL_VERSION') > 0;
        }
        else if (feature === 'WEBGL_VERSION') {
            if (environment_util_1.isWebGLVersionEnabled(2)) {
                return 2;
            }
            else if (environment_util_1.isWebGLVersionEnabled(1)) {
                return 1;
            }
            return 0;
        }
        else if (feature === 'WEBGL_RENDER_FLOAT32_ENABLED') {
            return environment_util_1.isRenderToFloatTextureEnabled(this.get('WEBGL_VERSION'));
        }
        else if (feature === 'WEBGL_DOWNLOAD_FLOAT_ENABLED') {
            return environment_util_1.isDownloadFloatTextureEnabled(this.get('WEBGL_VERSION'));
        }
        else if (feature === 'WEBGL_FENCE_API_ENABLED') {
            return environment_util_1.isWebGLFenceEnabled(this.get('WEBGL_VERSION'));
        }
        else if (feature === 'WEBGL_SIZE_UPLOAD_UNIFORM') {
            var useUniforms = this.get('WEBGL_RENDER_FLOAT32_ENABLED');
            return useUniforms ? 4 : 0;
        }
        else if (feature === 'TEST_EPSILON') {
            return this.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
                TEST_EPSILON_FLOAT16;
        }
        else if (feature === 'EPSILON') {
            return this.backend.floatPrecision() === 32 ? EPSILON_FLOAT32 :
                EPSILON_FLOAT16;
        }
        else if (feature === 'PROD') {
            return false;
        }
        else if (feature === 'TENSORLIKE_CHECK_SHAPE_CONSISTENCY') {
            return !this.get('PROD');
        }
        throw new Error("Unknown feature " + feature + ".");
    };
    Environment.prototype.setFeatures = function (features) {
        this.features = Object.assign({}, features);
    };
    Environment.prototype.reset = function () {
        this.features = environment_util_1.getFeaturesFromURL();
        if (this.globalEngine != null) {
            this.globalEngine = null;
        }
    };
    Object.defineProperty(Environment.prototype, "backend", {
        get: function () {
            return this.engine.backend;
        },
        enumerable: true,
        configurable: true
    });
    Environment.prototype.findBackend = function (name) {
        if (!(name in this.registry)) {
            return null;
        }
        return this.registry[name].backend;
    };
    Environment.prototype.registerBackend = function (name, factory, priority, setTensorTrackerFn) {
        var _this = this;
        if (priority === void 0) { priority = 1; }
        if (name in this.registry) {
            console.warn(name + " backend was already registered. Reusing existing backend");
            if (setTensorTrackerFn != null) {
                setTensorTrackerFn(function () { return _this.engine; });
            }
            return false;
        }
        try {
            var backend = factory();
            backend.setDataMover({ moveData: function (dataId) { return _this.engine.moveData(dataId); } });
            this.registry[name] = { backend: backend, priority: priority };
            return true;
        }
        catch (err) {
            console.warn("Registration of backend " + name + " failed");
            console.warn(err.stack || err.message);
            return false;
        }
    };
    Environment.prototype.removeBackend = function (name) {
        if (!(name in this.registry)) {
            throw new Error(name + " backend not found in registry");
        }
        this.registry[name].backend.dispose();
        delete this.registry[name];
    };
    Object.defineProperty(Environment.prototype, "engine", {
        get: function () {
            this.initEngine();
            return this.globalEngine;
        },
        enumerable: true,
        configurable: true
    });
    Environment.prototype.initEngine = function () {
        var _this = this;
        if (this.globalEngine == null) {
            this.backendName = this.get('BACKEND');
            var backend = this.findBackend(this.backendName);
            this.globalEngine =
                new engine_1.Engine(backend, false, function () { return _this.get('DEBUG'); });
        }
    };
    return Environment;
}());
exports.Environment = Environment;
function getGlobalNamespace() {
    var ns;
    if (typeof (window) !== 'undefined') {
        ns = window;
    }
    else if (typeof (process) !== 'undefined') {
        ns = process;
    }
    else {
        throw new Error('Could not find a global object');
    }
    return ns;
}
function getOrMakeEnvironment() {
    var ns = getGlobalNamespace();
    if (ns.ENV == null) {
        ns.ENV = new Environment(environment_util_1.getFeaturesFromURL());
        tensor_1.setTensorTracker(function () { return ns.ENV.engine; });
    }
    return ns.ENV;
}
exports.ENV = getOrMakeEnvironment();
//# sourceMappingURL=environment.js.map