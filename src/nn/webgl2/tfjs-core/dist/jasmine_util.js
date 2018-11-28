"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("./environment");
var backend_cpu_1 = require("./kernels/backend_cpu");
var backend_webgl_1 = require("./kernels/backend_webgl");
Error.stackTraceLimit = Infinity;
function envSatisfiesConstraints(constraints) {
    for (var key in constraints) {
        var value = constraints[key];
        if (environment_1.ENV.get(key) !== value) {
            return false;
        }
    }
    return true;
}
exports.envSatisfiesConstraints = envSatisfiesConstraints;
function parseKarmaFlags(args) {
    var features;
    var backend;
    var name = '';
    args.forEach(function (arg, i) {
        if (arg === '--features') {
            features = JSON.parse(args[i + 1]);
        }
        else if (arg === '--backend') {
            var type = args[i + 1];
            name = type;
            if (type.toLowerCase() === 'cpu') {
                backend = function () { return new backend_cpu_1.MathBackendCPU(); };
                features = features || {};
                features['HAS_WEBGL'] = false;
            }
            else if (type.toLowerCase() === 'webgl') {
                backend = function () { return new backend_webgl_1.MathBackendWebGL(); };
            }
            else {
                throw new Error("Unknown value " + type + " for flag --backend. " +
                    "Allowed values are 'cpu' or 'webgl'.");
            }
        }
    });
    if (features == null && backend == null) {
        return null;
    }
    if (features != null && backend == null) {
        throw new Error('--backend flag is required when --features is present. ' +
            'Available values are "webgl" or "cpu".');
    }
    return { features: features || {}, factory: backend, name: name };
}
exports.parseKarmaFlags = parseKarmaFlags;
function describeWithFlags(name, constraints, tests) {
    exports.TEST_ENVS.forEach(function (testEnv) {
        environment_1.ENV.setFeatures(testEnv.features);
        if (envSatisfiesConstraints(constraints)) {
            var testName = name + ' ' + testEnv.name + ' ' + JSON.stringify(testEnv.features);
            executeTests(testName, tests, testEnv);
        }
    });
}
exports.describeWithFlags = describeWithFlags;
exports.TEST_ENVS = [
    {
        name: 'test-webgl1',
        factory: function () { return new backend_webgl_1.MathBackendWebGL(); },
        features: { 'WEBGL_VERSION': 1, 'WEBGL_CPU_FORWARD': false }
    },
    {
        name: 'test-webgl2',
        factory: function () { return new backend_webgl_1.MathBackendWebGL(); },
        features: { 'WEBGL_VERSION': 2, 'WEBGL_CPU_FORWARD': false }
    },
    {
        name: 'test-cpu',
        factory: function () { return new backend_cpu_1.MathBackendCPU(); },
        features: { 'HAS_WEBGL': false }
    }
];
exports.CPU_FACTORY = function () { return new backend_cpu_1.MathBackendCPU(); };
if (typeof __karma__ !== 'undefined') {
    var testEnv = parseKarmaFlags(__karma__.config.args);
    if (testEnv) {
        setTestEnvs([testEnv]);
    }
}
function setTestEnvs(testEnvs) {
    exports.TEST_ENVS = testEnvs;
}
exports.setTestEnvs = setTestEnvs;
function executeTests(testName, tests, testEnv) {
    describe(testName, function () {
        var backendName = 'test-' + testEnv.name;
        beforeAll(function () {
            environment_1.ENV.reset();
            environment_1.ENV.setFeatures(testEnv.features);
            environment_1.ENV.set('IS_TEST', true);
            environment_1.ENV.registerBackend(backendName, testEnv.factory, 1000);
            environment_1.Environment.setBackend(backendName);
        });
        beforeEach(function () {
            environment_1.ENV.engine.startScope();
        });
        afterEach(function () {
            environment_1.ENV.engine.endScope();
            environment_1.Environment.disposeVariables();
        });
        afterAll(function () {
            environment_1.ENV.removeBackend(backendName);
            environment_1.ENV.reset();
        });
        tests(testEnv);
    });
}
//# sourceMappingURL=jasmine_util.js.map