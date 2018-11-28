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
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var modelTopology1 = {
    'class_name': 'Sequential',
    'keras_version': '2.1.4',
    'config': [{
            'class_name': 'Dense',
            'config': {
                'kernel_initializer': {
                    'class_name': 'VarianceScaling',
                    'config': {
                        'distribution': 'uniform',
                        'scale': 1.0,
                        'seed': null,
                        'mode': 'fan_avg'
                    }
                },
                'name': 'dense',
                'kernel_constraint': null,
                'bias_regularizer': null,
                'bias_constraint': null,
                'dtype': 'float32',
                'activation': 'linear',
                'trainable': true,
                'kernel_regularizer': null,
                'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                'units': 1,
                'batch_input_shape': [null, 3],
                'use_bias': true,
                'activity_regularizer': null
            }
        }],
    'backend': 'tensorflow'
};
var weightSpecs1 = [
    {
        name: 'dense/kernel',
        shape: [3, 1],
        dtype: 'float32',
    },
    {
        name: 'dense/bias',
        shape: [1],
        dtype: 'float32',
    }
];
var weightData1 = new ArrayBuffer(16);
var artifacts1 = {
    modelTopology: modelTopology1,
    weightSpecs: weightSpecs1,
    weightData: weightData1,
};
jasmine_util_1.describeWithFlags('Passthrough Saver', test_util_1.BROWSER_ENVS, function () {
    it('passes provided arguments through on save', function () { return __awaiter(_this, void 0, void 0, function () {
        function saveHandler(artifacts) {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    savedArtifacts = artifacts;
                    return [2, {
                            modelArtifactsInfo: {
                                dateSaved: testStartDate,
                                modelTopologyType: 'JSON',
                                modelTopologyBytes: JSON.stringify(modelTopology1).length,
                                weightSpecsBytes: JSON.stringify(weightSpecs1).length,
                                weightDataBytes: weightData1.byteLength,
                            }
                        }];
                });
            });
        }
        var testStartDate, savedArtifacts, saveTrigger, saveResult, artifactsInfo;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    testStartDate = new Date();
                    savedArtifacts = null;
                    saveTrigger = tf.io.withSaveHandler(saveHandler);
                    return [4, saveTrigger.save(artifacts1)];
                case 1:
                    saveResult = _a.sent();
                    expect(saveResult.errors).toEqual(undefined);
                    artifactsInfo = saveResult.modelArtifactsInfo;
                    expect(artifactsInfo.dateSaved.getTime())
                        .toBeGreaterThanOrEqual(testStartDate.getTime());
                    expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
                        .toEqual(JSON.stringify(modelTopology1).length);
                    expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
                        .toEqual(JSON.stringify(weightSpecs1).length);
                    expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(16);
                    expect(savedArtifacts.modelTopology).toEqual(modelTopology1);
                    expect(savedArtifacts.weightSpecs).toEqual(weightSpecs1);
                    expect(savedArtifacts.weightData).toEqual(weightData1);
                    return [2];
            }
        });
    }); });
});
jasmine_util_1.describeWithFlags('Passthrough Loader', test_util_1.BROWSER_ENVS, function () {
    it('load topology and weights', function () { return __awaiter(_this, void 0, void 0, function () {
        var passthroughHandler, modelArtifacts;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    passthroughHandler = tf.io.fromMemory(modelTopology1, weightSpecs1, weightData1);
                    return [4, passthroughHandler.load()];
                case 1:
                    modelArtifacts = _a.sent();
                    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                    expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
                    expect(modelArtifacts.weightData).toEqual(weightData1);
                    return [2];
            }
        });
    }); });
    it('load model topology only', function () { return __awaiter(_this, void 0, void 0, function () {
        var passthroughHandler, modelArtifacts;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    passthroughHandler = tf.io.fromMemory(modelTopology1);
                    return [4, passthroughHandler.load()];
                case 1:
                    modelArtifacts = _a.sent();
                    expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                    expect(modelArtifacts.weightSpecs).toEqual(undefined);
                    expect(modelArtifacts.weightData).toEqual(undefined);
                    return [2];
            }
        });
    }); });
});
//# sourceMappingURL=passthrough_test.js.map