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
var tf = require("./index");
var jasmine_util_1 = require("./jasmine_util");
var backend_cpu_1 = require("./kernels/backend_cpu");
var backend_webgl_1 = require("./kernels/backend_webgl");
var test_util_1 = require("./test_util");
jasmine_util_1.describeWithFlags('fromPixels + regular math op', test_util_1.WEBGL_ENVS, function () {
    it('debug mode does not error when no nans', function () {
        var pixels = new ImageData(2, 2);
        for (var i = 0; i < 8; i++) {
            pixels.data[i] = 100;
        }
        for (var i = 8; i < 16; i++) {
            pixels.data[i] = 250;
        }
        var a = tf.fromPixels(pixels, 4);
        var b = tf.scalar(20, 'int32');
        var res = tf.add(a, b);
        test_util_1.expectArraysEqual(res, [
            120, 120, 120, 120, 120, 120, 120, 120, 270, 270, 270, 270, 270, 270, 270,
            270
        ]);
    });
});
jasmine_util_1.describeWithFlags('gradients', test_util_1.ALL_ENVS, function () {
    it('matmul + relu', function () {
        var a = tf.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
        var b = tf.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);
        var _a = tf.grads(function (a, b) {
            var m = tf.matMul(a, b);
            var y = tf.relu(m);
            return tf.sum(y);
        })([a, b]), da = _a[0], db = _a[1];
        var dedm = tf.step(tf.matMul(a, b));
        expect(da.shape).toEqual(a.shape);
        var transposeA = false;
        var transposeB = true;
        test_util_1.expectArraysClose(da, tf.matMul(dedm, b, transposeA, transposeB));
        expect(db.shape).toEqual(b.shape);
        transposeA = true;
        transposeB = false;
        test_util_1.expectArraysClose(db, tf.matMul(a, dedm, transposeA, transposeB));
    });
    it('grad(f)', function () {
        var grad = tf.grad(function (x) { return x.square(); });
        var result = grad(tf.tensor1d([.1, .2]));
        test_util_1.expectArraysClose(result, [.2, .4]);
    });
    it('calling grad(f) twice works', function () {
        var grad = tf.grad(function (x) { return x.square(); });
        var result = grad(tf.tensor1d([.1, .2]));
        var result2 = grad(tf.tensor1d([.1, .4]));
        test_util_1.expectArraysClose(result, [.2, .4]);
        test_util_1.expectArraysClose(result2, [.2, .8]);
    });
    it('grads(f)', function () {
        var grads = tf.grads(function (x) { return x.square(); });
        var result = grads([tf.tensor1d([.1, .2])]);
        test_util_1.expectArraysClose(result[0], [.2, .4]);
    });
    it('calling grads(f) twice works', function () {
        var grads = tf.grads(function (x) { return x.square(); });
        var result = grads([tf.tensor1d([.1, .2])]);
        var result2 = grads([tf.tensor1d([.1, .4])]);
        test_util_1.expectArraysClose(result[0], [.2, .4]);
        test_util_1.expectArraysClose(result2[0], [.2, .8]);
    });
    it('works with reshape', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
        var exponent = tf.tensor1d([2, 2, 2, 2], 'int32');
        var da = tf.grad(function (a) {
            var b = a.flatten();
            var m = tf.pow(b, exponent);
            return tf.sum(m);
        })(a);
        expect(da.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(da, [2, 4, 6, 8]);
    });
    it('reshape outside tf.grads() throws error', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
        var b = a.flatten();
        var exponent = tf.tensor1d([2, 2, 2, 2], 'int32');
        var f = function () {
            tf.grads(function (a, b) {
                var m = tf.pow(b, exponent);
                return tf.sum(m);
            })([a, b]);
        };
        expect(f).toThrowError();
    });
    it('does not error if irrelevant (pruned) ops are missing grads', function () {
        var a = tf.tensor1d([true, true], 'bool');
        var b = tf.tensor1d([false, true], 'bool');
        var da = tf.grad(function (a) {
            a.logicalAnd(b);
            return a.sum();
        })(a);
        test_util_1.expectArraysClose(da, [1, 1]);
    });
    it('errors if relevant ops are missing grads', function () {
        var a = tf.tensor1d([true, true], 'bool');
        var b = tf.tensor1d([false, true], 'bool');
        var dfda = tf.grad(function (a) {
            return a.logicalAnd(b);
        });
        expect(function () { return dfda(a); }).toThrowError();
    });
    it('works with asType', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
        var exponent = tf.tensor2d([2, 2, 2, 2], [2, 2], 'int32');
        var da = tf.grad(function (a) {
            var b = a.toFloat();
            var m = tf.pow(b, exponent);
            return tf.sum(m);
        })(a);
        expect(da.shape).toEqual([2, 2]);
        expect(da.dtype).toEqual('float32');
        test_util_1.expectArraysClose(da, [2, 4, 6, 8]);
    });
    it('asType outside of tf.grads() throws error', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
        var b = a.toFloat();
        var exponent = tf.tensor2d([2, 2, 2, 2], [2, 2], 'int32');
        var f = function () {
            tf.grad(function (a) {
                var m = tf.pow(b, exponent);
                return tf.sum(m);
            })(a);
        };
        expect(f).toThrowError();
    });
});
jasmine_util_1.describeWithFlags('valueAndGradients', test_util_1.ALL_ENVS, function () {
    it('matmul + relu', function () {
        var a = tf.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
        var b = tf.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);
        var _a = tf.valueAndGrads(function (a, b) {
            var m = tf.matMul(a, b);
            var y = tf.relu(m);
            return tf.sum(y);
        })([a, b]), value = _a.value, grads = _a.grads;
        test_util_1.expectNumbersClose(value.get(), 10);
        var dedm = tf.step(tf.matMul(a, b));
        var da = grads[0], db = grads[1];
        var transposeA = false;
        var transposeB = true;
        test_util_1.expectArraysClose(da, tf.matMul(dedm, b, transposeA, transposeB));
        transposeA = true;
        transposeB = false;
        test_util_1.expectArraysClose(db, tf.matMul(a, dedm, transposeA, transposeB));
    });
    it('matmul + relu + inner tidy', function () {
        var a = tf.tensor2d([-1, 2, -3, 10, -20, 30], [2, 3]);
        var b = tf.tensor2d([2, -3, 4, -1, 2, -3], [3, 2]);
        var _a = tf.valueAndGrads(function (a, b) {
            var m = tf.matMul(a, b);
            return tf.tidy(function () {
                var y = tf.relu(m);
                return tf.sum(y);
            });
        })([a, b]), value = _a.value, grads = _a.grads;
        test_util_1.expectNumbersClose(value.get(), 10);
        var dedm = tf.step(tf.matMul(a, b));
        var da = grads[0], db = grads[1];
        var transposeA = false;
        var transposeB = true;
        test_util_1.expectArraysClose(da, tf.matMul(dedm, b, transposeA, transposeB));
        transposeA = true;
        transposeB = false;
        test_util_1.expectArraysClose(db, tf.matMul(a, dedm, transposeA, transposeB));
    });
});
jasmine_util_1.describeWithFlags('higher-order gradients', test_util_1.ALL_ENVS, function () {
    it('grad(grad(f))', function () {
        var gradgrad = tf.grad(tf.grad(function (x) { return x.mul(x).mul(x); }));
        var result = gradgrad(tf.tensor1d([.1, .2]));
        test_util_1.expectArraysClose(result, [.6, 1.2]);
    });
    it('grads(grads(f))', function () {
        var grads = tf.grads(function (x) { return x.mul(x).mul(x); });
        var gradsgrads = tf.grads(function (x) { return grads([x])[0]; });
        var result = gradsgrads([tf.tensor1d([.1, .2])]);
        test_util_1.expectArraysClose(result[0], [.6, 1.2]);
    });
});
jasmine_util_1.describeWithFlags('customGradient', test_util_1.ALL_ENVS, function () {
    it('basic', function () {
        var a = tf.scalar(3);
        var b = tf.scalar(2, 'int32');
        var dy = tf.scalar(4);
        var customPow = tf.customGrad(function (a) {
            var value = tf.pow(a, b);
            var gradFunc = function (dy) { return dy.mul(tf.scalar(0.1)); };
            return { value: value, gradFunc: gradFunc };
        });
        var _a = tf.valueAndGrad(function (a) { return customPow(a); })(a, dy), value = _a.value, grad = _a.grad;
        expect(value.shape).toEqual(a.shape);
        test_util_1.expectArraysClose(value, [9]);
        expect(grad.shape).toEqual(a.shape);
        test_util_1.expectArraysClose(grad, [.4]);
    });
    it('second order derivative through customGradient', function () {
        var a = tf.scalar(3);
        var b = tf.scalar(2, 'int32');
        var dy = tf.scalar(5);
        var customPow = tf.customGrad(function (a) {
            var value = tf.pow(a, b);
            var gradFunc = function (dy) { return dy.mul(a); };
            return { value: value, gradFunc: gradFunc };
        });
        var dda = tf.grad(tf.grad(function (a) { return customPow(a); }))(a, dy);
        expect(dda.shape).toEqual(a.shape);
        test_util_1.expectArraysClose(dda, dy);
    });
    it('calling gradient of custom op twice works', function () {
        var customOp = tf.customGrad(function (x) {
            return { value: x.square(), gradFunc: function (dy) { return dy.mul(x.abs()); } };
        });
        var x = tf.tensor1d([-1, -2, 3]);
        var grad = tf.grad(function (x) { return customOp(x); });
        test_util_1.expectArraysClose(grad(x), [1, 2, 3]);
        test_util_1.expectArraysClose(grad(x), [1, 2, 3]);
    });
});
jasmine_util_1.describeWithFlags('memory', test_util_1.ALL_ENVS, function () {
    it('Sum(float)', function () {
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
        var sum = tf.tidy(function () {
            var a = tf.tensor1d([1, 2, 3, 4]);
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4 * 4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        test_util_1.expectArraysClose(sum, [1 + 2 + 3 + 4]);
    });
    it('Sum(bool)', function () {
        var sum = tf.tidy(function () {
            var a = tf.tensor1d([true, true, false, true], 'bool');
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expect(sum.dtype).toBe('int32');
        test_util_1.expectArraysClose(sum, [1 + 1 + 0 + 1]);
    });
    it('Sum(int32)', function () {
        var sum = tf.tidy(function () {
            var a = tf.tensor1d([1, 1, 0, 1], 'int32');
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4 * 4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expect(sum.dtype).toBe('int32');
        test_util_1.expectArraysClose(sum, [1 + 1 + 0 + 1]);
    });
});
jasmine_util_1.describeWithFlags('profile', test_util_1.ALL_ENVS, function () {
    it('squaring', function () { return __awaiter(_this, void 0, void 0, function () {
        var profile, result;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4, tf.profile(function () {
                        var x = tf.tensor1d([1, 2, 3]);
                        var x2 = x.square();
                        x2.dispose();
                        x2 = x.square();
                        x2.dispose();
                        return x;
                    })];
                case 1:
                    profile = _a.sent();
                    result = profile.result;
                    expect(profile.newBytes).toBe(12);
                    expect(profile.peakBytes).toBe(24);
                    expect(profile.newTensors).toBe(1);
                    test_util_1.expectArraysClose(result, [1, 2, 3]);
                    expect(profile.kernels).toEqual([
                        {
                            'name': 'square',
                            'bytesAdded': 12,
                            'totalBytesSnapshot': 24,
                            'tensorsAdded': 1,
                            'totalTensorsSnapshot': 2,
                            'inputShapes': [[3]],
                            'outputShape': [3]
                        },
                        {
                            'name': 'square',
                            'bytesAdded': 12,
                            'totalBytesSnapshot': 24,
                            'tensorsAdded': 1,
                            'totalTensorsSnapshot': 2,
                            'inputShapes': [[3]],
                            'outputShape': [3]
                        }
                    ]);
                    return [2];
            }
        });
    }); });
    it('squaring without disposing', function () { return __awaiter(_this, void 0, void 0, function () {
        var profile, result;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4, tf.profile(function () {
                        var x = tf.tensor1d([1, 2, 3]);
                        var x2 = x.square();
                        return x2;
                    })];
                case 1:
                    profile = _a.sent();
                    result = profile.result;
                    expect(profile.newBytes).toBe(24);
                    expect(profile.peakBytes).toBe(24);
                    expect(profile.newTensors).toBe(2);
                    test_util_1.expectArraysClose(result, [1, 4, 9]);
                    expect(profile.kernels).toEqual([{
                            'name': 'square',
                            'bytesAdded': 12,
                            'totalBytesSnapshot': 24,
                            'tensorsAdded': 1,
                            'totalTensorsSnapshot': 2,
                            'inputShapes': [[3]],
                            'outputShape': [3]
                        }]);
                    return [2];
            }
        });
    }); });
});
jasmine_util_1.describeWithFlags('disposeVariables', test_util_1.ALL_ENVS, function () {
    it('reuse same name variable', function () {
        tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        tf.tensor1d([1, 2, 3]).variable(true, 'v2');
        expect(function () {
            tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        }).toThrowError();
        tf.disposeVariables();
        tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        tf.tensor1d([1, 2, 3]).variable(true, 'v2');
    });
});
describe('Switching cpu backends', function () {
    beforeEach(function () {
        tf.ENV.registerBackend('cpu1', function () { return new backend_cpu_1.MathBackendCPU(); });
        tf.ENV.registerBackend('cpu2', function () { return new backend_cpu_1.MathBackendCPU(); });
    });
    afterEach(function () {
        tf.ENV.removeBackend('cpu1');
        tf.ENV.removeBackend('cpu2');
    });
    it('Move data from cpu1 to cpu2 backend', function () {
        tf.setBackend('cpu1');
        var a = tf.scalar(5);
        tf.setBackend('cpu2');
        var b = tf.scalar(3);
        expect(tf.memory().numDataBuffers).toBe(2);
        expect(tf.memory().numTensors).toBe(2);
        expect(tf.memory().numBytes).toBe(8);
        test_util_1.expectArraysClose(a, [5]);
        test_util_1.expectArraysClose(b, [3]);
        tf.setBackend('cpu1');
        test_util_1.expectArraysClose(a, [5]);
        test_util_1.expectArraysClose(b, [3]);
        tf.dispose([a, b]);
        expect(tf.memory().numDataBuffers).toBe(0);
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
    });
    it('can execute op with data from mixed backends', function () {
        tf.setBackend('cpu1');
        var a = tf.scalar(5);
        tf.setBackend('cpu2');
        var b = tf.scalar(3);
        tf.tidy(function () {
            tf.setBackend('cpu1');
            test_util_1.expectArraysClose(tf.add(a, b), [8]);
            tf.setBackend('cpu2');
            test_util_1.expectArraysClose(tf.add(a, b), [8]);
        });
        expect(tf.memory().numTensors).toBe(2);
        expect(tf.memory().numDataBuffers).toBe(2);
        tf.dispose([a, b]);
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numDataBuffers).toBe(0);
    });
});
jasmine_util_1.describeWithFlags('Switching WebGL + CPU backends', test_util_1.WEBGL_ENVS, function () {
    beforeEach(function () {
        tf.ENV.registerBackend('webgl1', function () { return new backend_webgl_1.MathBackendWebGL(); });
        tf.ENV.registerBackend('webgl2', function () { return new backend_webgl_1.MathBackendWebGL(); });
        tf.ENV.registerBackend('cpu1', function () { return new backend_cpu_1.MathBackendCPU(); });
    });
    afterEach(function () {
        tf.ENV.removeBackend('webgl1');
        tf.ENV.removeBackend('webgl2');
        tf.ENV.removeBackend('cpu1');
    });
    it('can execute op with data from mixed backends', function () {
        tf.setBackend('webgl1');
        var a = tf.scalar(5);
        tf.setBackend('webgl2');
        var b = tf.scalar(3);
        tf.setBackend('cpu1');
        var c = tf.scalar(2);
        tf.tidy(function () {
            tf.setBackend('webgl1');
            test_util_1.expectArraysClose(tf.addN([a, b, c]), [10]);
            tf.setBackend('webgl2');
            test_util_1.expectArraysClose(tf.addN([a, b, c]), [10]);
            tf.setBackend('cpu1');
            test_util_1.expectArraysClose(tf.addN([a, b, c]), [10]);
        });
        expect(tf.memory().numTensors).toBe(3);
        expect(tf.memory().numDataBuffers).toBe(3);
        tf.dispose([a, b, c]);
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numDataBuffers).toBe(0);
    });
    it('fromPixels with mixed backends works', function () {
        tf.setBackend('webgl1');
        var a = tf.fromPixels(new ImageData(new Uint8ClampedArray([1, 2, 3, 4]), 1, 1));
        tf.setBackend('webgl2');
        var b = tf.fromPixels(new ImageData(new Uint8ClampedArray([5, 6, 7, 8]), 1, 1));
        test_util_1.expectArraysClose(tf.add(a, b), [6, 8, 10]);
    });
    it('single tidy multiple backends', function () {
        expect(tf.memory().numTensors).toBe(0);
        tf.tidy(function () {
            tf.setBackend('webgl1');
            var a = tf.scalar(1);
            a.square();
            tf.setBackend('webgl2');
            var b = tf.scalar(1);
            b.square();
            expect(tf.memory().numTensors).toBe(4);
        });
        expect(tf.memory().numTensors).toBe(0);
    });
});
//# sourceMappingURL=engine_test.js.map