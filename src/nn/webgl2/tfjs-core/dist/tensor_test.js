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
var tensor_1 = require("./tensor");
var test_util_1 = require("./test_util");
var types_1 = require("./types");
jasmine_util_1.describeWithFlags('tensor', test_util_1.ALL_ENVS, function () {
    it('Tensors of arbitrary size', function () {
        var t = tf.tensor1d([1, 2, 3]);
        expect(t.rank).toBe(1);
        expect(t.size).toBe(3);
        test_util_1.expectArraysClose(t, [1, 2, 3]);
        expect(t.get(4)).toBeUndefined();
        t = tf.tensor2d([1, 2, 3], [1, 3]);
        expect(t.rank).toBe(2);
        expect(t.size).toBe(3);
        test_util_1.expectArraysClose(t, [1, 2, 3]);
        expect(t.get(0, 4)).toBeUndefined();
        t = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
        expect(t.rank).toBe(2);
        expect(t.size).toBe(6);
        test_util_1.expectArraysClose(t, [1, 2, 3, 4, 5, 6]);
        expect(t.get(5, 3)).toBeUndefined();
        expect(function () { return tf.tensor2d([1], [1, 2]); }).toThrowError();
    });
    it('Tensors of explicit size', function () {
        var t = tf.tensor1d([5, 3, 2]);
        expect(t.rank).toBe(1);
        expect(t.shape).toEqual([3]);
        test_util_1.expectNumbersClose(t.get(1), 3);
        expect(function () { return tf.tensor3d([1, 2], [1, 2, 3, 5]); }).toThrowError();
        var t4 = tf.tensor4d([1, 2, 3, 4], [1, 2, 1, 2]);
        test_util_1.expectNumbersClose(t4.get(0, 0, 0, 0), 1);
        test_util_1.expectNumbersClose(t4.get(0, 0, 0, 1), 2);
        test_util_1.expectNumbersClose(t4.get(0, 1, 0, 0), 3);
        test_util_1.expectNumbersClose(t4.get(0, 1, 0, 1), 4);
        var x = tf.ones([3, 4, 2]);
        expect(x.rank).toBe(3);
        expect(x.size).toBe(24);
        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 4; j++) {
                for (var k = 0; k < 2; k++) {
                    test_util_1.expectNumbersClose(x.get(i, j, k), 1);
                }
            }
        }
        var z = tf.zeros([3, 4, 2]);
        expect(z.rank).toBe(3);
        expect(z.size).toBe(24);
        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 4; j++) {
                for (var k = 0; k < 2; k++) {
                    test_util_1.expectNumbersClose(z.get(i, j, k), 0);
                }
            }
        }
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
        test_util_1.expectNumbersClose(a.get(1, 2), 6);
    });
    it('Tensor dataSync CPU --> GPU', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        test_util_1.expectArraysClose(a.dataSync(), new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('Tensor.data() CPU --> GPU', function () { return __awaiter(_this, void 0, void 0, function () {
        var a, _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    a = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
                    _a = test_util_1.expectArraysClose;
                    return [4, a.data()];
                case 1:
                    _a.apply(void 0, [_b.sent(), new Float32Array([1, 2, 3, 4, 5, 6])]);
                    return [2];
            }
        });
    }); });
    it('Scalar basic methods', function () {
        var a = tf.scalar(5);
        test_util_1.expectNumbersClose(a.get(), 5);
        test_util_1.expectArraysClose(a, [5]);
        expect(a.rank).toBe(0);
        expect(a.size).toBe(1);
        expect(a.shape).toEqual([]);
    });
    it('indexToLoc Scalar', function () {
        var a = tf.scalar(0).buffer();
        expect(a.indexToLoc(0)).toEqual([]);
        var b = tf.zeros([]).buffer();
        expect(b.indexToLoc(0)).toEqual([]);
    });
    it('indexToLoc Tensor1D', function () {
        var a = tf.zeros([3]).buffer();
        expect(a.indexToLoc(0)).toEqual([0]);
        expect(a.indexToLoc(1)).toEqual([1]);
        expect(a.indexToLoc(2)).toEqual([2]);
        var b = tf.zeros([3]).buffer();
        expect(b.indexToLoc(0)).toEqual([0]);
        expect(b.indexToLoc(1)).toEqual([1]);
        expect(b.indexToLoc(2)).toEqual([2]);
    });
    it('indexToLoc Tensor2D', function () {
        var a = tf.zeros([3, 2]).buffer();
        expect(a.indexToLoc(0)).toEqual([0, 0]);
        expect(a.indexToLoc(1)).toEqual([0, 1]);
        expect(a.indexToLoc(2)).toEqual([1, 0]);
        expect(a.indexToLoc(3)).toEqual([1, 1]);
        expect(a.indexToLoc(4)).toEqual([2, 0]);
        expect(a.indexToLoc(5)).toEqual([2, 1]);
        var b = tf.zeros([3, 2]).buffer();
        expect(b.indexToLoc(0)).toEqual([0, 0]);
        expect(b.indexToLoc(1)).toEqual([0, 1]);
        expect(b.indexToLoc(2)).toEqual([1, 0]);
        expect(b.indexToLoc(3)).toEqual([1, 1]);
        expect(b.indexToLoc(4)).toEqual([2, 0]);
        expect(b.indexToLoc(5)).toEqual([2, 1]);
    });
    it('indexToLoc Tensor3D', function () {
        var a = tf.zeros([3, 2, 2]).buffer();
        expect(a.indexToLoc(0)).toEqual([0, 0, 0]);
        expect(a.indexToLoc(1)).toEqual([0, 0, 1]);
        expect(a.indexToLoc(2)).toEqual([0, 1, 0]);
        expect(a.indexToLoc(3)).toEqual([0, 1, 1]);
        expect(a.indexToLoc(4)).toEqual([1, 0, 0]);
        expect(a.indexToLoc(5)).toEqual([1, 0, 1]);
        expect(a.indexToLoc(11)).toEqual([2, 1, 1]);
        var b = tf.zeros([3, 2, 2]).buffer();
        expect(b.indexToLoc(0)).toEqual([0, 0, 0]);
        expect(b.indexToLoc(1)).toEqual([0, 0, 1]);
        expect(b.indexToLoc(2)).toEqual([0, 1, 0]);
        expect(b.indexToLoc(3)).toEqual([0, 1, 1]);
        expect(b.indexToLoc(4)).toEqual([1, 0, 0]);
        expect(b.indexToLoc(5)).toEqual([1, 0, 1]);
        expect(b.indexToLoc(11)).toEqual([2, 1, 1]);
    });
    it('indexToLoc Tensor 5D', function () {
        var values = new Float32Array([1, 2, 3, 4]);
        var a = tensor_1.Tensor.make([2, 1, 1, 1, 2], { values: values }).buffer();
        expect(a.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
        expect(a.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
        expect(a.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
        expect(a.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
    });
    it('locToIndex Scalar', function () {
        var a = tf.scalar(0).buffer();
        expect(a.locToIndex([])).toEqual(0);
        var b = tf.zeros([]).buffer();
        expect(b.locToIndex([])).toEqual(0);
    });
    it('locToIndex Tensor1D', function () {
        var a = tf.zeros([3]).buffer();
        expect(a.locToIndex([0])).toEqual(0);
        expect(a.locToIndex([1])).toEqual(1);
        expect(a.locToIndex([2])).toEqual(2);
        var b = tf.zeros([3]).buffer();
        expect(b.locToIndex([0])).toEqual(0);
        expect(b.locToIndex([1])).toEqual(1);
        expect(b.locToIndex([2])).toEqual(2);
    });
    it('locToIndex Tensor2D', function () {
        var a = tf.zeros([3, 2]).buffer();
        expect(a.locToIndex([0, 0])).toEqual(0);
        expect(a.locToIndex([0, 1])).toEqual(1);
        expect(a.locToIndex([1, 0])).toEqual(2);
        expect(a.locToIndex([1, 1])).toEqual(3);
        expect(a.locToIndex([2, 0])).toEqual(4);
        expect(a.locToIndex([2, 1])).toEqual(5);
        var b = tf.zeros([3, 2]).buffer();
        expect(b.locToIndex([0, 0])).toEqual(0);
        expect(b.locToIndex([0, 1])).toEqual(1);
        expect(b.locToIndex([1, 0])).toEqual(2);
        expect(b.locToIndex([1, 1])).toEqual(3);
        expect(b.locToIndex([2, 0])).toEqual(4);
        expect(b.locToIndex([2, 1])).toEqual(5);
    });
    it('locToIndex Tensor3D', function () {
        var a = tf.zeros([3, 2, 2]).buffer();
        expect(a.locToIndex([0, 0, 0])).toEqual(0);
        expect(a.locToIndex([0, 0, 1])).toEqual(1);
        expect(a.locToIndex([0, 1, 0])).toEqual(2);
        expect(a.locToIndex([0, 1, 1])).toEqual(3);
        expect(a.locToIndex([1, 0, 0])).toEqual(4);
        expect(a.locToIndex([1, 0, 1])).toEqual(5);
        expect(a.locToIndex([2, 1, 1])).toEqual(11);
        var b = tf.zeros([3, 2, 2]).buffer();
        expect(b.locToIndex([0, 0, 0])).toEqual(0);
        expect(b.locToIndex([0, 0, 1])).toEqual(1);
        expect(b.locToIndex([0, 1, 0])).toEqual(2);
        expect(b.locToIndex([0, 1, 1])).toEqual(3);
        expect(b.locToIndex([1, 0, 0])).toEqual(4);
        expect(b.locToIndex([1, 0, 1])).toEqual(5);
        expect(b.locToIndex([2, 1, 1])).toEqual(11);
    });
    it('Tensor assignability (asserts compiler)', function () {
        var a = null;
        var b = a;
        expect(b).toBeNull();
        var a1 = null;
        var b1 = a1;
        expect(b1).toBeNull();
        var a2 = null;
        var b2 = a2;
        expect(b2).toBeNull();
        var a3 = null;
        var b3 = a3;
        expect(b3).toBeNull();
        var a4 = null;
        var b4 = a4;
        expect(b4).toBeNull();
    });
    it('tf.tensor1d() from number[]', function () {
        var a = tf.tensor1d([1, 2, 3]);
        test_util_1.expectArraysClose(a, [1, 2, 3]);
    });
    it('tf.tensor1d() throw error with null input value', function () {
        expect(function () { return tf.tensor1d(null); })
            .toThrowError('The input to the tensor constructor ' +
            'must be a non-null value.');
    });
    it('tf.tensor1d() from number[][], shape mismatch', function () {
        expect(function () { return tf.tensor1d([[1], [2], [3]]); }).toThrowError();
    });
    it('tf.tensor2d() from number[][]', function () {
        var a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
    });
    it('tf.tensor2d() requires shape to be of length 2', function () {
        var shape = [4];
        expect(function () { return tf.tensor2d([1, 2, 3, 4], shape); }).toThrowError();
    });
    it('tf.tensor2d() from number[][], but shape does not match', function () {
        expect(function () { return tf.tensor2d([[1, 2, 3], [4, 5, 6]], [3, 2]); }).toThrowError();
    });
    it('tf.tensor2d() from number[], but no shape throws error', function () {
        expect(function () { return tf.tensor2d([1, 2, 3, 4]); }).toThrowError();
    });
    it('tf.tensor2d() throw error with null input value', function () {
        expect(function () { return tf.tensor2d(null); })
            .toThrowError('The input to the tensor constructor ' +
            'must be a non-null value.');
    });
    it('tensor3d() from number[][][]', function () {
        var a = tf.tensor3d([[[1], [2], [3]], [[4], [5], [6]]], [2, 3, 1]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4, 5, 6]);
    });
    it('tensor3d() from number[][][], but shape does not match', function () {
        var values = [[[1], [2], [3]], [[4], [5], [6]]];
        expect(function () { return tf.tensor3d(values, [3, 2, 1]); }).toThrowError();
    });
    it('tf.tensor3d() from number[], but no shape throws error', function () {
        expect(function () { return tf.tensor3d([1, 2, 3, 4]); }).toThrowError();
    });
    it('tf.tensor3d() requires shape to be of length 3', function () {
        var shape = [4, 1];
        expect(function () { return tf.tensor3d([1, 2, 3, 4], shape); }).toThrowError();
    });
    it('tf.tensor3d() throw error with null input value', function () {
        expect(function () { return tf.tensor3d(null); })
            .toThrowError('The input to the tensor constructor ' +
            'must be a non-null value.');
    });
    it('tensor4d() from number[][][][]', function () {
        var a = tf.tensor4d([[[[1]], [[2]]], [[[4]], [[5]]]], [2, 2, 1, 1]);
        test_util_1.expectArraysClose(a, [1, 2, 4, 5]);
    });
    it('tensor4d() from number[][][][], but shape does not match', function () {
        var f = function () {
            tf.tensor4d([[[[1]], [[2]]], [[[4]], [[5]]]], [2, 1, 2, 1]);
        };
        expect(f).toThrowError();
    });
    it('tf.tensor4d() from number[], but no shape throws error', function () {
        expect(function () { return tf.tensor4d([1, 2, 3, 4]); }).toThrowError();
    });
    it('tf.tensor4d() requires shape to be of length 4', function () {
        var shape = [4, 1];
        expect(function () { return tf.tensor4d([1, 2, 3, 4], shape); }).toThrowError();
    });
    it('tf.tensor4d() throw error with null input value', function () {
        expect(function () { return tf.tensor4d(null); })
            .toThrowError('The input to the tensor constructor ' +
            'must be a non-null value.');
    });
    it('tf.tensor5d() throw error with null input value', function () {
        expect(function () { return tf.tensor5d(null); })
            .toThrowError('The input to the tensor constructor ' +
            'must be a non-null value.');
    });
    it('tf.tensor6d() throw error with null input value', function () {
        expect(function () { return tf.tensor6d(null); })
            .toThrowError('The input to the tensor constructor ' +
            'must be a non-null value.');
    });
    it('default dtype', function () {
        var a = tf.scalar(3);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [3]);
    });
    it('float32 dtype', function () {
        var a = tf.scalar(3, 'float32');
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [3]);
    });
    it('int32 dtype', function () {
        var a = tf.scalar(3, 'int32');
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [3]);
    });
    it('int32 dtype, 3.9 => 3, like numpy', function () {
        var a = tf.scalar(3.9, 'int32');
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [3]);
    });
    it('int32 dtype, -3.9 => -3, like numpy', function () {
        var a = tf.scalar(-3.9, 'int32');
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [-3]);
    });
    it('bool dtype, 3 => true, like numpy', function () {
        var a = tf.scalar(3, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(1);
    });
    it('bool dtype, -2 => true, like numpy', function () {
        var a = tf.scalar(-2, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(1);
    });
    it('bool dtype, 0 => false, like numpy', function () {
        var a = tf.scalar(0, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(0);
    });
    it('bool dtype from boolean', function () {
        var a = tf.scalar(false, 'bool');
        expect(a.get()).toBe(0);
        expect(a.dtype).toBe('bool');
        var b = tf.scalar(true, 'bool');
        expect(b.get()).toBe(1);
        expect(b.dtype).toBe('bool');
    });
    it('int32 dtype from boolean', function () {
        var a = tf.scalar(true, 'int32');
        expect(a.get()).toBe(1);
        expect(a.dtype).toBe('int32');
    });
    it('default dtype from boolean', function () {
        var a = tf.scalar(false);
        test_util_1.expectNumbersClose(a.get(), 0);
        expect(a.dtype).toBe('float32');
    });
    it('default dtype', function () {
        var a = tf.tensor1d([1, 2, 3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [1, 2, 3]);
    });
    it('float32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [1, 2, 3]);
    });
    it('int32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysEqual(a, [1, 2, 3]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = tf.tensor1d([1.1, 2.5, 3.9], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysEqual(a, [1, 2, 3]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = tf.tensor1d([-1.1, -2.5, -3.9], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysEqual(a, [-1, -2, -3]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = tf.tensor1d([1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([4]);
        expect(a.get(0)).toBe(1);
        expect(a.get(1)).toBe(1);
        expect(a.get(2)).toBe(0);
        expect(a.get(3)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = tf.tensor1d([false, false, true]);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [0, 0, 1]);
    });
    it('float32 dtype from boolean[]', function () {
        var a = tf.tensor1d([false, false, true], 'float32');
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [0, 0, 1]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = tf.tensor1d([false, false, true], 'int32');
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [0, 0, 1]);
    });
    it('bool dtype from boolean[]', function () {
        var a = tf.tensor1d([false, false, true], 'bool');
        expect(a.dtype).toBe('bool');
        test_util_1.expectArraysEqual(a, [0, 0, 1]);
    });
    it('default dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('float32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('int32 dtype', function () {
        var a = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = tf.tensor2d([1.1, 2.5, 3.9, 4.0], [2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = tf.tensor2d([-1.1, -2.5, -3.9, -4.0], [2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(a, [-1, -2, -3, -4]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = tf.tensor2d([1, -2, 0, 3], [2, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2]);
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(0);
        expect(a.get(1, 1)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = tf.tensor2d([[false, false], [true, false]], [2, 2]);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('float32 dtype from boolean[]', function () {
        var a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'float32');
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('bool dtype from boolean[]', function () {
        var a = tf.tensor2d([[false, false], [true, false]], [2, 2], 'bool');
        expect(a.dtype).toBe('bool');
        test_util_1.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('default dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('float32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('int32 dtype', function () {
        var a = tf.tensor3d([[[1], [2]], [[3], [4]]], [2, 2, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = tf.tensor3d([1.1, 2.5, 3.9, 4.0], [2, 2, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = tf.tensor3d([-1.1, -2.5, -3.9, -4.0], [2, 2, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(a, [-1, -2, -3, -4]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = tf.tensor3d([1, -2, 0, 3], [2, 2, 1], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.get(0, 0, 0)).toBe(1);
        expect(a.get(0, 1, 0)).toBe(1);
        expect(a.get(1, 0, 0)).toBe(0);
        expect(a.get(1, 1, 0)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1]);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('float32 dtype from boolean[]', function () {
        var a = tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1], 'float32');
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1], 'int32');
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('bool dtype from boolean[]', function () {
        var a = tf.tensor3d([[[false], [false]], [[true], [false]]], [2, 2, 1], 'bool');
        expect(a.dtype).toBe('bool');
        test_util_1.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('default dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('float32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('int32 dtype', function () {
        var a = tf.tensor4d([[[[1]], [[2]]], [[[3]], [[4]]]], [2, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = tf.tensor4d([1.1, 2.5, 3.9, 4.0], [2, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(a, [1, 2, 3, 4]);
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = tf.tensor4d([-1.1, -2.5, -3.9, -4.0], [2, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(a, [-1, -2, -3, -4]);
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = tf.tensor4d([1, -2, 0, 3], [2, 2, 1, 1], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.get(0, 0, 0, 0)).toBe(1);
        expect(a.get(0, 1, 0, 0)).toBe(1);
        expect(a.get(1, 0, 0, 0)).toBe(0);
        expect(a.get(1, 1, 0, 0)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = tf.tensor4d([[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1]);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('float32 dtype from boolean[]', function () {
        var a = tf.tensor4d([[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'float32');
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [0, 0, 1, 0]);
    });
    it('int32 dtype from boolean[]', function () {
        var a = tf.tensor4d([[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'int32');
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('bool dtype from boolean[]', function () {
        var a = tf.tensor4d([[[[false], [false]], [[true], [false]]]], [1, 2, 2, 1], 'bool');
        expect(a.dtype).toBe('bool');
        test_util_1.expectArraysEqual(a, [0, 0, 1, 0]);
    });
    it('Scalar default dtype', function () {
        var a = tf.scalar(4);
        var b = a.reshape([1, 1]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 1]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Scalar float32 dtype', function () {
        var a = tf.scalar(4, 'float32');
        var b = a.reshape([1, 1]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 1]);
    });
    it('Scalar int32 dtype', function () {
        var a = tf.scalar(4, 'int32');
        var b = a.reshape([1, 1]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([1, 1]);
    });
    it('Scalar bool dtype', function () {
        var a = tf.scalar(4, 'bool');
        var b = a.reshape([1, 1, 1]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([1, 1, 1]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Scalar complex64 dtype', function () {
        var a = tf.complex(4, 5);
        var b = a.reshape([1, 1]);
        test_util_1.expectArraysClose(a, [4, 5]);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([1, 1]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor1D default dtype', function () {
        var a = tf.tensor1d([1, 2, 3, 4]);
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor1D float32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3, 4], 'float32');
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('Tensor1D int32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3, 4], 'int32');
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor1D complex64 dtype', function () {
        var a = tf.complex([1, 3, 5, 7], [2, 4, 6, 8]);
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor2D default dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor2D float32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3], 'float32');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
    });
    it('Tensor2D int32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3], 'int32');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([6]);
    });
    it('Tensor2D bool dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3], 'bool');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([6]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor2D complex64 dtype', function () {
        var a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([6]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor3D default dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor3D float32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1], 'float32');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
    });
    it('Tensor3D int32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1], 'int32');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([6]);
    });
    it('Tensor3D bool dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6], [2, 3, 1], 'bool');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([6]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor3D complex64 dtype', function () {
        var a = tf.complex([[[1], [3], [5]], [[7], [9], [11]]], [[[2], [4], [6]], [[8], [10], [12]]]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([6]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor4D default dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1]);
        var b = a.reshape([2, 3]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 3]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor4D float32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1], 'float32');
        var b = a.reshape([2, 3]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 3]);
    });
    it('Tensor4D int32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1], 'int32');
        var b = a.reshape([3, 2]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor4D complex64 dtype', function () {
        var a = tf.complex([[[[1]], [[3]], [[5]]], [[[7]], [[9]], [[11]]]], [[[[2]], [[4]], [[6]]], [[[8]], [[10]], [[12]]]]);
        var b = a.reshape([3, 2]);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
    });
    it('Tensor4D bool dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4, 5, 6], [2, 3, 1, 1], 'bool');
        var b = a.reshape([3, 2]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3, 2]);
    });
    it('reshape is functional', function () {
        var a = tf.scalar(2.4);
        var b = a.reshape([]);
        expect(a.id).not.toBe(b.id);
        b.dispose();
        test_util_1.expectArraysClose(a, [2.4]);
    });
    it('reshape throws when passed a non-tensor', function () {
        expect(function () { return tf.reshape({}, []); })
            .toThrowError(/Argument 'x' passed to 'reshape' must be a Tensor/);
    });
    it('reshape accepts a tensor-like object', function () {
        var res = tf.reshape([[1, 2, 3], [4, 5, 6]], [3, 2]);
        expect(res.dtype).toBe('float32');
        expect(res.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('cast bool -> bool', function () {
        var a = tf.tensor1d([1, 0], 'bool');
        expect(a.cast('bool').dtype).toEqual('bool');
    });
    it('cast bool -> int32', function () {
        var a = tf.tensor1d([1, 0], 'bool');
        expect(a.cast('int32').dtype).toEqual('int32');
    });
    it('cast bool -> float32', function () {
        var a = tf.tensor1d([1, 0], 'bool');
        expect(a.cast('float32').dtype).toEqual('float32');
    });
    it('cast int32 -> bool', function () {
        var a = tf.tensor1d([1, 0], 'int32');
        expect(a.cast('bool').dtype).toEqual('bool');
    });
    it('cast int32 -> int32', function () {
        var a = tf.tensor1d([1, 2], 'int32');
        expect(a.cast('int32').dtype).toEqual('int32');
    });
    it('cast int32 -> float32', function () {
        var a = tf.tensor1d([1, 2], 'int32');
        expect(a.cast('float32').dtype).toEqual('float32');
    });
    it('cast float32 -> bool', function () {
        var a = tf.tensor1d([1.0, 0.0]);
        expect(a.cast('bool').dtype).toEqual('bool');
    });
    it('cast float32 -> int32', function () {
        var a = tf.tensor1d([1.0, 2.0]);
        expect(a.cast('int32').dtype).toEqual('int32');
    });
    it('cast float32 -> int32. async download', function () { return __awaiter(_this, void 0, void 0, function () {
        var a, aInt, asyncData;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    a = tf.tensor1d([1, 2]);
                    aInt = a.cast('int32');
                    expect(aInt.dtype).toEqual('int32');
                    return [4, aInt.data()];
                case 1:
                    asyncData = _a.sent();
                    expect(asyncData instanceof Int32Array).toEqual(true);
                    return [2];
            }
        });
    }); });
    it('cast float32 -> int32. queued async download', function () { return __awaiter(_this, void 0, void 0, function () {
        var a, aInt, _a, first, second;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    a = tf.tensor1d([1, 2]);
                    aInt = a.cast('int32');
                    expect(aInt.dtype).toEqual('int32');
                    return [4, Promise.all([aInt.data(), aInt.data()])];
                case 1:
                    _a = _b.sent(), first = _a[0], second = _a[1];
                    expect(first instanceof Int32Array).toEqual(true);
                    expect(second instanceof Int32Array).toEqual(true);
                    return [2];
            }
        });
    }); });
    it('cast float32 -> int32. sync download', function () {
        var a = tf.tensor1d([1, 2]).cast('int32');
        expect(a.dtype).toEqual('int32');
        var data = a.dataSync();
        expect(data instanceof Int32Array).toEqual(true);
    });
    it('cast float32 -> float32', function () {
        var a = tf.tensor1d([1.0, 2.0]);
        expect(a.cast('float32').dtype).toEqual('float32');
    });
    it('cast complex64 -> float32', function () {
        var a = tf.complex([1.0, 2.0], [3.0, 4.0]);
        var result = a.cast('float32');
        expect(result.dtype).toEqual('float32');
        test_util_1.expectArraysClose(result, [1.0, 2.0]);
    });
    it('cast complex64 -> int32', function () {
        var a = tf.complex([1.0, 2.0], [3.0, 4.0]);
        var result = a.cast('int32');
        expect(result.dtype).toEqual('int32');
        test_util_1.expectArraysEqual(result, [1, 2]);
    });
    it('cast complex64 -> bool', function () {
        var a = tf.complex([1.0, 0.0], [1.0, 1.0]);
        var result = a.cast('bool');
        expect(result.dtype).toEqual('bool');
        test_util_1.expectArraysEqual(result, [true, false]);
    });
    it('cast throws when passed a non-tensor', function () {
        expect(function () { return tf.cast({}, 'float32'); })
            .toThrowError(/Argument 'x' passed to 'cast' must be a Tensor/);
    });
    it('cast accepts a tensor-like object', function () {
        var a = [1.0, 2.0];
        var res = tf.cast(a, 'int32');
        expect(res.dtype).toEqual('int32');
        test_util_1.expectArraysClose(res, [1, 2]);
    });
    it('scalar bool -> int32', function () {
        var a = tf.scalar(true, 'bool').toInt();
        expect(a.dtype).toBe('int32');
        expect(a.get()).toBe(1);
    });
    it('Tensor1D float32 -> int32', function () {
        var a = tf.tensor1d([1.1, 3.9, -2.9, 0]).toInt();
        expect(a.dtype).toBe('int32');
        test_util_1.expectArraysEqual(a, [1, 3, -2, 0]);
    });
    it('Tensor2D float32 -> bool', function () {
        var a = tf.tensor2d([1.1, 3.9, -2.9, 0], [2, 2]).asType(types_1.DType.bool);
        expect(a.dtype).toBe('bool');
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(1);
        expect(a.get(1, 1)).toBe(0);
    });
    it('Tensor2D int32 -> bool', function () {
        var a = tf.tensor2d([1, 3, 0, -1], [2, 2], 'int32').toBool();
        expect(a.dtype).toBe('bool');
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(0);
        expect(a.get(1, 1)).toBe(1);
    });
    it('Tensor3D bool -> float32', function () {
        var a = tf.tensor3d([true, false, false, true], [2, 2, 1], 'bool').toFloat();
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [1, 0, 0, 1]);
    });
    it('bool CPU -> GPU -> CPU', function () {
        var a = tf.tensor1d([1, 2, 0, 0, 5], 'bool');
        test_util_1.expectArraysEqual(a, [1, 1, 0, 0, 1]);
    });
    it('int32 CPU -> GPU -> CPU', function () {
        var a = tf.tensor1d([1, 2, 0, 0, 5], 'int32');
        test_util_1.expectArraysEqual(a, [1, 2, 0, 0, 5]);
    });
    it('asType is functional', function () {
        var a = tf.scalar(2.4, 'float32');
        var b = a.toFloat();
        expect(a.id).not.toBe(b.id);
        b.dispose();
        test_util_1.expectArraysClose(a, [2.4]);
    });
    it('squeeze no axis', function () {
        var a = tf.tensor2d([4, 2, 1], [3, 1], 'bool');
        var b = a.squeeze();
        expect(b.shape).toEqual([3]);
    });
    it('squeeze with axis', function () {
        var a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
        var b = a.squeeze([1]);
        expect(b.shape).toEqual([3, 1]);
    });
    it('squeeze wrong axis', function () {
        var a = tf.tensor3d([4, 2, 1], [3, 1, 1], 'bool');
        expect(function () { return a.squeeze([0, 1]); }).toThrowError();
    });
    it('squeeze throws when passed a non-tensor', function () {
        expect(function () { return tf.squeeze({}); })
            .toThrowError(/Argument 'x' passed to 'squeeze' must be a Tensor/);
    });
    it('squeeze accepts a tensor-like object', function () {
        var res = tf.squeeze([[[4]], [[2]], [[1]]]);
        expect(res.shape).toEqual([3]);
        test_util_1.expectArraysClose(res, [4, 2, 1]);
    });
    it('squeeze a zero-sized tensor', function () {
        var a = tf.tensor3d([], [0, 1, 0]);
        var res = tf.squeeze(a);
        expect(res.shape).toEqual([0, 0]);
    });
    it('squeeze a complex64 tensor', function () {
        var a = tf.complex([[4], [1], [5]], [[2], [3], [6]]);
        var b = a.squeeze();
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [4, 2, 1, 3, 5, 6]);
    });
    it('scalar -> 2d', function () {
        var a = tf.scalar(4, 'int32');
        var b = a.as2D(1, 1);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([1, 1]);
    });
    it('1d -> 2d', function () {
        var a = tf.tensor1d([4, 2, 1], 'bool');
        var b = a.as2D(3, 1);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3, 1]);
    });
    it('2d -> 4d', function () {
        var a = tf.tensor2d([4, 2, 1, 3], [2, 2]);
        var b = a.as4D(1, 1, 2, 2);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 1, 2, 2]);
    });
    it('3d -> 2d', function () {
        var a = tf.tensor3d([4, 2, 1, 3], [2, 2, 1], 'float32');
        var b = a.as2D(2, 2);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('4d -> 1d', function () {
        var a = tf.tensor4d([4, 2, 1, 3], [2, 2, 1, 1], 'bool');
        var b = a.as1D();
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([4]);
    });
});
describe('tensor.toString', function () {
    it('scalar verbose', function () {
        var verbose = true;
        var str = tf.scalar(5).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: float32\n' +
            '  rank: 0\n' +
            '  shape: []\n' +
            '  values:\n' +
            '    5');
    });
    it('1d tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([4]).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: float32\n' +
            '  rank: 1\n' +
            '  shape: [4]\n' +
            '  values:\n' +
            '    [0, 0, 0, 0]');
    });
    it('2d tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([3, 3]).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: float32\n' +
            '  rank: 2\n' +
            '  shape: [3,3]\n' +
            '  values:\n' +
            '    [[0, 0, 0],\n' +
            '     [0, 0, 0],\n' +
            '     [0, 0, 0]]');
    });
    it('3d tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([3, 3, 2]).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: float32\n' +
            '  rank: 3\n' +
            '  shape: [3,3,2]\n' +
            '  values:\n' +
            '    [[[0, 0],\n' +
            '      [0, 0],\n' +
            '      [0, 0]],\n\n' +
            '     [[0, 0],\n' +
            '      [0, 0],\n' +
            '      [0, 0]],\n\n' +
            '     [[0, 0],\n' +
            '      [0, 0],\n' +
            '      [0, 0]]]');
    });
    it('1d long tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([100]).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: float32\n' +
            '  rank: 1\n' +
            '  shape: [100]\n' +
            '  values:\n' +
            '    [0, 0, 0, ..., 0, 0, 0]');
    });
    it('2d long tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([100, 100]).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: float32\n' +
            '  rank: 2\n' +
            '  shape: [100,100]\n' +
            '  values:\n' +
            '    [[0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     ...,\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0]]');
    });
    it('2d with padding to align columns verbose', function () {
        var verbose = true;
        var str = tf.tensor([
            [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
            [1.991, 0.0640865, 0.2983858]
        ]).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: float32\n' +
            '  rank: 2\n' +
            '  shape: [3,3]\n' +
            '  values:\n' +
            '    [[0.8597712, 3        , 0.2740789],\n' +
            '     [0.6696132, 0.4825962, 2.75     ],\n' +
            '     [1.9910001, 0.0640865, 0.2983858]]');
    });
    it('scalar', function () {
        var str = tf.scalar(5).toString();
        expect(str).toEqual('Tensor\n' +
            '    5');
    });
    it('1d tensor', function () {
        var str = tf.zeros([4]).toString();
        expect(str).toEqual('Tensor\n' +
            '    [0, 0, 0, 0]');
    });
    it('2d tensor', function () {
        var str = tf.zeros([3, 3]).toString();
        expect(str).toEqual('Tensor\n' +
            '    [[0, 0, 0],\n' +
            '     [0, 0, 0],\n' +
            '     [0, 0, 0]]');
    });
    it('3d tensor', function () {
        var str = tf.zeros([3, 3, 2]).toString();
        expect(str).toEqual('Tensor\n' +
            '    [[[0, 0],\n' +
            '      [0, 0],\n' +
            '      [0, 0]],\n\n' +
            '     [[0, 0],\n' +
            '      [0, 0],\n' +
            '      [0, 0]],\n\n' +
            '     [[0, 0],\n' +
            '      [0, 0],\n' +
            '      [0, 0]]]');
    });
    it('1d long tensor', function () {
        var str = tf.zeros([100]).toString();
        expect(str).toEqual('Tensor\n' +
            '    [0, 0, 0, ..., 0, 0, 0]');
    });
    it('2d long tensor', function () {
        var str = tf.zeros([100, 100]).toString();
        expect(str).toEqual('Tensor\n' +
            '    [[0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     ...,\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0],\n' +
            '     [0, 0, 0, ..., 0, 0, 0]]');
    });
    it('2d with padding to align columns', function () {
        var str = tf.tensor([
            [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
            [1.991, 0.0640865, 0.2983858]
        ]).toString();
        expect(str).toEqual('Tensor\n' +
            '    [[0.8597712, 3        , 0.2740789],\n' +
            '     [0.6696132, 0.4825962, 2.75     ],\n' +
            '     [1.9910001, 0.0640865, 0.2983858]]');
    });
    it('scalar complex64 verbose', function () {
        var verbose = true;
        var str = tf.complex(5, 6).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: complex64\n' +
            '  rank: 0\n' +
            '  shape: []\n' +
            '  values:\n' +
            '    5 + 6j');
    });
    it('1d complex64 tensor verbose', function () {
        var verbose = true;
        var str = tf.complex([3, 5], [4, 6]).toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: complex64\n' +
            '  rank: 1\n' +
            '  shape: [2]\n' +
            '  values:\n' +
            '    [3 + 4j, 5 + 6j]');
    });
    it('2d complex64 tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([3, 3], 'complex64').toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: complex64\n' +
            '  rank: 2\n' +
            '  shape: [3,3]\n' +
            '  values:\n' +
            '    [[0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j]]');
    });
    it('3d complex64 tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([3, 3, 2], 'complex64').toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: complex64\n' +
            '  rank: 3\n' +
            '  shape: [3,3,2]\n' +
            '  values:\n' +
            '    [[[0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j]],\n\n' +
            '     [[0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j]],\n\n' +
            '     [[0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j]]]');
    });
    it('1d long tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([100], 'complex64').toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: complex64\n' +
            '  rank: 1\n' +
            '  shape: [100]\n' +
            '  values:\n' +
            '    [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]');
    });
    it('2d long tensor verbose', function () {
        var verbose = true;
        var str = tf.zeros([100, 100], 'complex64').toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: complex64\n' +
            '  rank: 2\n' +
            '  shape: [100,100]\n' +
            '  values:\n' +
            '    [[0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     ...,\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]]');
    });
    it('2d complex64 with padding to align columns verbose', function () {
        var verbose = true;
        var str = tf.complex([
            [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
            [1.991, 0.0640865, 0.2983858]
        ], [[1, 1.0102332, 3], [2, 5, 2.34424], [1.23, 2, 0.123]])
            .toString(verbose);
        expect(str).toEqual('Tensor\n' +
            '  dtype: complex64\n' +
            '  rank: 2\n' +
            '  shape: [3,3]\n' +
            '  values:\n' +
            '    [[0.8597712 + 1j   , 3 + 1.0102332j, 0.2740789 + 3j    ],\n' +
            '     [0.6696132 + 2j   , 0.4825962 + 5j, 2.75 + 2.34424j   ],\n' +
            '     [1.9910001 + 1.23j, 0.0640865 + 2j, 0.2983858 + 0.123j]]');
    });
    it('scalar complex64', function () {
        var str = tf.complex(5, 4).toString();
        expect(str).toEqual('Tensor\n' +
            '    5 + 4j');
    });
    it('1d complex64 tensor', function () {
        var str = tf.zeros([4], 'complex64').toString();
        expect(str).toEqual('Tensor\n' +
            '    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]');
    });
    it('2d complex64 tensor', function () {
        var str = tf.zeros([3, 3], 'complex64').toString();
        expect(str).toEqual('Tensor\n' +
            '    [[0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j]]');
    });
    it('3d complex64 tensor', function () {
        var str = tf.zeros([3, 3, 2], 'complex64').toString();
        expect(str).toEqual('Tensor\n' +
            '    [[[0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j]],\n\n' +
            '     [[0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j]],\n\n' +
            '     [[0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j],\n' +
            '      [0 + 0j, 0 + 0j]]]');
    });
    it('1d long complex64 tensor', function () {
        var str = tf.zeros([100], 'complex64').toString();
        expect(str).toEqual('Tensor\n' +
            '    [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]');
    });
    it('2d long complex64 tensor', function () {
        var str = tf.zeros([100, 100], 'complex64').toString();
        expect(str).toEqual('Tensor\n' +
            '    [[0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     ...,\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j],\n' +
            '     [0 + 0j, 0 + 0j, 0 + 0j, ..., 0 + 0j, 0 + 0j, 0 + 0j]]');
    });
    it('2d complex64 with padding to align columns', function () {
        var str = tf.complex([
            [0.8597712, 3, 0.2740789], [0.6696132, 0.4825962, 2.75],
            [1.991, 0.0640865, 0.2983858]
        ], [[1, 1.0102332, 3], [2, 5, 2.34424], [1.23, 2, 0.123]])
            .toString();
        expect(str).toEqual('Tensor\n' +
            '    [[0.8597712 + 1j   , 3 + 1.0102332j, 0.2740789 + 3j    ],\n' +
            '     [0.6696132 + 2j   , 0.4825962 + 5j, 2.75 + 2.34424j   ],\n' +
            '     [1.9910001 + 1.23j, 0.0640865 + 2j, 0.2983858 + 0.123j]]');
    });
});
jasmine_util_1.describeWithFlags('tensor grad', test_util_1.ALL_ENVS, function () {
    it('grad with second derivative', function () {
        var f = function (x) { return x.pow(tf.scalar(3, 'int32')); };
        var g = tf.grad(f);
        var gg = tf.grad(g);
        var x = tf.tensor1d([2, 3]);
        var data = gg(x);
        test_util_1.expectArraysClose(data, [12, 18]);
    });
});
jasmine_util_1.describeWithFlags('tensor.data', test_util_1.ALL_ENVS, function () {
    it('interleaving .data() and .dataSync()', function () { return __awaiter(_this, void 0, void 0, function () {
        var a, b, ra, rb, _a, _b, _c;
        return __generator(this, function (_d) {
            switch (_d.label) {
                case 0:
                    a = tf.tensor1d([1, 2, 3]);
                    b = tf.tensor1d([4, 5, 6]);
                    ra = a.square().data();
                    rb = b.square().dataSync();
                    test_util_1.expectArraysClose(a, [1, 2, 3]);
                    test_util_1.expectArraysClose(b, [4, 5, 6]);
                    test_util_1.expectArraysClose(Array.from(rb), [16, 25, 36]);
                    _a = test_util_1.expectArraysClose;
                    _c = (_b = Array).from;
                    return [4, ra];
                case 1:
                    _a.apply(void 0, [_c.apply(_b, [_d.sent()]), [1, 4, 9]]);
                    return [2];
            }
        });
    }); });
    it('.data() postpones disposal of tensor', function (done) {
        expect(tf.memory().numTensors).toBe(0);
        tf.tidy(function () {
            var a = tf.scalar(5);
            expect(tf.memory().numTensors).toBe(1);
            a.square();
            a.data().then(function (vals) {
                test_util_1.expectNumbersClose(vals[0], 5);
            });
        });
        setTimeout(function () {
            expect(tf.memory().numTensors).toBe(0);
            done();
        });
    });
    it('calling .data() twice works (2 subscribers to a single read)', function (done) {
        tf.tidy(function () {
            var a = tf.scalar(5);
            a.square();
            a.data().then(function (vals) {
                test_util_1.expectNumbersClose(vals[0], 5);
            });
            a.data()
                .then(function (vals) {
                test_util_1.expectNumbersClose(vals[0], 5);
            })
                .then(done);
        });
    });
});
jasmine_util_1.describeWithFlags('x instanceof Tensor', test_util_1.ALL_ENVS, function () {
    it('x: Tensor', function () {
        var t = tf.scalar(1);
        expect(t instanceof tensor_1.Tensor).toBe(true);
    });
    it('x: Tensor-like', function () {
        var t = { shape: [2], dtype: 'float32' };
        expect(t instanceof tensor_1.Tensor).toBe(true);
    });
    it('x: other object, fails', function () {
        var t = { something: 'else' };
        expect(t instanceof tensor_1.Tensor).toBe(false);
    });
    it('x: undefined or null, fails', function () {
        expect(undefined instanceof tensor_1.Tensor).toBe(false);
        expect(null instanceof tensor_1.Tensor).toBe(false);
    });
});
jasmine_util_1.describeWithFlags('tensor with 0 in shape', test_util_1.ALL_ENVS, function () {
    it('1d of shape [0]', function () {
        var a = tf.tensor1d([]);
        expect(a.dtype).toBe('float32');
        expect(a.rank).toBe(1);
        expect(a.shape).toEqual([0]);
        test_util_1.expectArraysEqual(a, []);
    });
    it('1d throws when values are not empty', function () {
        var values = new Float32Array([1, 2, 3]);
        expect(function () { return tensor_1.Tensor.make([0], { values: values }, 'float32'); })
            .toThrowError('Based on the provided shape, [0], and dtype float32, the tensor ' +
            'should have 0 values but has 3');
    });
    it('2d of shape [0, 5]', function () {
        var a = tf.tensor2d([], [0, 5]);
        expect(a.dtype).toBe('float32');
        expect(a.rank).toBe(2);
        expect(a.shape).toEqual([0, 5]);
        test_util_1.expectArraysEqual(a, []);
    });
    it('2d throws when values are not empty', function () {
        var values = [1, 2, 3, 4];
        expect(function () { return tf.tensor2d(values, [0, 5], 'float32'); })
            .toThrowError('Based on the provided shape, [0,5], and dtype float32, the ' +
            'tensor should have 0 values but has 4');
    });
    it('3d of shape [0, 3, 0]', function () {
        var a = tf.tensor3d([], [0, 3, 0]);
        expect(a.dtype).toBe('float32');
        expect(a.rank).toBe(3);
        expect(a.shape).toEqual([0, 3, 0]);
        test_util_1.expectArraysEqual(a, []);
    });
    it('3d throws when values are not empty', function () {
        var values = [1, 2, 3];
        expect(function () { return tf.tensor3d(values, [0, 3, 0], 'float32'); })
            .toThrowError('Based on the provided shape, [0,3,0], and dtype float32, the ' +
            'tensor should have 0 values but has 3');
    });
    it('4d of shape [1, 3, 0, 5]', function () {
        var a = tf.tensor4d([], [1, 3, 0, 5]);
        expect(a.dtype).toBe('float32');
        expect(a.rank).toBe(4);
        expect(a.shape).toEqual([1, 3, 0, 5]);
        test_util_1.expectArraysEqual(a, []);
    });
    it('4d throws when values are not empty', function () {
        var values = [1, 2, 3];
        expect(function () { return tf.tensor4d(values, [1, 3, 0, 5], 'float32'); })
            .toThrowError('Based on the provided shape, [1,3,0,5], and dtype float32, the ' +
            'tensor should have 0 values but has 3');
    });
    it('complex64 with 0 in shape', function () {
        var areal = tf.tensor2d([], [0, 5]);
        var breal = tf.tensor2d([], [0, 5]);
        var a = tf.complex(areal, breal);
        expect(a.dtype).toBe('complex64');
        expect(a.rank).toBe(2);
        expect(a.shape).toEqual([0, 5]);
        test_util_1.expectArraysEqual(a, []);
    });
});
jasmine_util_1.describeWithFlags('buffer', test_util_1.ALL_ENVS, function () {
    it('float32', function () {
        var buff = tf.buffer([1, 2, 3], 'float32');
        buff.set(1, 0, 0, 0);
        buff.set(2, 0, 1, 0);
        expect(buff.get(0, 0, 0)).toEqual(1);
        expect(buff.get(0, 0, 1)).toEqual(0);
        expect(buff.get(0, 0, 2)).toEqual(0);
        expect(buff.get(0, 1, 0)).toEqual(2);
        expect(buff.get(0, 1, 1)).toEqual(0);
        expect(buff.get(0, 1, 2)).toEqual(0);
        test_util_1.expectArraysClose(buff.toTensor(), [1, 0, 0, 2, 0, 0]);
    });
});
//# sourceMappingURL=tensor_test.js.map