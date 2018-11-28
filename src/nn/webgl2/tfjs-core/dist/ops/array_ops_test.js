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
var util = require("../util");
var rand_util_1 = require("./rand_util");
jasmine_util_1.describeWithFlags('zeros', test_util_1.ALL_ENVS, function () {
    it('1D default dtype', function () {
        var a = tf.zeros([3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [0, 0, 0]);
    });
    it('1D float32 dtype', function () {
        var a = tf.zeros([3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [0, 0, 0]);
    });
    it('1D int32 dtype', function () {
        var a = tf.zeros([3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysEqual(a, [0, 0, 0]);
    });
    it('1D bool dtype', function () {
        var a = tf.zeros([3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysEqual(a, [0, 0, 0]);
    });
    it('2D default dtype', function () {
        var a = tf.zeros([3, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('2D float32 dtype', function () {
        var a = tf.zeros([3, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('2D int32 dtype', function () {
        var a = tf.zeros([3, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
    it('2D bool dtype', function () {
        var a = tf.zeros([3, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
    it('3D default dtype', function () {
        var a = tf.zeros([2, 2, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('3D float32 dtype', function () {
        var a = tf.zeros([2, 2, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('3D int32 dtype', function () {
        var a = tf.zeros([2, 2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('3D bool dtype', function () {
        var a = tf.zeros([2, 2, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysEqual(a, [0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('4D default dtype', function () {
        var a = tf.zeros([3, 2, 1, 1]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('4D float32 dtype', function () {
        var a = tf.zeros([3, 2, 1, 1], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysClose(a, [0, 0, 0, 0, 0, 0]);
    });
    it('4D int32 dtype', function () {
        var a = tf.zeros([3, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
    it('4D bool dtype', function () {
        var a = tf.zeros([3, 2, 1, 1], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysEqual(a, [0, 0, 0, 0, 0, 0]);
    });
});
jasmine_util_1.describeWithFlags('ones', test_util_1.ALL_ENVS, function () {
    it('1D default dtype', function () {
        var a = tf.ones([3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [1, 1, 1]);
    });
    it('1D float32 dtype', function () {
        var a = tf.ones([3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [1, 1, 1]);
    });
    it('1D int32 dtype', function () {
        var a = tf.ones([3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysEqual(a, [1, 1, 1]);
    });
    it('1D bool dtype', function () {
        var a = tf.ones([3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysEqual(a, [1, 1, 1]);
    });
    it('2D default dtype', function () {
        var a = tf.ones([3, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('2D float32 dtype', function () {
        var a = tf.ones([3, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('2D int32 dtype', function () {
        var a = tf.ones([3, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
    it('2D bool dtype', function () {
        var a = tf.ones([3, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
    it('3D default dtype', function () {
        var a = tf.ones([2, 2, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('3D float32 dtype', function () {
        var a = tf.ones([2, 2, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('3D int32 dtype', function () {
        var a = tf.ones([2, 2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('3D bool dtype', function () {
        var a = tf.ones([2, 2, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysEqual(a, [1, 1, 1, 1, 1, 1, 1, 1]);
    });
    it('4D default dtype', function () {
        var a = tf.ones([3, 2, 1, 1]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('4D float32 dtype', function () {
        var a = tf.ones([3, 2, 1, 1], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysClose(a, [1, 1, 1, 1, 1, 1]);
    });
    it('4D int32 dtype', function () {
        var a = tf.ones([3, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
    it('4D bool dtype', function () {
        var a = tf.ones([3, 2, 1, 1], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        test_util_1.expectArraysEqual(a, [1, 1, 1, 1, 1, 1]);
    });
});
jasmine_util_1.describeWithFlags('zerosLike', test_util_1.ALL_ENVS, function () {
    it('1D default dtype', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [0, 0, 0]);
    });
    it('chainable 1D default dtype', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var b = a.zerosLike();
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [0, 0, 0]);
    });
    it('1D float32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'float32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [0, 0, 0]);
    });
    it('1D int32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'int32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysEqual(b, [0, 0, 0]);
    });
    it('1D bool dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'bool');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysEqual(b, [0, 0, 0]);
    });
    it('2D default dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('2D float32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('2D int32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('2D bool dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('3D default dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('3D float32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('3D int32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('3D bool dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('4D default dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('4D float32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('4D int32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('4D bool dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('4D default dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('5D float32 dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'float32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('5D int32 dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'int32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('5D bool dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'bool');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('5D default dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1]);
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('6D float32 dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'float32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 2, 2, 1, 1, 1]);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('6D int32 dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'int32');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual(a.shape);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('6D bool dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'bool');
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual(a.shape);
        test_util_1.expectArraysEqual(b, [0, 0, 0, 0]);
    });
    it('6D default dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1]);
        var b = tf.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual(a.shape);
        test_util_1.expectArraysClose(b, [0, 0, 0, 0]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.zerosLike({}); })
            .toThrowError(/Argument 'x' passed to 'zerosLike' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var res = tf.zerosLike([[1, 2], [3, 4]]);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(res, [0, 0, 0, 0]);
    });
});
jasmine_util_1.describeWithFlags('onesLike', test_util_1.ALL_ENVS, function () {
    it('1D default dtype', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [1, 1, 1]);
    });
    it('chainable 1D default dtype', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var b = a.onesLike();
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [1, 1, 1]);
    });
    it('1D float32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'float32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [1, 1, 1]);
    });
    it('1D int32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'int32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysEqual(b, [1, 1, 1]);
    });
    it('1D bool dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'bool');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysEqual(b, [1, 1, 1]);
    });
    it('2D default dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('2D float32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('2D int32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('2D bool dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'bool');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('3D default dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('3D float32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('3D int32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('3D bool dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'bool');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('4D default dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('4D float32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('4D int32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('4D bool dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'bool');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('4D default dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('5D float32 dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'float32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('5D int32 dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'int32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('5D bool dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1], 'bool');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('5D default dtype', function () {
        var a = tf.tensor5d([1, 2, 3, 4], [1, 2, 2, 1, 1]);
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('6D int32 dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'int32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual(a.shape);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('6D bool dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'bool');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual(a.shape);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 1]);
    });
    it('6D default dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1]);
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual(a.shape);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('6D float32 dtype', function () {
        var a = tf.tensor6d([1, 2, 3, 4], [1, 2, 2, 1, 1, 1], 'float32');
        var b = tf.onesLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual(a.shape);
        test_util_1.expectArraysClose(b, [1, 1, 1, 1]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.onesLike({}); })
            .toThrowError(/Argument 'x' passed to 'onesLike' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var res = tf.onesLike([[1, 2], [3, 4]]);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(res, [1, 1, 1, 1]);
    });
});
jasmine_util_1.describeWithFlags('rand', test_util_1.ALL_ENVS, function () {
    it('should return a random 1D float32 array', function () {
        var shape = [10];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2); });
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2);
        result = tf.rand(shape, function () { return util.randUniform(0, 1.5); });
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 1D int32 array', function () {
        var shape = [10];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 1D bool array', function () {
        var shape = [10];
        var result = tf.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 2D float32 array', function () {
        var shape = [3, 4];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2.5); });
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.rand(shape, function () { return util.randUniform(0, 1.5); }, 'float32');
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 2D int32 array', function () {
        var shape = [3, 4];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 2D bool array', function () {
        var shape = [3, 4];
        var result = tf.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 3D float32 array', function () {
        var shape = [3, 4, 5];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2.5); });
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.rand(shape, function () { return util.randUniform(0, 1.5); }, 'float32');
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 3D int32 array', function () {
        var shape = [3, 4, 5];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 3D bool array', function () {
        var shape = [3, 4, 5];
        var result = tf.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 4D float32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2.5); });
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.rand(shape, function () { return util.randUniform(0, 1.5); });
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 4D int32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = tf.rand(shape, function () { return util.randUniform(0, 2); }, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 4D bool array', function () {
        var shape = [3, 4, 5, 6];
        var result = tf.rand(shape, function () { return util.randUniform(0, 1); }, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
});
jasmine_util_1.describeWithFlags('eye', test_util_1.ALL_ENVS, function () {
    it('1x1', function () {
        test_util_1.expectArraysClose(tf.eye(1), tf.tensor2d([[1]]));
    });
    it('2x2', function () {
        test_util_1.expectArraysClose(tf.eye(2), tf.tensor2d([[1, 0], [0, 1]]));
    });
    it('3x3', function () {
        test_util_1.expectArraysClose(tf.eye(3), tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
    });
    it('3x4', function () {
        test_util_1.expectArraysClose(tf.eye(3, 4), tf.tensor2d([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]));
    });
    it('4x3', function () {
        test_util_1.expectArraysClose(tf.eye(4, 3), tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]));
    });
    it('with 1D batchShape', function () {
        test_util_1.expectArraysClose(tf.eye(2, 2, [3]), tf.tensor3d([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]));
    });
    it('with 2D batchShape', function () {
        test_util_1.expectArraysClose(tf.eye(2, 2, [2, 3]), tf.tensor4d([
            [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
            [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]
        ]));
    });
    it('with 3D batchShape', function () {
        test_util_1.expectArraysClose(tf.eye(2, 2, [2, 2, 3]), tf.tensor5d([
            [
                [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
                [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            ],
            [
                [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
                [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            ]
        ]));
    });
    it('3x3, int32', function () {
        test_util_1.expectArraysClose(tf.eye(3, 3, null, 'int32'), tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [3, 3], 'int32'));
    });
    it('3x3, bool', function () {
        test_util_1.expectArraysClose(tf.eye(3, 3, null, 'bool'), tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [3, 3], 'bool'));
    });
});
jasmine_util_1.describeWithFlags('randomNormal', test_util_1.ALL_ENVS, function () {
    var SEED = 2002;
    var EPSILON = 0.05;
    it('should return a float32 1D of random normal values', function () {
        var SAMPLES = 10000;
        var result = tf.randomNormal([SAMPLES], 0, 0.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES]);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);
        result = tf.randomNormal([SAMPLES], 0, 1.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES]);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
    });
    it('should return a int32 1D of random normal values', function () {
        var SAMPLES = 10000;
        var result = tf.randomNormal([SAMPLES], 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([SAMPLES]);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
    it('should return a float32 2D of random normal values', function () {
        var SAMPLES = 100;
        var result = tf.randomNormal([SAMPLES, SAMPLES], 0, 2.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES]);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 2.5, EPSILON);
        result = tf.randomNormal([SAMPLES, SAMPLES], 0, 3.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES]);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
    });
    it('should return a int32 2D of random normal values', function () {
        var SAMPLES = 100;
        var result = tf.randomNormal([SAMPLES, SAMPLES], 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([SAMPLES, SAMPLES]);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
    it('should return a float32 3D of random normal values', function () {
        var SAMPLES_SHAPE = [20, 20, 20];
        var result = tf.randomNormal(SAMPLES_SHAPE, 0, 0.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual(SAMPLES_SHAPE);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);
        result = tf.randomNormal(SAMPLES_SHAPE, 0, 1.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual(SAMPLES_SHAPE);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
    });
    it('should return a int32 3D of random normal values', function () {
        var SAMPLES_SHAPE = [20, 20, 20];
        var result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual(SAMPLES_SHAPE);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
    it('should return a float32 4D of random normal values', function () {
        var SAMPLES_SHAPE = [10, 10, 10, 10];
        var result = tf.randomNormal(SAMPLES_SHAPE, 0, 0.5, null, SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual(SAMPLES_SHAPE);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 0.5, EPSILON);
        result = tf.randomNormal(SAMPLES_SHAPE, 0, 1.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        expect(result.shape).toEqual(SAMPLES_SHAPE);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 1.5, EPSILON);
    });
    it('should return a int32 4D of random normal values', function () {
        var SAMPLES_SHAPE = [10, 10, 10, 10];
        var result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual(SAMPLES_SHAPE);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
    it('should return a int32 5D of random normal values', function () {
        var SAMPLES_SHAPE = [10, 10, 10, 10, 10];
        var result = tf.randomNormal(SAMPLES_SHAPE, 0, 2, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual(SAMPLES_SHAPE);
        rand_util_1.jarqueBeraNormalityTest(result);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 2, EPSILON);
    });
});
jasmine_util_1.describeWithFlags('truncatedNormal', test_util_1.ALL_ENVS, function () {
    var EPSILON = 0.60;
    var SEED = 2002;
    function assertTruncatedValues(array, mean, stdv) {
        var bounds = mean + stdv * 2;
        var values = array.dataSync();
        for (var i = 0; i < values.length; i++) {
            expect(Math.abs(values[i])).toBeLessThanOrEqual(bounds);
        }
    }
    it('should return a random 1D float32 array', function () {
        var shape = [1000];
        var result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a randon 1D int32 array', function () {
        var shape = [1000];
        var result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
    it('should return a 2D float32 array', function () {
        var shape = [50, 50];
        var result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a 2D int32 array', function () {
        var shape = [50, 50];
        var result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
    it('should return a 3D float32 array', function () {
        var shape = [10, 10, 10];
        var result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a 3D int32 array', function () {
        var shape = [10, 10, 10];
        var result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
    it('should return a 4D float32 array', function () {
        var shape = [5, 5, 5, 5];
        var result = tf.truncatedNormal(shape, 0, 3.5, null, SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 3.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 3.5, EPSILON);
        result = tf.truncatedNormal(shape, 0, 4.5, 'float32', SEED);
        expect(result.dtype).toBe('float32');
        assertTruncatedValues(result, 0, 4.5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 4.5, EPSILON);
    });
    it('should return a 4D int32 array', function () {
        var shape = [5, 5, 5, 5];
        var result = tf.truncatedNormal(shape, 0, 5, 'int32', SEED);
        expect(result.dtype).toBe('int32');
        assertTruncatedValues(result, 0, 5);
        rand_util_1.expectArrayInMeanStdRange(result, 0, 5, EPSILON);
    });
});
jasmine_util_1.describeWithFlags('randomUniform', test_util_1.ALL_ENVS, function () {
    it('should return a random 1D float32 array', function () {
        var shape = [10];
        var result = tf.randomUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.randomUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 1D int32 array', function () {
        var shape = [10];
        var result = tf.randomUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 1D bool array', function () {
        var shape = [10];
        var result = tf.randomUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 2D float32 array', function () {
        var shape = [3, 4];
        var result = tf.randomUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.randomUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 2D int32 array', function () {
        var shape = [3, 4];
        var result = tf.randomUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 2D bool array', function () {
        var shape = [3, 4];
        var result = tf.randomUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 3D float32 array', function () {
        var shape = [3, 4, 5];
        var result = tf.randomUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.randomUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 3D int32 array', function () {
        var shape = [3, 4, 5];
        var result = tf.randomUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 3D bool array', function () {
        var shape = [3, 4, 5];
        var result = tf.randomUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 4D float32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = tf.randomUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.randomUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 4D int32 array', function () {
        var shape = [3, 4, 5, 6];
        var result = tf.randomUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 4D bool array', function () {
        var shape = [3, 4, 5, 6];
        var result = tf.randomUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
    it('should return a random 5D float32 array', function () {
        var shape = [2, 3, 4, 5, 6];
        var result = tf.randomUniform(shape, 0, 2.5);
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 2.5);
        result = tf.randomUniform(shape, 0, 1.5, 'float32');
        expect(result.dtype).toBe('float32');
        test_util_1.expectValuesInRange(result, 0, 1.5);
    });
    it('should return a random 5D int32 array', function () {
        var shape = [2, 3, 4, 5, 6];
        var result = tf.randomUniform(shape, 0, 2, 'int32');
        expect(result.dtype).toBe('int32');
        test_util_1.expectValuesInRange(result, 0, 2);
    });
    it('should return a random 5D bool array', function () {
        var shape = [2, 3, 4, 5, 6];
        var result = tf.randomUniform(shape, 0, 1, 'bool');
        expect(result.dtype).toBe('bool');
        test_util_1.expectValuesInRange(result, 0, 1);
    });
});
var MockContext = (function () {
    function MockContext() {
    }
    MockContext.prototype.getImageData = function (x, y, width, height) {
        var data = new Uint8ClampedArray(width * height * 4);
        for (var i = 0; i < data.length; ++i) {
            data[i] = i + 1;
        }
        return { data: data };
    };
    return MockContext;
}());
var MockCanvas = (function () {
    function MockCanvas(width, height) {
        this.width = width;
        this.height = height;
    }
    MockCanvas.prototype.getContext = function (type) {
        return new MockContext();
    };
    return MockCanvas;
}());
jasmine_util_1.describeWithFlags('fromPixels, mock canvas', test_util_1.NODE_ENVS, function () {
    it('accepts a canvas-like element', function () {
        var c = new MockCanvas(2, 2);
        var t = tf.fromPixels(c);
        expect(t.dtype).toBe('int32');
        expect(t.shape).toEqual([2, 2, 3]);
        tf.test_util.expectArraysEqual(t, [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]);
    });
    it('accepts a canvas-like element, numChannels=4', function () {
        var c = new MockCanvas(2, 2);
        var t = tf.fromPixels(c, 4);
        expect(t.dtype).toBe('int32');
        expect(t.shape).toEqual([2, 2, 4]);
        tf.test_util.expectArraysEqual(t, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    });
    it('errors when passed a non-canvas object', function () {
        expect(function () { return tf.fromPixels(5); }).toThrowError();
    });
});
jasmine_util_1.describeWithFlags('fromPixels', test_util_1.BROWSER_ENVS, function () {
    it('ImageData 1x1x3', function () {
        var pixels = new ImageData(1, 1);
        pixels.data[0] = 0;
        pixels.data[1] = 80;
        pixels.data[2] = 160;
        pixels.data[3] = 240;
        var array = tf.fromPixels(pixels, 3);
        test_util_1.expectArraysEqual(array, [0, 80, 160]);
    });
    it('ImageData 1x1x4', function () {
        var pixels = new ImageData(1, 1);
        pixels.data[0] = 0;
        pixels.data[1] = 80;
        pixels.data[2] = 160;
        pixels.data[3] = 240;
        var array = tf.fromPixels(pixels, 4);
        test_util_1.expectArraysEqual(array, [0, 80, 160, 240]);
    });
    it('ImageData 2x2x3', function () {
        var pixels = new ImageData(2, 2);
        for (var i = 0; i < 8; i++) {
            pixels.data[i] = i * 2;
        }
        for (var i = 8; i < 16; i++) {
            pixels.data[i] = i * 2;
        }
        var array = tf.fromPixels(pixels, 3);
        test_util_1.expectArraysEqual(array, [0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]);
    });
    it('ImageData 2x2x4', function () {
        var pixels = new ImageData(2, 2);
        for (var i = 0; i < 8; i++) {
            pixels.data[i] = i * 2;
        }
        for (var i = 8; i < 16; i++) {
            pixels.data[i] = i * 2;
        }
        var array = tf.fromPixels(pixels, 4);
        test_util_1.expectArraysClose(array, new Int32Array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
    });
    it('fromPixels, 3 channels', function () {
        var pixels = new ImageData(1, 2);
        pixels.data[0] = 2;
        pixels.data[1] = 3;
        pixels.data[2] = 4;
        pixels.data[3] = 255;
        pixels.data[4] = 5;
        pixels.data[5] = 6;
        pixels.data[6] = 7;
        pixels.data[7] = 255;
        var res = tf.fromPixels(pixels, 3);
        expect(res.shape).toEqual([2, 1, 3]);
        expect(res.dtype).toBe('int32');
        test_util_1.expectArraysClose(res, [2, 3, 4, 5, 6, 7]);
    });
    it('fromPixels, reshape, then do tf.add()', function () {
        var pixels = new ImageData(1, 1);
        pixels.data[0] = 2;
        pixels.data[1] = 3;
        pixels.data[2] = 4;
        pixels.data[3] = 255;
        var a = tf.fromPixels(pixels, 3).reshape([1, 1, 1, 3]);
        var res = a.add(tf.scalar(2, 'int32'));
        expect(res.shape).toEqual([1, 1, 1, 3]);
        expect(res.dtype).toBe('int32');
        test_util_1.expectArraysClose(res, [4, 5, 6]);
    });
    it('fromPixels + fromPixels', function () {
        var pixelsA = new ImageData(1, 1);
        pixelsA.data[0] = 255;
        pixelsA.data[1] = 3;
        pixelsA.data[2] = 4;
        pixelsA.data[3] = 255;
        var pixelsB = new ImageData(1, 1);
        pixelsB.data[0] = 5;
        pixelsB.data[1] = 6;
        pixelsB.data[2] = 7;
        pixelsB.data[3] = 255;
        var a = tf.fromPixels(pixelsA, 3).toFloat();
        var b = tf.fromPixels(pixelsB, 3).toFloat();
        var res = a.add(b);
        expect(res.shape).toEqual([1, 1, 3]);
        expect(res.dtype).toBe('float32');
        test_util_1.expectArraysClose(res, [260, 9, 11]);
    });
    it('throws when passed a primitive number', function () {
        expect(function () { return tf.fromPixels(3); })
            .toThrowError(/pixels passed to tf.fromPixels\(\) must be either/);
    });
    it('throws when passed a string', function () {
        expect(function () { return tf.fromPixels('test'); })
            .toThrowError(/pixels passed to tf.fromPixels\(\) must be either/);
    });
});
jasmine_util_1.describeWithFlags('toPixels no canvas', test_util_1.ALL_ENVS, function () {
    it('draws a rank-2 float32 tensor', function (done) {
        var x = tf.tensor2d([.15, .2], [2, 1], 'float32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255),
                255, Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255),
                255
            ]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-2 int32 tensor', function (done) {
        var x = tf.tensor2d([10, 20], [2, 1], 'int32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 float32 tensor, 1 channel', function (done) {
        var x = tf.tensor3d([.15, .2], [2, 1, 1], 'float32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255),
                255, Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255),
                255
            ]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 int32 tensor, 1 channel', function (done) {
        var x = tf.tensor3d([10, 20], [2, 1, 1], 'int32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 float32 tensor, 3 channel', function (done) {
        var x = tf.tensor3d([.05, .1001, .15, .2, .25, .3001], [2, 1, 3], 'float32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
                255, Math.round(.2 * 255), Math.round(.25 * 255),
                Math.round(.3001 * 255), 255
            ]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 int32 tensor, 3 channel', function (done) {
        var x = tf.tensor3d([10, 20, 30, 40, 50, 60], [2, 1, 3], 'int32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 float32 tensor, 4 channel', function (done) {
        var x = tf.tensor3d([.05, .1001, .15, .2, .25, .3001, .35, .4], [2, 1, 4], 'float32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
                Math.round(.20 * 255), Math.round(.25 * 255), Math.round(.3001 * 255),
                Math.round(.35 * 255), Math.round(.4 * 255)
            ]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 int32 tensor, 4 channel', function (done) {
        var x = tf.tensor3d([10, 20, 30, 40, 50, 60, 70, 80], [2, 1, 4], 'int32');
        tf.toPixels(x).then(function (data) {
            var expected = new Uint8ClampedArray([10, 20, 30, 40, 50, 60, 70, 80]);
            expect(data).toEqual(expected);
            done();
        });
    });
    it('throws for scalars', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.scalar(1)); }, done);
    });
    it('throws for rank-1 tensors', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.tensor1d([1])); }, done);
    });
    it('throws for rank-4 tensors', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.tensor4d([1], [1, 1, 1, 1])); }, done);
    });
    it('throws for bool dtype', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.tensor2d([1], [1, 1], 'bool')); }, done);
    });
    it('throws for rank-3 depth = 2', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.tensor3d([1, 2], [1, 1, 2])); }, done);
    });
    it('throws for rank-3 depth = 5', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.tensor3d([1, 2, 3, 4, 5], [1, 1, 5])); }, done);
    });
    it('throws for float32 tensor with values not in [0 - 1]', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.tensor2d([-1, .5], [1, 2])); }, done);
    });
    it('throws for int32 tensor with values not in [0 - 255]', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels(tf.tensor2d([-1, 100], [1, 2], 'int32')); }, done);
    });
    it('throws when passed a non-tensor', function (done) {
        test_util_1.expectPromiseToFail(function () { return tf.toPixels({}); }, done);
    });
    it('accepts a tensor-like object', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, data, expected;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    x = [[10], [20]];
                    return [4, tf.toPixels(x)];
                case 1:
                    data = _a.sent();
                    expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
                    expect(data).toEqual(expected);
                    return [2];
            }
        });
    }); });
});
jasmine_util_1.describeWithFlags('toPixels', test_util_1.WEBGL_ENVS, function () {
    it('draws a rank-2 float32 tensor, canvas', function (done) {
        var x = tf.tensor2d([.15, .2], [2, 1], 'float32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255),
                255, Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255),
                255
            ]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-2 int32 tensor, canvas', function (done) {
        var x = tf.tensor2d([10, 20], [2, 1], 'int32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 float32 tensor, 1 channel, canvas', function (done) {
        var x = tf.tensor3d([.15, .2], [2, 1, 1], 'float32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.15 * 255), Math.round(.15 * 255), Math.round(.15 * 255),
                255, Math.round(.2 * 255), Math.round(.2 * 255), Math.round(.2 * 255),
                255
            ]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 int32 tensor, 1 channel, canvas', function (done) {
        var x = tf.tensor3d([10, 20], [2, 1, 1], 'int32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([10, 10, 10, 255, 20, 20, 20, 255]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 float32 tensor, 3 channel, canvas', function (done) {
        var x = tf.tensor3d([.05, .1001, .15, .20, .25, .3001], [2, 1, 3], 'float32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
                255, Math.round(.2 * 255), Math.round(.25 * 255),
                Math.round(.3001 * 255), 255
            ]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 int32 tensor, 3 channel, canvas', function (done) {
        var x = tf.tensor3d([10, 20, 30, 40, 50, 60], [2, 1, 3], 'int32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 float32 tensor, 4 channel, canvas', function (done) {
        var x = tf.tensor3d([.05, .1001, .15, 1, .20, .25, .3001, 1], [2, 1, 4], 'float32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([
                Math.round(.05 * 255), Math.round(.1001 * 255), Math.round(.15 * 255),
                255, Math.round(.20 * 255), Math.round(.25 * 255),
                Math.round(.3001 * 255), 255
            ]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('draws a rank-3 int32 tensor, 4 channel, canvas', function (done) {
        var x = tf.tensor3d([10, 20, 30, 255, 50, 60, 70, 255], [2, 1, 4], 'int32');
        var canvas = document.createElement('canvas');
        tf.toPixels(x, canvas).then(function (data) {
            var expected = new Uint8ClampedArray([10, 20, 30, 255, 50, 60, 70, 255]);
            expect(data).toEqual(expected);
            var ctx = canvas.getContext('2d');
            var imgData = ctx.getImageData(0, 0, 1, 2);
            expect(imgData.data).toEqual(expected);
            done();
        });
    });
    it('accepts a tensor-like object', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, canvas, data, expected, ctx, imgData;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    x = [[127], [100]];
                    canvas = document.createElement('canvas');
                    return [4, tf.toPixels(x, canvas)];
                case 1:
                    data = _a.sent();
                    expected = new Uint8ClampedArray([127, 127, 127, 255, 100, 100, 100, 255]);
                    expect(data).toEqual(expected);
                    ctx = canvas.getContext('2d');
                    imgData = ctx.getImageData(0, 0, 1, 2);
                    expect(imgData.data).toEqual(expected);
                    return [2];
            }
        });
    }); });
});
jasmine_util_1.describeWithFlags('clone', test_util_1.ALL_ENVS, function () {
    it('1D default dtype', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [1, 2, 3]);
    });
    it('1D float32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'float32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysClose(b, [1, 2, 3]);
    });
    it('1D int32 dtype', function () {
        var a = tf.tensor1d([1, 2, 3], 'int32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysEqual(b, [1, 2, 3]);
    });
    it('1D bool dtype', function () {
        var a = tf.tensor1d([1, 1, 0], 'bool');
        var b = tf.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        test_util_1.expectArraysEqual(b, [1, 1, 0]);
    });
    it('1D complex64 dtype', function () {
        var a = tf.complex([1], [1]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([1]);
        test_util_1.expectArraysEqual(b, [1, 1]);
    });
    it('2D default dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('2D float32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('2D int32 dtype', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(b, [1, 2, 3, 4]);
    });
    it('2D bool dtype', function () {
        var a = tf.tensor2d([1, 1, 1, 0], [2, 2], 'bool');
        var b = tf.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 0]);
    });
    it('2D complex64 dtype', function () {
        var a = tf.complex([[1, 3], [5, 7]], [[2, 4], [6, 8]]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(b, [1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('3D default dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('3D float32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'float32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('3D int32 dtype', function () {
        var a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1], 'int32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(b, [1, 2, 3, 4]);
    });
    it('3D bool dtype', function () {
        var a = tf.tensor3d([1, 1, 1, 0], [2, 2, 1], 'bool');
        var b = tf.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 0]);
    });
    it('3D complex64 dtype', function () {
        var a = tf.complex([[[1], [3]], [[5], [7]]], [[[2], [4]], [[6], [8]]]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysEqual(b, [1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('4D default dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('4D float32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'float32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysClose(b, [1, 2, 3, 4]);
    });
    it('4D int32 dtype', function () {
        var a = tf.tensor4d([1, 2, 3, 4], [2, 2, 1, 1], 'int32');
        var b = tf.clone(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [1, 2, 3, 4]);
    });
    it('4D bool dtype', function () {
        var a = tf.tensor4d([1, 1, 1, 0], [2, 2, 1, 1], 'bool');
        var b = tf.clone(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [1, 1, 1, 0]);
    });
    it('4D complex64 dtype', function () {
        var a = tf.complex([[[[1]], [[3]]], [[[5]], [[7]]]], [[[[2]], [[4]]], [[[6]], [[8]]]]);
        var b = tf.clone(a);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(b, [1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('gradient: 1D', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var dy = tf.tensor1d([4, 5, 6]);
        var da = tf.grad(function (x) { return tf.clone(x); })(a, dy);
        expect(da.dtype).toBe('float32');
        expect(da.shape).toEqual([3]);
        test_util_1.expectArraysClose(da, [4, 5, 6]);
    });
    it('gradient: 2D int32', function () {
        var a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
        var dy = tf.tensor2d([5, 6, 7, 8], [2, 2], 'float32');
        var da = tf.grad(function (x) { return tf.clone(x); })(a, dy);
        expect(da.dtype).toBe('float32');
        expect(da.shape).toEqual([2, 2]);
        test_util_1.expectArraysEqual(da, [5, 6, 7, 8]);
    });
    it('gradient: 4D bool', function () {
        var a = tf.tensor4d([1, 1, 1, 0], [2, 2, 1, 1], 'bool');
        var dy = tf.tensor4d([5, 6, 7, 8], [2, 2, 1, 1], 'float32');
        var da = tf.grad(function (x) { return tf.clone(x); })(a, dy);
        expect(da.dtype).toBe('float32');
        expect(da.shape).toEqual([2, 2, 1, 1]);
        test_util_1.expectArraysEqual(da, [5, 6, 7, 8]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.clone({}); })
            .toThrowError(/Argument 'x' passed to 'clone' must be a Tensor/);
    });
});
jasmine_util_1.describeWithFlags('tile', test_util_1.ALL_ENVS, function () {
    it('1D (tile)', function () {
        var t = tf.tensor1d([1, 2, 3]);
        var t2 = tf.tile(t, [2]);
        expect(t2.shape).toEqual([6]);
        test_util_1.expectArraysClose(t2, [1, 2, 3, 1, 2, 3]);
    });
    it('2D (tile)', function () {
        var t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
        var t2 = tf.tile(t, [1, 2]);
        expect(t2.shape).toEqual([2, 4]);
        test_util_1.expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22]);
        t2 = tf.tile(t, [2, 1]);
        expect(t2.shape).toEqual([4, 2]);
        test_util_1.expectArraysClose(t2, [1, 11, 2, 22, 1, 11, 2, 22]);
        t2 = tf.tile(t, [2, 2]);
        expect(t2.shape).toEqual([4, 4]);
        test_util_1.expectArraysClose(t2, [1, 11, 1, 11, 2, 22, 2, 22, 1, 11, 1, 11, 2, 22, 2, 22]);
    });
    it('3D (tile)', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var t2 = tf.tile(t, [1, 2, 1]);
        expect(t2.shape).toEqual([2, 4, 2]);
        test_util_1.expectArraysClose(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
    });
    it('4D (tile)', function () {
        var t = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2]);
        var t2 = tf.tile(t, [1, 2, 1, 1]);
        expect(t2.shape).toEqual([1, 4, 2, 2]);
        test_util_1.expectArraysClose(t2, [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('5D (tile)', function () {
        var t = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2]);
        var t2 = tf.tile(t, [1, 2, 1, 1, 1]);
        expect(t2.shape).toEqual([1, 2, 2, 2, 2]);
        test_util_1.expectArraysClose(t2, [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('propagates NaNs', function () {
        var t = tf.tensor1d([1, 2, NaN]);
        var t2 = tf.tile(t, [2]);
        expect(t2.shape).toEqual([6]);
        test_util_1.expectArraysClose(t2, [1, 2, NaN, 1, 2, NaN]);
    });
    it('1D bool (tile)', function () {
        var t = tf.tensor1d([true, false, true], 'bool');
        var t2 = tf.tile(t, [2]);
        expect(t2.shape).toEqual([6]);
        expect(t2.dtype).toBe('bool');
        test_util_1.expectArraysEqual(t2, [1, 0, 1, 1, 0, 1]);
    });
    it('2D bool (tile)', function () {
        var t = tf.tensor2d([true, false, true, true], [2, 2], 'bool');
        var t2 = tf.tile(t, [1, 2]);
        expect(t2.shape).toEqual([2, 4]);
        expect(t2.dtype).toBe('bool');
        test_util_1.expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1]);
        t2 = tf.tile(t, [2, 1]);
        expect(t2.shape).toEqual([4, 2]);
        expect(t2.dtype).toBe('bool');
        test_util_1.expectArraysEqual(t2, [1, 0, 1, 1, 1, 0, 1, 1]);
        t2 = tf.tile(t, [2, 2]);
        expect(t2.shape).toEqual([4, 4]);
        expect(t2.dtype).toBe('bool');
        test_util_1.expectArraysEqual(t2, [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]);
    });
    it('3D bool (tile)', function () {
        var t = tf.tensor3d([true, false, true, false, true, false, true, false], [2, 2, 2], 'bool');
        var t2 = tf.tile(t, [1, 2, 1]);
        expect(t2.shape).toEqual([2, 4, 2]);
        expect(t2.dtype).toBe('bool');
        test_util_1.expectArraysEqual(t2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
    });
    it('1D int32 (tile)', function () {
        var t = tf.tensor1d([1, 2, 5], 'int32');
        var t2 = tf.tile(t, [2]);
        expect(t2.shape).toEqual([6]);
        expect(t2.dtype).toBe('int32');
        test_util_1.expectArraysEqual(t2, [1, 2, 5, 1, 2, 5]);
    });
    it('2D int32 (tile)', function () {
        var t = tf.tensor2d([1, 2, 3, 4], [2, 2], 'int32');
        var t2 = tf.tile(t, [1, 2]);
        expect(t2.shape).toEqual([2, 4]);
        expect(t2.dtype).toBe('int32');
        test_util_1.expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4]);
        t2 = tf.tile(t, [2, 1]);
        expect(t2.shape).toEqual([4, 2]);
        expect(t2.dtype).toBe('int32');
        test_util_1.expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4]);
        t2 = tf.tile(t, [2, 2]);
        expect(t2.shape).toEqual([4, 4]);
        expect(t2.dtype).toBe('int32');
        test_util_1.expectArraysEqual(t2, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
    });
    it('3D int32 (tile)', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], 'int32');
        var t2 = tf.tile(t, [1, 2, 1]);
        expect(t2.shape).toEqual([2, 4, 2]);
        expect(t2.dtype).toBe('int32');
        test_util_1.expectArraysEqual(t2, [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8]);
    });
    it('1D (tile) gradient', function () {
        var x = tf.tensor1d([1, 2, 3]);
        var dy = tf.tensor1d([0.1, 0.2, 0.3, 1, 2, 3, 10, 20, 30]);
        var gradients = tf.grad(function (x) { return tf.tile(x, [3]); })(x, dy);
        test_util_1.expectArraysClose(gradients, tf.tensor1d([11.1, 22.2, 33.3]));
    });
    it('2D (tile) gradient', function () {
        var x = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
        var dy = tf.tensor2d([[1, 2, 10, 20], [3, 4, 30, 40]], [2, 4]);
        var gradients = tf.grad(function (x) { return tf.tile(x, [1, 2]); })(x, dy);
        test_util_1.expectArraysClose(gradients, tf.tensor2d([[11, 22], [33, 44]], [2, 2]));
    });
    it('3D (tile) gradient', function () {
        var x = tf.tensor3d([[[1], [2]], [[3], [4]]], [2, 2, 1]);
        var dy = tf.tensor3d([[[1, 10], [2, 20]], [[3, 30], [4, 40]]], [2, 2, 2]);
        var gradients = tf.grad(function (x) { return tf.tile(x, [1, 1, 2]); })(x, dy);
        test_util_1.expectArraysClose(gradients, tf.tensor3d([[[11], [22]], [[33], [44]]], [2, 2, 1]));
    });
    it('4D (tile) gradient', function () {
        var x = tf.tensor4d([[[[1]], [[2]]], [[[3]], [[4]]]], [2, 2, 1, 1]);
        var dy = tf.tensor4d([
            [[[.01, .1], [1, 10]], [[.02, .2], [2, 20]]],
            [[[.03, .3], [3, 30]], [[.04, .4], [4, 40]]]
        ], [2, 2, 2, 2]);
        var gradients = tf.grad(function (x) { return tf.tile(x, [1, 1, 2, 2]); })(x, dy);
        test_util_1.expectArraysClose(gradients, tf.tensor4d([[[[11.11]], [[22.22]]], [[[33.33]], [[44.44]]]], [2, 2, 1, 1]));
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.tile({}, [1]); })
            .toThrowError(/Argument 'x' passed to 'tile' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var res = tf.tile([1, 2, 3], [2]);
        expect(res.shape).toEqual([6]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 1, 2, 3]);
    });
});
jasmine_util_1.describeWithFlags('gather', test_util_1.ALL_ENVS, function () {
    it('1D (gather)', function () {
        var t = tf.tensor1d([1, 2, 3]);
        var t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);
        expect(t2.shape).toEqual([4]);
        test_util_1.expectArraysClose(t2, [1, 3, 1, 2]);
    });
    it('2D (gather)', function () {
        var t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
        var t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 0);
        expect(t2.shape).toEqual([4, 2]);
        test_util_1.expectArraysClose(t2, [2, 22, 1, 11, 1, 11, 2, 22]);
        t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 1);
        expect(t2.shape).toEqual([2, 4]);
        test_util_1.expectArraysClose(t2, [11, 1, 1, 11, 22, 2, 2, 22]);
    });
    it('3D (gather)', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var t2 = tf.gather(t, tf.tensor1d([1, 0, 0, 1], 'int32'), 2);
        expect(t2.shape).toEqual([2, 2, 4]);
        test_util_1.expectArraysClose(t2, [2, 1, 1, 2, 4, 3, 3, 4, 6, 5, 5, 6, 8, 7, 7, 8]);
    });
    it('bool (gather)', function () {
        var t = tf.tensor1d([true, false, true], 'bool');
        var t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);
        expect(t2.shape).toEqual([4]);
        expect(t2.dtype).toBe('bool');
        expect(t2.dataSync()).toEqual(new Uint8Array([1, 1, 1, 0]));
    });
    it('int32 (gather)', function () {
        var t = tf.tensor1d([1, 2, 5], 'int32');
        var t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);
        expect(t2.shape).toEqual([4]);
        expect(t2.dtype).toBe('int32');
        expect(t2.dataSync()).toEqual(new Int32Array([1, 5, 1, 2]));
    });
    it('propagates NaNs', function () {
        var t = tf.tensor1d([1, 2, NaN]);
        var t2 = tf.gather(t, tf.tensor1d([0, 2, 0, 1], 'int32'), 0);
        expect(t2.shape).toEqual([4]);
        test_util_1.expectArraysClose(t2, [1, NaN, 1, 2]);
    });
    it('chaining, axis=1', function () {
        var x = tf.zeros([2, 4, 6]);
        var indices = tf.range(0, 6, 2, 'int32');
        var axis = 2;
        expect(x.gather(indices, axis).shape).toEqual([2, 4, 3]);
    });
    it('indices not int32 throws error', function () {
        var x = tf.zeros([2, 4, 6]);
        var indices = tf.range(0, 6, 2);
        var axis = 2;
        expect(function () { return x.gather(indices, axis); }).toThrowError();
    });
    it('throws when passed x as a non-tensor', function () {
        expect(function () { return tf.gather({}, tf.tensor1d([1])); })
            .toThrowError(/Argument 'x' passed to 'gather' must be a Tensor/);
    });
    it('throws when passed indices as a non-tensor', function () {
        expect(function () { return tf.gather(tf.tensor1d([1]), {}); })
            .toThrowError(/Argument 'indices' passed to 'gather' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var res = tf.gather([1, 2, 3], [0, 2, 0, 1], 0);
        expect(res.shape).toEqual([4]);
        test_util_1.expectArraysClose(res, [1, 3, 1, 2]);
    });
    it('gradient 1D (gather)', function () {
        var t = tf.tensor1d([1, 2, 3]);
        var indices = tf.tensor1d([0, 2, 0, 1], 'int32');
        var dy = tf.tensor([3, 4, 5, 6]);
        var gradients = tf.grad(function (t) { return tf.gather(t, indices); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [8, 6, 4]);
    });
    it('gradient 2D (gather) axis=0 shape=[2, 2]', function () {
        var t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
        var indices = tf.tensor1d([1, 0, 0, 1], 'int32');
        var dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [4, 2]);
        var axis = 0;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [12, 14, 12, 14]);
    });
    it('gradient 2D (gather) axis=0 shape=[4, 1]', function () {
        var t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
        var indices = tf.tensor1d([1, 0, 0, 1], 'int32');
        var dy = tf.tensor([23, 7, 19, 13], [4, 1]);
        var axis = 0;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [26, 36, 0, 0]);
    });
    it('gradient 2D (gather) axis=1 shape=[2, 2]', function () {
        var t = tf.tensor2d([1, 11, 2, 22], [2, 2]);
        var indices = tf.tensor1d([1, 0, 0, 1], 'int32');
        var dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10], [2, 4]);
        var axis = 1;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [9, 9, 17, 17]);
    });
    it('gradient 2D (gather) axis=1 shape=[4, 1]', function () {
        var t = tf.tensor2d([1, 11, 2, 22], [4, 1]);
        var indices = tf.tensor1d([0, 0, 0, 0], 'int32');
        var dy = tf.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [4, 4]);
        var axis = 1;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [18, 34, 50, 66]);
    });
    it('gradient 3D (gather) axis=0 shape=[2, 3, 2]', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
        var indices = tf.tensor1d([1, 0, 0, 1], 'int32');
        var dy = tf.tensor([
            2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13,
            4, 15, 12, -7, 18, 19, 2, 21, 6, 23, 24, 25
        ], [4, 3, 2]);
        var axis = 0;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [5, 33, 12.01, -7, 30, 32, 4, 18, 10, 38, 30, 25.7]);
    });
    it('gradient 3D (gather) axis=0 shape=[1, 4, 4]', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
        var indices = tf.tensor1d([0, 0], 'int32');
        var dy = tf.tensor([
            2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7,
            18, 19, 2, 21, 6, 23, 24, 25, 101, 31, 34, 54, 1, 0, -3, -4
        ], [2, 4, 4]);
        var axis = 0;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [20, 16, 6, 36, 12, 23.7, 25, 43, 101.01, 31, 46, 67, 5, 15, 9, -11]);
    });
    it('gradient 3D (gather) axis=1 shape=[2, 3, 2]', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
        var indices = tf.tensor1d([1, 2, 2, 1], 'int32');
        var dy = tf.tensor([2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7], [2, 4, 2]);
        var axis = 1;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [0, 0, 3, 15, 10, 15.7, 0, 0, 12.01, -7, 16, 28]);
    });
    it('gradient 3D (gather) axis=1 shape=[1, 4, 4]', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4]);
        var indices = tf.tensor1d([1, 2, 2, 1], 'int32');
        var dy = tf.tensor([2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13, 4, 15, 12, -7], [1, 4, 4]);
        var axis = 1;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [0, 0, 0, 0, 6, 12, 16, 8, 6.01, .7, 13, 31, 0, 0, 0, 0]);
    });
    it('gradient 3D (gather) axis=2 shape=[2, 3, 2]', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 3, 2]);
        var indices = tf.tensor1d([1, 0, 1, 0], 'int32');
        var dy = tf.tensor([
            2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 12, 13,
            4, 15, 12, -7, 18, 19, 2, 21, 6, 23, 24, 25
        ], [2, 3, 4]);
        var axis = 2;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [12, 6, 18.7, 7, 13, 12.01, 8, 16, 40, 20, 48, 30]);
    });
    it('gradient 3D (gather) axis=2 shape=[4, 1, 4]', function () {
        var t = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 1, 4]);
        var indices = tf.tensor1d([1, 3, 1], 'int32');
        var dy = tf.tensor([2, -3, 4, 15, 6, 0.7, 1, 18, 0.01, 0, 4, 15], [4, 1, 3]);
        var axis = 2;
        var gradients = tf.grad(function (t) { return tf.gather(t, indices, axis); })(t, dy);
        expect(gradients.shape).toEqual(t.shape);
        test_util_1.expectArraysClose(gradients, [0, 6, 0, -3, 0, 15.7, 0, 6, 0, 1.01, 0, 18, 0, 15, 0, 4]);
    });
});
jasmine_util_1.describeWithFlags('oneHot', test_util_1.ALL_ENVS, function () {
    it('Depth 1 throws error', function () {
        var indices = tf.tensor1d([0, 0, 0], 'int32');
        expect(function () { return tf.oneHot(indices, 1); }).toThrowError();
    });
    it('Depth 2, diagonal', function () {
        var indices = tf.tensor1d([0, 1], 'int32');
        var res = tf.oneHot(indices, 2);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [1, 0, 0, 1]);
    });
    it('Depth 2, transposed diagonal', function () {
        var indices = tf.tensor1d([1, 0], 'int32');
        var res = tf.oneHot(indices, 2);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [0, 1, 1, 0]);
    });
    it('Depth 3, 4 events', function () {
        var indices = tf.tensor1d([2, 1, 2, 0], 'int32');
        var res = tf.oneHot(indices, 3);
        expect(res.shape).toEqual([4, 3]);
        test_util_1.expectArraysClose(res, [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
    });
    it('Out of range events do not trigger onValue', function () {
        var indices = tf.tensor1d([-1, 5, 12345], 'int32');
        var res = tf.oneHot(indices, 5);
        expect(res.shape).toEqual([3, 5]);
        test_util_1.expectArraysClose(res, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    });
    it('Depth 2 onValue=3, offValue=-2', function () {
        var indices = tf.tensor1d([0, 1], 'int32');
        var res = tf.oneHot(indices, 2, 3, -2);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [3, -2, -2, 3]);
    });
    it('indices not int32 throws error', function () {
        var indices = tf.tensor1d([0, 1], 'float32');
        expect(function () { return tf.oneHot(indices, 2); }).toThrowError();
    });
    it('check output dtype', function () {
        var expectedType = 'int32';
        var indices = tf.tensor1d([0, 1], 'int32');
        var res = tf.oneHot(indices, 2);
        expect(res.dtype).toEqual(expectedType);
    });
    it('oneHot accepts a tensor-like object', function () {
        var res = tf.oneHot([0, 1], 2);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [1, 0, 0, 1]);
    });
    it('has gradient', function () {
        var a = tf.tensor1d([0, 1, 2], 'int32');
        var dy = tf.ones([3, 3], 'float32');
        var da = tf.grad(function (x) { return tf.oneHot(x, 3); })(a, dy);
        expect(da.dtype).toBe('int32');
        expect(da.shape).toEqual([3]);
        test_util_1.expectArraysClose(da, [0, 0, 0]);
    });
});
jasmine_util_1.describeWithFlags('linspace', test_util_1.ALL_ENVS, function () {
    it('start stop', function () {
        var a = tf.linspace(1, 10, 10);
        test_util_1.expectArraysEqual(a, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
        expect(a.shape).toEqual([10]);
        var b = tf.linspace(12, 17, 8);
        test_util_1.expectArraysClose(b, [
            12., 12.71428571, 13.42857143, 14.14285714, 14.85714286, 15.57142857,
            16.28571429, 17.
        ]);
        expect(b.shape).toEqual([8]);
        var c = tf.linspace(9, 0, 6);
        test_util_1.expectArraysClose(c, [9., 7.2, 5.4, 3.6, 1.8, 0.]);
        expect(c.shape).toEqual([6]);
    });
    it('negative start stop', function () {
        var a = tf.linspace(-4, 5, 6);
        test_util_1.expectArraysClose(a, [-4., -2.2, -0.4, 1.4, 3.2, 5.]);
        expect(a.shape).toEqual([6]);
    });
    it('start negative stop', function () {
        var a = tf.linspace(4, -5, 6);
        test_util_1.expectArraysClose(a, [4., 2.2, 0.4, -1.4, -3.2, -5.]);
        expect(a.shape).toEqual([6]);
    });
    it('negative start negative stop', function () {
        var a = tf.linspace(-4, -5, 6);
        test_util_1.expectArraysClose(a, [-4., -4.2, -4.4, -4.6, -4.8, -5.]);
        expect(a.shape).toEqual([6]);
        var b = tf.linspace(-9, -4, 5);
        test_util_1.expectArraysClose(b, [-9., -7.75, -6.5, -5.25, -4.]);
        expect(b.shape).toEqual([5]);
    });
    it('should throw with no samples', function () {
        expect(function () { return tf.linspace(2, 10, 0); }).toThrow();
    });
});
jasmine_util_1.describeWithFlags('range', test_util_1.ALL_ENVS, function () {
    it('start stop', function () {
        var a = tf.range(0, 3);
        test_util_1.expectArraysEqual(a, [0, 1, 2]);
        expect(a.shape).toEqual([3]);
        var b = tf.range(3, 8);
        test_util_1.expectArraysEqual(b, [3, 4, 5, 6, 7]);
        expect(b.shape).toEqual([5]);
    });
    it('start stop negative', function () {
        var a = tf.range(-2, 3);
        test_util_1.expectArraysEqual(a, [-2, -1, 0, 1, 2]);
        expect(a.shape).toEqual([5]);
        var b = tf.range(4, -2);
        test_util_1.expectArraysEqual(b, [4, 3, 2, 1, 0, -1]);
        expect(b.shape).toEqual([6]);
    });
    it('start stop step', function () {
        var a = tf.range(4, 15, 4);
        test_util_1.expectArraysEqual(a, [4, 8, 12]);
        expect(a.shape).toEqual([3]);
        var b = tf.range(4, 11, 4);
        test_util_1.expectArraysEqual(b, [4, 8]);
        expect(b.shape).toEqual([2]);
        var c = tf.range(4, 17, 4);
        test_util_1.expectArraysEqual(c, [4, 8, 12, 16]);
        expect(c.shape).toEqual([4]);
        var d = tf.range(0, 30, 5);
        test_util_1.expectArraysEqual(d, [0, 5, 10, 15, 20, 25]);
        expect(d.shape).toEqual([6]);
        var e = tf.range(-3, 9, 2);
        test_util_1.expectArraysEqual(e, [-3, -1, 1, 3, 5, 7]);
        expect(e.shape).toEqual([6]);
        var f = tf.range(3, 3);
        test_util_1.expectArraysEqual(f, new Float32Array(0));
        expect(f.shape).toEqual([0]);
        var g = tf.range(3, 3, 1);
        test_util_1.expectArraysEqual(g, new Float32Array(0));
        expect(g.shape).toEqual([0]);
        var h = tf.range(3, 3, 4);
        test_util_1.expectArraysEqual(h, new Float32Array(0));
        expect(h.shape).toEqual([0]);
        var i = tf.range(-18, -2, 5);
        test_util_1.expectArraysEqual(i, [-18, -13, -8, -3]);
        expect(i.shape).toEqual([4]);
    });
    it('start stop large step', function () {
        var a = tf.range(3, 10, 150);
        test_util_1.expectArraysEqual(a, [3]);
        expect(a.shape).toEqual([1]);
        var b = tf.range(10, 500, 205);
        test_util_1.expectArraysEqual(b, [10, 215, 420]);
        expect(b.shape).toEqual([3]);
        var c = tf.range(3, -10, -150);
        test_util_1.expectArraysEqual(c, [3]);
        expect(c.shape).toEqual([1]);
        var d = tf.range(-10, -500, -205);
        test_util_1.expectArraysEqual(d, [-10, -215, -420]);
        expect(d.shape).toEqual([3]);
    });
    it('start stop negative step', function () {
        var a = tf.range(0, -10, -1);
        test_util_1.expectArraysEqual(a, [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
        expect(a.shape).toEqual([10]);
        var b = tf.range(0, -10);
        test_util_1.expectArraysEqual(b, [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]);
        expect(b.shape).toEqual([10]);
        var c = tf.range(3, -4, -2);
        test_util_1.expectArraysEqual(c, [3, 1, -1, -3]);
        expect(c.shape).toEqual([4]);
        var d = tf.range(-3, -18, -5);
        test_util_1.expectArraysEqual(d, [-3, -8, -13]);
        expect(d.shape).toEqual([3]);
    });
    it('start stop incompatible step', function () {
        var a = tf.range(3, 10, -2);
        test_util_1.expectArraysEqual(a, new Float32Array(0));
        expect(a.shape).toEqual([0]);
        var b = tf.range(40, 3, 2);
        test_util_1.expectArraysEqual(b, new Float32Array(0));
        expect(b.shape).toEqual([0]);
    });
    it('zero step', function () {
        expect(function () { return tf.range(2, 10, 0); }).toThrow();
    });
    it('should have default dtype', function () {
        var a = tf.range(1, 4);
        test_util_1.expectArraysEqual(a, [1, 2, 3]);
        expect(a.dtype).toEqual('float32');
        expect(a.shape).toEqual([3]);
    });
    it('should have float32 dtype', function () {
        var a = tf.range(1, 4, undefined, 'float32');
        test_util_1.expectArraysEqual(a, [1, 2, 3]);
        expect(a.dtype).toEqual('float32');
        expect(a.shape).toEqual([3]);
    });
    it('should have int32 dtype', function () {
        var a = tf.range(1, 4, undefined, 'int32');
        test_util_1.expectArraysEqual(a, [1, 2, 3]);
        expect(a.dtype).toEqual('int32');
        expect(a.shape).toEqual([3]);
    });
});
jasmine_util_1.describeWithFlags('fill', test_util_1.ALL_ENVS, function () {
    it('1D fill', function () {
        var a = tf.fill([3], 2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [2, 2, 2]);
    });
    it('2D fill', function () {
        var a = tf.fill([3, 2], 2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
    });
    it('3D fill', function () {
        var a = tf.fill([3, 2, 1], 2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1]);
        test_util_1.expectArraysClose(a, [2, 2, 2, 2, 2, 2]);
    });
    it('4D fill', function () {
        var a = tf.fill([3, 2, 1, 2], 2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 2]);
        test_util_1.expectArraysClose(a, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
    });
    it('5D fill', function () {
        var a = tf.fill([2, 1, 2, 1, 2], 2);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 1, 2, 1, 2]);
        test_util_1.expectArraysClose(a, [2, 2, 2, 2, 2, 2, 2, 2]);
    });
});
jasmine_util_1.describeWithFlags('stack', test_util_1.ALL_ENVS, function () {
    it('scalars 3, 5 and 7', function () {
        var a = tf.scalar(3);
        var b = tf.scalar(5);
        var c = tf.scalar(7);
        var res = tf.stack([a, b, c]);
        expect(res.shape).toEqual([3]);
        test_util_1.expectArraysClose(res, [3, 5, 7]);
    });
    it('scalars 3, 5 and 7 along axis=1 throws error', function () {
        var a = tf.scalar(3);
        var b = tf.scalar(5);
        var c = tf.scalar(7);
        var f = function () { return tf.stack([a, b, c], 1); };
        expect(f).toThrowError();
    });
    it('non matching shapes throws error', function () {
        var a = tf.scalar(3);
        var b = tf.tensor1d([5]);
        var f = function () { return tf.stack([a, b]); };
        expect(f).toThrowError();
    });
    it('non matching dtypes throws error', function () {
        var a = tf.scalar(3);
        var b = tf.scalar(5, 'bool');
        var f = function () { return tf.stack([a, b]); };
        expect(f).toThrowError();
    });
    it('2d but axis=3 throws error', function () {
        var a = tf.zeros([2, 2]);
        var b = tf.zeros([2, 2]);
        var f = function () { return tf.stack([a, b], 3); };
        expect(f).toThrowError();
    });
    it('[1,2], [3,4] and [5,6], axis=0', function () {
        var a = tf.tensor1d([1, 2]);
        var b = tf.tensor1d([3, 4]);
        var c = tf.tensor1d([5, 6]);
        var res = tf.stack([a, b, c], 0);
        expect(res.shape).toEqual([3, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('[1,2], [3,4] and [5,6], axis=1', function () {
        var a = tf.tensor1d([1, 2]);
        var b = tf.tensor1d([3, 4]);
        var c = tf.tensor1d([5, 6]);
        var res = tf.stack([a, b, c], 1);
        expect(res.shape).toEqual([2, 3]);
        test_util_1.expectArraysClose(res, [1, 3, 5, 2, 4, 6]);
    });
    it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=0', function () {
        var a = tf.tensor2d([[1, 2], [3, 4]]);
        var b = tf.tensor2d([[5, 6], [7, 8]]);
        var res = tf.stack([a, b], 0);
        expect(res.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('[[1,2],[3,4]] and [[5, 6], [7, 8]], axis=2', function () {
        var a = tf.tensor2d([[1, 2], [3, 4]]);
        var b = tf.tensor2d([[5, 6], [7, 8]]);
        var c = tf.tensor2d([[9, 10], [11, 12]]);
        var res = tf.stack([a, b, c], 2);
        expect(res.shape).toEqual([2, 2, 3]);
        test_util_1.expectArraysClose(res, [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]);
    });
    it('single tensor', function () {
        var a = tf.tensor2d([[1, 2], [3, 4]]);
        var res = tf.stack([a], 2);
        expect(res.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.stack([{}]); })
            .toThrowError(/Argument 'tensors\[0\]' passed to 'stack' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var a = [[1, 2], [3, 4]];
        var res = tf.stack([a], 2);
        expect(res.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
});
jasmine_util_1.describeWithFlags('unstack', test_util_1.ALL_ENVS, function () {
    it('unstack by default', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        var res = tf.unstack(x);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(1);
        expect(res[0].shape).toEqual([4]);
        test_util_1.expectArraysClose(res[0], [1, 2, 3, 4]);
        expect(res[1].rank).toEqual(1);
        expect(res[1].shape).toEqual([4]);
        test_util_1.expectArraysClose(res[1], [5, 6, 7, 8]);
    });
    it('unstack into 3 tensors', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        var res = tf.unstack(x, 0);
        expect(res.length).toEqual(3);
        expect(res[0].rank).toEqual(1);
        expect(res[0].shape).toEqual([2]);
        test_util_1.expectArraysClose(res[0], [1, 2]);
        expect(res[1].rank).toEqual(1);
        expect(res[1].shape).toEqual([2]);
        test_util_1.expectArraysClose(res[1], [3, 4]);
        expect(res[2].rank).toEqual(1);
        expect(res[2].shape).toEqual([2]);
        test_util_1.expectArraysClose(res[2], [5, 6]);
    });
    it('unstack by axis=1', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        var res = tf.unstack(x, 1);
        expect(res.length).toEqual(4);
        expect(res[0].rank).toEqual(1);
        expect(res[0].shape).toEqual([2]);
        test_util_1.expectArraysClose(res[0], [1, 5]);
        expect(res[1].rank).toEqual(1);
        expect(res[1].shape).toEqual([2]);
        test_util_1.expectArraysClose(res[1], [2, 6]);
        expect(res[2].rank).toEqual(1);
        expect(res[2].shape).toEqual([2]);
        test_util_1.expectArraysClose(res[2], [3, 7]);
        expect(res[3].rank).toEqual(1);
        expect(res[3].shape).toEqual([2]);
        test_util_1.expectArraysClose(res[3], [4, 8]);
    });
    it('unstack rank 3 tensor', function () {
        var x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var res = tf.unstack(x);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(2);
        expect(res[0].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[0], [1, 2, 3, 4]);
        expect(res[1].rank).toEqual(2);
        expect(res[1].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[1], [5, 6, 7, 8]);
    });
    it('unstack rank 3 tensor with axis=1', function () {
        var x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var res = tf.unstack(x, 1);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(2);
        expect(res[0].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[0], [1, 2, 5, 6]);
        expect(res[1].rank).toEqual(2);
        expect(res[1].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[1], [3, 4, 7, 8]);
    });
    it('unstack rank 3 tensor with axis=2', function () {
        var x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var res = tf.unstack(x, 2);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(2);
        expect(res[0].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[0], [1, 3, 5, 7]);
        expect(res[1].rank).toEqual(2);
        expect(res[1].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[1], [2, 4, 6, 8]);
    });
    it('unstack rank 4 tensor', function () {
        var x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
        var res = tf.unstack(x);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(3);
        expect(res[0].shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res[0], [1, 2, 3, 4]);
        expect(res[1].rank).toEqual(3);
        expect(res[1].shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res[1], [5, 6, 7, 8]);
    });
    it('unstack rank 4 tensor with axis=1', function () {
        var x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
        var res = tf.unstack(x, 1);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(3);
        expect(res[0].shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res[0], [1, 2, 5, 6]);
        expect(res[1].rank).toEqual(3);
        expect(res[1].shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res[1], [3, 4, 7, 8]);
    });
    it('unstack rank 4 tensor with axis=2', function () {
        var x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
        var res = tf.unstack(x, 2);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(3);
        expect(res[0].shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res[0], [1, 3, 5, 7]);
        expect(res[1].rank).toEqual(3);
        expect(res[1].shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(res[1], [2, 4, 6, 8]);
    });
    it('unstack rank 4 tensor with axis=3', function () {
        var x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
        var res = tf.unstack(x, 3);
        expect(res.length).toEqual(1);
        expect(res[0].rank).toEqual(3);
        expect(res[0].shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(res[0], [1, 2, 3, 4, 5, 6, 7, 8]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.unstack({}); })
            .toThrowError(/Argument 'x' passed to 'unstack' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var x = [[1, 2, 3, 4], [5, 6, 7, 8]];
        var res = tf.unstack(x);
        expect(res.length).toEqual(2);
        expect(res[0].rank).toEqual(1);
        expect(res[0].shape).toEqual([4]);
        test_util_1.expectArraysClose(res[0], [1, 2, 3, 4]);
        expect(res[1].rank).toEqual(1);
        expect(res[1].shape).toEqual([4]);
        test_util_1.expectArraysClose(res[1], [5, 6, 7, 8]);
    });
});
jasmine_util_1.describeWithFlags('split', test_util_1.ALL_ENVS, function () {
    it('split by number', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        var res = tf.split(x, 2, 1);
        expect(res.length).toEqual(2);
        expect(res[0].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[0], [1, 2, 5, 6]);
        expect(res[1].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[1], [3, 4, 7, 8]);
    });
    it('split by sizes', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        var res = tf.split(x, [1, 2, 1], 1);
        expect(res.length).toEqual(3);
        expect(res[0].shape).toEqual([2, 1]);
        test_util_1.expectArraysClose(res[0], [1, 5]);
        expect(res[1].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[1], [2, 3, 6, 7]);
        expect(res[2].shape).toEqual([2, 1]);
        test_util_1.expectArraysClose(res[2], [4, 8]);
    });
    it('chainable split by sizes', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        var res = x.split([1, 2, 1], 1);
        expect(res.length).toEqual(3);
        expect(res[0].shape).toEqual([2, 1]);
        test_util_1.expectArraysClose(res[0], [1, 5]);
        expect(res[1].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[1], [2, 3, 6, 7]);
        expect(res[2].shape).toEqual([2, 1]);
        test_util_1.expectArraysClose(res[2], [4, 8]);
    });
    it('sizes to not sum to axis size throws error', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        var f = function () { return tf.split(x, [1, 2], 1); };
        expect(f).toThrowError();
    });
    it('number of splits does not evenly divide axis', function () {
        var x = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
        var f = function () { return tf.split(x, 3, 1); };
        expect(f).toThrowError();
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.split({}, 1); })
            .toThrowError(/Argument 'x' passed to 'split' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var x = [[1, 2, 3, 4], [5, 6, 7, 8]];
        var res = tf.split(x, 2, 1);
        expect(res.length).toEqual(2);
        expect(res[0].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[0], [1, 2, 5, 6]);
        expect(res[1].shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res[1], [3, 4, 7, 8]);
    });
    it('gradient of 1st output', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var da = tf.grad(function (x) { return tf.split(x, [1, 2])[0]; })(a);
        expect(da.shape).toEqual([3]);
        test_util_1.expectArraysClose(da, [1, 0, 0]);
    });
    it('gradient of 2nd output', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var da = tf.grad(function (x) { return tf.split(x, [1, 2])[1]; })(a);
        expect(da.shape).toEqual([3]);
        test_util_1.expectArraysClose(da, [0, 1, 1]);
    });
});
jasmine_util_1.describeWithFlags('expandDims', test_util_1.ALL_ENVS, function () {
    it('scalar, default axis is 0', function () {
        var res = tf.scalar(1).expandDims();
        expect(res.shape).toEqual([1]);
        test_util_1.expectArraysClose(res, [1]);
    });
    it('scalar, axis is out of bounds throws error', function () {
        var f = function () { return tf.scalar(1).expandDims(1); };
        expect(f).toThrowError();
    });
    it('1d, axis=-3', function () {
        expect(function () {
            tf.tensor1d([1, 2, 3]).expandDims(-3);
        }).toThrowError('Axis must be in the interval [-2, 1]');
    });
    it('1d, axis=-2', function () {
        var res = tf.tensor1d([1, 2, 3]).expandDims(-2);
        expect(res.shape).toEqual([1, 3]);
        test_util_1.expectArraysClose(res, [1, 2, 3]);
    });
    it('1d, axis=-1', function () {
        var res = tf.tensor1d([1, 2, 3]).expandDims(-1);
        expect(res.shape).toEqual([3, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3]);
    });
    it('1d, axis=0', function () {
        var res = tf.tensor1d([1, 2, 3]).expandDims(0);
        expect(res.shape).toEqual([1, 3]);
        test_util_1.expectArraysClose(res, [1, 2, 3]);
    });
    it('1d, axis=1', function () {
        var res = tf.tensor1d([1, 2, 3]).expandDims(1);
        expect(res.shape).toEqual([3, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3]);
    });
    it('2d, axis=-4', function () {
        expect(function () {
            tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-4);
        }).toThrowError('Axis must be in the interval [-3, 2]');
    });
    it('2d, axis=-3', function () {
        var res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-3);
        expect(res.shape).toEqual([1, 3, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('2d, axis=-2', function () {
        var res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-2);
        expect(res.shape).toEqual([3, 1, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('2d, axis=-1', function () {
        var res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(-1);
        expect(res.shape).toEqual([3, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('2d, axis=0', function () {
        var res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(0);
        expect(res.shape).toEqual([1, 3, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('2d, axis=1', function () {
        var res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(1);
        expect(res.shape).toEqual([3, 1, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('2d, axis=2', function () {
        var res = tf.tensor2d([[1, 2], [3, 4], [5, 6]]).expandDims(2);
        expect(res.shape).toEqual([3, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
    it('4d, axis=0', function () {
        var res = tf.tensor4d([[[[4]]]]).expandDims();
        expect(res.shape).toEqual([1, 1, 1, 1, 1]);
        test_util_1.expectArraysClose(res, [4]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.expandDims({}); })
            .toThrowError(/Argument 'x' passed to 'expandDims' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var res = tf.expandDims(7);
        expect(res.shape).toEqual([1]);
        test_util_1.expectArraysClose(res, [7]);
    });
    it('works with 0 in shape', function () {
        var a = tf.tensor2d([], [0, 3]);
        var res = a.expandDims();
        expect(res.shape).toEqual([1, 0, 3]);
        test_util_1.expectArraysClose(res, []);
        var res2 = a.expandDims(1);
        expect(res2.shape).toEqual([0, 1, 3]);
        test_util_1.expectArraysClose(res2, []);
        var res3 = a.expandDims(2);
        expect(res3.shape).toEqual([0, 3, 1]);
        test_util_1.expectArraysClose(res3, []);
    });
});
jasmine_util_1.describeWithFlags('cumsum', test_util_1.ALL_ENVS, function () {
    it('1D standard', function () {
        var res = tf.tensor1d([1, 2, 3, 4]).cumsum();
        expect(res.shape).toEqual([4]);
        test_util_1.expectArraysClose(res, [1, 3, 6, 10]);
    });
    it('1D reverse', function () {
        var reverse = true;
        var exclusive = false;
        var res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive, reverse);
        expect(res.shape).toEqual([4]);
        test_util_1.expectArraysClose(res, [10, 9, 7, 4]);
    });
    it('1D exclusive', function () {
        var exclusive = true;
        var res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive);
        expect(res.shape).toEqual([4]);
        test_util_1.expectArraysClose(res, [0, 1, 3, 6]);
    });
    it('1D exclusive reverse', function () {
        var reverse = true;
        var exclusive = true;
        var res = tf.tensor1d([1, 2, 3, 4]).cumsum(0, exclusive, reverse);
        expect(res.shape).toEqual([4]);
        test_util_1.expectArraysClose(res, [9, 7, 4, 0]);
    });
    it('gradient: 1D', function () {
        var a = tf.tensor1d([1, 2, 3]);
        var dy = tf.tensor1d([4, 5, 6]);
        var da = tf.grad(function (x) { return tf.cumsum(x); })(a, dy);
        expect(da.shape).toEqual([3]);
        test_util_1.expectArraysClose(da, [15, 11, 6]);
    });
    it('2D standard', function () {
        var res = tf.tensor2d([[1, 2], [3, 4]]).cumsum(1);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [1, 3, 3, 7]);
    });
    it('2D reverse exclusive', function () {
        var reverse = true;
        var exclusive = true;
        var res = tf.tensor2d([[1, 2], [3, 4]]).cumsum(1, exclusive, reverse);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [2, 0, 4, 0]);
    });
    it('2D axis=0', function () {
        var res = tf.tensor2d([[1, 2], [3, 4]]).cumsum();
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 4, 6]);
    });
    it('3D standard', function () {
        var res = tf.tensor3d([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]).cumsum(2);
        expect(res.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(res, [0, 1, 2, 5, 4, 9, 6, 13]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.cumsum({}); })
            .toThrowError(/Argument 'x' passed to 'cumsum' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var res = tf.cumsum([1, 2, 3, 4]);
        expect(res.shape).toEqual([4]);
        test_util_1.expectArraysClose(res, [1, 3, 6, 10]);
    });
});
jasmine_util_1.describeWithFlags('batchToSpaceND', test_util_1.ALL_ENVS, function () {
    it('tensor4d, input shape=[4, 1, 1, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([1, 2, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
    it('tensor4d, input shape=[4, 1, 1, 3], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 1, 1, 3]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([1, 2, 2, 3]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    });
    it('tensor4d, input shape=[4, 2, 2, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16], [4, 2, 2, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([1, 4, 4, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    });
    it('tensor4d, input shape=[8, 1, 3, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([
            0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12,
            0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16
        ], [8, 1, 3, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [2, 0]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([2, 2, 4, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    });
    it('tensor2d, blockShape [1]', function () {
        var t = tf.tensor2d([1, 2, 3, 4], [2, 2]);
        var blockShape = [2];
        var crops = [[0, 0]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([1, 4]);
        test_util_1.expectArraysClose(res, [1, 3, 2, 4]);
    });
    it('tensor3d,  blockSHape [1]', function () {
        var t = tf.tensor([
            -61, 37, -68, 72, 31, 62, 0, -13, 28, 54, 96,
            44, -55, -64, -88, -94, 65, -32, -96, -73, -2, -77,
            -14, 47, 33, 15, 70, 20, 75, 28, 84, -13
        ], [8, 2, 2]);
        var blockShape = [2];
        var crops = [[0, 2]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([4, 2, 2]);
        test_util_1.expectArraysClose(res, [-61, 37, 65, -32, 31, 62, -2, -77, 28, 54, 33, 15, -55, -64, 75, 28]);
    });
    it('tensor3d, blockShape [2]', function () {
        var t = tf.tensor([
            -61, 37, -68, 72, 31, 62, 0, -13, 28, 54, 96,
            44, -55, -64, -88, -94, 65, -32, -96, -73, -2, -77,
            -14, 47, 33, 15, 70, 20, 75, 28, 84, -13
        ], [8, 2, 2]);
        var blockShape = [2, 2];
        var crops = [[2, 0], [2, 0]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([2, 2, 2]);
        test_util_1.expectArraysClose(res, [72, 44, -73, 20, -13, -94, 47, -13]);
    });
    it('throws when blockShape equal to input rank', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
        var blockShape = [2, 2, 2, 2];
        var crops = [[0, 0], [0, 0], [0, 0], [0, 0]];
        expect(function () { return tf.batchToSpaceND(t, blockShape, crops); })
            .toThrowError("input rank is " + t.rank + " but should be > than blockShape.length " + blockShape.length);
    });
    it('throws when crops row dimension not equal to blockshape', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0]];
        expect(function () { return tf.batchToSpaceND(t, blockShape, crops); })
            .toThrowError("crops.length is " + crops.length + " but should be equal to blockShape.length  " + blockShape.length);
    });
    it('throws when input tensor batch not divisible by prod(blockShape)', function () {
        var t = tf.tensor4d([1, 2, 3, 4, 5], [5, 1, 1, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var prod = blockShape.reduce(function (a, b) { return a * b; });
        expect(function () { return tf.batchToSpaceND(t, blockShape, crops); })
            .toThrowError("input tensor batch is " + t.shape[0] + " but is not divisible by the " +
            ("product of the elements of blockShape " + blockShape.join(' * ') + " === " + prod));
    });
    it('accepts a tensor-like object', function () {
        var t = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]];
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var res = tf.batchToSpaceND(t, blockShape, crops);
        expect(res.shape).toEqual([1, 2, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
    it('gradients,  input shape=[4, 2, 2], block shape=[2]', function () {
        var t = tf.tensor([-61, 37, -68, 72, 31, 62, 0, -13, 28, 54, 96, 44, -55, -64, -88, -94], [4, 2, 2]);
        var blockShape = [2];
        var crops = [[0, 2]];
        var dy = tf.tensor([.01, .02, .03, .04, .05, .06, .07, .08], [2, 2, 2]);
        var gradient = tf.grad(function (t) { return tf.batchToSpaceND(t, blockShape, crops); })(t, dy);
        expect(gradient.shape).toEqual([4, 2, 2]);
        test_util_1.expectArraysClose(gradient, [
            0.01, 0.02, 0, 0, 0.05, 0.06, 0, 0, 0.03, 0.04, 0, 0, 0.07, 0.08, 0, 0
        ]);
    });
    it('gradients, input shape=[4, 2, 2, 1], block shape=[2, 2]', function () {
        var t = tf.tensor4d([1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16], [4, 2, 2, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var dy = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 4, 1]);
        var gradient = tf.grad(function (t) { return tf.batchToSpaceND(t, blockShape, crops); })(t, dy);
        expect(gradient.shape).toEqual([4, 2, 2, 1]);
        test_util_1.expectArraysClose(gradient, [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16]);
    });
});
jasmine_util_1.describeWithFlags('spaceToBatchND', test_util_1.ALL_ENVS, function () {
    it('tensor4d, input shape=[1, 2, 2, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([[[[1], [2]], [[3], [4]]]], [1, 2, 2, 1]);
        var blockShape = [2, 2];
        var paddings = [[0, 0], [0, 0]];
        var res = tf.spaceToBatchND(t, blockShape, paddings);
        expect(res.shape).toEqual([4, 1, 1, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
    it('tensor4d, input shape=[1, 2, 2, 3], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], [1, 2, 2, 3]);
        var blockShape = [2, 2];
        var paddings = [[0, 0], [0, 0]];
        var res = tf.spaceToBatchND(t, blockShape, paddings);
        expect(res.shape).toEqual([4, 1, 1, 3]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    });
    it('tensor4d, input shape=[1, 4, 4, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([[
                [[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]],
                [[13], [14], [15], [16]]
            ]], [1, 4, 4, 1]);
        var blockShape = [2, 2];
        var paddings = [[0, 0], [0, 0]];
        var res = tf.spaceToBatchND(t, blockShape, paddings);
        expect(res.shape).toEqual([4, 2, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16]);
    });
    it('tensor4d, input shape=[2, 6, 6, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ], [2, 6, 6, 1]);
        var blockShape = [2, 2];
        var paddings = [[0, 0], [0, 0]];
        var res = tf.spaceToBatchND(t, blockShape, paddings);
        expect(res.shape).toEqual([8, 3, 3, 1]);
        test_util_1.expectArraysClose(res, [
            1, 3, 5, 13, 15, 17, 25, 27, 29, 37, 39, 41, 49, 51, 53, 61, 63, 65,
            2, 4, 6, 14, 16, 18, 26, 28, 30, 38, 40, 42, 50, 52, 54, 62, 64, 66,
            7, 9, 11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71,
            8, 10, 12, 20, 22, 24, 32, 34, 36, 44, 46, 48, 56, 58, 60, 68, 70, 72
        ]);
    });
    it('tensor4d, input shape=[2, 2, 4, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([
            [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
            [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]
        ], [2, 2, 4, 1]);
        var blockShape = [2, 2];
        var paddings = [[0, 0], [2, 0]];
        var res = tf.spaceToBatchND(t, blockShape, paddings);
        expect(res.shape).toEqual([8, 1, 3, 1]);
        test_util_1.expectArraysClose(res, [
            0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12,
            0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16
        ]);
    });
    it('tensor2d, blockShape [2]', function () {
        var t = tf.tensor2d([1, 3, 2, 4], [1, 4]);
        var blockShape = [2];
        var paddings = [[0, 0]];
        var res = tf.spaceToBatchND(t, blockShape, paddings);
        expect(res.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
    it('throws when blockShape equal to input rank', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
        var blockShape = [2, 2, 2, 2];
        var paddings = [[0, 0], [0, 0], [0, 0], [0, 0]];
        expect(function () { return tf.spaceToBatchND(t, blockShape, paddings); })
            .toThrowError('input rank 4 should be > than [blockShape] 4');
    });
    it('throws when paddings row dimension not equal to blockshape', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
        var blockShape = [2, 2];
        var paddings = [[0, 0]];
        expect(function () { return tf.spaceToBatchND(t, blockShape, paddings); })
            .toThrowError('paddings.shape[0] 1 must be equal to [blockShape] 2');
    });
    it('throws when input tensor spatial dimension not divisible by blockshapes', function () {
        var t = tf.tensor4d([1, 2, 3, 4, 5, 6], [1, 2, 3, 1]);
        var blockShape = [2, 2];
        var paddings = [[0, 0], [0, 0]];
        expect(function () { return tf.spaceToBatchND(t, blockShape, paddings); })
            .toThrowError('input spatial dimensions 2,3,1 with paddings 0,0,0,0 must be ' +
            'divisible by blockShapes 2,2');
    });
    it('accepts a tensor-like object', function () {
        var t = [[[[1], [2]], [[3], [4]]]];
        var blockShape = [2, 2];
        var paddings = [[0, 0], [0, 0]];
        var res = tf.spaceToBatchND(t, blockShape, paddings);
        expect(res.shape).toEqual([4, 1, 1, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
});
jasmine_util_1.describeWithFlags('batchToSpaceND X spaceToBatchND', test_util_1.ALL_ENVS, function () {
    it('tensor4d, input shape=[4, 1, 1, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var paddings = [[0, 0], [0, 0]];
        var b2s = tf.batchToSpaceND(t, blockShape, crops);
        expect(b2s.shape).toEqual([1, 2, 2, 1]);
        test_util_1.expectArraysClose(b2s, [1, 2, 3, 4]);
        var s2b = tf.spaceToBatchND(b2s, blockShape, paddings);
        expect(s2b.shape).toEqual([4, 1, 1, 1]);
        test_util_1.expectArraysClose(s2b, [1, 2, 3, 4]);
    });
    it('tensor4d, input shape=[2, 6, 6, 1], blockShape=[2, 2]', function () {
        var t = tf.tensor4d([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ], [2, 6, 6, 1]);
        var blockShape = [2, 2];
        var crops = [[0, 0], [0, 0]];
        var paddings = [[0, 0], [0, 0]];
        var s2b = tf.spaceToBatchND(t, blockShape, paddings);
        expect(s2b.shape).toEqual([8, 3, 3, 1]);
        test_util_1.expectArraysClose(s2b, [
            1, 3, 5, 13, 15, 17, 25, 27, 29, 37, 39, 41, 49, 51, 53, 61, 63, 65,
            2, 4, 6, 14, 16, 18, 26, 28, 30, 38, 40, 42, 50, 52, 54, 62, 64, 66,
            7, 9, 11, 19, 21, 23, 31, 33, 35, 43, 45, 47, 55, 57, 59, 67, 69, 71,
            8, 10, 12, 20, 22, 24, 32, 34, 36, 44, 46, 48, 56, 58, 60, 68, 70, 72
        ]);
        var b2s = tf.batchToSpaceND(s2b, blockShape, crops);
        expect(b2s.shape).toEqual([2, 6, 6, 1]);
        test_util_1.expectArraysClose(b2s, [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ]);
    });
    it('gradients,  input shape=[4, 2, 2], block shape=[2]', function () {
        var t = tf.tensor([-61, 37, -68, 72, 31, 62, 0, -13, 28, 54, 96, 44, -55, -64, -88, -94], [4, 2, 2]);
        var blockShape = [2];
        var paddings = [[0, 2]];
        var dy = tf.tensor([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ], [8, 2, 2]);
        var gradient = tf.grad(function (t) { return tf.spaceToBatchND(t, blockShape, paddings); })(t, dy);
        expect(gradient.shape).toEqual([4, 2, 2]);
        test_util_1.expectArraysClose(gradient, [1, 2, 17, 18, 5, 6, 21, 22, 9, 10, 25, 26, 13, 14, 29, 30]);
    });
    it('gradients, input shape=[2, 2, 4, 1], block shape=[2, 2]', function () {
        var t = tf.tensor4d([
            [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
            [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]
        ], [2, 2, 4, 1]);
        var blockShape = [2, 2];
        var paddings = [[0, 0], [2, 0]];
        var dy = tf.tensor([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ], [8, 1, 3, 1]);
        var gradient = tf.grad(function (t) { return tf.spaceToBatchND(t, blockShape, paddings); })(t, dy);
        expect(gradient.shape).toEqual([2, 2, 4, 1]);
        test_util_1.expectArraysClose(gradient, [2, 8, 3, 9, 14, 20, 15, 21, 5, 11, 6, 12, 17, 23, 18, 24]);
    });
});
jasmine_util_1.describeWithFlags('depthToSpace', test_util_1.ALL_ENVS, function () {
    it('tensor4d, input shape=[1, 1, 1, 4], blockSize=2, format=NHWC', function () {
        var t = tf.tensor4d([[[[1, 2, 3, 4]]]]);
        var blockSize = 2;
        var dataFormat = 'NHWC';
        var res = tf.depthToSpace(t, blockSize, dataFormat);
        expect(res.shape).toEqual([1, 2, 2, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
    it('tensor4d, input shape=[1, 1, 1, 12], blockSize=2, format=NHWC', function () {
        var t = tf.tensor4d([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]);
        var blockSize = 2;
        var dataFormat = 'NHWC';
        var res = tf.depthToSpace(t, blockSize, dataFormat);
        expect(res.shape).toEqual([1, 2, 2, 3]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    });
    it('tensor4d, input shape=[1, 2, 2, 4], blockSize=2, format=NHWC', function () {
        var t = tf.tensor4d([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]]);
        var blockSize = 2;
        var dataFormat = 'NHWC';
        var res = tf.depthToSpace(t, blockSize, dataFormat);
        expect(res.shape).toEqual([1, 4, 4, 1]);
        test_util_1.expectArraysClose(res, [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]);
    });
    it('throws when depth not divisible by blockSize * blockSize', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
        var blockSize = 3;
        expect(function () { return tf.depthToSpace(t, blockSize); })
            .toThrowError("Dimension size must be evenly divisible by " + blockSize * blockSize + " but is " + t.shape[3] + " for depthToSpace with input shape " + t.shape);
    });
});
jasmine_util_1.describeWithFlags('depthToSpace', test_util_1.BROWSER_ENVS, function () {
    it('throws when blocksize < 2', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [1, 1, 1, 4]);
        var blockSize = 1;
        expect(function () { return tf.depthToSpace(t, blockSize); })
            .toThrowError("blockSize should be > 1 for depthToSpace, but was: " + blockSize);
    });
});
jasmine_util_1.describeWithFlags('depthToSpace', test_util_1.CPU_ENVS, function () {
    it('throws when CPU backend used with data format NCHW', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [1, 4, 1, 1]);
        var blockSize = 2;
        var dataFormat = 'NCHW';
        expect(function () { return tf.depthToSpace(t, blockSize, dataFormat); })
            .toThrowError("Only NHWC dataFormat supported on CPU for depthToSpace. Got " + dataFormat);
    });
});
jasmine_util_1.describeWithFlags('depthToSpace', test_util_1.WEBGL_ENVS, function () {
    it('tensor4d, input shape=[1, 4, 1, 1], blockSize=2, format=NCHW', function () {
        var t = tf.tensor4d([1, 2, 3, 4], [1, 4, 1, 1]);
        var blockSize = 2;
        var dataFormat = 'NCHW';
        var res = tf.depthToSpace(t, blockSize, dataFormat);
        expect(res.shape).toEqual([1, 1, 2, 2]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4]);
    });
    it('tensor4d, input shape=[1, 12, 1, 1], blockSize=2, format=NCHW', function () {
        var t = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 12, 1, 1]);
        var blockSize = 2;
        var dataFormat = 'NCHW';
        var res = tf.depthToSpace(t, blockSize, dataFormat);
        expect(res.shape).toEqual([1, 3, 2, 2]);
        test_util_1.expectArraysClose(res, [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]);
    });
    it('tensor4d, input shape=[1, 4, 2, 2], blockSize=2, format=NCHW', function () {
        var t = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [1, 4, 2, 2]);
        var blockSize = 2;
        var dataFormat = 'NCHW';
        var res = tf.depthToSpace(t, blockSize, dataFormat);
        expect(res.shape).toEqual([1, 1, 4, 4]);
        test_util_1.expectArraysClose(res, [1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16]);
    });
    it('tensor4d, input shape=[1, 8, 2, 2], blockSize=2, format=NCHW', function () {
        var t = tf.tensor4d([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ], [1, 8, 2, 2]);
        var blockSize = 2;
        var dataFormat = 'NCHW';
        var res = tf.depthToSpace(t, blockSize, dataFormat);
        expect(res.shape).toEqual([1, 2, 4, 4]);
        test_util_1.expectArraysClose(res, [
            1, 9, 2, 10, 17, 25, 18, 26, 3, 11, 4, 12, 19, 27, 20, 28,
            5, 13, 6, 14, 21, 29, 22, 30, 7, 15, 8, 16, 23, 31, 24, 32
        ]);
    });
});
jasmine_util_1.describeWithFlags('setdiff1dAsync', test_util_1.ALL_ENVS, function () {
    it('1d int32 tensor', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, y, _a, out, indices;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    x = tf.tensor1d([1, 2, 3, 4], 'int32');
                    y = tf.tensor1d([1, 2], 'int32');
                    return [4, tf.setdiff1dAsync(x, y)];
                case 1:
                    _a = _b.sent(), out = _a[0], indices = _a[1];
                    expect(out.dtype).toBe('int32');
                    expect(indices.dtype).toBe('int32');
                    expect(out.shape).toEqual([2]);
                    expect(indices.shape).toEqual([2]);
                    test_util_1.expectArraysClose(out, [3, 4]);
                    test_util_1.expectArraysClose(indices, [2, 3]);
                    return [2];
            }
        });
    }); });
    it('1d float32 tensor', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, y, _a, out, indices;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    x = tf.tensor1d([1, 2, 3, 4], 'float32');
                    y = tf.tensor1d([1, 3], 'float32');
                    return [4, tf.setdiff1dAsync(x, y)];
                case 1:
                    _a = _b.sent(), out = _a[0], indices = _a[1];
                    expect(out.dtype).toBe('float32');
                    expect(indices.dtype).toBe('int32');
                    expect(out.shape).toEqual([2]);
                    expect(indices.shape).toEqual([2]);
                    test_util_1.expectArraysClose(out, [2, 4]);
                    test_util_1.expectArraysClose(indices, [1, 3]);
                    return [2];
            }
        });
    }); });
    it('empty output', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, y, _a, out, indices;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    x = tf.tensor1d([1, 2, 3, 4], 'float32');
                    y = tf.tensor1d([1, 2, 3, 4], 'float32');
                    return [4, tf.setdiff1dAsync(x, y)];
                case 1:
                    _a = _b.sent(), out = _a[0], indices = _a[1];
                    expect(out.dtype).toBe('float32');
                    expect(indices.dtype).toBe('int32');
                    expect(out.shape).toEqual([0]);
                    expect(indices.shape).toEqual([0]);
                    test_util_1.expectArraysClose(out, []);
                    test_util_1.expectArraysClose(indices, []);
                    return [2];
            }
        });
    }); });
    it('tensor like', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, y, _a, out, indices;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    x = [1, 2, 3, 4];
                    y = [1, 3];
                    return [4, tf.setdiff1dAsync(x, y)];
                case 1:
                    _a = _b.sent(), out = _a[0], indices = _a[1];
                    expect(out.dtype).toBe('float32');
                    expect(indices.dtype).toBe('int32');
                    expect(out.shape).toEqual([2]);
                    expect(indices.shape).toEqual([2]);
                    test_util_1.expectArraysClose(out, [2, 4]);
                    test_util_1.expectArraysClose(indices, [1, 3]);
                    return [2];
            }
        });
    }); });
    it('should throw if x is not 1d', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, y, ex_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    x = tf.tensor2d([1, 2, 3, 4], [4, 1], 'float32');
                    y = tf.tensor1d([1, 2, 3, 4], 'float32');
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, tf.setdiff1dAsync(x, y)];
                case 2:
                    _a.sent();
                    throw new Error('The line above should have thrown an error');
                case 3:
                    ex_1 = _a.sent();
                    expect(ex_1.message).toBe('x should be 1D tensor, but got x (4,1).');
                    return [3, 4];
                case 4: return [2];
            }
        });
    }); });
    it('should throw if y is not 1d', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, y, ex_2;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    x = tf.tensor1d([1, 2, 3, 4], 'float32');
                    y = tf.tensor2d([1, 2, 3, 4], [4, 1], 'float32');
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, tf.setdiff1dAsync(x, y)];
                case 2:
                    _a.sent();
                    throw new Error('The line above should have thrown an error');
                case 3:
                    ex_2 = _a.sent();
                    expect(ex_2.message).toBe('y should be 1D tensor, but got y (4,1).');
                    return [3, 4];
                case 4: return [2];
            }
        });
    }); });
    it('should throw if x and y dtype mismatch', function () { return __awaiter(_this, void 0, void 0, function () {
        var x, y, ex_3;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    x = tf.tensor1d([1, 2, 3, 4], 'float32');
                    y = tf.tensor1d([1, 2, 3, 4], 'int32');
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, tf.setdiff1dAsync(x, y)];
                case 2:
                    _a.sent();
                    throw new Error('The line above should have thrown an error');
                case 3:
                    ex_3 = _a.sent();
                    expect(ex_3.message)
                        .toBe('x and y should have the same dtype,' +
                        ' but got x (float32) and y (int32).');
                    return [3, 4];
                case 4: return [2];
            }
        });
    }); });
});
//# sourceMappingURL=array_ops_test.js.map