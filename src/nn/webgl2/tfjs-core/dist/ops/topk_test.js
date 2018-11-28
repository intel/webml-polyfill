"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var tensor_ops_1 = require("./tensor_ops");
jasmine_util_1.describeWithFlags('topk', test_util_1.ALL_ENVS, function () {
    it('1d array with default k', function () {
        var a = tensor_ops_1.tensor1d([20, 10, 40, 30]);
        var _a = tf.topk(a), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([1]);
        expect(indices.shape).toEqual([1]);
        expect(values.dtype).toBe('float32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [40]);
        test_util_1.expectArraysClose(indices, [2]);
    });
    it('1d array with default k from tensor.topk', function () {
        var a = tensor_ops_1.tensor1d([20, 10, 40, 30]);
        var _a = a.topk(), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([1]);
        expect(indices.shape).toEqual([1]);
        expect(values.dtype).toBe('float32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [40]);
        test_util_1.expectArraysClose(indices, [2]);
    });
    it('2d array with default k', function () {
        var a = tensor_ops_1.tensor2d([[10, 50], [40, 30]]);
        var _a = tf.topk(a), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([2, 1]);
        expect(indices.shape).toEqual([2, 1]);
        expect(values.dtype).toBe('float32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [50, 40]);
        test_util_1.expectArraysClose(indices, [1, 0]);
    });
    it('2d array with k=2', function () {
        var a = tensor_ops_1.tensor2d([
            [1, 5, 2],
            [4, 3, 6],
            [3, 2, 1],
            [1, 2, 3],
        ]);
        var k = 2;
        var _a = tf.topk(a, k), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([4, 2]);
        expect(indices.shape).toEqual([4, 2]);
        expect(values.dtype).toBe('float32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [5, 2, 6, 4, 3, 2, 3, 2]);
        test_util_1.expectArraysClose(indices, [1, 2, 2, 0, 0, 1, 2, 1]);
    });
    it('2d array with k=2 from tensor.topk', function () {
        var a = tensor_ops_1.tensor2d([
            [1, 5, 2],
            [4, 3, 6],
            [3, 2, 1],
            [1, 2, 3],
        ]);
        var k = 2;
        var _a = a.topk(k), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([4, 2]);
        expect(indices.shape).toEqual([4, 2]);
        expect(values.dtype).toBe('float32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [5, 2, 6, 4, 3, 2, 3, 2]);
        test_util_1.expectArraysClose(indices, [1, 2, 2, 0, 0, 1, 2, 1]);
    });
    it('3d array with k=3', function () {
        var a = tensor_ops_1.tensor3d([
            [[1, 5, 2], [4, 3, 6]],
            [[3, 2, 1], [1, 2, 3]],
        ]);
        var k = 3;
        var _a = tf.topk(a, k), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([2, 2, 3]);
        expect(indices.shape).toEqual([2, 2, 3]);
        expect(values.dtype).toBe('float32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [5, 2, 1, 6, 4, 3, 3, 2, 1, 3, 2, 1]);
        test_util_1.expectArraysClose(indices, [1, 2, 0, 2, 0, 1, 0, 1, 2, 2, 1, 0]);
    });
    it('topk(int32) propagates int32 dtype', function () {
        var a = tensor_ops_1.tensor1d([2, 3, 1, 4], 'int32');
        var _a = tf.topk(a), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([1]);
        expect(indices.shape).toEqual([1]);
        expect(values.dtype).toBe('int32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [4]);
        test_util_1.expectArraysClose(indices, [3]);
    });
    it('lower-index element appears first, k=4', function () {
        var a = tensor_ops_1.tensor1d([1, 2, 2, 1], 'int32');
        var k = 4;
        var _a = tf.topk(a, k), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([4]);
        expect(indices.shape).toEqual([4]);
        expect(values.dtype).toBe('int32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [2, 2, 1, 1]);
        test_util_1.expectArraysClose(indices, [1, 2, 0, 3]);
    });
    it('throws when k > size of array', function () {
        var a = tensor_ops_1.tensor2d([[10, 50], [40, 30]]);
        expect(function () { return tf.topk(a, 3); })
            .toThrowError(/'k' passed to topk\(\) must be <= the last dimension/);
    });
    it('throws when passed a scalar', function () {
        var a = tensor_ops_1.scalar(2);
        expect(function () { return tf.topk(a); })
            .toThrowError(/topk\(\) expects the input to be of rank 1 or higher/);
    });
    it('accepts a tensor-like object, k=2', function () {
        var a = [20, 10, 40, 30];
        var k = 2;
        var _a = tf.topk(a, k), values = _a.values, indices = _a.indices;
        expect(values.shape).toEqual([2]);
        expect(indices.shape).toEqual([2]);
        expect(values.dtype).toBe('float32');
        expect(indices.dtype).toBe('int32');
        test_util_1.expectArraysClose(values, [40, 30]);
        test_util_1.expectArraysClose(indices, [2, 3]);
    });
});
//# sourceMappingURL=topk_test.js.map