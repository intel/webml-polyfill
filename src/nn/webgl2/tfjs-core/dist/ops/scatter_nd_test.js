"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('scatterND', test_util_1.ALL_ENVS, function () {
    it('should work for 2d', function () {
        var indices = tf.tensor1d([0, 4, 2], 'int32');
        var updates = tf.tensor2d([100, 101, 102, 777, 778, 779, 1000, 1001, 1002], [3, 3], 'int32');
        var shape = [5, 3];
        var result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        test_util_1.expectArraysClose(result, [100, 101, 102, 0, 0, 0, 1000, 1001, 1002, 0, 0, 0, 777, 778, 779]);
    });
    it('should work for simple 1d', function () {
        var indices = tf.tensor1d([3], 'int32');
        var updates = tf.tensor1d([101], 'float32');
        var shape = [5];
        var result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        test_util_1.expectArraysClose(result, [0, 0, 0, 101, 0]);
    });
    it('should work for multiple 1d', function () {
        var indices = tf.tensor1d([0, 4, 2], 'int32');
        var updates = tf.tensor1d([100, 101, 102], 'float32');
        var shape = [5];
        var result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        test_util_1.expectArraysClose(result, [100, 0, 102, 0, 101]);
    });
    it('should work for high rank updates', function () {
        var indices = tf.tensor2d([0, 2], [2, 1], 'int32');
        var updates = tf.tensor3d([
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ], [2, 4, 4], 'float32');
        var shape = [4, 4, 4];
        var result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        test_util_1.expectArraysClose(result, [
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
            8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]);
    });
    it('should work for high rank indices', function () {
        var indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'int32');
        var updates = tf.tensor1d([10, 20], 'float32');
        var shape = [3, 3];
        var result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        test_util_1.expectArraysClose(result, [0, 20, 10, 0, 0, 0, 0, 0, 0]);
    });
    it('should sum the duplicated indices', function () {
        var indices = tf.tensor1d([0, 4, 2, 1, 3, 0], 'int32');
        var updates = tf.tensor1d([10, 20, 30, 40, 50, 60], 'float32');
        var shape = [8];
        var result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(updates.dtype);
        test_util_1.expectArraysClose(result, [70, 40, 30, 50, 20, 0, 0, 0]);
    });
    it('should work for tensorLike input', function () {
        var indices = [0, 4, 2];
        var updates = [100, 101, 102];
        var shape = [5];
        var result = tf.scatterND(indices, updates, shape);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual('float32');
        test_util_1.expectArraysClose(result, [100, 0, 102, 0, 101]);
    });
    it('should throw error when indices type is not int32', function () {
        var indices = tf.tensor2d([0, 2, 0, 1], [2, 2], 'float32');
        var updates = tf.tensor1d([10, 20], 'float32');
        var shape = [3, 3];
        expect(function () { return tf.scatterND(indices, updates, shape); }).toThrow();
    });
    it('should throw error when indices and update mismatch', function () {
        var indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
        var updates = tf.tensor2d([100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004], [3, 4], 'float32');
        var shape = [5, 3];
        expect(function () { return tf.scatterND(indices, updates, shape); }).toThrow();
    });
    it('should throw error when indices and update count mismatch', function () {
        var indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
        var updates = tf.tensor2d([100, 101, 102, 10000, 10001, 10002], [2, 3], 'float32');
        var shape = [5, 3];
        expect(function () { return tf.scatterND(indices, updates, shape); }).toThrow();
    });
    it('should throw error when indices are scalar', function () {
        var indices = tf.scalar(1, 'int32');
        var updates = tf.tensor2d([100, 101, 102, 10000, 10001, 10002], [2, 3], 'float32');
        var shape = [5, 3];
        expect(function () { return tf.scatterND(indices, updates, shape); }).toThrow();
    });
    it('should throw error when update is scalar', function () {
        var indices = tf.tensor2d([0, 4, 2], [3, 1], 'int32');
        var updates = tf.scalar(1, 'float32');
        var shape = [5, 3];
        expect(function () { return tf.scatterND(indices, updates, shape); }).toThrow();
    });
});
jasmine_util_1.describeWithFlags('scatterND CPU', test_util_1.CPU_ENVS, function () {
    it('should throw error when index out of range', function () {
        var indices = tf.tensor2d([0, 4, 99], [3, 1], 'int32');
        var updates = tf.tensor2d([100, 101, 102, 777, 778, 779, 10000, 10001, 10002], [3, 3], 'float32');
        var shape = [5, 3];
        expect(function () { return tf.scatterND(indices, updates, shape); }).toThrow();
    });
    it('should throw error when indices has wrong dimension', function () {
        var indices = tf.tensor2d([0, 4, 99], [3, 1], 'int32');
        var updates = tf.tensor2d([100, 101, 102, 777, 778, 779, 10000, 10001, 10002], [3, 3], 'float32');
        var shape = [2, 3];
        expect(function () { return tf.scatterND(indices, updates, shape); }).toThrow();
    });
});
//# sourceMappingURL=scatter_nd_test.js.map