"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var gather_nd_1 = require("./gather_nd");
var tensor_ops_1 = require("./tensor_ops");
jasmine_util_1.describeWithFlags('gatherND', test_util_1.ALL_ENVS, function () {
    it('should work for simple slice', function () {
        var indices = tensor_ops_1.tensor2d([0, 4, 8], [3, 1], 'int32');
        var input = tensor_ops_1.tensor1d([100, 101, 102, 777, 778, 779, 1000, 1001, 1002], 'int32');
        var shape = [3];
        var result = gather_nd_1.gatherND(input, indices);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(input.dtype);
        test_util_1.expectArraysClose(result, [100, 778, 1002]);
    });
    it('should work for indexing 2d', function () {
        var indices = tensor_ops_1.tensor2d([0, 2], [2, 1], 'int32');
        var input = tensor_ops_1.tensor2d([
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ], [8, 4], 'float32');
        var shape = [2, 4];
        var result = gather_nd_1.gatherND(input, indices);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(input.dtype);
        test_util_1.expectArraysClose(result, [5, 5, 5, 5, 7, 7, 7, 7]);
    });
    it('should work for indexing 3d', function () {
        var indices = tensor_ops_1.tensor2d([0, 2, 1, 1], [2, 2], 'int32');
        var input = tensor_ops_1.tensor3d([
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ], [2, 4, 4], 'float32');
        var shape = [2, 4];
        var result = gather_nd_1.gatherND(input, indices);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(input.dtype);
        test_util_1.expectArraysClose(result, [7, 7, 7, 7, 6, 6, 6, 6]);
    });
    it('should work for batch slice', function () {
        var indices = tensor_ops_1.tensor3d([0, 4, 2], [3, 1, 1], 'int32');
        var input = tensor_ops_1.tensor1d([100, 101, 102, 777, 778, 779, 10000, 10001, 10002], 'int32');
        var shape = [3, 1];
        var result = gather_nd_1.gatherND(input, indices);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(input.dtype);
        test_util_1.expectArraysClose(result, [100, 778, 102]);
    });
    it('should work for batch indexing 2d', function () {
        var indices = tensor_ops_1.tensor3d([0, 2], [2, 1, 1], 'int32');
        var input = tensor_ops_1.tensor2d([
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ], [8, 4], 'float32');
        var shape = [2, 1, 4];
        var result = gather_nd_1.gatherND(input, indices);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(input.dtype);
        test_util_1.expectArraysClose(result, [5, 5, 5, 5, 7, 7, 7, 7]);
    });
    it('should work for batch indexing 3d', function () {
        var indices = tensor_ops_1.tensor3d([0, 2, 1, 1], [2, 1, 2], 'int32');
        var input = tensor_ops_1.tensor3d([
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
            5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8
        ], [2, 4, 4], 'float32');
        var shape = [2, 1, 4];
        var result = gather_nd_1.gatherND(input, indices);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(input.dtype);
        test_util_1.expectArraysClose(result, [7, 7, 7, 7, 6, 6, 6, 6]);
    });
    it('should work for TensorLike inputs', function () {
        var indices = [[0], [4], [8]];
        var input = [100, 101, 102, 777, 778, 779, 1000, 1001, 1002];
        var shape = [3];
        var result = gather_nd_1.gatherND(input, indices);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual('float32');
        test_util_1.expectArraysClose(result, [100, 778, 1002]);
    });
    it('should throw error when indices are not int32', function () {
        var indices = tensor_ops_1.tensor1d([1], 'float32');
        var input = tensor_ops_1.tensor2d([100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004], [3, 4], 'float32');
        expect(function () { return gather_nd_1.gatherND(input, indices); }).toThrow();
    });
    it('should throw error when indices are scalar', function () {
        var indices = tensor_ops_1.scalar(1, 'int32');
        var input = tensor_ops_1.tensor2d([100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004], [3, 4], 'float32');
        expect(function () { return gather_nd_1.gatherND(input, indices); }).toThrow();
    });
    it('should throw error when x is scalar', function () {
        var indices = tensor_ops_1.tensor2d([0, 4, 2], [3, 1], 'int32');
        var input = tensor_ops_1.scalar(1.0, 'float32');
        expect(function () { return gather_nd_1.gatherND(input, indices); }).toThrow();
    });
    it('should throw error when indices inner dim > x shape length', function () {
        var indices = tensor_ops_1.tensor2d([0, 4, 2], [1, 3], 'int32');
        var input = tensor_ops_1.tensor2d([100, 101, 102, 10000, 10001, 10002], [3, 2], 'float32');
        expect(function () { return gather_nd_1.gatherND(input, indices); }).toThrow();
    });
});
jasmine_util_1.describeWithFlags('gatherND CPU', test_util_1.CPU_ENVS, function () {
    it('should throw error when index out of range', function () {
        var indices = tensor_ops_1.tensor2d([0, 2, 99], [3, 1], 'int32');
        var input = tensor_ops_1.tensor2d([100, 101, 102, 777, 778, 779, 10000, 10001, 10002], [3, 3], 'float32');
        expect(function () { return gather_nd_1.gatherND(input, indices); }).toThrow();
    });
});
//# sourceMappingURL=gather_nd_test.js.map