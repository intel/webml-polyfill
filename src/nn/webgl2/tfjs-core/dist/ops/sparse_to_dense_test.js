"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var sparse_to_dense_1 = require("./sparse_to_dense");
var tensor_ops_1 = require("./tensor_ops");
var defaultValue;
jasmine_util_1.describeWithFlags('sparseToDense', test_util_1.ALL_ENVS, function () {
    beforeEach(function () { return defaultValue = tensor_ops_1.scalar(0, 'int32'); });
    it('should work for scalar indices', function () {
        var indices = tensor_ops_1.scalar(2, 'int32');
        var values = tensor_ops_1.scalar(100, 'int32');
        var shape = [6];
        var result = sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(values.dtype);
        test_util_1.expectArraysClose(result, [0, 0, 100, 0, 0, 0]);
    });
    it('should work for vector', function () {
        var indices = tensor_ops_1.tensor1d([0, 2, 4], 'int32');
        var values = tensor_ops_1.tensor1d([100, 101, 102], 'int32');
        var shape = [6];
        var result = sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(values.dtype);
        test_util_1.expectArraysClose(result, [100, 0, 101, 0, 102, 0]);
    });
    it('should work for scalar value', function () {
        var indices = tensor_ops_1.tensor1d([0, 2, 4], 'int32');
        var values = tensor_ops_1.scalar(10, 'int32');
        var shape = [6];
        var result = sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue);
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(values.dtype);
        test_util_1.expectArraysClose(result, [10, 0, 10, 0, 10, 0]);
    });
    it('should work for matrix', function () {
        var indices = tensor_ops_1.tensor2d([0, 1, 1, 1], [2, 2], 'int32');
        var values = tensor_ops_1.tensor1d([5, 6], 'float32');
        var shape = [2, 2];
        var result = sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue.toFloat());
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(values.dtype);
        test_util_1.expectArraysClose(result, [0, 5, 0, 6]);
    });
    it('should throw exception if default value does not match dtype', function () {
        var indices = tensor_ops_1.tensor2d([0, 1, 1, 1], [2, 2], 'int32');
        var values = tensor_ops_1.tensor1d([5, 6], 'float32');
        var shape = [2, 2];
        expect(function () { return sparse_to_dense_1.sparseToDense(indices, values, shape, tensor_ops_1.scalar(1, 'int32')); })
            .toThrowError();
    });
    it('should allow setting default value', function () {
        var indices = tensor_ops_1.tensor2d([0, 1, 1, 1], [2, 2], 'int32');
        var values = tensor_ops_1.tensor1d([5, 6], 'float32');
        var shape = [2, 2];
        var result = sparse_to_dense_1.sparseToDense(indices, values, shape, tensor_ops_1.scalar(1));
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual(values.dtype);
        test_util_1.expectArraysClose(result, [1, 5, 1, 6]);
    });
    it('should support TensorLike inputs', function () {
        var indices = [[0, 1], [1, 1]];
        var values = [5, 6];
        var shape = [2, 2];
        var result = sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue.toFloat());
        expect(result.shape).toEqual(shape);
        expect(result.dtype).toEqual('float32');
        test_util_1.expectArraysClose(result, [0, 5, 0, 6]);
    });
    it('should throw error when indices are not int32', function () {
        var indices = tensor_ops_1.scalar(2, 'float32');
        var values = tensor_ops_1.scalar(100, 'int32');
        var shape = [6];
        expect(function () { return sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue); }).toThrow();
    });
    it('should throw error when indices rank > 2', function () {
        var indices = tensor_ops_1.tensor3d([1], [1, 1, 1], 'int32');
        var values = tensor_ops_1.tensor1d([100], 'float32');
        var shape = [6];
        expect(function () { return sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue); }).toThrow();
    });
    it('should throw error when values has rank > 1', function () {
        var indices = tensor_ops_1.tensor1d([0, 4, 2], 'int32');
        var values = tensor_ops_1.tensor2d([1.0, 2.0, 3.0], [3, 1], 'float32');
        var shape = [6];
        expect(function () { return sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue); }).toThrow();
    });
    it('should throw error when values has wrong size', function () {
        var indices = tensor_ops_1.tensor1d([0, 4, 2], 'int32');
        var values = tensor_ops_1.tensor1d([1.0, 2.0, 3.0, 4.0], 'float32');
        var shape = [6];
        expect(function () { return sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue); }).toThrow();
    });
});
jasmine_util_1.describeWithFlags('sparseToDense CPU', test_util_1.CPU_ENVS, function () {
    it('should throw error when index out of range', function () {
        var indices = tensor_ops_1.tensor1d([0, 2, 6], 'int32');
        var values = tensor_ops_1.tensor1d([100, 101, 102], 'int32');
        var shape = [6];
        expect(function () { return sparse_to_dense_1.sparseToDense(indices, values, shape, defaultValue); }).toThrow();
    });
});
//# sourceMappingURL=sparse_to_dense_test.js.map