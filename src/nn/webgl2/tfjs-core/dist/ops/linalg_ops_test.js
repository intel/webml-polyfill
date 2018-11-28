"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var ops_1 = require("./ops");
jasmine_util_1.describeWithFlags('gramSchmidt-tiny', test_util_1.ALL_ENVS, function () {
    it('2x2, Array of Tensor1D', function () {
        var xs = [
            tf.randomNormal([2], 0, 1, 'float32', 1),
            tf.randomNormal([2], 0, 1, 'float32', 2)
        ];
        var ys = tf.linalg.gramSchmidt(xs);
        var y = tf.stack(ys);
        test_util_1.expectArraysClose(y.transpose().matMul(y), tf.eye(2));
        test_util_1.expectArraysClose(tf.sum(xs[0].mul(ys[0])), tf.norm(xs[0]).mul(tf.norm(ys[0])));
    });
    it('3x3, Array of Tensor1D', function () {
        var xs = [
            tf.randomNormal([3], 0, 1, 'float32', 1),
            tf.randomNormal([3], 0, 1, 'float32', 2),
            tf.randomNormal([3], 0, 1, 'float32', 3)
        ];
        var ys = tf.linalg.gramSchmidt(xs);
        var y = tf.stack(ys);
        test_util_1.expectArraysClose(y.transpose().matMul(y), tf.eye(3));
        test_util_1.expectArraysClose(tf.sum(xs[0].mul(ys[0])), tf.norm(xs[0]).mul(tf.norm(ys[0])));
    });
    it('3x3, Matrix', function () {
        var xs = tf.randomNormal([3, 3], 0, 1, 'float32', 1);
        var y = tf.linalg.gramSchmidt(xs);
        test_util_1.expectArraysClose(y.transpose().matMul(y), tf.eye(3));
    });
    it('2x3, Matrix', function () {
        var xs = tf.randomNormal([2, 3], 0, 1, 'float32', 1);
        var y = tf.linalg.gramSchmidt(xs);
        test_util_1.expectArraysClose(y.matMul(y.transpose()), tf.eye(2));
    });
    it('3x2 Matrix throws Error', function () {
        var xs = tf.tensor2d([[1, 2], [3, -1], [5, 1]]);
        expect(function () { return tf.linalg.gramSchmidt(xs); })
            .toThrowError(/Number of vectors \(3\) exceeds number of dimensions \(2\)/);
    });
    it('Mismatching dimensions input throws Error', function () {
        var xs = [tf.tensor1d([1, 2, 3]), tf.tensor1d([-1, 5, 1]), tf.tensor1d([0, 0])];
        expect(function () { return tf.linalg.gramSchmidt(xs); }).toThrowError(/Non-unique/);
    });
    it('Empty input throws Error', function () {
        expect(function () { return tf.linalg.gramSchmidt([]); }).toThrowError(/empty/);
    });
});
jasmine_util_1.describeWithFlags('gramSchmidt-non-tiny', test_util_1.WEBGL_ENVS, function () {
    it('32x512', function () {
        var xs = tf.randomUniform([32, 512]);
        var y = tf.linalg.gramSchmidt(xs);
        test_util_1.expectArraysClose(y.matMul(y.transpose()), tf.eye(32));
    });
});
jasmine_util_1.describeWithFlags('qr', test_util_1.ALL_ENVS, function () {
    it('1x1', function () {
        var x = ops_1.tensor2d([[10]], [1, 1]);
        var _a = tf.linalg.qr(x), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor2d([[-1]], [1, 1]));
        test_util_1.expectArraysClose(r, ops_1.tensor2d([[-10]], [1, 1]));
    });
    it('2x2', function () {
        var x = ops_1.tensor2d([[1, 3], [-2, -4]], [2, 2]);
        var _a = tf.linalg.qr(x), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor2d([[-0.4472, -0.8944], [0.8944, -0.4472]], [2, 2]));
        test_util_1.expectArraysClose(r, ops_1.tensor2d([[-2.2361, -4.9193], [0, -0.8944]], [2, 2]));
    });
    it('2x2x2', function () {
        var x = ops_1.tensor3d([[[-1, -3], [2, 4]], [[1, 3], [-2, -4]]], [2, 2, 2]);
        var _a = tf.linalg.qr(x), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor3d([
            [[-0.4472, -0.8944], [0.8944, -0.4472]],
            [[-0.4472, -0.8944], [0.8944, -0.4472]]
        ], [2, 2, 2]));
        test_util_1.expectArraysClose(r, ops_1.tensor3d([
            [[2.2361, 4.9193], [0, 0.8944]],
            [[-2.2361, -4.9193], [0, -0.8944]]
        ], [2, 2, 2]));
    });
    it('2x1x2x2', function () {
        var x = ops_1.tensor4d([[[[-1, -3], [2, 4]]], [[[1, 3], [-2, -4]]]], [2, 1, 2, 2]);
        var _a = tf.linalg.qr(x), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor4d([
            [[[-0.4472, -0.8944], [0.8944, -0.4472]]],
            [[[-0.4472, -0.8944], [0.8944, -0.4472]]],
        ], [2, 1, 2, 2]));
        test_util_1.expectArraysClose(r, ops_1.tensor4d([
            [[[2.2361, 4.9193], [0, 0.8944]]],
            [[[-2.2361, -4.9193], [0, -0.8944]]]
        ], [2, 1, 2, 2]));
    });
    it('3x3', function () {
        var x = ops_1.tensor2d([[1, 3, 2], [-2, 0, 7], [8, -9, 4]], [3, 3]);
        var _a = tf.linalg.qr(x), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor2d([
            [-0.1204, 0.8729, 0.4729], [0.2408, -0.4364, 0.8669],
            [-0.9631, -0.2182, 0.1576]
        ], [3, 3]));
        test_util_1.expectArraysClose(r, ops_1.tensor2d([[-8.3066, 8.3066, -2.4077], [0, 4.5826, -2.1822], [0, 0, 7.6447]], [3, 3]));
    });
    it('3x2, fullMatrices = default false', function () {
        var x = ops_1.tensor2d([[1, 2], [3, -3], [-2, 1]], [3, 2]);
        var _a = tf.linalg.qr(x), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor2d([[-0.2673, 0.9221], [-0.8018, -0.3738], [0.5345, -0.0997]], [3, 2]));
        test_util_1.expectArraysClose(r, ops_1.tensor2d([[-3.7417, 2.4054], [0, 2.8661]], [2, 2]));
    });
    it('3x2, fullMatrices = true', function () {
        var x = ops_1.tensor2d([[1, 2], [3, -3], [-2, 1]], [3, 2]);
        var _a = tf.linalg.qr(x, true), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor2d([
            [-0.2673, 0.9221, 0.2798], [-0.8018, -0.3738, 0.4663],
            [0.5345, -0.0997, 0.8393]
        ], [3, 3]));
        test_util_1.expectArraysClose(r, ops_1.tensor2d([[-3.7417, 2.4054], [0, 2.8661], [0, 0]], [3, 2]));
    });
    it('2x3, fullMatrices = default false', function () {
        var x = ops_1.tensor2d([[1, 2, 3], [-3, -2, 1]], [2, 3]);
        var _a = tf.linalg.qr(x), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor2d([[-0.3162278, -0.9486833], [0.9486833, -0.31622773]], [2, 2]));
        test_util_1.expectArraysClose(r, ops_1.tensor2d([[-3.162, -2.5298, -2.3842e-07], [0, -1.2649, -3.162]], [2, 3]));
    });
    it('2x3, fullMatrices = true', function () {
        var x = ops_1.tensor2d([[1, 2, 3], [-3, -2, 1]], [2, 3]);
        var _a = tf.linalg.qr(x, true), q = _a[0], r = _a[1];
        test_util_1.expectArraysClose(q, ops_1.tensor2d([[-0.3162278, -0.9486833], [0.9486833, -0.31622773]], [2, 2]));
        test_util_1.expectArraysClose(r, ops_1.tensor2d([[-3.162, -2.5298, -2.3842e-07], [0, -1.2649, -3.162]], [2, 3]));
    });
    it('Does not leak memory', function () {
        var x = ops_1.tensor2d([[1, 3], [-2, -4]], [2, 2]);
        tf.linalg.qr(x);
        var numTensors = tf.memory().numTensors;
        tf.linalg.qr(x);
        expect(tf.memory().numTensors).toEqual(numTensors + 2);
    });
    it('Insuffient input tensor rank leads to error', function () {
        var x1 = ops_1.scalar(12);
        expect(function () { return tf.linalg.qr(x1); }).toThrowError(/rank >= 2.*got rank 0/);
        var x2 = ops_1.tensor1d([12]);
        expect(function () { return tf.linalg.qr(x2); }).toThrowError(/rank >= 2.*got rank 1/);
    });
});
//# sourceMappingURL=linalg_ops_test.js.map