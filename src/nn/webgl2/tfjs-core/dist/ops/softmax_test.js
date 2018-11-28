"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('softmax', test_util_1.ALL_ENVS, function () {
    it('regular test', function () {
        var y = tf.softmax(tf.tensor1d([2, 1, 3]));
        test_util_1.expectArraysClose(y, [0.24472847, 0.09003057, 0.66524095]);
        test_util_1.expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
    });
    it('overflow', function () {
        var y = tf.softmax(tf.tensor1d([100, 100]));
        test_util_1.expectArraysClose(y, [0.5, 0.5]);
    });
    it('underflow', function () {
        var y = tf.softmax(tf.tensor1d([-100, -100]));
        test_util_1.expectArraysClose(y, [0.5, 0.5]);
    });
    it('Huge difference between probabilities', function () {
        var y = tf.softmax(tf.tensor1d([-1000, +1000]));
        test_util_1.expectArraysClose(y, [0, 1]);
    });
    it('Propagates NaNs', function () {
        var a = tf.tensor1d([2, 1, NaN]);
        var y = tf.softmax(a);
        test_util_1.expectArraysClose(y, [NaN, NaN, NaN]);
    });
    it('2D, dim=1', function () {
        var y = tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 1);
        var expected = [
            0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
        ];
        expect(y.rank).toBe(2);
        test_util_1.expectArraysClose(y, expected);
    });
    it('2D, implicit dim=1', function () {
        var y = tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]));
        var expected = [
            0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
        ];
        expect(y.rank).toBe(2);
        test_util_1.expectArraysClose(y, expected);
    });
    it('2D, dim=0 throws error', function () {
        var f = function () {
            tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 0);
        };
        expect(f).toThrowError();
    });
    it('1D gradient', function () {
        var x = tf.tensor1d([10, 0, -1]);
        var y = tf.softmax(x);
        var dy = tf.tensor1d([1, 2, 3]);
        var dx = tf.grad(function (x) { return x.softmax(); })(x, dy);
        var totalSum = tf.sum(tf.mul(dy, y));
        expect(dx.shape).toEqual(x.shape);
        test_util_1.expectArraysClose(dx, [
            (dy.get(0) - totalSum.get()) * y.get(0),
            (dy.get(1) - totalSum.get()) * y.get(1),
            (dy.get(2) - totalSum.get()) * y.get(2)
        ]);
    });
    it('2D gradient', function () {
        var x = tf.tensor2d([10, 0, -1, 5, 4, 3], [2, 3]);
        var y = tf.softmax(x);
        var dy = tf.tensor2d([3, 2, 1, 1, 2, 3], [2, 3]);
        var dx = tf.grad(function (x) { return x.softmax(); })(x, dy);
        var axis = -1;
        var totalSum = tf.sum(tf.mulStrict(dy, y), axis);
        expect(dx.shape).toEqual(x.shape);
        test_util_1.expectArraysClose(dx, [
            (dy.get(0, 0) - totalSum.get(0)) * y.get(0, 0),
            (dy.get(0, 1) - totalSum.get(0)) * y.get(0, 1),
            (dy.get(0, 2) - totalSum.get(0)) * y.get(0, 2),
            (dy.get(1, 0) - totalSum.get(1)) * y.get(1, 0),
            (dy.get(1, 1) - totalSum.get(1)) * y.get(1, 1),
            (dy.get(1, 2) - totalSum.get(1)) * y.get(1, 2)
        ]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.softmax({}); })
            .toThrowError(/Argument 'logits' passed to 'softmax' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var y = tf.softmax([2, 1, 3]);
        test_util_1.expectArraysClose(y, [0.24472847, 0.09003057, 0.66524095]);
        test_util_1.expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
    });
});
jasmine_util_1.describeWithFlags('logSoftmax', test_util_1.ALL_ENVS, function () {
    it('regular test', function () {
        var y = tf.logSoftmax(tf.tensor1d([2, 1, 3]));
        test_util_1.expectArraysClose(y, [-1.407606, -2.4076061, -0.407606]);
    });
    it('Huge difference', function () {
        var y = tf.logSoftmax(tf.tensor1d([-1000, +1000]));
        test_util_1.expectArraysClose(y, [-2000, 0]);
    });
    it('Propagates NaNs', function () {
        var a = tf.tensor1d([2, 1, NaN]);
        var y = tf.logSoftmax(a);
        test_util_1.expectArraysClose(y, [NaN, NaN, NaN]);
    });
    it('2D, axis=1', function () {
        var y = tf.logSoftmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 1);
        var expected = [-1.407606, -2.4076061, -0.407606, -2.4076061, -0.4076061, -1.4076061];
        expect(y.rank).toBe(2);
        test_util_1.expectArraysClose(y, expected);
    });
    it('2D, implicit axis=1', function () {
        var y = tf.logSoftmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]));
        var expected = [-1.407606, -2.4076061, -0.407606, -2.4076061, -0.4076061, -1.4076061];
        expect(y.rank).toBe(2);
        test_util_1.expectArraysClose(y, expected);
    });
    it('1D gradient', function () {
        var x = tf.tensor1d([1, 2, 10]);
        var dy = tf.tensor1d([1, 2, 3]);
        var dx = tf.grad(function (x) { return x.logSoftmax(); })(x, dy);
        expect(dx.shape).toEqual(x.shape);
        test_util_1.expectArraysClose(dx, [0.9992599, 1.9979881, -2.9972477]);
    });
    it('2D, axis=0 throws error', function () {
        var f = function () {
            tf.logSoftmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 0);
        };
        expect(f).toThrowError();
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.logSoftmax({}); })
            .toThrowError(/Argument 'logits' passed to 'logSoftmax' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var y = tf.logSoftmax([2, 1, 3]);
        test_util_1.expectArraysClose(y, [-1.407606, -2.4076061, -0.407606]);
    });
});
//# sourceMappingURL=softmax_test.js.map