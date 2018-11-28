"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('confusionMatrix', test_util_1.ALL_ENVS, function () {
    it('3x3 all cases present in both labels and predictions', function () {
        var labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
        var predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
        var numClasses = 3;
        var out = tf.math.confusionMatrix(labels, predictions, numClasses);
        test_util_1.expectArraysEqual(out, tf.tensor2d([[2, 0, 0], [0, 1, 1], [0, 0, 1]], [3, 3], 'int32'));
    });
    it('float32 arguments are accepted', function () {
        var labels = tf.tensor1d([0, 1, 2, 1, 0], 'float32');
        var predictions = tf.tensor1d([0, 2, 2, 1, 0], 'float32');
        var numClasses = 3;
        var out = tf.math.confusionMatrix(labels, predictions, numClasses);
        test_util_1.expectArraysEqual(out, tf.tensor2d([[2, 0, 0], [0, 1, 1], [0, 0, 1]], [3, 3], 'int32'));
    });
    it('4x4 all cases present in labels, but not predictions', function () {
        var labels = tf.tensor1d([3, 3, 2, 2, 1, 1, 0, 0], 'int32');
        var predictions = tf.tensor1d([2, 2, 2, 2, 0, 0, 0, 0], 'int32');
        var numClasses = 4;
        var out = tf.math.confusionMatrix(labels, predictions, numClasses);
        test_util_1.expectArraysEqual(out, tf.tensor2d([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 2, 0]], [4, 4], 'int32'));
    });
    it('4x4 all cases present in predictions, but not labels', function () {
        var labels = tf.tensor1d([2, 2, 2, 2, 0, 0, 0, 0], 'int32');
        var predictions = tf.tensor1d([3, 3, 2, 2, 1, 1, 0, 0], 'int32');
        var numClasses = 4;
        var out = tf.math.confusionMatrix(labels, predictions, numClasses);
        test_util_1.expectArraysEqual(out, tf.tensor2d([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 2, 2], [0, 0, 0, 0]], [4, 4], 'int32'));
    });
    it('Plain arrays as inputs', function () {
        var labels = [3, 3, 2, 2, 1, 1, 0, 0];
        var predictions = [2, 2, 2, 2, 0, 0, 0, 0];
        var numClasses = 4;
        var out = tf.math.confusionMatrix(labels, predictions, numClasses);
        test_util_1.expectArraysEqual(out, tf.tensor2d([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 2, 0]], [4, 4], 'int32'));
    });
    it('Int32Arrays as inputs', function () {
        var labels = new Int32Array([3, 3, 2, 2, 1, 1, 0, 0]);
        var predictions = new Int32Array([2, 2, 2, 2, 0, 0, 0, 0]);
        var numClasses = 4;
        var out = tf.math.confusionMatrix(labels, predictions, numClasses);
        test_util_1.expectArraysEqual(out, tf.tensor2d([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 2, 0]], [4, 4], 'int32'));
    });
    it('5x5 predictions and labels both missing some cases', function () {
        var labels = tf.tensor1d([0, 4], 'int32');
        var predictions = tf.tensor1d([4, 0], 'int32');
        var numClasses = 5;
        var out = tf.math.confusionMatrix(labels, predictions, numClasses);
        test_util_1.expectArraysEqual(out, tf.tensor2d([
            [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]
        ], [5, 5], 'int32'));
    });
    it('Invalid numClasses leads to Error', function () {
        expect(function () { return tf.math.confusionMatrix(tf.tensor1d([0, 1]), tf.tensor1d([1, 0]), 2.5); })
            .toThrowError(/numClasses .* positive integer.* got 2\.5/);
    });
    it('Incorrect tensor rank leads to Error', function () {
        expect(function () { return tf.math.confusionMatrix(tf.scalar(0), tf.scalar(0), 1); })
            .toThrowError(/rank .* 1.*got 0/);
        expect(function () {
            return tf.math.confusionMatrix(tf.zeros([3, 3]), tf.zeros([9]), 2);
        })
            .toThrowError(/rank .* 1.*got 2/);
        expect(function () {
            return tf.math.confusionMatrix(tf.zeros([9]), tf.zeros([3, 3]), 2);
        })
            .toThrowError(/rank .* 1.*got 2/);
    });
    it('Mismatch in lengths leads to Error', function () {
        expect(function () { return tf.math.confusionMatrix(tf.zeros([3]), tf.zeros([9]), 2); })
            .toThrowError(/Mismatch .* 3 vs.* 9/);
    });
});
//# sourceMappingURL=confusion_matrix_test.js.map