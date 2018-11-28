"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('conv2dTranspose', test_util_1.ALL_ENVS, function () {
    it('input=2x2x1,d2=1,f=2,s=1,p=0', function () {
        var origInputDepth = 1;
        var origOutputDepth = 1;
        var inputShape = [1, 1, origOutputDepth];
        var fSize = 2;
        var origPad = 0;
        var origStride = 1;
        var x = tf.tensor3d([2], inputShape);
        var w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        var result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
        var expected = [6, 2, 10, 0];
        expect(result.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(result, expected);
    });
    it('input=2x2x1,d2=1,f=2,s=1,p=0, batch=2', function () {
        var origInputDepth = 1;
        var origOutputDepth = 1;
        var inputShape = [2, 1, 1, origOutputDepth];
        var fSize = 2;
        var origPad = 0;
        var origStride = 1;
        var x = tf.tensor4d([2, 3], inputShape);
        var w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        var result = tf.conv2dTranspose(x, w, [2, 2, 2, 1], origStride, origPad);
        var expected = [6, 2, 10, 0, 9, 3, 15, 0];
        expect(result.shape).toEqual([2, 2, 2, 1]);
        test_util_1.expectArraysClose(result, expected);
    });
    it('gradient input=[1,3,3,1] f=[2,2,2,1] s=1 padding=valid', function () {
        var inputDepth = 1;
        var outputDepth = 2;
        var inputShape = [1, 3, 3, inputDepth];
        var filterSize = 2;
        var stride = 1;
        var pad = 'valid';
        var filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        var x = tf.tensor4d([[
                [[-0.14656299], [0.32942239], [-1.90302866]],
                [[-0.06487813], [-2.02637842], [-1.83669377]],
                [[0.82650784], [-0.89249092], [0.01207666]]
            ]], inputShape);
        var filt = tf.tensor4d([
            [[[-0.48280062], [1.26770487]], [[-0.83083738], [0.54341856]]],
            [[[-0.274904], [0.73111374]], [[2.01885189], [-2.68975237]]]
        ], filterShape);
        var grads = tf.grads(function (x, filter) {
            return tf.conv2dTranspose(x, filter, [1, 4, 4, outputDepth], stride, pad);
        });
        var dy = tf.ones([1, 4, 4, outputDepth]);
        var _a = grads([x, filt], dy), xGrad = _a[0], filtGrad = _a[1];
        test_util_1.expectArraysClose(xGrad, tf.ones([1, 3, 3, 1]).mul(tf.scalar(0.2827947)));
        test_util_1.expectArraysClose(filtGrad, tf.ones([2, 2, 2, 1]).mul(tf.scalar(-5.70202599)));
    });
    it('gradient input=[1,2,2,1] f=[2,2,2,1] s=[2,2] padding=valid', function () {
        var inputDepth = 1;
        var outputDepth = 2;
        var inputShape = [1, 2, 2, inputDepth];
        var filterSize = 2;
        var stride = [2, 2];
        var pad = 'valid';
        var filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        var x = tf.tensor4d([[[[-0.36541713], [-0.53973116]], [[0.01731674], [0.90227772]]]], inputShape);
        var filt = tf.tensor4d([
            [[[-0.01423461], [-1.00267384]], [[1.61163029], [0.66302646]]],
            [[[-0.46900087], [-0.78649444]], [[0.87780536], [-0.84551637]]]
        ], filterShape);
        var grads = tf.grads(function (x, filter) {
            return tf.conv2dTranspose(x, filter, [1, 4, 4, outputDepth], stride, pad);
        });
        var dy = tf.ones([1, 4, 4, outputDepth]).mul(tf.scalar(-1));
        var _a = grads([x, filt], dy), xGrad = _a[0], filtGrad = _a[1];
        test_util_1.expectArraysClose(xGrad, tf.ones([1, 2, 2, 1]).mul(tf.scalar(-0.03454196)));
        test_util_1.expectArraysClose(filtGrad, tf.ones([2, 2, 2, 1]).mul(tf.scalar(-0.01444618)));
    });
    it('gradient input=[1,3,3,1] f=[2,2,2,1] s=[1,1] padding=same', function () {
        var inputDepth = 1;
        var outputDepth = 2;
        var inputShape = [1, 3, 3, inputDepth];
        var filterSize = 2;
        var stride = [1, 1];
        var pad = 'same';
        var filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        var x = tf.tensor4d([[
                [[1.52433065], [-0.77053435], [-0.64562341]],
                [[0.77962889], [1.58413887], [-0.25581856]],
                [[-0.58966221], [0.05411662], [0.70749138]]
            ]], inputShape);
        var filt = tf.tensor4d([
            [[[0.11178388], [-0.96654977]], [[1.21021296], [0.84121729]]],
            [[[0.34968338], [-0.42306114]], [[1.27395733], [-1.09014535]]]
        ], filterShape);
        var grads = tf.grads(function (x, filter) {
            return tf.conv2dTranspose(x, filter, [1, 3, 3, outputDepth], stride, pad);
        });
        var dy = tf.ones([1, 3, 3, outputDepth]);
        var _a = grads([x, filt], dy), xGrad = _a[0], filtGrad = _a[1];
        test_util_1.expectArraysClose(xGrad, tf.tensor4d([[
                [[1.30709858], [1.30709858], [-0.92814366]],
                [[1.30709858], [1.30709858], [-0.92814366]],
                [[1.19666437], [1.19666437], [-0.85476589]]
            ]]));
        test_util_1.expectArraysClose(filtGrad, tf.tensor4d([
            [[[2.38806788], [2.38806788]], [[2.58201847], [2.58201847]]],
            [[[2.2161221], [2.2161221]], [[3.11756406], [3.11756406]]]
        ]));
    });
    it('gradient input=[1,2,2,2] f=[2,2,2,1] s=[2,2] padding=same', function () {
        var inputDepth = 2;
        var outputDepth = 2;
        var inputShape = [1, 2, 2, inputDepth];
        var filterSize = 2;
        var stride = [2, 2];
        var pad = 'same';
        var filterShape = [filterSize, filterSize, outputDepth, inputDepth];
        var x = tf.tensor4d([[
                [[-1.81506593, 1.00900095], [-0.05199118, 0.26311377]],
                [[-1.18469792, -0.34780521], [2.04971242, -0.65154692]]
            ]], inputShape);
        var filt = tf.tensor4d([
            [
                [[0.19529686, -0.79594708], [0.70314057, -0.06081263]],
                [[0.28724744, 0.88522715], [-0.51824096, -0.97120989]]
            ],
            [
                [[0.51872197, -1.17569193], [1.28316791, -0.81225092]],
                [[-0.44221532, 0.70058174], [-0.4849217, 0.03806348]]
            ]
        ], filterShape);
        var grads = tf.grads(function (x, filter) {
            return tf.conv2dTranspose(x, filter, [1, 3, 3, outputDepth], stride, pad);
        });
        var dy = tf.ones([1, 3, 3, outputDepth]);
        var _a = grads([x, filt], dy), xGrad = _a[0], filtGrad = _a[1];
        test_util_1.expectArraysClose(xGrad, tf.tensor4d([[
                [[1.54219678, -2.19204008], [2.70032732, -2.84470257]],
                [[0.66744391, -0.94274245], [0.89843743, -0.85675972]]
            ]]));
        test_util_1.expectArraysClose(filtGrad, tf.tensor4d([
            [
                [[-1.00204261, 0.27276259], [-1.00204261, 0.27276259]],
                [[-2.99976385, 0.66119574], [-2.99976385, 0.66119574]]
            ],
            [
                [[-1.86705711, 1.27211472], [-1.86705711, 1.27211472]],
                [[-1.81506593, 1.00900095], [-1.81506593, 1.00900095]]
            ]
        ]));
    });
    it('throws when x is not rank 3', function () {
        var origInputDepth = 1;
        var origOutputDepth = 1;
        var fSize = 2;
        var origPad = 0;
        var origStride = 1;
        var x = tf.tensor2d([2, 2], [2, 1]);
        var w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(function () { return tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad); })
            .toThrowError();
    });
    it('throws when weights is not rank 4', function () {
        var origInputDepth = 1;
        var origOutputDepth = 1;
        var inputShape = [1, 1, origOutputDepth];
        var fSize = 2;
        var origPad = 0;
        var origStride = 1;
        var x = tf.tensor3d([2], inputShape);
        var w = tf.tensor3d([3, 1, 5, 0], [fSize, fSize, origInputDepth]);
        expect(function () { return tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad); })
            .toThrowError();
    });
    it('throws when x depth does not match weights original output depth', function () {
        var origInputDepth = 1;
        var origOutputDepth = 2;
        var wrongOrigOutputDepth = 3;
        var inputShape = [1, 1, origOutputDepth];
        var fSize = 2;
        var origPad = 0;
        var origStride = 1;
        var x = tf.tensor3d([2, 2], inputShape);
        var w = tf.randomNormal([fSize, fSize, origInputDepth, wrongOrigOutputDepth]);
        expect(function () { return tf.conv2dTranspose(x, w, [2, 2, 2], origStride, origPad); })
            .toThrowError();
    });
    it('throws when passed x as a non-tensor', function () {
        var origInputDepth = 1;
        var origOutputDepth = 1;
        var fSize = 2;
        var origPad = 0;
        var origStride = 1;
        var w = tf.tensor4d([3, 1, 5, 0], [fSize, fSize, origInputDepth, origOutputDepth]);
        expect(function () { return tf.conv2dTranspose({}, w, [2, 2, 1], origStride, origPad); })
            .toThrowError(/Argument 'x' passed to 'conv2dTranspose' must be a Tensor/);
    });
    it('throws when passed filter as a non-tensor', function () {
        var origOutputDepth = 1;
        var inputShape = [1, 1, origOutputDepth];
        var origPad = 0;
        var origStride = 1;
        var x = tf.tensor3d([2], inputShape);
        expect(function () { return tf.conv2dTranspose(x, {}, [2, 2, 1], origStride, origPad); })
            .toThrowError(/Argument 'filter' passed to 'conv2dTranspose' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var origPad = 0;
        var origStride = 1;
        var x = [[[2]]];
        var w = [[[[3]], [[1]]], [[[5]], [[0]]]];
        var result = tf.conv2dTranspose(x, w, [2, 2, 1], origStride, origPad);
        var expected = [6, 2, 10, 0];
        expect(result.shape).toEqual([2, 2, 1]);
        test_util_1.expectArraysClose(result, expected);
    });
});
//# sourceMappingURL=conv2d_transpose_test.js.map