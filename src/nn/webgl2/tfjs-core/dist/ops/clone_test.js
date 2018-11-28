"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('clone', test_util_1.ALL_ENVS, function () {
    it('returns a tensor with the same shape and value', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
        var aPrime = tf.clone(a);
        expect(aPrime.shape).toEqual(a.shape);
        test_util_1.expectArraysClose(aPrime, a);
    });
    it('accepts a tensor-like object', function () {
        var res = tf.clone([[1, 2, 3], [4, 5, 6]]);
        expect(res.dtype).toBe('float32');
        expect(res.shape).toEqual([2, 3]);
        test_util_1.expectArraysClose(res, [1, 2, 3, 4, 5, 6]);
    });
});
//# sourceMappingURL=clone_test.js.map