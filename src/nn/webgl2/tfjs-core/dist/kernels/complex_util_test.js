"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var complex_util = require("./complex_util");
describe('complex_util', function () {
    it('mergeRealAndImagArrays', function () {
        var real = new Float32Array([1, 2, 3]);
        var imag = new Float32Array([4, 5, 6]);
        var complex = complex_util.mergeRealAndImagArrays(real, imag);
        expect(complex).toEqual(new Float32Array([1, 4, 2, 5, 3, 6]));
    });
    it('splitRealAndImagArrays', function () {
        var complex = new Float32Array([1, 4, 2, 5, 3, 6]);
        var result = complex_util.splitRealAndImagArrays(complex);
        expect(result.real).toEqual(new Float32Array([1, 2, 3]));
        expect(result.imag).toEqual(new Float32Array([4, 5, 6]));
    });
    it('complexWithEvenIndex', function () {
        var complex = new Float32Array([1, 2, 3, 4, 5, 6]);
        var result = complex_util.complexWithEvenIndex(complex);
        expect(result.real).toEqual(new Float32Array([1, 5]));
        expect(result.imag).toEqual(new Float32Array([2, 6]));
    });
    it('complexWithOddIndex', function () {
        var complex = new Float32Array([1, 2, 3, 4, 5, 6]);
        var result = complex_util.complexWithOddIndex(complex);
        expect(result.real).toEqual(new Float32Array([3]));
        expect(result.imag).toEqual(new Float32Array([4]));
    });
});
jasmine_util_1.describeWithFlags('complex_util exponents', test_util_1.ALL_ENVS, function () {
    it('exponents inverse=false', function () {
        var inverse = false;
        var result = complex_util.exponents(5, inverse);
        test_util_1.expectArraysClose(result.real, new Float32Array([1, 0.30901700258255005]));
        test_util_1.expectArraysClose(result.imag, new Float32Array([0, -0.9510565400123596]));
    });
    it('exponents inverse=true', function () {
        var inverse = true;
        var result = complex_util.exponents(5, inverse);
        test_util_1.expectArraysClose(result.real, new Float32Array([1, 0.30901700258255005]));
        test_util_1.expectArraysClose(result.imag, new Float32Array([0, 0.9510565400123596]));
    });
});
jasmine_util_1.describeWithFlags('complex_util assignment', test_util_1.ALL_ENVS, function () {
    it('assign complex value in TypedArray', function () {
        var t = new Float32Array(4);
        complex_util.assignToTypedArray(t, 1, 2, 0);
        complex_util.assignToTypedArray(t, 3, 4, 1);
        test_util_1.expectArraysClose(t, new Float32Array([1, 2, 3, 4]));
    });
});
//# sourceMappingURL=complex_util_test.js.map