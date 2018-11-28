"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('complex64', test_util_1.ALL_ENVS, function () {
    it('tf.complex', function () {
        var real = tf.tensor1d([3, 30]);
        var imag = tf.tensor1d([4, 40]);
        var complex = tf.complex(real, imag);
        expect(complex.dtype).toBe('complex64');
        expect(complex.shape).toEqual(real.shape);
        test_util_1.expectArraysClose(complex, [3, 4, 30, 40]);
    });
    it('tf.real', function () {
        var complex = tf.complex([3, 30], [4, 40]);
        var real = tf.real(complex);
        expect(real.dtype).toBe('float32');
        expect(real.shape).toEqual([2]);
        test_util_1.expectArraysClose(real, [3, 30]);
    });
    it('tf.imag', function () {
        var complex = tf.complex([3, 30], [4, 40]);
        var imag = tf.imag(complex);
        expect(imag.dtype).toBe('float32');
        expect(imag.shape).toEqual([2]);
        test_util_1.expectArraysClose(imag, [4, 40]);
    });
    it('throws when shapes dont match', function () {
        var real = tf.tensor1d([3, 30]);
        var imag = tf.tensor1d([4, 40, 50]);
        var re = /real and imag shapes, 2 and 3, must match in call to tf.complex\(\)/;
        expect(function () { return tf.complex(real, imag); }).toThrowError(re);
    });
});
var BYTES_PER_COMPLEX_ELEMENT = 4 * 2;
jasmine_util_1.describeWithFlags('complex64 memory', test_util_1.BROWSER_ENVS, function () {
    it('usage', function () {
        var numTensors = tf.memory().numTensors;
        var numBuffers = tf.memory().numDataBuffers;
        var startTensors = numTensors;
        var real1 = tf.tensor1d([1]);
        var imag1 = tf.tensor1d([2]);
        var complex1 = tf.complex(real1, imag1);
        expect(tf.memory().numTensors).toBe(numTensors + 5);
        expect(tf.memory().numDataBuffers).toBe(numBuffers + 3);
        numTensors = tf.memory().numTensors;
        numBuffers = tf.memory().numDataBuffers;
        var real2 = tf.tensor1d([3]);
        var imag2 = tf.tensor1d([4]);
        var complex2 = tf.complex(real2, imag2);
        expect(tf.memory().numTensors).toBe(numTensors + 5);
        expect(tf.memory().numDataBuffers).toBe(numBuffers + 3);
        numTensors = tf.memory().numTensors;
        numBuffers = tf.memory().numDataBuffers;
        var result = complex1.add(complex2);
        expect(tf.memory().numTensors).toBe(numTensors + 3);
        numTensors = tf.memory().numTensors;
        expect(result.dtype).toBe('complex64');
        expect(result.shape).toEqual([1]);
        test_util_1.expectArraysClose(result, [4, 6]);
        var real = tf.real(result);
        expect(tf.memory().numTensors).toBe(numTensors + 1);
        numTensors = tf.memory().numTensors;
        test_util_1.expectArraysClose(real, [4]);
        var imag = tf.imag(result);
        expect(tf.memory().numTensors).toBe(numTensors + 1);
        numTensors = tf.memory().numTensors;
        test_util_1.expectArraysClose(imag, [6]);
        real1.dispose();
        imag1.dispose();
        real2.dispose();
        imag2.dispose();
        complex1.dispose();
        complex2.dispose();
        result.dispose();
        real.dispose();
        imag.dispose();
        expect(tf.memory().numTensors).toBe(startTensors);
    });
    it('tf.complex disposing underlying tensors', function () {
        var numTensors = tf.memory().numTensors;
        var real = tf.tensor1d([3, 30]);
        var imag = tf.tensor1d([4, 40]);
        expect(tf.memory().numTensors).toEqual(numTensors + 2);
        var complex = tf.complex(real, imag);
        expect(tf.memory().numTensors).toEqual(numTensors + 5);
        real.dispose();
        imag.dispose();
        expect(tf.memory().numTensors).toEqual(numTensors + 3);
        expect(complex.dtype).toBe('complex64');
        expect(complex.shape).toEqual(real.shape);
        test_util_1.expectArraysClose(complex, [3, 4, 30, 40]);
        complex.dispose();
        expect(tf.memory().numTensors).toEqual(numTensors);
    });
    it('reshape', function () {
        var memoryBefore = tf.memory();
        var a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
        expect(tf.memory().numBytes)
            .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);
        var b = a.reshape([6]);
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 4);
        expect(tf.memory().numBytes)
            .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);
        expect(b.dtype).toBe('complex64');
        expect(b.shape).toEqual([6]);
        test_util_1.expectArraysClose(a.dataSync(), b.dataSync());
        b.dispose();
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
        expect(tf.memory().numBytes)
            .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);
        a.dispose();
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors);
        expect(tf.memory().numBytes).toBe(memoryBefore.numBytes);
    });
    it('clone', function () {
        var memoryBefore = tf.memory();
        var a = tf.complex([[1, 3, 5], [7, 9, 11]], [[2, 4, 6], [8, 10, 12]]);
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
        expect(tf.memory().numBytes)
            .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);
        var b = a.clone();
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 4);
        expect(tf.memory().numBytes)
            .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);
        expect(b.dtype).toBe('complex64');
        test_util_1.expectArraysClose(a, b);
        b.dispose();
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors + 3);
        expect(tf.memory().numBytes)
            .toBe(memoryBefore.numBytes + 6 * BYTES_PER_COMPLEX_ELEMENT);
        a.dispose();
        expect(tf.memory().numTensors).toBe(memoryBefore.numTensors);
        expect(tf.memory().numBytes).toBe(memoryBefore.numBytes);
    });
});
//# sourceMappingURL=complex_ops_test.js.map