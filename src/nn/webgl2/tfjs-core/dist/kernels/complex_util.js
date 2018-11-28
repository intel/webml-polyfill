"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function mergeRealAndImagArrays(real, imag) {
    if (real.length !== imag.length) {
        throw new Error("Cannot merge real and imag arrays of different lengths. real:" +
            (real.length + ", imag: " + imag.length + "."));
    }
    var result = new Float32Array(real.length * 2);
    for (var i = 0; i < result.length; i += 2) {
        result[i] = real[i / 2];
        result[i + 1] = imag[i / 2];
    }
    return result;
}
exports.mergeRealAndImagArrays = mergeRealAndImagArrays;
function splitRealAndImagArrays(complex) {
    var real = new Float32Array(complex.length / 2);
    var imag = new Float32Array(complex.length / 2);
    for (var i = 0; i < complex.length; i += 2) {
        real[i / 2] = complex[i];
        imag[i / 2] = complex[i + 1];
    }
    return { real: real, imag: imag };
}
exports.splitRealAndImagArrays = splitRealAndImagArrays;
function complexWithEvenIndex(complex) {
    var len = Math.ceil(complex.length / 4);
    var real = new Float32Array(len);
    var imag = new Float32Array(len);
    for (var i = 0; i < complex.length; i += 4) {
        real[Math.floor(i / 4)] = complex[i];
        imag[Math.floor(i / 4)] = complex[i + 1];
    }
    return { real: real, imag: imag };
}
exports.complexWithEvenIndex = complexWithEvenIndex;
function complexWithOddIndex(complex) {
    var len = Math.floor(complex.length / 4);
    var real = new Float32Array(len);
    var imag = new Float32Array(len);
    for (var i = 2; i < complex.length; i += 4) {
        real[Math.floor(i / 4)] = complex[i];
        imag[Math.floor(i / 4)] = complex[i + 1];
    }
    return { real: real, imag: imag };
}
exports.complexWithOddIndex = complexWithOddIndex;
function getComplexWithIndex(complex, index) {
    var real = complex[index * 2];
    var imag = complex[index * 2 + 1];
    return { real: real, imag: imag };
}
exports.getComplexWithIndex = getComplexWithIndex;
function assignToTypedArray(data, real, imag, index) {
    data[index * 2] = real;
    data[index * 2 + 1] = imag;
}
exports.assignToTypedArray = assignToTypedArray;
function exponents(n, inverse) {
    var real = new Float32Array(n / 2);
    var imag = new Float32Array(n / 2);
    for (var i = 0; i < Math.ceil(n / 2); i++) {
        var x = (inverse ? 2 : -2) * Math.PI * (i / n);
        real[i] = Math.cos(x);
        imag[i] = Math.sin(x);
    }
    return { real: real, imag: imag };
}
exports.exponents = exponents;
function exponent(k, n, inverse) {
    var x = (inverse ? 2 : -2) * Math.PI * (k / n);
    var real = Math.cos(x);
    var imag = Math.sin(x);
    return { real: real, imag: imag };
}
exports.exponent = exponent;
//# sourceMappingURL=complex_util.js.map