"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.COMPLEX_FFT = {
    REAL: 'return real * expR - imag * expI;',
    IMAG: 'return real * expI + imag * expR;'
};
var FFTProgram = (function () {
    function FFTProgram(op, inputShape, inverse) {
        this.variableNames = ['real', 'imag'];
        var innerDim = inputShape[1];
        this.outputShape = inputShape;
        var exponentMultiplierSnippet = inverse ? "2.0 * " + Math.PI : "-2.0 * " + Math.PI;
        var resultDenominator = inverse ? innerDim + ".0" : '1.0';
        this.userCode = "\n      const float exponentMultiplier = " + exponentMultiplierSnippet + ";\n\n      float unaryOpComplex(float real, float expR, float imag, float expI) {\n        " + op + "\n      }\n\n      float mulMatDFT(int batch, int index) {\n        float indexRatio = float(index) / float(" + innerDim + ");\n        float exponentMultiplierTimesIndexRatio =\n            exponentMultiplier * indexRatio;\n\n        float result = 0.0;\n\n        for (int i = 0; i < " + innerDim + "; i++) {\n          // x = (-2|2 * PI / N) * index * i;\n          float x = exponentMultiplierTimesIndexRatio * float(i);\n          float expR = cos(x);\n          float expI = sin(x);\n          float real = getReal(batch, i);\n          float imag = getImag(batch, i);\n\n          result +=\n              unaryOpComplex(real, expR, imag, expI) / " + resultDenominator + ";\n        }\n\n        return result;\n      }\n\n      void main() {\n        ivec2 coords = getOutputCoords();\n        setOutput(mulMatDFT(coords[0], coords[1]));\n      }\n    ";
    }
    return FFTProgram;
}());
exports.FFTProgram = FFTProgram;
//# sourceMappingURL=fft_gpu.js.map