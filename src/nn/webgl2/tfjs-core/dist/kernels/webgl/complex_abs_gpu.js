"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ComplexAbsProgram = (function () {
    function ComplexAbsProgram(shape) {
        this.variableNames = ['real', 'imag'];
        this.outputShape = shape;
        this.userCode = "\n      void main() {\n        float real = getRealAtOutCoords();\n        float imag = getImagAtOutCoords();\n        vec2 v = vec2(real, imag);\n\n        setOutput(sqrt(dot(v, v)));\n      }\n    ";
    }
    return ComplexAbsProgram;
}());
exports.ComplexAbsProgram = ComplexAbsProgram;
//# sourceMappingURL=complex_abs_gpu.js.map