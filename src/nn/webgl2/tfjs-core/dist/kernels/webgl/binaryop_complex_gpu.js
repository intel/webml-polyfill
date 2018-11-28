"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var broadcast_util = require("../../ops/broadcast_util");
exports.COMPLEX_MULTIPLY = {
    REAL: 'return areal * breal - aimag * bimag;',
    IMAG: 'return areal * bimag + aimag * breal;'
};
var BinaryOpComplexProgram = (function () {
    function BinaryOpComplexProgram(op, aShape, bShape) {
        this.variableNames = ['AReal', 'AImag', 'BReal', 'BImag'];
        this.supportsBroadcasting = true;
        this.outputShape =
            broadcast_util.assertAndGetBroadcastShape(aShape, bShape);
        this.userCode = "\n      float binaryOpComplex(\n          float areal, float aimag, float breal, float bimag) {\n        " + op + "\n      }\n\n      void main() {\n        float areal = getARealAtOutCoords();\n        float aimag = getAImagAtOutCoords();\n        float breal = getBRealAtOutCoords();\n        float bimag = getBImagAtOutCoords();\n        setOutput(binaryOpComplex(areal, aimag, breal, bimag));\n      }\n    ";
    }
    return BinaryOpComplexProgram;
}());
exports.BinaryOpComplexProgram = BinaryOpComplexProgram;
//# sourceMappingURL=binaryop_complex_gpu.js.map