"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var MatMulPackedProgram = (function () {
    function MatMulPackedProgram(aShape, bShape, outputShape, transposeA, transposeB) {
        if (transposeA === void 0) { transposeA = false; }
        if (transposeB === void 0) { transposeB = false; }
        this.variableNames = ['matrixA', 'matrixB'];
        this.usesPackedTextures = true;
        this.outputShape = outputShape;
        var sharedDim = transposeA ? aShape[0] : aShape[1];
        var sharedDimensionPacked = Math.ceil(sharedDim / 2);
        var aSample = transposeA ? 'i * 2, rc.x' : 'rc.x, i * 2';
        var bSample = transposeB ? 'rc.y, i * 2' : 'i * 2, rc.y';
        var aSwizzle = transposeA ? ['a.xxyy', 'a.zzww'] : ['a.xxzz', 'a.yyww'];
        var bSwizzle = transposeB ? ['b.xzxz', 'b.ywyw'] : ['b.xyxy', 'b.zwzw'];
        this.userCode = "\n      const float sharedDimension = " + sharedDimensionPacked + ".0;\n\n      vec4 dot2x2ARowBCol(ivec2 rc) {\n        vec4 result = vec4(0);\n        for (int i = 0; i < " + sharedDimensionPacked + "; i++) {\n          vec4 a = getMatrixA(" + aSample + ");\n          vec4 b = getMatrixB(" + bSample + ");\n\n          result += (" + aSwizzle[0] + " * " + bSwizzle[0] + ") + (" + aSwizzle[1] + " * " + bSwizzle[1] + ");\n        }\n        return result;\n      }\n\n      void main() {\n        ivec2 rc = getOutputCoords();\n        setOutput(dot2x2ARowBCol(rc));\n      }\n    ";
    }
    return MatMulPackedProgram;
}());
exports.MatMulPackedProgram = MatMulPackedProgram;
//# sourceMappingURL=mulmat_packed_gpu.js.map