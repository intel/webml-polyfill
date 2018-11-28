"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var shader_compiler_1 = require("./shader_compiler");
var GatherNDProgram = (function () {
    function GatherNDProgram(sliceDim, strides, shape) {
        this.sliceDim = sliceDim;
        this.strides = strides;
        this.variableNames = ['x', 'indices'];
        this.outputShape = shape;
        var stridesType = shader_compiler_1.getCoordsDataType(strides.length);
        var dtype = shader_compiler_1.getCoordsDataType(shape.length);
        var strideString = this.sliceDim > 1 ? 'strides[j]' : 'strides';
        this.userCode = "\n        " + stridesType + " strides = " + stridesType + "(" + this.strides + ");\n         void main() {\n          " + dtype + " coords = getOutputCoords();\n          int flattenIndex = 0;\n          for (int j = 0; j < " + this.sliceDim + "; j++) {\n            int index = round(getIndices(coords[0], j));\n            flattenIndex += index * " + strideString + ";\n          }\n          setOutput(getX(flattenIndex, coords[1]));\n        }\n      ";
    }
    return GatherNDProgram;
}());
exports.GatherNDProgram = GatherNDProgram;
//# sourceMappingURL=gather_nd_gpu.js.map