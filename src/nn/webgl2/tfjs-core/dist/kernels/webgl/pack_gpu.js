"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var packing_util_1 = require("../packing_util");
var shader_compiler_1 = require("./shader_compiler");
var PackProgram = (function () {
    function PackProgram(outputShape) {
        this.variableNames = ['A'];
        this.outputShape = outputShape;
        var rank = outputShape.length;
        var channels = packing_util_1.getChannels('rc', rank);
        var dtype = shader_compiler_1.getCoordsDataType(rank);
        var outOfBoundsCondition = getOutOfBoundsCondition(rank, outputShape, channels);
        var setup = getSetup(rank, outputShape[outputShape.length - 1], outputShape[outputShape.length - 2], channels);
        var output = getOutput(outputShape, channels);
        this.userCode = "\n      void main() {\n        " + dtype + " rc = getOutputCoords();\n\n        if(" + outOfBoundsCondition + ") {\n          outputColor = vec4(0);\n        } else {\n          " + setup + "\n\n          setOutput(vec4(" + output + "));\n        }\n      }\n    ";
    }
    return PackProgram;
}());
exports.PackProgram = PackProgram;
function getSourceCoordsArr(rank, dims) {
    var coords = [];
    for (var row = 0; row <= 1; row++) {
        for (var col = 0; col <= 1; col++) {
            var coord = (row === 0 ? 'r' : 'rp1') + ", " + (col === 0 ? 'c' : 'cp1');
            for (var d = 2; d < rank; d++) {
                coord = dims[dims.length - 1 - d] + "," + coord;
            }
            coords.push(coord);
        }
    }
    return coords;
}
function getOutOfBoundsCondition(rank, shape, dims) {
    if (rank === 1) {
        return "rc > " + shape[0];
    }
    var cond = '';
    for (var i = rank - 2; i < rank; i++) {
        cond += dims[i] + " >= " + shape[i];
        if (i < rank - 1) {
            cond += '||';
        }
    }
    return cond;
}
function getSetup(rank, cols, rows, dims) {
    if (rank === 1) {
        return '';
    }
    var innerDims = dims.slice(-2);
    return "\n    int r = " + innerDims[0] + ";\n    int c = " + innerDims[1] + ";\n    int rp1 = r + 1;\n    int cp1 = c + 1;\n\n    bool cEdge = cp1 >= " + cols + ";\n    bool rEdge = rp1 >= " + rows + ";\n  ";
}
function getOutput(shape, dims) {
    var rank = shape.length;
    var sourceCoords = getSourceCoordsArr(rank, dims);
    if (rank === 1) {
        return "getA(rc),\n            rc + 1 >= " + shape[0] + " ? 0. : getA(rc + 1),\n            0, 0";
    }
    return "getA(" + sourceCoords[0] + "),\n          cEdge ? 0. : getA(" + sourceCoords[1] + "),\n          rEdge ? 0. : getA(" + sourceCoords[2] + "),\n          rEdge || cEdge ? 0. : getA(" + sourceCoords[3] + ")";
}
//# sourceMappingURL=pack_gpu.js.map