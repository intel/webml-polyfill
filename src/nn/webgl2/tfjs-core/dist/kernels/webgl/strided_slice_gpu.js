"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var shader_compiler_1 = require("./shader_compiler");
var StridedSliceProgram = (function () {
    function StridedSliceProgram(begin, strides, size, shrinkAxis) {
        this.variableNames = ['x'];
        var shape = size.filter(function (v, index) { return shrinkAxis.indexOf(index) === -1; });
        this.outputShape = shape;
        var rank = size.length;
        var inputDtype = shader_compiler_1.getCoordsDataType(size.length);
        var dtype = shader_compiler_1.getCoordsDataType(shape.length);
        var newCoords = '';
        if (rank === 1) {
            newCoords = 'coords * strides + begin';
        }
        else {
            var outputAxis_1 = 0;
            newCoords =
                size.map(function (_, i) {
                    if (shrinkAxis.indexOf(i) === -1) {
                        outputAxis_1++;
                        return shape.length === 1 ?
                            "coords * strides[" + i + "] + begin[" + i + "]" :
                            "coords[" + (outputAxis_1 - 1) + "] * strides[" + i + "] + begin[" + i + "]";
                    }
                    else {
                        return "begin[" + i + "]";
                    }
                })
                    .join(',');
        }
        this.userCode = "\n      " + inputDtype + " begin = " + inputDtype + "(" + begin + ");\n      " + inputDtype + " strides = " + inputDtype + "(" + strides + ");\n\n      void main() {\n        " + dtype + " coords = getOutputCoords();\n        setOutput(getX(" + newCoords + "));\n      }\n    ";
    }
    return StridedSliceProgram;
}());
exports.StridedSliceProgram = StridedSliceProgram;
//# sourceMappingURL=strided_slice_gpu.js.map