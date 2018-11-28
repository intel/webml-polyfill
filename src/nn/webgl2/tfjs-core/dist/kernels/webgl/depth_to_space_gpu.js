"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var DepthToSpaceProgram = (function () {
    function DepthToSpaceProgram(outputShape, blockSize, dataFormat) {
        this.variableNames = ['x'];
        this.outputShape = [];
        this.outputShape = outputShape;
        this.blockSize = blockSize;
        this.dataFormat = dataFormat;
        this.userCode = "\n    void main() {\n      ivec4 coords = getOutputCoords();\n      int b = coords[0];\n      int h = " + this.getHeightCoordString() + ";\n      int w = " + this.getWidthCoordString() + ";\n      int d = " + this.getDepthCoordString() + ";\n\n      int in_h = h / " + blockSize + ";\n      int offset_h = imod(h, " + blockSize + ");\n      int in_w = w / " + blockSize + ";\n      int offset_w = imod(w, " + blockSize + ");\n      int offset_d = (offset_h * " + blockSize + " + offset_w) *\n        " + this.getOutputDepthSize() + ";\n      int in_d = d + offset_d;\n\n      float result = " + this.getInputSamplingString() + ";\n      setOutput(result);\n    }\n  ";
    }
    DepthToSpaceProgram.prototype.getHeightCoordString = function () {
        if (this.dataFormat === 'NHWC') {
            return "coords[1]";
        }
        else {
            return "coords[2]";
        }
    };
    DepthToSpaceProgram.prototype.getWidthCoordString = function () {
        if (this.dataFormat === 'NHWC') {
            return "coords[2]";
        }
        else {
            return "coords[3]";
        }
    };
    DepthToSpaceProgram.prototype.getDepthCoordString = function () {
        if (this.dataFormat === 'NHWC') {
            return "coords[3]";
        }
        else {
            return "coords[1]";
        }
    };
    DepthToSpaceProgram.prototype.getOutputDepthSize = function () {
        if (this.dataFormat === 'NHWC') {
            return this.outputShape[3];
        }
        else {
            return this.outputShape[1];
        }
    };
    DepthToSpaceProgram.prototype.getInputSamplingString = function () {
        if (this.dataFormat === 'NHWC') {
            return "getX(b, in_h, in_w, in_d)";
        }
        else {
            return "getX(b, in_d, in_h, in_w)";
        }
    };
    return DepthToSpaceProgram;
}());
exports.DepthToSpaceProgram = DepthToSpaceProgram;
//# sourceMappingURL=depth_to_space_gpu.js.map