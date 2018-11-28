"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var shader_util = require("./shader_compiler_util");
var ReshapePackedProgram = (function () {
    function ReshapePackedProgram(outputShape, inputShape) {
        this.variableNames = ['A'];
        this.usesPackedTextures = true;
        this.outputShape = outputShape;
        var mainLoop = "";
        for (var i = 0; i < 4; i++) {
            var thisRC = "thisRC = rc;";
            if (i % 2 === 1) {
                thisRC += "thisRC.z += 1;";
            }
            if (i > 1) {
                thisRC += "thisRC.y += 1;";
            }
            mainLoop += "\n        " + thisRC + "\n        " + (i > 0 ? "if(thisRC.y < rows && thisRC.z < cols){" : '') + "\n          int flatIndex = getFlatIndex(thisRC);\n\n          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);\n          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));\n\n          result[" + i + "] =\n            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);\n        " + (i > 0 ? '}' : '') + "\n      ";
        }
        this.userCode = "\n      " + getReshapedInputCoords(inputShape) + "\n      " + getFlatIndex(outputShape) + "\n\n      void main() {\n        ivec3 rc = getOutputCoords();\n\n        vec4 result = vec4(0.);\n\n        ivec3 thisRC;\n        int rows = " + outputShape[1] + ";\n        int cols = " + outputShape[2] + ";\n\n        " + mainLoop + "\n\n        setOutput(result);\n      }\n    ";
    }
    return ReshapePackedProgram;
}());
exports.ReshapePackedProgram = ReshapePackedProgram;
function getFlatIndex(shape) {
    var dotCoordsWithStrides = shader_util.dotify(['coords.x', 'coords.y', 'coords.z'], util.computeStrides(shape).map(function (d) { return d.toString(); }).concat(['1.']));
    return "\n    int getFlatIndex(ivec3 coords) {\n      return round(" + dotCoordsWithStrides + ");\n    }\n  ";
}
function getReshapedInputCoords(shape) {
    var coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
    return "\n    ivec3 inputCoordsFromReshapedOutCoords(int index) {\n      " + coordsFromIndexSnippet + "\n      return ivec3(r, c, d);\n    }\n  ";
}
//# sourceMappingURL=reshape_packed_gpu.js.map