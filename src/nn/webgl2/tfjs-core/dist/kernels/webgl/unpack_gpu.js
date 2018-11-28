"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var packing_util_1 = require("../packing_util");
var shader_compiler_1 = require("./shader_compiler");
var UnpackProgram = (function () {
    function UnpackProgram(outputShape) {
        this.variableNames = ['A'];
        this.usesPackedTextures = true;
        this.outputShape = outputShape;
        var rank = outputShape.length;
        var channels = packing_util_1.getChannels('rc', rank);
        var dtype = shader_compiler_1.getCoordsDataType(rank);
        var sourceCoords = packing_util_1.getSourceCoords(rank, channels);
        var innerDims = channels.slice(-2);
        var coords = rank === 1 ? 'rc' : "vec2(" + innerDims.join(',') + ")";
        this.userCode = "\n      void main() {\n        " + dtype + " rc = getOutputCoords();\n        vec4 packedInput = getA(" + sourceCoords + ");\n\n        setOutput(getChannel(packedInput, " + coords + "));\n      }\n    ";
    }
    return UnpackProgram;
}());
exports.UnpackProgram = UnpackProgram;
//# sourceMappingURL=unpack_gpu.js.map