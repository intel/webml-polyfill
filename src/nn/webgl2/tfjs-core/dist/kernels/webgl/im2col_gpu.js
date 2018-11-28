"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Im2ColProgram = (function () {
    function Im2ColProgram(outputShape, inputShape, convInfo) {
        this.variableNames = ['A'];
        this.outputShape = outputShape;
        var filterWidth = convInfo.filterWidth, inChannels = convInfo.inChannels, strideWidth = convInfo.strideWidth, strideHeight = convInfo.strideHeight, padInfo = convInfo.padInfo, outWidth = convInfo.outWidth, dilationWidth = convInfo.dilationWidth, dilationHeight = convInfo.dilationHeight;
        var left = padInfo.left, top = padInfo.top;
        var itemsPerBlockRow = inChannels * filterWidth;
        this.userCode = "\n      void main() {\n        ivec2 rc = getOutputCoords();\n\n        vec4 result = vec4(0);\n\n        for(int row=0; row<=1; row++) {\n          for(int col=0; col<=1; col++) {\n            int blockIndex = rc.y + col;\n            int pos = rc.x + row;\n\n            if(blockIndex >= " + outputShape[1] + " || pos >= " + outputShape[0] + ") continue;\n\n            int offsetY = int(blockIndex / (" + outWidth + ")) * " + strideHeight + " - " + top + ";\n            int d0 = offsetY + " + dilationHeight + " * (pos / " + itemsPerBlockRow + ");\n\n            if(d0 >= " + inputShape[0] + " || d0 < 0) continue;\n\n            int offsetX = int(mod(float(blockIndex), " + outWidth + ".) * " + strideWidth + ". - " + left + ".);\n            int d1 = offsetX + " + dilationWidth + " * (int(mod(float(pos), " + itemsPerBlockRow + ".) / " + inChannels + ".));\n\n            if(d1 >= " + inputShape[1] + " || d1 < 0) continue;\n\n            result[row * 2 + col] = getA(d0, d1, int(mod(float(pos), " + inChannels + ".)));\n          }\n        }\n\n        outputColor = result;\n      }\n    ";
    }
    return Im2ColProgram;
}());
exports.Im2ColProgram = Im2ColProgram;
//# sourceMappingURL=im2col_gpu.js.map