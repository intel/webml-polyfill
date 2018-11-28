"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var LRNGradProgram = (function () {
    function LRNGradProgram(inputShape, depthRadius, bias, alpha, beta) {
        this.variableNames = ['inputImage', 'outputImage', 'dy'];
        this.outputShape = [];
        this.outputShape = inputShape;
        this.depth = inputShape[3];
        this.depthRadius = depthRadius;
        this.bias = bias;
        this.alpha = alpha;
        this.beta = beta;
        this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int r = coords[1];\n        int c = coords[2];\n\n        float result = 0.0;\n        for (int d = 0; d < " + this.depth + "; ++d) {\n          int depthBegin = int(max(0.0, float(d - " + depthRadius + ")));\n          int depthEnd = int(min(float(" + this.depth + "),\n              float(d + " + depthRadius + " + 1)));\n\n          const int MIN_DEPTH_BEGIN = 0;\n          const int MAX_DEPTH_END = " + this.depth + ";\n\n          float norm = 0.0;\n          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {\n            if (k < depthBegin){\n              continue;\n            }\n            else if (k >= depthBegin && k < depthEnd) {\n              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);\n            }\n            else {\n              break;\n            }\n          }\n\n          norm = float(" + alpha + ") * norm + float(" + bias + ");\n\n          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){\n            if (k < depthBegin){\n              continue;\n            }\n            else if (k >= depthBegin && k < depthEnd){\n              float dyi = -2.0 * float(" + alpha + ")\n                * float(" + beta + ")\n                * getInputImage(b ,r ,c, k) * getOutputImage(b, r, c, d)\n                / norm;\n              if (k == d) {\n                dyi += pow(norm, -1.0 * " + beta + ");\n              }\n              if (k == coords[3]) {\n                dyi *= getDy(b, r, c, d);\n                result += dyi;\n              }\n            }\n            else {\n              break;\n            }\n          }\n      }\n      setOutput(result);\n      }\n    ";
    }
    return LRNGradProgram;
}());
exports.LRNGradProgram = LRNGradProgram;
//# sourceMappingURL=lrn_grad_gpu.js.map