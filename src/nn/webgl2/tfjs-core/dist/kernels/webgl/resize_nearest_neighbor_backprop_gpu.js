"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ResizeNearestNeigborBackpropProgram = (function () {
    function ResizeNearestNeigborBackpropProgram(dy, x, alignCorners) {
        this.variableNames = ['dy'];
        this.outputShape = [];
        this.outputShape = x.shape;
        var _a = x.shape, xHeight = _a[1], xWidth = _a[2];
        var _b = dy.shape, yHeight = _b[1], yWidth = _b[2];
        var effectiveXSize = [
            (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
            (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
        ];
        var effectiveYSize = [
            (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
            (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
        ];
        var heightScale = effectiveXSize[0] / effectiveYSize[0];
        var widthScale = effectiveXSize[1] / effectiveYSize[1];
        var invHeightScale = 1 / heightScale;
        var invWidthScale = 1 / widthScale;
        var winHeight = (Math.ceil(invHeightScale) * 2) + 2;
        var winWidth = (Math.ceil(invWidthScale) * 2) + 2;
        this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int b = coords[0];\n        int d = coords[3];\n        int r = coords[1];\n        int c = coords[2];\n\n        float accumulator = 0.0;\n\n        const float heightScale = float(" + heightScale + ");\n        const float widthScale = float(" + widthScale + ");\n\n        const float invHeightScale = float(" + invHeightScale + ");\n        const float invWidthScale = float(" + invWidthScale + ");\n\n        const int winHeight = int(" + winHeight + ");\n        const int winWidth = int(" + winWidth + ");\n\n        // Compute bounds for where in dy we will look\n        float startRLerp = floor(float(r) * invHeightScale);\n        int startDyR = int(floor(startRLerp - float(winHeight / 2)));\n\n        float startCLerp = floor(float(c) * invWidthScale);\n        int startDyC = int(floor(startCLerp - float(winWidth / 2)));\n\n        // Loop over dy\n        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {\n          int dyR = dyROffset + startDyR;\n\n          // Guard against the window exceeding the bounds of dy\n          if (dyR < 0 || dyR >= " + yHeight + ") {\n            continue;\n          }\n\n          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {\n            int dyC = dyCOffset + startDyC;\n\n            // Guard against the window exceeding the bounds of dy\n            if (dyC < 0 || dyC >= " + yWidth + ") {\n              continue;\n            }\n\n            float sourceFracRow =\n              float(" + effectiveXSize[0] + ") *\n                (float(dyR) / float(" + effectiveYSize[0] + "));\n\n            float sourceFracCol =\n                float(" + effectiveXSize[1] + ") *\n                  (float(dyC) / float(" + effectiveYSize[1] + "));\n\n            int sourceNearestRow = int(min(\n                float(int(" + xHeight + ") - 1),\n                " + alignCorners + " ? float(round(sourceFracRow)) :\n                                  float(floor(sourceFracRow))));\n\n            int sourceNearestCol = int(min(\n                float(int(" + xWidth + ") - 1),\n                " + alignCorners + " ? float(round(sourceFracCol)) :\n                                  float(floor(sourceFracCol))));\n\n            if (r == sourceNearestRow && c == sourceNearestCol) {\n              accumulator += getDy(b, dyR, dyC, d);\n            }\n          }\n        }\n        // End loop over dy\n\n        setOutput(accumulator);\n      }\n    ";
    }
    return ResizeNearestNeigborBackpropProgram;
}());
exports.ResizeNearestNeigborBackpropProgram = ResizeNearestNeigborBackpropProgram;
//# sourceMappingURL=resize_nearest_neighbor_backprop_gpu.js.map